import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
#from langchain.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from fastapi import FastAPI
from langserve import add_routes


#from langchain_unstructured import UnstructuredLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory


# 导入pdf，进行问答分析
# 使用自行构建的deepseek-R1-8b

# 解决 Intel OpenMP 库（如 MKL、TBB）的运行时冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 设置Ollama的主机和端口（可选，如果已在环境变量中设置则不需要）
os.environ["OLLAMA_HOST"] = "127.0.0.1"
os.environ["OLLAMA_PORT"] = "11434"

# 初始化ollama
def get_chat_llm() -> OllamaLLM:
    chat_llm = OllamaLLM(
        model="deepseek-r1:8b"
    )
    return chat_llm


# 1 将文档加载至向量数据库
model = get_chat_llm()
embeddings = OllamaEmbeddings(model="nomic-embed-text")

persist_dir = 'chroma_monitor_dir'  # 存放向量数据库的目录

# 一些产品文档
pdfs = [
    "./docs/install.pdf",
    "./docs/use.pdf"
]
''' 此处为文档加载代码，在完成代码加载即完成任务了
docs = []  # document的数组
for pdf in pdfs:
    # 一个Youtube的视频对应一个document
    docs.extend(PDFPlumberLoader(pdf).load())


# 根据多个doc构建向量数据库
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=30)
split_doc = text_splitter.split_documents(docs)
# 向量数据库的持久化
vectorstore = Chroma.from_documents(split_doc, embeddings, persist_directory=persist_dir)  # 并且把向量数据库持久化到磁盘
'''

# 加载磁盘中的向量数据库
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
# 创建检索器
retriever = vectorstore.as_retriever()

# 创建一个问题的模板
system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer 
the question. If you don't know the answer, say that you 
don't know. \n

{context}
"""
prompt = ChatPromptTemplate.from_messages(  # 提问和回答的 历史记录  模板
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),  #
        ("human", "{input}"),
    ]
)

# 问答chain
chain1 = create_stuff_documents_chain(model, prompt)

# 子链的提示模板，历史记录搜素
contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, 
formulate a standalone question which can be understood 
without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

retriever_history_temp = ChatPromptTemplate.from_messages(
    [
        ('system', contextualize_q_system_prompt),
        MessagesPlaceholder('chat_history'),
        ("human", "{input}"),
    ]
)

# 创建一个子链
history_chain = create_history_aware_retriever(model, retriever, retriever_history_temp)

# 创建父链chain: 把前两个链整合
chain = create_retrieval_chain(history_chain, chain1)

# 保持问答的历史记录
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

result_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)

app = FastAPI(title='问答服务', version='V1.0', description='基于xxxx平台的产品和安装问答的服务器')

add_routes(
    app,
    result_chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8091)
