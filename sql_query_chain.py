import os
from operator import itemgetter

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM
from langchain_community.tools import QuerySQLDataBaseTool


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

model = get_chat_llm()
# sqlalchemy 初始化MySQL数据库连接
HOSTNAME='127.0.0.1'
PORT = '3306'
DATABASE = 'test'
USERNAME = 'root'
PASS = '123456'
MYSQL_URI = 'mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME,PASS,HOSTNAME,PORT,DATABASE)
db = SQLDatabase.from_uri(MYSQL_URI)

print(db.get_usable_table_names())


test_chain = create_sql_query_chain(model,db)

answer_prompt = PromptTemplate.from_template(
    """给定以下用户问题、SQL语句和SQL执行后的结果，回答用户问题。
    Question：{question}
    SQL Query:{query}
    SQL Result: {result}
    回答："""
)
# 创建一个执行SQL语句的工具
execute_sql_tool = QuerySQLDataBaseTool(db=db)

# 生成sql，执行sql
chain = (RunnablePassthrough.assign(query=test_chain).assign(result=itemgetter('query')|execute_sql_tool)
         |answer_prompt|model|StrOutputParser())

resp = chain.invoke(input={'question':'请问：员工表有多少条数据'})
print(resp)
