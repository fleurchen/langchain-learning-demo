import os
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import OllamaLLM
from langgraph.prebuilt import chat_agent_executor


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
DATABASE = 'test_db8'
USERNAME = 'root'
PASS = '123456'
MYSQL_URI = 'mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf9mb4'.format(USERNAME,PASS,HOSTNAME,PORT,DATABASE)
db = SQLDatabase.from_uri(MYSQL_URI)

#print(db.get_usable_table_names())

toolkit = SQLDatabaseToolkit(db=db,llm=model)
tools = toolkit.get_tools()

system_prompt = """
您是一个被设计用来与SQL数据库交互的代理。
给定一个输入问题，创建一个语法正确的SQL语句并执行，然后查看查询结果并返回答案。
除非用户指定了他们想要获得的示例的具体数量，否则始终将SQL查询限制为最多10个结果。
你可以按相关列对结果进行排序，以返回MySQL数据库中最匹配的数据。
您可以使用与数据库交互的工具。在执行查询之前，你必须仔细检查。如果在执行查询时出现错误，请重写查询SQL并重试。
不要对数据库做任何DML语句(插入，更新，删除，删除等)。

首先，你应该查看数据库中的表，看看可以查询什么。
不要跳过这一步。
然后查询最相关的表的模式。
"""
system_message = SystemMessage(content=system_prompt)

agent_executor = chat_agent_executor.create_tool_calling_executor(model,tools,system_message)

result = agent_executor.invoke({'messages':[HumanMessage(content='哪个部门下员工人数最多')]})

re1 = result['messages']
print(re1[len(re1) - 1])
