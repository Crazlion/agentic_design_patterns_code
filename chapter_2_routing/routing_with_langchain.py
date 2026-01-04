# encoding=utf-8
# from langchain_google_genai import ChatGoogleGenerativeAI 

import os
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


# 载入环境变量
load_dotenv()
# 初始化LLM，书中用的ChatGoogleGenerativeAI
llm = ChatOpenAI(
    model_name="doubao-seed-1-6-251015",
    openai_api_key=os.getenv("ARK_API_KEY"),
    openai_api_base="https://ark.cn-beijing.volces.com/api/v3",
    temperature=0.1  # RAG场景建议调低温度，保证回答精准度
)

# 定义子Agent的处理
def booking_handler(request: str) -> str: 
   """模拟预订Agent处理请求""" 
   print("\n--- 委托给预订处理程序 ---") 
   return f"Booking Handler processed request: '{request}'. Result: Simulated booking action." 

def info_handler(request: str) -> str: 
   """模拟信息Agent处理请求.""" 
   print("\n--- 委托信息处理器 ---") 
   return f"Info Handler processed request: '{request}'. Result: Simulated information retrieval." 

def unclear_handler(request: str) -> str: 
   """处理无法分配的请求""" 
   print("\n--- 处理不明确的请求 ---") 
   return f"Coordinator could not delegate request: '{request}'. Please clarify." 

# --- Define Coordinator Router Chain (equivalent to ADK coordinator's instruction) --- 
# This chain decides which handler to delegate to. 

coordinator_router_prompt = ChatPromptTemplate.from_messages([ 
    ("system", """Analyze the user's request and determine which specialist handler should process it. - If the request is related to booking flights or hotels, output 'booker'. - For all other general information questions, output 'info'. - If the request is unclear or doesn't fit either category, output 'unclear'. ONLY output one word: 'booker', 'info', or 'unclear'."""),
    ("user", "{request}") ])

if llm:
    coordinator_router_chain = coordinator_router_prompt|llm|StrOutputParser()


# 输入一个request，调用对应的函数，并将output = 函数的输出
branches = { 
   "booker": RunnablePassthrough.assign(output=lambda x: booking_handler(x['request']['request'])),  
   "info": RunnablePassthrough.assign(output=lambda x: info_handler(x['request']['request'])), 
   "unclear": RunnablePassthrough.assign(output=lambda x: unclear_handler(x['request']['request'])), 
}

delegation_branch = RunnableBranch( 
   (lambda x: x['decision'].strip() == 'booker', branches["booker"]), # Added .strip() 
   (lambda x: x['decision'].strip() == 'info', branches["info"]),     # Added .strip() 
   branches["unclear"] # Default branch for 'unclear' or any other output 
) 

#  RunnablePassthrough的使用说明 https://python.langchain.com.cn/docs/expression_language/how_to/passthrough
#
coordinator_agent = { 
   "decision": coordinator_router_chain, # LLM的意图识别
   "request": RunnablePassthrough()  # 传递输入，相当于 "requset":{"request": request_a} ,所以上面的branches的函数都需要x['request']['request']才能获取到原始输入
} | delegation_branch | (lambda x: x['output']) # 提取最后的输出

def main(): 
   if not llm: 
       print("\nSkipping execution due to LLM initialization failure.") 
       return 
 
   print("--- 正在运行预订请求 ---") 
   request_a = "Book me a flight to London." 
   result_a = coordinator_agent.invoke({"request": request_a})
   print(f"Final Result A: {result_a}") 
   print("\n--- 执行信息请求 ---") 
   request_b = "What is the capital of Italy?" 
   result_b = coordinator_agent.invoke({"request": request_b}) 
   print(f"Final Result B: {result_b}") 
   print("\n--- 执行不清晰请求 ---") 
   request_c = "xxfdjklsfjdksalufi." 
   result_c = coordinator_agent.invoke({"request": request_c}) 
   print(f"Final Result C: {result_c}") 

if __name__ == "__main__":
    main()