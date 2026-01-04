# encoding=utf-8

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json


# 写一个记账的Agent

# 载入环境变量
load_dotenv()
# 初始化LLM
llm = ChatOpenAI(
    model_name="doubao-seed-1-6-251015",
    openai_api_key=os.getenv("ARK_API_KEY"),
    openai_api_base="https://ark.cn-beijing.volces.com/api/v3",
    temperature=0.1  # RAG场景建议调低温度，保证回答精准度
)

# 计算工具
def calc_handler(request: str) -> str:
    print("\n--- 委托给计算工具 ---")
    print(f"计算工具输入：'{request}'")
    objs = json.loads(request)
    print("objs", objs)
    print("details", objs.get('details'))
    total = 0
    for obj in objs.get('details'):
        single_price = obj.get('unit_price')
        num = obj.get('nums')
        all_price = obj.get('total_price')
        if not all_price:
            total += all_price
        else:
            total += num*single_price
    return f"总花销:{total}元"


def err_handler(request: str) -> str:
    print("\n--- 错误处理 ---")
    return f"本次请求里没有对应的消费项目: '{request}'.请确认."


# 提取
extract_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        你是一个记账大师,分析用户的输入,将输入转化成json数组,包含key:unit_price(单价),nums(数量),total_price(总价)，
        输出结果为一个json格式的字符串，key:err_handle,details(上述得到json数组)
        如果提取出对应的信息，err_handle=false
        否则err_handle=true
    """),
    ("user", "{request}")])
extract_chain = extract_prompt | llm | StrOutputParser()

# 计算或者错误处理
branches = {
    "calc_handle": RunnablePassthrough.assign(output=lambda x: calc_handler(x['decision'])),
    "err_handle": RunnablePassthrough.assign(output=lambda x: err_handler(x['request'])),
}

delegation_branch = RunnableBranch(
    (lambda x: len(json.loads(x['decision']).get(
        'details')) > 0, branches["calc_handle"]),
    branches["err_handle"]
)

coordinator_agent = {
    "decision": extract_chain,  # 提取信息
    # 传递输入，相当于 "requset":{"request": request_a} ,所以上面的branches的函数都需要x['request']['request']才能获取到原始输入
    "request": RunnablePassthrough()
} | delegation_branch | (lambda x: x['output'])


def main():
    if not llm:
        print("\nSkipping execution due to LLM initialization failure.")
        return

    print("--- 只有总价 ---")
    request_a = "我买了2只铅笔花费4块，买了一个橡皮花费1块钱"
    result_a = coordinator_agent.invoke({"request": request_a})
    print(f"Final Result A: {result_a}")
    print("--- 只有单价 ---")
    request_b = "我买了2只铅笔每只花费2块，买了5个橡皮每个花费1块钱"
    result_b = coordinator_agent.invoke({"request": request_b})
    print(f"Final Result B: {result_b}")
    print("--- 异常 ---")
    request_b = "xxxxyyyy"
    result_b = coordinator_agent.invoke({"request": request_b})
    print(f"Final Result B: {result_b}")


if __name__ == "__main__":
    main()
