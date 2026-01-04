# encoding=utf-8
import litellm
from dotenv import load_dotenv
import os

load_dotenv()

# ========== 配置豆包信息【仅需改这2处】 ==========
DOUBAO_API_KEY = os.getenv("ARK_API_KEY")  # 替换成自己的
DOUBAO_MODEL ="openai/doubao-seed-1-6-251015"     # 可选：doubao-lite/doubao-turbo/doubao-4
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

# ========== 核心调用代码（一行不改，通用所有模型） ==========
response = litellm.completion(
    model=DOUBAO_MODEL,          # 指定豆包模型名
    messages=[
        {"role": "user", "content": "用简洁的语言介绍一下LiteLLM的核心优势"}
    ],
    api_key=DOUBAO_API_KEY,      # 豆包的API密钥
    base_url=BASE_URL,           # 豆包的兼容地址
    temperature=0.7,             # 生成随机性，0~1
    max_tokens=2048              # 最大生成token数
)

# ========== 解析返回结果 ==========
print("✅ 豆包模型返回结果：")
print(response.choices[0].message.content)