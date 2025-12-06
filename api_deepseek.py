# cover.py
from openai import OpenAI

# 创建DeepSeek客户端并导出
client = OpenAI(
    api_key='sk-a49001abe22b430fa441ef79316391b4',
    base_url="https://api.deepseek.com"
)

# AI助教功能函数
def ask_ai_assistant(question, context=""):
    """向AI助教提问"""
    try:
        system_prompt = f"""
        你是一个友好的机器学习助教，专门教授机器学习知识。
        你正在一个交互式学习平台中帮助学生。        
        当前上下文: {context}        
        请用中文回答学生的问题，保持专业但友好，且饱含鼓励肯定的语气。解释概念时请尽量使用比喻和直观的例子。
        回答字数控制在500字以内。
        """
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"出错了: {str(e)}。请检查您的API密钥是否正确或网络连接。"

