from langchain.chat_models import ChatOpenAI

def get_llm():
    return ChatOpenAI(temperature=0)