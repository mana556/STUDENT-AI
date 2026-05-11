from langchain.schema import BaseRetriever
print([m for m in dir(BaseRetriever) if 'relevant' in m or 'documents' in m or 'retrieve' in m])
