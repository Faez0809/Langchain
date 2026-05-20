from langchain_community.document_loaders import WebBaseLoader

url = 'https://www.geeksforgeeks.org/python/python-programming-language-tutorial/'

loader = WebBaseLoader(url)

docs = loader.load()

print(docs[0].page_content)