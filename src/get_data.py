import bs4
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
import json
from indexer import fetch_page_content, get_mainURL_results, get_page_content, get_query_answer
import requests
from bs4 import BeautifulSoup


# def get_user_query_context(result_link, result_db, query):
#   user_query_db = fetch_page_content(result_link, result_db)

#   embeds = OllamaEmbeddings(model="nomic-embed-text")

#   vector_db = QdrantVectorStore.from_existing_collection(
#     embedding=embeds,
#     url="http://localhost:6333",
#     collection_name=result_db
#   )

#   search_result = vector_db.similarity_search(query=query)
#   print(search_result)


#   # context = "\n\n".join(for result in search_result)

print("URL - https://docs.chaicode.com/")
print("DB - chaiDocs")  
query = "What is PostgreSQL in SQl?"

ingest_db = fetch_page_content("https://docs.chaicode.com/", "chaiDocs")
get_url_result = get_mainURL_results("chaiDocs", "What is PostgreSQL in SQl?")
print(type(get_url_result))

content_url = get_url_result["link"]
content_db = get_url_result["db_collection_name"]

get_query_result = get_page_content(content_url, content_db, "What is PostgreSQL in SQl?")
print(get_query_result)
get_answer = get_query_answer(get_query_result, query)
print(get_answer)