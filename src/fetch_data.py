import bs4
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from indexer import fetch_page_content

embeds = OllamaEmbeddings(model="nomic-embed-text")

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embeds,
    url="http://localhost:6333",
    collection_name="website_db"
)

llm = ChatOllama(
    model="llama3.1:8b",
    format="json"
)

query = input(">")
search_result = vector_db.similarity_search(query=query)

context = "\n\n".join([f"Source: {result.metadata["source"]} \n HTML Content: {result.page_content}" for result in search_result])
print(f"🫣: Fetching relevant context")

system_prompt = f"""
    You are a smart html parser. You have are given a user query and an HTML context. 
    You have to find the relevant links to the query asked by user. You will be given with source url and the html content in the context itself.
    Always find the relevant links from the given context only. You can find links best in <a> tag

    User Query: {query}
    Context: {context}
    Main URL: "https://docs.chaicode.com/"

    You have to give output in the specific format. Always return the output in JSON format.
    "link": "Use the relevant links here which are in the context", "db_collection_name": "whatever collection name"
    Never use links from other sources. Always use it from the context. But always the relevant ones.
    
    Example of output:
    Below is the output format. Since all the href links are part of Main URL you can merge like below.
    "link": "https://docs.chaicode.com/youtube/chai-aur-git/introduction/", "db_collection_name": "chai_docs"
"""

print(system_prompt)
messages = [
    ("system", system_prompt),
    ("human", query)
]

result = llm.invoke(messages)

print(result.content)





