import bs4
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
import json
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

def fetch_page_content(main_url, db_collection_name):
    loader = RecursiveUrlLoader(main_url)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    split_docs = text_splitter.split_documents(documents=docs)

    embeds = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = QdrantVectorStore.from_documents(
        embedding=embeds,
        url = "http://localhost:6333",
        collection_name = db_collection_name,
        documents=docs
    )

    return db_collection_name



def get_mainURL_results(db_collection_name, query):
    """
    Get relevant links from the QdrantDB.
    """
    embeds = OllamaEmbeddings(model="nomic-embed-text")

    vector_db = QdrantVectorStore.from_existing_collection(
        embedding=embeds,
        url = "http://localhost:6333",
        collection_name = db_collection_name
    )

    llm = ChatOpenAI(
        model = "gpt-4.1-nano"
    )

    print(f"🙈: Searching for relevant results")
    search_result = vector_db.similarity_search(query=query)

    context = "\n\n".join([f"Page Source Link: {result.metadata["source"]} \n HTML Content: {result.page_content}" for result in search_result])

    system_prompt = f"""
    You are a smart html parser. You have are given a user query and an HTML context. 
    You have to find the relevant links to the query asked by user. You will be given with source url and the html content in the context itself.
    Always find the relevant links from the given context only. You can find links best in <a> tag

    User Query: {query}
    Context: {context}
    Main URL: "https://docs.chaicode.com/"
    db_collection_name: {db_collection_name}

    You have to give output in the specific format. Always return the output in JSON format.
    "link": "Use the relevant links here which are in the context", "db_collection_name": "whatever collection name has been provided to you."
    Never use links from other sources. Always use it from the context. But always the relevant ones.
    
    Example of output:
    Below is the output format. Since all the href links are part of Main URL you can merge like below.
    "link": "https://docs.chaicode.com/youtube/chai-aur-git/introduction/", "db_collection_name": "chai_docs"
    """

    messages = [
        ("system", system_prompt),
        ("human", query)
    ]

    result = llm.invoke(messages).content

    print(f"Result: {result}")

    json_result = json.loads(result)

    return json_result


def get_page_content(content_link, content_db, query):
    response = requests.get(content_link)
    html_content = response.text
    soup = BeautifulSoup(html_content, "html.parser")
    all_text = soup.get_text()

    print(f"Extracted the content")

    data = {
        "link": content_link,
        # "title": title,
        "text": all_text
    }

    document = Document(
        page_content = data["text"],
        metadata={
            # "title": data["title"],
            "link": data["link"]
        }
    )
    print(document)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs_splitted = text_splitter.split_documents([document])

    embeds = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = QdrantVectorStore.from_documents(
        embedding=embeds,
        url = "http://localhost:6333",
        collection_name = content_db,
        documents=docs_splitted
    )

    print(f"Stored the extracted content in vector DB")

    vector_results = QdrantVectorStore.from_existing_collection(
        embedding=embeds,
        url = "http://localhost:6333",
        collection_name=content_db
    )

    search_result = vector_results.similarity_search(query=query)

    # print(search_result)

    return search_result

def get_query_answer(search_result, query):
    context = "\n\n".join([f"Page Source Link: {result.metadata["link"]} \n HTML Content: {result.page_content}" for result in search_result])
    print("✍🏻: Preparing answer")

    system_prompt = f"""
        You are a professional documentation explainer. If a user asks you a query about something from a documentation related to any field you explain user in simplest terms.
        You are given few things below:
        - query: {query}: This is what user is asking. Always respond to user according to query.
        - context: {context}: It comprises of link and relevant answer to user query. Always use this to give an answer. Never go out of this. Always mention link of source if requried. If you never find context which is relevant to user context then deny that there is no information about the relevant query in this documentation.
        If required, always ask for a follow up question. 
    """

    llm = ChatOpenAI(
        model = "gpt-4.1-nano"
    )

    messages = [
        ("system", system_prompt),
        ("human", query)
    ]

    result = llm.invoke(messages).content
    print(result)

    return result



# print(get_mainURL_results("website_db", "What is PostgreSQL in SQL?"))



