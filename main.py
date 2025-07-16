import streamlit as st 
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from src.indexer import fetch_page_content, get_mainURL_results, get_page_content, get_query_answer

st.title("Chat with Docs")
st.write("Interact with any docuementation you want.")

st.markdown(f"""Welcome to ChatDocs.
            ChatDocs is a RAG based HTML content fetcher. It fetches relvant links and gives answers according to user quer.
            It saves a lot of time and gives user direct access to search in any documentation.
            """)


input_link = st.text_input("Enter documentation link:")
link_submit = st.button("Accept")

if link_submit:
    with st.spinner("⏳ Indexing website..."):
        db_name = fetch_page_content(input_link, "test_doc") #replace with session

user_query = st.text_input("What do you want to know from this documentation?")
user_result_button = st.button("Submit")

if user_result_button:
    with st.spinner("⏳ Fetching and Indexing relevant results..."):
        get_url_result = get_mainURL_results("test_doc", user_query)
        content_url = get_url_result["link"]
        content_db = get_url_result["db_collection_name"]

    with st.spinner("✍🏻: Summing up..."):
        get_query_result = get_page_content(content_url, content_db, user_query)
        get_answer = get_query_answer(get_query_result, user_query)
    
    st.write(get_answer)




