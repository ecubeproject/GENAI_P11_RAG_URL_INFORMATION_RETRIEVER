# URL_Info_Retriever.py
import os
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
url_file_path = "url_faiss_store_openai.pkl"

# Streamlit setup
st.set_page_config(
    page_title="AI BASED URL INFORMATION RETRIEVER",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("<h1 style='text-align: center; color: black;'>AI BASED URL INFORMATION RETRIEVER</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='text-align: center; color: black;'>Assistant Console</h3>", unsafe_allow_html=True)

# ---- URL Loading & Embedding ----
num_links = st.sidebar.slider("How many links do you want to input?", min_value=1, max_value=5, value=1)
urls = [st.sidebar.text_input(f"URL {i+1}", key=f"url{i}") for i in range(num_links)]
if urls and st.sidebar.button("Get Info From URL", key="load_urls_button"):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "."], chunk_size=1000)
    url_docs = text_splitter.split_documents(data)
    if url_docs:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        url_vectorindex_openai = FAISS.from_documents(url_docs, embeddings)
        url_vectorindex_openai.save_local("faiss_store")

# ---- Query Interface ----
llm = OpenAI(temperature=0.9, max_tokens=500, openai_api_key=OPENAI_API_KEY)
query_url = st.text_input('Ask your question about URLs:')
if query_url and st.button("Get Info From URL", key="query_url_button"):
    if os.path.exists(url_file_path):  # Ensure URL database exists
        vectorstore = FAISS.load_local("faiss_store", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query_url}, return_only_outputs=True)
        st.header("Answer based on URLs:")
        st.subheader(result['answer'])
