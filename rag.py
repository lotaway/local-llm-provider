import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

def load_docs(path):
    docs = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".txt") or f.endswith(".md"):
                loader = TextLoader(os.path.join(root, f), encoding="utf-8")
                docs.extend(loader.load())
    return docs

def build_vectorstore(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_args={"host": "localhost", "port": "19530"},
        collection_name="rag_docs"
    )
    return vectorstore

def get_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def create_qa_chain(retriever):
    llm = HuggingFacePipeline.from_model_id(model_id="tiiuae/falcon-7b-instruct", task="text-generation")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

def main():
    data_path = "./docs"
    milvus_host = "localhost"
    milvus_port = "19530"
    collection_name = "rag_docs"

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"host": milvus_host, "port": milvus_port},
        collection_name=collection_name
    )

    if not vectorstore.col_exists(collection_name):
        docs = load_docs(data_path)
        vectorstore = build_vectorstore(docs)

    retriever = get_retriever(vectorstore)
    qa_chain = create_qa_chain(retriever)

    while True:
        query = input("\n问：")
        if query.lower() in ["exit", "quit"]:
            break
        print("答：", qa_chain.run(query))
