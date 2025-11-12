import os
from pymilvus import connections, utility
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def load_docs(path):
    docs = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".txt") or f.endswith(".md"):
                loader = TextLoader(os.path.join(root, f), encoding="utf-8")
                docs.extend(loader.load())
    return docs

def build_vectorstore(docs, host="localhost", port="19530", collection="rag_docs"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_args={"host": host, "port": port},
        collection_name=collection
    )
    return vectorstore

def get_or_create_vectorstore(data_path, host="localhost", port="19530", collection="rag_docs"):
    connections.connect(host=host, port=port)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not utility.has_collection(collection):
        print(f"Collection '{collection}' not found, creating and inserting documents...")
        docs = load_docs(data_path)
        return build_vectorstore(docs, host, port, collection)
    else:
        print(f"Collection '{collection}' already exists, reusing it.")
        return Milvus(
            embedding_function=embeddings,
            connection_args={"host": host, "port": port},
            collection_name=collection
        )

def init_rag_app():
    data_path = "./docs"
    host = "localhost"
    port = "19530"
    collection = "rag_docs"

    vectorstore = get_or_create_vectorstore(data_path, host, port, collection)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = HuggingFacePipeline.from_model_id(
        model_id="tiiuae/falcon-7b-instruct",
        task="text-generation"
    )

    prompt = ChatPromptTemplate.from_template(
        "根据以下内容回答问题：\n\n{context}\n\n问题：{question}"
    )

    # 这是 LCEL 风格链：prompt → LLM → 输出解析
    rag_chain = (
        {"context": retriever, "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def command_line_rag():
    rag_chain = init_rag_app()
    while True:
        query = input("\n问：")
        if query.lower() in ["exit", "quit"]:
            break
        answer = rag_chain.invoke({"question": query})
        print("答：", answer)

if __name__ == "__main__":
    command_line_rag()