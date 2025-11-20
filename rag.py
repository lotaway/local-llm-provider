import os
from pymilvus import connections, utility
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from model_provider import LocalLLModel


class LocalRAG:

    def __init__(self, llm: LocalLLModel, data_path="./docs"):
        self.llm = llm
        self.data_path = data_path
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = os.getenv("DB_PORT", "19530")
        self.collection = os.getenv("DB_COLLECTION", "rag_docs")

        vectorstore = self.get_or_create_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # 将 LocalLLModel 的方法包装成 LangChain 兼容的 Runnable
        # generate_runnable = RunnableLambda(self.llm.generate)
        chat_runnable = RunnableLambda(self.llm.chat_at_once)

        prompt = ChatPromptTemplate.from_template(
            "根据以下内容回答问题：\n\n{context}\n\n问题：{question}"
        )

        # LCEL链
        self.rag_chain = (
            {"context": retriever, "question": lambda x: x}
            | prompt
            # | generate_runnable
            | chat_runnable
            | StrOutputParser()
        )

    def load_base_documents(self):
        docs = []
        for root, _, files in os.walk(self.data_path):
            for f in files:
                if f.endswith(".txt") or f.endswith(".md"):
                    try:
                        loader = TextLoader(os.path.join(root, f), encoding="utf-8")
                        docs.extend(loader.load())
                    except Exception as e:
                        print(f"加载文件 {f} 时出错: {e}")
        return docs

    def load_documents(self) -> list[Document]:
        """加载多种格式的文档"""
        docs = []

        for root, _, files in os.walk(self.data_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()

                try:
                    if file_ext in [".txt"]:
                        loader = TextLoader(file_path, encoding="utf-8")
                        docs.extend(loader.load())

                    elif file_ext in [".md", ".markdown"]:
                        loader = UnstructuredMarkdownLoader(file_path)
                        docs.extend(loader.load())

                    elif file_ext == ".pdf":
                        loader = PyPDFLoader(file_path)
                        docs.extend(loader.load())

                    elif file_ext in [".docx", ".doc"]:
                        loader = UnstructuredWordDocumentLoader(file_path)
                        docs.extend(loader.load())

                    elif file_ext in [".pptx", ".ppt"]:
                        loader = UnstructuredPowerPointLoader(file_path)
                        docs.extend(loader.load())

                    elif file_ext in [".xlsx", ".xls"]:
                        loader = UnstructuredExcelLoader(file_path)
                        docs.extend(loader.load())

                    elif file_ext == ".csv":
                        loader = CSVLoader(file_path)
                        docs.extend(loader.load())

                    elif file_ext == ".json":
                        # JSONLoader 需要指定 jq_schema 来提取内容
                        loader = JSONLoader(
                            file_path=file_path,
                            jq_schema=".",  # 提取所有内容，根据实际结构调整
                            text_content=True,  # 确保输出是文本而不是字典
                        )
                        docs.extend(loader.load())

                    elif file_ext in [".py", ".java", ".js", ".html", ".css"]:
                        # 代码文件作为纯文本处理
                        loader = TextLoader(file_path, encoding="utf-8")
                        docs.extend(loader.load())

                    else:
                        print(f"跳过不支持的文件格式: {file}")

                except Exception as e:
                    print(f"加载文件 {file} 时出错: {e}")

        print(f"成功加载 {len(docs)} 个文档片段")
        return docs

    def get_embeddings(self):
        if (
            not hasattr(self.llm, "embedding_model_name")
            or not self.llm.embedding_model_name
        ):
            raise ValueError("LocalLLModel 必须提供 embedding_model_name 属性")

        embeddings = HuggingFaceEmbeddings(model_name=self.llm.embedding_model_name)

        return embeddings

    def build_vectorstore(self, docs: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        embeddings = self.get_embeddings()
        vectorstore = Milvus.from_documents(
            documents=chunks,
            embedding=embeddings,
            connection_args={"host": self.host, "port": self.port},
            collection_name=self.collection,
        )
        return vectorstore

    def get_or_create_vectorstore(self):
        connections.connect(host=self.host, port=self.port)
        embeddings = self.get_embeddings()
        if not utility.has_collection(self.collection):
            print(
                f"Collection '{self.collection}' not found, creating and inserting documents..."
            )
            docs = self.load_documents()
            if not docs:
                raise ValueError(f"在路径 {self.data_path} 中没有找到任何文档")
            return self.build_vectorstore(docs)
        else:
            print(f"Collection '{self.collection}' already exists, reusing it.")
            return Milvus(
                embedding_function=embeddings,
                connection_args={"host": self.host, "port": self.port},
                collection_name=self.collection,
            )

    def invoke(self, question):
        """更方便的调用方法"""
        return self.rag_chain.invoke({"question": question})


def command_line_rag():
    local_model = LocalLLModel()
    local_rag = LocalRAG(local_model)

    print("RAG 系统已启动，输入 'exit' 或 'quit' 退出")
    while True:
        try:
            query = input("\n问：")
            if query.lower() in ["exit", "quit"]:
                break
            answer = local_rag.invoke(query)
            print("答：", answer)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    command_line_rag()
