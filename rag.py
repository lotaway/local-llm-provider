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
        format_messages_runnable = RunnableLambda(self.llm.format_messages)

        def prepare_messages(input_data) -> list[dict]:
            if hasattr(input_data, "messages"):
                return input_data.messages
            else:
                return input_data

        chat_runnable = RunnableLambda(
            lambda x: self.llm.chat_at_once(prepare_messages(x))
        )
        after_runnable = RunnableLambda(self.llm.extract_after_think)

        prompt_str = ChatPromptTemplate.from_template(
            """
你是一名检索增强问答系统的回答助手。

【规则】
1. 只能根据提供的内容（context）回答问题，不允许推测、臆断或使用外部知识。
2. 如果内容不足以回答，必须回复：`无法从提供的内容中找到答案。`
3. 不要编造事实或超出 context 的信息。
4. 回答必须简洁、准确、结构清晰。
5. 如果 context 中包含多个矛盾信息，仅选择最明确且重复出现的部分作为答案。
6. 如果内容是列表、步骤、定义等，需要保留原有格式再组织语言。
7. 禁止输出 context 本身，除非问题要求引用。
8. 如果问题与 context 无关，回复：`该问题与提供的内容无关。`

【context】
{context}

【question】
{question}

【回答格式要求】
- 如果有答案：直接回答，不要多余开头词。
- 如果无法回答：使用规则中对应内容。
            """
        )

        # LCEL链

        self.rag_chain = (
            {"context": retriever, "question": lambda x: x}
            | prompt_str
            | format_messages_runnable
            | chat_runnable
            | StrOutputParser()
            | after_runnable
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

    def generate_answer(self, question):
        return self.rag_chain.invoke(question)


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
