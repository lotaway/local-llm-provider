import os
from pymilvus import connections, utility
import gc
import torch
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
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence
from model_provider import LocalLLModel
from typing import Callable, List
from ExpandedRetriever import ExpandedRetriever
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.reranker import Reranker


class LocalRAG:

    def __init__(
        self,
        llm: LocalLLModel,
        data_path="./docs",
        use_hybrid_search: bool = True,
        use_reranking: bool = True,
        retrieval_strategy: str = "adaptive"
    ):
        self.llm = llm
        self.data_path = data_path
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = os.getenv("DB_PORT", "19530")
        self.collection = os.getenv("DB_COLLECTION", "rag_docs")
        self.rag_chain: RunnableSequence | None = None
        
        # Enhanced retrieval settings
        self.use_hybrid_search = use_hybrid_search
        self.use_reranking = use_reranking
        self.retrieval_strategy = retrieval_strategy  # "adaptive", "hybrid", "vector"
        self.reranker = Reranker() if use_reranking else None
        self.all_documents: List[Document] = []

    def init_rag_chain(self):
        vectorstore = self.get_or_create_vectorstore()
        # retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        #retriever = ExpandedRetriever(vectorstore, search_kwargs={"k": 3})
        # Choose retrieval strategy
        if self.use_hybrid_search and self.all_documents:
            retriever = HybridRetriever(
                vectorstore=vectorstore,
                documents=self.all_documents,
                vector_weight=0.7,
                bm25_weight=0.3,
                k=10 if self.use_reranking else 5
            )
        else:
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 10 if self.use_reranking else 5}
            )
        # Add reranking step if enabled
        if self.use_reranking and self.reranker:
            def retrieve_and_rerank(query: str) -> List[Document]:
                docs = retriever.invoke(query)
                return self.reranker.adaptive_rerank(query, docs, top_k=5)
            
            retrieval_runnable = RunnableLambda(retrieve_and_rerank)
        else:
            retrieval_runnable = retriever
        
        format_messages_runnable = RunnableLambda(self.llm.format_messages)

        def prepare_messages(input_data) -> list[dict]:
            if hasattr(input_data, "messages"):
                return input_data.messages
            else:
                return input_data

        # Set standard RAG sampling parameters for a balance of accuracy and creativity
        chat_runnable = RunnableLambda(
            lambda x: self.llm.chat_at_once(
                prepare_messages(x),
                # do_sample=False,
                temperature=0.1,
                top_p=0.95,
                top_k=40
            )
        )
        after_runnable = RunnableLambda(self.llm.extract_after_think)

        prompt_str = ChatPromptTemplate.from_template(
            """
你是检索增强问答助手，严格基于提供的内容回答问题。

【核心规则】
1. 仅根据 context 回答，禁止推测或使用外部知识
2. 内容不足时回复：`无法从提供的内容中找到答案`
3. 保持简洁准确，结构清晰
4. context 有矛盾时，选择最明确且重复的部分
5. 保留列表、步骤等原有格式
6. 不输出 context 原文（除非问题要求引用）
7. 问题与 context 无关时回复：`该问题与提供的内容无关`

【context】
{context}

【question】
{question}

【输出要求】
有答案：直接回答，保持礼貌用词，无需开场白
无答案：使用上述规则中的标准回复
            """
        )

        self.rag_chain = (
            {"context": retrieval_runnable, "question": lambda x: x}
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

    def load_documents(self, after_doc_load: Callable[[Document, str], Document] = lambda x: x) -> list[Document]:
        """加载多种格式的文档"""
        docs = []
        print(f"开始扫描文档目录: {self.data_path}")
        
        for root, _, files in os.walk(self.data_path):
            print(f"当前目录: {root}, 文件数: {len(files)}")
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                print(f"处理文件: {file}, 扩展名: {file_ext}, 完整路径: {file_path}")

                try:
                    if file_ext in [".txt"]:
                        print(f"加载文本文件: {file}")
                        loader = TextLoader(file_path, encoding="utf-8")
                        loaded = loader.load()
                        print(f"成功加载 {len(loaded)} 个文本片段")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext in [".md", ".markdown"]:
                        print(f"加载Markdown文件: {file}")
                        loader = UnstructuredMarkdownLoader(file_path)
                        loaded = loader.load()
                        print(f"成功加载 {len(loaded)} 个Markdown片段")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext == ".pdf":
                        print(f"加载PDF文件: {file}")
                        loader = PyPDFLoader(file_path)
                        loaded = loader.load()
                        print(f"成功加载 {len(loaded)} 个PDF页面")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext in [".docx", ".doc"]:
                        print(f"加载Word文档: {file}")
                        loader = UnstructuredWordDocumentLoader(file_path)
                        loaded = loader.load()
                        print(f"成功加载 {len(loaded)} 个Word文档片段")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext in [".pptx", ".ppt"]:
                        print(f"加载PPT文件: {file}")
                        loader = UnstructuredPowerPointLoader(file_path)
                        loaded = loader.load()
                        print(f"成功加载 {len(loaded)} 个PPT幻灯片")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext in [".xlsx", ".xls"]:
                        print(f"加载Excel文件: {file}")
                        loader = UnstructuredExcelLoader(file_path)
                        loaded = loader.load()
                        print(f"成功加载 {len(loaded)} 个Excel工作表")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext == ".csv":
                        print(f"加载CSV文件: {file}")
                        loader = CSVLoader(file_path)
                        loaded = loader.load()
                        print(f"成功加载 {len(loaded)} 个CSV记录")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext == ".json":
                        print(f"加载JSON文件: {file}")
                        # JSONLoader 需要指定 jq_schema 来提取内容
                        loader = JSONLoader(
                            file_path=file_path,
                            jq_schema=".",  # 提取所有内容，根据实际结构调整
                            text_content=True,  # 确保输出是文本而不是字典
                        )
                        loaded = loader.load()
                        print(f"成功加载 {len(loaded)} 个JSON条目")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext in [".py", ".java", ".kt", ".rs", ".js", ".ts", ".html", ".css", ".cs", ".swift"]:
                        print(f"加载代码文件: {file}")
                        loader = TextLoader(file_path, encoding="utf-8")
                        loaded = loader.load()
                        print(f"成功加载 {len(loaded)} 个代码片段")
                        docs.extend(after_doc_load(loaded, file))

                    else:
                        print(f"跳过不支持的文件格式: {file}")

                except Exception as e:
                    print(f"加载文件 {file} 时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()

        print(f"成功加载 {len(docs)} 个文档片段")
        return docs

    def get_embeddings(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            encode_kwargs={"batch_size": 8}
        )
        return embeddings

    def build_vectorstore(self, docs: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        embeddings = self.get_embeddings()
        print(f"共切分为 {len(chunks)} 个文本块，开始分批向量化...")

        # 分批处理以避免显存溢出
        batch_size = 50
        vectorstore = None

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            print(f"正在处理第 {i + 1} 到 {min(i + batch_size, len(chunks))} 个文本块...")

            if vectorstore is None:
                vectorstore = Milvus.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    connection_args={"host": self.host, "port": self.port},
                    collection_name=self.collection,
                )
            else:
                vectorstore.add_documents(batch)
            
            # 清理显存
            del batch
            gc.collect()
            torch.cuda.empty_cache()
        
        return vectorstore

    def get_or_create_vectorstore(self):
        connections.connect(host=self.host, port=self.port)
        embeddings = self.get_embeddings()
        if not utility.has_collection(self.collection):
            print(
                f"Collection '{self.collection}' not found, creating and inserting documents..."
            )
            docs = self.load_documents()
            # docs = self.load_documents(lambda doc, file : doc.metadata["author"] = file)
            if not docs:
                raise ValueError(f"在路径 {self.data_path} 中没有找到任何文档")
            
            # Store documents for hybrid search
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            self.all_documents = text_splitter.split_documents(docs)
            
            return self.build_vectorstore(docs)
        else:
            print(f"Collection '{self.collection}' already exists, reusing it.")
            return Milvus(
                embedding_function=embeddings,
                connection_args={"host": self.host, "port": self.port},
                collection_name=self.collection,
            )

    def generate_answer(self, question):
        if self.rag_chain is None:
            self.init_rag_chain()
        return self.rag_chain.invoke(question)

    def release_memory(self):
        """释放显存资源"""
        if hasattr(self, "rag_chain"):
            del self.rag_chain
            self.rag_chain = None # Set to None after deletion
        gc.collect()
        torch.cuda.empty_cache()
        print("已释放 RAG 相关显存资源")


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
