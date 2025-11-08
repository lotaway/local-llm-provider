import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import PGVector
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 加载环境变量
load_dotenv()

# 配置 PostgreSQL 连接字符串
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.getenv("PGVECTOR_DRIVER", "psycopg2"),
    host=os.getenv("PGVECTOR_HOST", "localhost"),
    port=int(os.getenv("PGVECTOR_PORT", "5432")),
    database=os.getenv("PGVECTOR_DATABASE", "postgres"),
    user=os.getenv("PGVECTOR_USER", "postgres"),
    password=os.getenv("PGVECTOR_PASSWORD", "password"),
)

# 加载文档
loader = TextLoader("document.txt")
documents = loader.load()

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# 初始化嵌入模型
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 将文档存入 PGVector
vectorstore = PGVector.from_documents(
    embedding=embedding_model,
    documents=texts,
    collection_name="rag_documents",
    connection_string=CONNECTION_STRING,
)

# 初始化检索器
retriever = vectorstore.as_retriever()

# 初始化本地 LLM（示例使用 GPT-2，可替换为其他模型）
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)
llm = HuggingFacePipeline(pipeline=pipe)

# 定义生成回答的函数
def generate_response(query):
    # 检索相关文档
    docs = retriever.get_relevant_documents(query)

    # 提取内容
    context = " ".join([doc.page_content for doc in docs])

    # 构造提示
    prompt = f"基于以下上下文：{context}\n\n请回答：{query}"

    # 调用 LLM 生成回答
    response = llm(prompt)
    return response

# 示例使用
if __name__ == "__main__":
    response = generate_response("你的查询问题")
    print(response)