import sys
import os

# Add current directory to path so imports work
sys.path.append(os.getcwd())

from rag import LocalRAG

class DummyLLModel:
    def format_messages(self, messages):
        return messages
    def chat_at_once(self, messages, **kwargs):
        return "Dummy response"
    def extract_after_think(self, text):
        return text

def test_loader():
    # Point to the directory containing the sample file
    # The user said the file is at docs/note/chatgpt.data.ex.json
    # We assume we are running from /home/wayluk/local-llm-provider/
    data_path = os.path.abspath("docs/note")
    
    print(f"Testing loader with data path: {data_path}")
    
    rag = LocalRAG(llm=DummyLLModel(), data_path=data_path)
    
    # We only want to test loading, so we call load_documents directly
    # Note: load_documents scans the directory.
    docs = rag.load_documents()
    
    print(f"\nTotal documents loaded: {len(docs)}")
    
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        print(f"Metadata: {doc.metadata}")
        print(f"Content Preview:\n{doc.page_content[:500]}...")
        print("-" * 20)

if __name__ == "__main__":
    test_loader()
