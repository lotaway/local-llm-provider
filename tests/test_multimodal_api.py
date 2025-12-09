#!/usr/bin/env python3
"""
Test script for multimodal chat completions API
Demonstrates file upload and analysis with LLM
"""
import requests
import json
import sys
import os


BASE_URL = "http://localhost:11434/v1"


def upload_file(file_path: str) -> str:
    """Upload a file and return its ID"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"📤 Uploading file: {file_path}")
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/upload", files=files)
        response.raise_for_status()
        
    result = response.json()
    file_id = result['id']
    filename = result['filename']
    
    print(f"✅ File uploaded successfully!")
    print(f"   ID: {file_id}")
    print(f"   Name: {filename}")
    
    return file_id


def analyze_file(file_id: str, question: str, stream: bool = False) -> dict:
    """Analyze a file using chat completions API"""
    print(f"\n🤖 Analyzing file with question: {question}")
    
    data = {
        "model": "deepseek-r1:16b",
        "messages": [
            {"role": "user", "content": question}
        ],
        "files": [file_id],
        "stream": stream
    }
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={'Content-Type': 'application/json'},
        json=data,
        stream=stream
    )
    response.raise_for_status()
    
    if stream:
        print("\n📝 Streaming response:")
        print("-" * 60)
        full_content = ""
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str == '[DONE]':
                        break
                    
                    try:
                        chunk_data = json.loads(data_str)
                        content = chunk_data['choices'][0].get('delta', {}).get('content', '')
                        if content:
                            print(content, end='', flush=True)
                            full_content += content
                    except json.JSONDecodeError:
                        pass
        
        print("\n" + "-" * 60)
        return {"content": full_content}
    else:
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        print("\n📝 Response:")
        print("-" * 60)
        print(content)
        print("-" * 60)
        
        return result


def analyze_multiple_files(file_ids: list[str], question: str) -> dict:
    """Analyze multiple files together"""
    print(f"\n🤖 Analyzing {len(file_ids)} files with question: {question}")
    
    data = {
        "model": "deepseek-r1:16b",
        "messages": [
            {"role": "user", "content": question}
        ],
        "files": file_ids,
        "stream": False
    }
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={'Content-Type': 'application/json'},
        json=data
    )
    response.raise_for_status()
    
    result = response.json()
    content = result['choices'][0]['message']['content']
    
    print("\n📝 Response:")
    print("-" * 60)
    print(content)
    print("-" * 60)
    
    return result


def test_text_file():
    """Test with a text file"""
    print("\n" + "=" * 60)
    print("TEST 1: Analyzing a text file")
    print("=" * 60)
    
    # Create a sample text file
    test_file = "/tmp/test_document.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("""
Python Programming Best Practices

1. Write Clean Code
   - Use meaningful variable names
   - Follow PEP 8 style guide
   - Add docstrings to functions

2. Error Handling
   - Use try-except blocks
   - Provide helpful error messages
   - Log errors appropriately

3. Testing
   - Write unit tests
   - Use pytest or unittest
   - Aim for high code coverage
        """)
    
    file_id = upload_file(test_file)
    analyze_file(file_id, "请总结这个文档的主要内容，用中文回答")


def test_json_file():
    """Test with a JSON file"""
    print("\n" + "=" * 60)
    print("TEST 2: Analyzing a JSON file")
    print("=" * 60)
    
    # Create a sample JSON file
    test_file = "/tmp/test_data.json"
    data = {
        "users": [
            {"id": 1, "name": "Alice", "age": 30, "city": "Beijing"},
            {"id": 2, "name": "Bob", "age": 25, "city": "Shanghai"},
            {"id": 3, "name": "Charlie", "age": 35, "city": "Guangzhou"}
        ],
        "total": 3
    }
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    file_id = upload_file(test_file)
    analyze_file(file_id, "这个JSON文件包含什么数据？请描述其结构和内容")


def test_python_file():
    """Test with a Python source file"""
    print("\n" + "=" * 60)
    print("TEST 3: Analyzing Python source code")
    print("=" * 60)
    
    # Create a sample Python file
    test_file = "/tmp/sample_code.py"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("""
def fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number\"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    \"\"\"Calculate the factorial of n\"\"\"
    if n == 0:
        return 1
    return n * factorial(n-1)

if __name__ == "__main__":
    print(f"Fibonacci(10) = {fibonacci(10)}")
    print(f"Factorial(5) = {factorial(5)}")
        """)
    
    file_id = upload_file(test_file)
    analyze_file(file_id, "请分析这段Python代码的功能，并指出可能的性能问题")


def test_multiple_files():
    """Test with multiple files"""
    print("\n" + "=" * 60)
    print("TEST 4: Analyzing multiple files together")
    print("=" * 60)
    
    # Create multiple test files
    file1 = "/tmp/readme.md"
    with open(file1, 'w', encoding='utf-8') as f:
        f.write("# My Project\n\nThis is a sample project for testing.")
    
    file2 = "/tmp/config.json"
    with open(file2, 'w', encoding='utf-8') as f:
        json.dump({"version": "1.0.0", "name": "test-app"}, f)
    
    # Upload both files
    file_id1 = upload_file(file1)
    file_id2 = upload_file(file2)
    
    # Analyze together
    analyze_multiple_files(
        [file_id1, file_id2],
        "这两个文件分别是什么？它们可能属于同一个项目吗？"
    )


def test_streaming():
    """Test streaming response"""
    print("\n" + "=" * 60)
    print("TEST 5: Streaming analysis")
    print("=" * 60)
    
    test_file = "/tmp/story.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("""
Once upon a time, in a land far away, there lived a brave knight.
The knight embarked on a quest to find the legendary dragon.
After many adventures, the knight finally found the dragon sleeping in a cave.
        """)
    
    file_id = upload_file(test_file)
    analyze_file(file_id, "请用一句话总结这个故事", stream=True)


def main():
    """Run all tests"""
    print("🚀 Starting Multimodal Chat Completions API Tests")
    print(f"📍 Base URL: {BASE_URL}")
    
    try:
        # Run all tests
        test_text_file()
        test_json_file()
        test_python_file()
        test_multiple_files()
        test_streaming()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the API server")
        print(f"   Please make sure the server is running at {BASE_URL}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
