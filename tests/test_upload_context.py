import sys
import os
import json
import requests
import time

# Use requests instead of TestClient since we want to test the running server if possible,
# or we can mock it. But given the previous error, let's try to use requests against localhost
# assuming the server is running. If not, we might need to start it.

BASE_URL = "http://localhost:11434"

def test_upload_and_tool_use():
    filename = "test_secret_tool.txt"
    content = "The secret code is: NEBULA_STORM_2025"
    
    with open(filename, "w") as f:
        f.write(content)
        
    try:
        # 1. Upload the file
        print(f"Uploading {filename}...")
        with open(filename, "rb") as f:
            try:
                response = requests.post(f"{BASE_URL}/v1/upload", files={"file": f})
                response.raise_for_status()
            except requests.exceptions.ConnectionError:
                print("Error: Server not running on localhost:11434. Please start the server first.")
                return
            
        result = response.json()
        print("Upload response:", result)
        
        file_id = result.get("id")
        assert file_id, "Upload response missing 'id'"
        
        # 2. Run agent with the file ID
        print(f"Running agent with file ID: {file_id}...")
        payload = {
            "model": "qwen2.5-7b-instruct",
            "messages": ["What is the secret code in the file?"],
            "files": [file_id]
        }
        
        response = requests.post(f"{BASE_URL}/v1/agents/run", json=payload, stream=True)
        response.raise_for_status()
        
        # Collect streaming output
        full_output = ""
        print("Agent Output Stream:")
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        
                        # Check for agent metadata to see tool usage
                        if "agent_metadata" in data:
                            meta = data["agent_metadata"]
                            event_type = meta.get("event_type")
                            if event_type == "agent_start":
                                print(f"\n[Agent Start] {meta.get('current_agent')}")
                            elif event_type == "agent_complete":
                                print(f"\n[Agent Complete] {meta.get('current_agent')}: {meta.get('status')}")
                                if meta.get("message"):
                                    print(f"  Message: {meta.get('message')}")
                        
                        # Check for LLM chunks
                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_output += content
                                print(content, end="", flush=True)
                                
                    except json.JSONDecodeError:
                        pass
        
        print("\n\nFull output:", full_output)
        
        # 3. Verify the secret is in the output
        if "NEBULA_STORM_2025" in full_output:
            print("\nSUCCESS: Secret code found in agent response!")
        else:
            print("\nFAILURE: Secret code NOT found in agent response.")
            
    finally:
        # Cleanup
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    test_upload_and_tool_use()
