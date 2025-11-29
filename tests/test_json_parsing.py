"""Test JSON parsing with control characters"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agents.agent_base import BaseAgent, AgentResult, AgentStatus


class MockLLM:
    """Mock LLM for testing"""
    def __init__(self):
        pass


class TestAgent(BaseAgent):
    """Test agent to access _parse_json_response"""
    def execute(self, input_data, context, stream_callback=None):
        return AgentResult(status=AgentStatus.SUCCESS, data=None)


def test_json_with_control_characters():
    """Test parsing JSON with newlines and other control characters"""
    print("=" * 60)
    print("Test: JSON Parsing with Control Characters")
    print("=" * 60)
    
    llm = MockLLM()
    agent = TestAgent(llm)
    
    # Test case 1: JSON with newlines in string value (the actual error case)
    json_with_newlines = """{
    "completed": true,
    "final_answer": "我已经收集了您需要观看的视频列表和偏好，请提供以下信息以制定详细的观看规划：

1. **观看目标**：您希望观看这些视频是为了休闲娱乐、学习知识，还是为了完成某个特定的任务？
2. **已观看的视频**：您已经看过哪些视频？（如果有的话）
3. **想观看的视频列表**：您有哪些具体的视频想看？（可以是电影、电视剧、教程、讲座等）"
}"""
    
    print("\nTest 1: JSON with newlines in string value")
    try:
        result = agent._parse_json_response(json_with_newlines)
        print(f"✓ Successfully parsed JSON with newlines")
        print(f"  completed: {result.get('completed')}")
        print(f"  final_answer length: {len(result.get('final_answer', ''))}")
        assert result.get("completed") == True
        assert "观看目标" in result.get("final_answer", "")
    except Exception as e:
        print(f"✗ Failed to parse: {e}")
        return False
    
    # Test case 2: JSON with tabs
    json_with_tabs = """{
    "plan": [
        {
            "task_id": "task_1",
            "description": "This has\ttabs\tin it"
        }
    ]
}"""
    
    print("\nTest 2: JSON with tabs in string value")
    try:
        result = agent._parse_json_response(json_with_tabs)
        print(f"✓ Successfully parsed JSON with tabs")
        assert "task_1" in str(result)
    except Exception as e:
        print(f"✗ Failed to parse: {e}")
        return False
    
    # Test case 3: JSON in markdown code block
    json_in_markdown = """Here is the result:
```json
{
    "completed": false,
    "plan": [
        {
            "task_id": "task_1",
            "description": "Test task with
multiple lines"
        }
    ]
}
```
Some other text"""
    
    print("\nTest 3: JSON in markdown code block with newlines")
    try:
        result = agent._parse_json_response(json_in_markdown)
        print(f"✓ Successfully parsed JSON from markdown")
        assert result.get("completed") == False
        assert len(result.get("plan", [])) > 0
    except Exception as e:
        print(f"✗ Failed to parse: {e}")
        return False
    
    # Test case 4: Normal JSON (should still work)
    normal_json = """{
    "completed": true,
    "final_answer": "This is a normal answer without control characters"
}"""
    
    print("\nTest 4: Normal JSON without control characters")
    try:
        result = agent._parse_json_response(normal_json)
        print(f"✓ Successfully parsed normal JSON")
        assert result.get("completed") == True
    except Exception as e:
        print(f"✗ Failed to parse: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = test_json_with_control_characters()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
