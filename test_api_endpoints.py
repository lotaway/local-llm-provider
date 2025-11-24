"""
测试 /mcp 和 /agent 端点的区别

运行前请确保：
1. 服务器已启动: python main.py
2. 已有文档数据在 ./docs 目录
"""

import requests
import json
import time

BASE_URL = "http://localhost:11434"


def test_mcp_endpoint():
    """测试 /mcp 端点（直接 RAG）"""
    print("\n" + "="*60)
    print("测试 /mcp 端点（直接 RAG 查询）")
    print("="*60)
    
    query = "什么是机器学习？"
    print(f"\n查询: {query}")
    
    start_time = time.time()
    response = requests.get(
        f"{BASE_URL}/mcp",
        params={"query": query}
    )
    elapsed = time.time() - start_time
    
    print(f"\n响应时间: {elapsed:.2f}秒")
    print(f"状态码: {response.status_code}")
    print(f"\n回答:\n{response.text}")
    
    return response.status_code == 200


def test_agent_endpoint():
    """测试 /agent 端点（Agent 系统）"""
    print("\n" + "="*60)
    print("测试 /agent 端点（Agent 系统）")
    print("="*60)
    
    query = "分析一下机器学习的概念，并总结关键要点"
    print(f"\n查询: {query}")
    
    start_time = time.time()
    response = requests.get(
        f"{BASE_URL}/agent",
        params={"query": query}
    )
    elapsed = time.time() - start_time
    
    print(f"\n响应时间: {elapsed:.2f}秒")
    print(f"状态码: {response.status_code}")
    print(f"\n回答:\n{response.text}")
    
    return response.status_code == 200


def test_agent_chat_endpoint():
    """测试 /agent/chat 端点（OpenAI 兼容）"""
    print("\n" + "="*60)
    print("测试 /agent/chat 端点（OpenAI 兼容）")
    print("="*60)
    
    query = "简单解释一下深度学习"
    print(f"\n查询: {query}")
    
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/agent/chat",
        json={
            "model": "deepseek-r1:16b",
            "messages": [
                {"role": "user", "content": query}
            ]
        }
    )
    elapsed = time.time() - start_time
    
    print(f"\n响应时间: {elapsed:.2f}秒")
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n回答:\n{result['choices'][0]['message']['content']}")
        print(f"\nAgent 元数据:")
        print(f"  状态: {result['agent_metadata']['status']}")
        print(f"  迭代次数: {result['agent_metadata']['iterations']}")
        print(f"  历史记录长度: {result['agent_metadata']['history_length']}")
    else:
        print(f"\n错误: {response.text}")
    
    return response.status_code == 200


def test_agent_status():
    """测试 /agent/status 端点"""
    print("\n" + "="*60)
    print("测试 /agent/status 端点")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/agent/status")
    
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        status = response.json()
        print(f"\nAgent 运行时状态:")
        print(json.dumps(status, indent=2, ensure_ascii=False))
    else:
        print(f"\n错误: {response.text}")
    
    return response.status_code == 200


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("开始测试 API 端点")
    print("="*60)
    
    results = {
        "/mcp": False,
        "/agent": False,
        "/agent/chat": False,
        "/agent/status": False
    }
    
    try:
        # 测试 /mcp
        results["/mcp"] = test_mcp_endpoint()
        
        # 等待一下
        time.sleep(1)
        
        # 测试 /agent
        results["/agent"] = test_agent_endpoint()
        
        # 等待一下
        time.sleep(1)
        
        # 测试 /agent/chat
        results["/agent/chat"] = test_agent_chat_endpoint()
        
        # 等待一下
        time.sleep(1)
        
        # 测试 /agent/status
        results["/agent/status"] = test_agent_status()
        
    except requests.exceptions.ConnectionError:
        print("\n❌ 错误: 无法连接到服务器")
        print("请确保服务器已启动: python main.py")
        return False
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 打印测试结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for endpoint, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{endpoint}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✅ 所有测试通过！")
    else:
        print("❌ 部分测试失败")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
