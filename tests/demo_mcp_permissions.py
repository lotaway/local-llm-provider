"""
演示 MCP Agent 和 Permission Manager 的集成使用

展示如何：
1. 注册带权限的 MCP 工具
2. 权限检查如何工作
3. 人工审批流程
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model_provider import LocalLLModel
from permission_manager import PermissionManager, SafetyLevel
from agents.task_agents.mcp_agent import MCPTaskAgent
from agents.agent_base import AgentStatus


def example_web_search_tool(query: str, task: str, context: dict) -> str:
    """示例：网络搜索工具（模拟）"""
    return f"搜索结果：关于 '{query}' 的信息..."


def example_file_write_tool(query: str, task: str, context: dict) -> str:
    """示例：文件写入工具（模拟）"""
    return f"已写入文件：{task}"


def example_system_command_tool(query: str, task: str, context: dict) -> str:
    """示例：系统命令工具（模拟）"""
    return f"执行命令：{task}"


def demo_mcp_with_permissions():
    """演示 MCP Agent 与权限管理的集成"""
    
    print("\n" + "="*70)
    print("MCP Agent 与 Permission Manager 集成演示")
    print("="*70)
    
    # 1. 初始化组件
    print("\n1. 初始化组件...")
    llm = LocalLLModel()
    pm = PermissionManager(human_approval_threshold=SafetyLevel.HIGH)
    mcp_agent = MCPTaskAgent(llm)
    mcp_agent.permission_manager = pm
    
    # 2. 注册工具（带权限）
    print("\n2. 注册 MCP 工具...")
    mcp_agent.register_tool("web_search", example_web_search_tool, "mcp.web_search")
    mcp_agent.register_tool("file_write", example_file_write_tool, "mcp.file_write")
    mcp_agent.register_tool("system_command", example_system_command_tool, "mcp.system_command")
    
    print("   ✓ 已注册 3 个工具")
    
    # 3. 测试不同安全级别的工具
    test_cases = [
        {
            "name": "网络搜索（MEDIUM - 自动批准）",
            "task": {
                "tool_name": "web_search",
                "description": "搜索机器学习相关信息"
            }
        },
        {
            "name": "文件写入（HIGH - 需要人工审批）",
            "task": {
                "tool_name": "file_write",
                "description": "写入配置文件"
            }
        },
        {
            "name": "系统命令（CRITICAL - 需要人工审批）",
            "task": {
                "tool_name": "system_command",
                "description": "执行系统更新"
            }
        }
    ]
    
    print("\n3. 测试工具执行...")
    print("-" * 70)
    
    context = {"original_query": "测试查询"}
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test_case['name']}")
        print("-" * 70)
        
        result = mcp_agent.execute(test_case['task'], context)
        
        print(f"状态: {result.status.value}")
        print(f"消息: {result.message}")
        
        if result.status == AgentStatus.SUCCESS:
            print(f"✅ 工具执行成功")
            print(f"结果: {result.data}")
        elif result.status == AgentStatus.NEEDS_HUMAN:
            print(f"⚠️  需要人工审批")
            print(f"提示: {result.data.get('prompt', 'N/A')}")
            print(f"安全级别: {result.data.get('safety_level', 'N/A')}")
        else:
            print(f"❌ 执行失败")
    
    # 4. 显示审计日志
    print("\n" + "="*70)
    print("4. 审计日志")
    print("="*70)
    
    audit_log = pm.get_audit_log()
    for i, entry in enumerate(audit_log, 1):
        status = "✅ 自动批准" if not entry['needs_human'] else "⚠️  需要人工"
        print(f"{i}. {entry['permission']:25} | {entry['safety_level']:10} | {status}")
    
    # 5. 测试不同的安全阈值
    print("\n" + "="*70)
    print("5. 测试不同安全阈值的影响")
    print("="*70)
    
    # MEDIUM 阈值 - 更严格
    print("\n使用 MEDIUM 阈值（更严格）:")
    pm_strict = PermissionManager(human_approval_threshold=SafetyLevel.MEDIUM)
    mcp_agent.permission_manager = pm_strict
    
    task = {"tool_name": "web_search", "description": "搜索"}
    result = mcp_agent.execute(task, context)
    
    if result.status == AgentStatus.NEEDS_HUMAN:
        print("  ⚠️  网络搜索现在需要人工审批（因为阈值降低到 MEDIUM）")
    else:
        print("  ✅ 网络搜索仍然自动批准")
    
    # CRITICAL 阈值 - 更宽松
    print("\n使用 CRITICAL 阈值（更宽松）:")
    pm_loose = PermissionManager(human_approval_threshold=SafetyLevel.CRITICAL)
    mcp_agent.permission_manager = pm_loose
    
    task = {"tool_name": "file_write", "description": "写入文件"}
    result = mcp_agent.execute(task, context)
    
    if result.status == AgentStatus.SUCCESS:
        print("  ✅ 文件写入现在自动批准（因为阈值提高到 CRITICAL）")
    else:
        print("  ⚠️  文件写入仍需要人工审批")
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)
    
    print("\n总结:")
    print("- MCP Agent 可以注册带权限的工具")
    print("- Permission Manager 根据安全级别自动判断是否需要人工审批")
    print("- 可以通过调整阈值来控制审批策略")
    print("- 所有权限检查都会记录到审计日志")


def demo_permission_testing():
    """演示权限测试功能"""
    
    print("\n" + "="*70)
    print("Permission Manager 测试功能演示")
    print("="*70)
    
    pm = PermissionManager(human_approval_threshold=SafetyLevel.HIGH)
    
    # 使用 test_permission 方法
    print("\n使用 test_permission 方法测试单个权限:")
    pm.test_permission("mcp.web_search")
    
    print("\n测试高风险权限:")
    pm.test_permission("mcp.file_delete")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("选择演示:")
    print("1. MCP Agent 与 Permission Manager 集成")
    print("2. Permission Manager 测试功能")
    print("3. 运行所有演示")
    print("="*70)
    
    choice = input("\n请选择 (1/2/3，默认 3): ").strip() or "3"
    
    if choice == "1":
        demo_mcp_with_permissions()
    elif choice == "2":
        demo_permission_testing()
    else:
        demo_mcp_with_permissions()
        demo_permission_testing()
