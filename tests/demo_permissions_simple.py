"""
简化版演示：Permission Manager 的使用（不需要 LLM）

展示如何：
1. 创建和配置 Permission Manager
2. 检查不同工具的权限
3. 查看审计日志
4. 调整安全阈值
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from permission_manager import PermissionManager, SafetyLevel


def demo_basic_usage():
    """基础使用演示"""
    
    print("\n" + "="*70)
    print("1. 基础使用演示")
    print("="*70)
    
    # 创建 Permission Manager（HIGH 阈值）
    pm = PermissionManager(human_approval_threshold=SafetyLevel.HIGH)
    
    print("\n创建了 Permission Manager，安全阈值: HIGH")
    print("这意味着 HIGH 和 CRITICAL 级别的操作需要人工审批\n")
    
    # 测试各种权限
    tools = [
        ("llm.query", "直接 LLM 查询"),
        ("rag.query", "RAG 文档查询"),
        ("mcp.web_search", "网络搜索"),
        ("mcp.file_read", "文件读取"),
        ("mcp.file_write", "文件写入"),
        ("mcp.file_delete", "文件删除"),
        ("mcp.system_command", "系统命令"),
    ]
    
    print("检查各种工具的权限:")
    print("-" * 70)
    print(f"{'工具':<25} {'安全级别':<12} {'需要审批':<12} {'说明'}")
    print("-" * 70)
    
    for tool, description in tools:
        result = pm.check_permission(tool)
        needs_human = "是 ⚠️" if result['needs_human'] else "否 ✅"
        print(f"{tool:<25} {result['safety_level']:<12} {needs_human:<12} {description}")


def demo_different_thresholds():
    """不同阈值的影响"""
    
    print("\n" + "="*70)
    print("2. 不同安全阈值的影响")
    print("="*70)
    
    thresholds = [
        (SafetyLevel.LOW, "LOW（最宽松）"),
        (SafetyLevel.MEDIUM, "MEDIUM（中等）"),
        (SafetyLevel.HIGH, "HIGH（严格）"),
        (SafetyLevel.CRITICAL, "CRITICAL（最严格）"),
    ]
    
    test_tool = "mcp.file_write"
    
    print(f"\n测试工具: {test_tool} (安全级别: HIGH)")
    print("-" * 70)
    print(f"{'阈值设置':<20} {'是否需要人工审批'}")
    print("-" * 70)
    
    for threshold, name in thresholds:
        pm = PermissionManager(human_approval_threshold=threshold)
        result = pm.check_permission(test_tool)
        needs_human = "是 ⚠️" if result['needs_human'] else "否 ✅"
        print(f"{name:<20} {needs_human}")


def demo_audit_log():
    """审计日志演示"""
    
    print("\n" + "="*70)
    print("3. 审计日志功能")
    print("="*70)
    
    pm = PermissionManager(human_approval_threshold=SafetyLevel.HIGH)
    
    # 执行一系列权限检查
    operations = [
        "llm.query",
        "mcp.web_search",
        "mcp.file_write",
        "mcp.file_delete",
        "rag.query",
    ]
    
    print("\n执行一系列操作...")
    for op in operations:
        pm.check_permission(op)
    
    # 显示审计日志
    print("\n审计日志:")
    print("-" * 70)
    print(f"{'序号':<6} {'权限':<25} {'安全级别':<12} {'需要审批'}")
    print("-" * 70)
    
    for i, entry in enumerate(pm.get_audit_log(), 1):
        needs_human = "是" if entry['needs_human'] else "否"
        print(f"{i:<6} {entry['permission']:<25} {entry['safety_level']:<12} {needs_human}")
    
    print(f"\n总共记录了 {len(pm.get_audit_log())} 条操作")


def demo_custom_permission():
    """自定义权限演示"""
    
    print("\n" + "="*70)
    print("4. 添加自定义权限")
    print("="*70)
    
    from permission_manager import Permission
    
    pm = PermissionManager(human_approval_threshold=SafetyLevel.HIGH)
    
    # 添加自定义权限
    custom_permissions = [
        Permission(
            name="mcp.database_query",
            safety_level=SafetyLevel.MEDIUM,
            description="数据库查询操作"
        ),
        Permission(
            name="mcp.api_call",
            safety_level=SafetyLevel.MEDIUM,
            description="外部 API 调用"
        ),
        Permission(
            name="mcp.email_send",
            safety_level=SafetyLevel.HIGH,
            description="发送邮件",
            requires_human=True
        ),
    ]
    
    print("\n注册自定义权限:")
    for perm in custom_permissions:
        pm.register_permission(perm)
        print(f"  ✓ {perm.name} (级别: {perm.safety_level.name})")
    
    print("\n检查自定义权限:")
    print("-" * 70)
    
    for perm in custom_permissions:
        result = pm.check_permission(perm.name)
        needs_human = "是 ⚠️" if result['needs_human'] else "否 ✅"
        print(f"{perm.name:<25} {result['safety_level']:<12} {needs_human}")


def demo_test_method():
    """测试方法演示"""
    
    print("\n" + "="*70)
    print("5. 使用 test_permission 方法")
    print("="*70)
    
    pm = PermissionManager(human_approval_threshold=SafetyLevel.HIGH)
    
    print("\n测试安全操作:")
    pm.test_permission("rag.query")
    
    print("\n测试高风险操作:")
    pm.test_permission("mcp.file_delete")


def main():
    """运行所有演示"""
    
    print("\n" + "="*70)
    print("Permission Manager 完整演示")
    print("="*70)
    
    demo_basic_usage()
    demo_different_thresholds()
    demo_audit_log()
    demo_custom_permission()
    demo_test_method()
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)
    
    print("\n主要功能:")
    print("  ✓ 5 个安全级别（SAFE → CRITICAL）")
    print("  ✓ 可配置的审批阈值")
    print("  ✓ 自动审计日志")
    print("  ✓ 支持自定义权限")
    print("  ✓ 测试和调试方法")
    
    print("\n使用场景:")
    print("  • MCP 工具权限控制")
    print("  • 高风险操作人工审批")
    print("  • 操作审计和追踪")
    print("  • 安全策略管理")


if __name__ == "__main__":
    main()
