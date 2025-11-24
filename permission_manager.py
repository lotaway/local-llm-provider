"""Permission Manager - Tool safety and permission control"""

from typing import Dict, Any, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Tool safety levels"""
    SAFE = 0  # Read-only, no side effects
    LOW = 1  # Minimal risk (e.g., simple calculations)
    MEDIUM = 2  # Moderate risk (e.g., file read)
    HIGH = 3  # High risk (e.g., file write, network requests)
    CRITICAL = 4  # Critical risk (e.g., system commands, delete operations)


class Permission:
    """Permission definition"""
    
    def __init__(
        self,
        name: str,
        safety_level: SafetyLevel,
        description: str = "",
        requires_human: bool = False
    ):
        self.name = name
        self.safety_level = safety_level
        self.description = description
        self.requires_human = requires_human


class PermissionManager:
    """Manages tool permissions and safety checks"""
    
    def __init__(self, human_approval_threshold: SafetyLevel = SafetyLevel.HIGH):
        """
        Initialize permission manager
        
        Args:
            human_approval_threshold: Safety level that requires human approval
        """
        self.human_approval_threshold = human_approval_threshold
        self.permissions: Dict[str, Permission] = {}
        self.audit_log: List[Dict[str, Any]] = []
        
        # Register default permissions
        self._register_default_permissions()
    
    def _register_default_permissions(self):
        """Register default tool permissions"""
        # LLM operations
        self.register_permission(Permission(
            name="llm.query",
            safety_level=SafetyLevel.SAFE,
            description="Direct LLM query without tools"
        ))
        
        # RAG operations
        self.register_permission(Permission(
            name="rag.query",
            safety_level=SafetyLevel.SAFE,
            description="RAG document retrieval and query"
        ))
        
        # MCP tools
        self.register_permission(Permission(
            name="mcp.web_search",
            safety_level=SafetyLevel.MEDIUM,
            description="Web search via MCP"
        ))
        
        self.register_permission(Permission(
            name="mcp.file_read",
            safety_level=SafetyLevel.MEDIUM,
            description="Read local files"
        ))
        
        self.register_permission(Permission(
            name="mcp.file_write",
            safety_level=SafetyLevel.HIGH,
            description="Write to local files",
            requires_human=True
        ))
        
        self.register_permission(Permission(
            name="mcp.file_delete",
            safety_level=SafetyLevel.CRITICAL,
            description="Delete files",
            requires_human=True
        ))
        
        self.register_permission(Permission(
            name="mcp.system_command",
            safety_level=SafetyLevel.CRITICAL,
            description="Execute system commands",
            requires_human=True
        ))
        
        self.register_permission(Permission(
            name="mcp.ocr",
            safety_level=SafetyLevel.LOW,
            description="OCR image text extraction"
        ))
        
        self.register_permission(Permission(
            name="mcp.image_recognition",
            safety_level=SafetyLevel.LOW,
            description="Image recognition and analysis"
        ))
        
        self.register_permission(Permission(
            name="mcp.audio_recognition",
            safety_level=SafetyLevel.LOW,
            description="Audio transcription"
        ))
    
    def register_permission(self, permission: Permission):
        """Register a permission"""
        self.permissions[permission.name] = permission
        logger.info(f"Registered permission: {permission.name} (Level: {permission.safety_level.name})")
    
    def check_permission(self, permission_name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check if operation is permitted
        
        Args:
            permission_name: Name of permission to check
            context: Additional context for permission check
            
        Returns:
            Dict with 'allowed', 'needs_human', 'reason'
        """
        if permission_name not in self.permissions:
            logger.warning(f"Unknown permission: {permission_name}")
            return {
                "allowed": False,
                "needs_human": True,
                "reason": f"Unknown permission: {permission_name}",
                "safety_level": "UNKNOWN"
            }
        
        permission = self.permissions[permission_name]
        
        # Check if human approval required
        needs_human = (
            permission.requires_human or
            permission.safety_level.value >= self.human_approval_threshold.value
        )
        
        # Log the check
        self.audit_log.append({
            "permission": permission_name,
            "safety_level": permission.safety_level.name,
            "needs_human": needs_human,
            "context": context or {}
        })
        
        return {
            "allowed": True,
            "needs_human": needs_human,
            "reason": permission.description,
            "safety_level": permission.safety_level.name
        }
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get audit log of permission checks"""
        return self.audit_log
    
    def clear_audit_log(self):
        """Clear audit log"""
        self.audit_log = []
    
    def test_permission(self, permission_name: str, verbose: bool = True) -> bool:
        """
        Test a permission check (for debugging and testing)
        
        Args:
            permission_name: Permission to test
            verbose: Print detailed information
            
        Returns:
            True if allowed, False otherwise
        """
        result = self.check_permission(permission_name)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Permission Test: {permission_name}")
            print(f"{'='*60}")
            print(f"Allowed: {result['allowed']}")
            print(f"Needs Human: {result['needs_human']}")
            print(f"Safety Level: {result['safety_level']}")
            print(f"Reason: {result['reason']}")
            print(f"{'='*60}\n")
        
        return result['allowed']


def test_permission_manager():
    """Test and demonstrate PermissionManager usage"""
    print("\n" + "="*60)
    print("Testing Permission Manager")
    print("="*60)
    
    # Create permission manager with HIGH threshold
    pm = PermissionManager(human_approval_threshold=SafetyLevel.HIGH)
    
    # Test various permissions
    test_cases = [
        "llm.query",           # SAFE - should not need human
        "rag.query",           # SAFE - should not need human
        "mcp.web_search",      # MEDIUM - should not need human (threshold is HIGH)
        "mcp.file_read",       # MEDIUM - should not need human
        "mcp.file_write",      # HIGH - should need human
        "mcp.file_delete",     # CRITICAL - should need human
        "mcp.system_command",  # CRITICAL - should need human
        "unknown.permission",  # UNKNOWN - should need human
    ]
    
    print("\nTesting permissions with HIGH threshold:")
    print("-" * 60)
    
    for permission in test_cases:
        result = pm.check_permission(permission)
        status = "✅ Auto-approve" if not result['needs_human'] else "⚠️  Needs human"
        print(f"{permission:25} | {result['safety_level']:10} | {status}")
    
    # Show audit log
    print("\n" + "="*60)
    print("Audit Log:")
    print("="*60)
    for i, entry in enumerate(pm.get_audit_log(), 1):
        print(f"{i}. {entry['permission']:25} | Level: {entry['safety_level']:10} | Human: {entry['needs_human']}")
    
    # Test with MEDIUM threshold
    print("\n" + "="*60)
    print("Testing with MEDIUM threshold:")
    print("="*60)
    
    pm_medium = PermissionManager(human_approval_threshold=SafetyLevel.MEDIUM)
    
    for permission in ["mcp.web_search", "mcp.file_read", "mcp.file_write"]:
        result = pm_medium.check_permission(permission)
        status = "✅ Auto-approve" if not result['needs_human'] else "⚠️  Needs human"
        print(f"{permission:25} | {result['safety_level']:10} | {status}")
    
    print("\n" + "="*60)
    print("Permission Manager Test Complete!")
    print("="*60)


if __name__ == "__main__":
    test_permission_manager()
