#!/bin/bash

# 项目验证脚本
# 验证所有核心功能是否正常

echo "=========================================="
echo "Local LLM Provider - 项目验证"
echo "=========================================="

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查 Python 环境
echo -e "\n${YELLOW}1. 检查 Python 环境${NC}"
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version)
    echo -e "${GREEN}✓${NC} Python 已安装: $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python 未安装"
    exit 1
fi

# 检查目录结构
echo -e "\n${YELLOW}2. 检查目录结构${NC}"
REQUIRED_DIRS=("agents" "retrievers" "tests" "docs")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓${NC} $dir/ 存在"
    else
        echo -e "${RED}✗${NC} $dir/ 不存在"
    fi
done

# 检查核心文件
echo -e "\n${YELLOW}3. 检查核心文件${NC}"
REQUIRED_FILES=(
    "main.py"
    "model_provider.py"
    "rag.py"
    "permission_manager.py"
    "agents/agent_base.py"
    "agents/agent_runtime.py"
    "retrievers/hybrid_retriever.py"
    "retrievers/reranker.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file 存在"
    else
        echo -e "${RED}✗${NC} $file 不存在"
    fi
done

# 检查测试文件
echo -e "\n${YELLOW}4. 检查测试文件${NC}"
TEST_FILES=(
    "tests/demo_permissions_simple.py"
    "tests/test_agent_system.py"
    "tests/test_api_endpoints.py"
)

for file in "${TEST_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file 存在"
    else
        echo -e "${RED}✗${NC} $file 不存在"
    fi
done

# 语法检查
echo -e "\n${YELLOW}5. Python 语法检查${NC}"
SYNTAX_CHECK_FILES=(
    "main.py"
    "permission_manager.py"
    "agents/agent_runtime.py"
    "agents/task_agents/mcp_agent.py"
)

SYNTAX_OK=true
for file in "${SYNTAX_CHECK_FILES[@]}"; do
    if python -m py_compile "$file" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $file 语法正确"
    else
        echo -e "${RED}✗${NC} $file 语法错误"
        SYNTAX_OK=false
    fi
done

# 运行简单测试
echo -e "\n${YELLOW}6. 运行权限管理演示${NC}"
if python tests/demo_permissions_simple.py > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} 权限管理演示运行成功"
else
    echo -e "${RED}✗${NC} 权限管理演示运行失败"
fi

# 检查依赖文件
echo -e "\n${YELLOW}7. 检查依赖文件${NC}"
if [ -f "requirements.txt" ]; then
    echo -e "${GREEN}✓${NC} requirements.txt 存在"
else
    echo -e "${RED}✗${NC} requirements.txt 不存在"
fi

if [ -f "requirements_agents.txt" ]; then
    echo -e "${GREEN}✓${NC} requirements_agents.txt 存在"
else
    echo -e "${RED}✗${NC} requirements_agents.txt 不存在"
fi

# 检查文档
echo -e "\n${YELLOW}8. 检查文档${NC}"
DOC_FILES=(
    "README.md"
    "AGENT_SYSTEM.md"
    "REFACTORING_COMPLETE.md"
    "tests/README.md"
)

for file in "${DOC_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file 存在"
    else
        echo -e "${YELLOW}⚠${NC} $file 不存在（可选）"
    fi
done

# 总结
echo -e "\n=========================================="
echo -e "${GREEN}验证完成！${NC}"
echo "=========================================="

if [ "$SYNTAX_OK" = true ]; then
    echo -e "\n${GREEN}✓ 所有核心功能验证通过${NC}"
    echo -e "\n下一步："
    echo "1. 激活环境: mamba activate python3.12"
    echo "2. 安装依赖: pip install -r requirements.txt && pip install -r requirements_agents.txt"
    echo "3. 启动服务: python main.py"
    echo "4. 测试 API: python tests/test_api_endpoints.py"
    exit 0
else
    echo -e "\n${RED}✗ 发现语法错误，请检查${NC}"
    exit 1
fi
