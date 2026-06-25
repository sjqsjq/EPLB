#!/bin/bash

# SSH免密登录配置脚本
# 本机IP: 11.139.21.79
# 密钥文件: id_rsa79

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置变量
LOCAL_IP="11.139.21.79"
SSH_USER="henry.sjq"
SSH_PASSWORD="617178Sjq"
SSH_KEY_NAME="id_rsa79"
SSH_KEY_PATH="$HOME/.ssh/${SSH_KEY_NAME}"
SSH_PUB_KEY_PATH="${SSH_KEY_PATH}.pub"

# 目标节点列表
TARGET_NODES=(
    "11.139.21.90"  # Priority 1 (Master/Rank 0)
    "11.139.21.86"  # Priority 2
    "11.139.21.89"  # Priority 3
    "11.139.21.91"  # Priority 4
    "11.139.21.78"
    "11.139.21.84"
    "11.139.21.76"
    "11.139.21.75"
    "11.139.21.74"
    "11.139.21.80"
    "11.139.21.77"
    "11.139.21.83"
    "11.139.21.81"
    "11.139.21.88"
    "11.139.21.87"
    "11.139.21.82"
    "11.139.21.79"
    "11.139.21.85"
)

echo -e "${GREEN}=== SSH免密登录配置脚本 ===${NC}"
echo -e "本机IP: ${LOCAL_IP}"
echo -e "用户名: ${SSH_USER}"
echo -e "密钥文件: ${SSH_KEY_NAME}"
echo -e "目标节点数: ${#TARGET_NODES[@]}"
echo ""

# 1. 检查并生成SSH密钥
echo -e "${YELLOW}[1/4] 检查SSH密钥...${NC}"
if [ ! -f "${SSH_KEY_PATH}" ]; then
    echo "生成新的SSH密钥对 ${SSH_KEY_NAME}..."
    ssh-keygen -t rsa -b 4096 -f "${SSH_KEY_PATH}" -N "" -C "${SSH_USER}@${LOCAL_IP}"
    echo -e "${GREEN}✓ SSH密钥生成成功: ${SSH_KEY_PATH}${NC}"
else
    echo -e "${GREEN}✓ SSH密钥已存在: ${SSH_KEY_PATH}${NC}"
fi

# 确保SSH目录和密钥文件权限正确
chmod 700 ~/.ssh
chmod 600 "${SSH_KEY_PATH}"
if [ -f "${SSH_PUB_KEY_PATH}" ]; then
    chmod 644 "${SSH_PUB_KEY_PATH}"
fi

# 2. 配置SSH客户端选项
echo -e "\n${YELLOW}[2/4] 配置SSH客户端选项...${NC}"
SSH_CONFIG=~/.ssh/config
if [ ! -f "$SSH_CONFIG" ]; then
    touch "$SSH_CONFIG"
    chmod 600 "$SSH_CONFIG"
fi

# 检查是否已有 11.139.21.* 的配置
if grep -q "^Host 11.139.21.\*" "$SSH_CONFIG"; then
    # 检查是否已配置 IdentityFile
    if grep -A 10 "^Host 11.139.21.\*" "$SSH_CONFIG" | grep -q "IdentityFile.*${SSH_KEY_NAME}"; then
        echo -e "${GREEN}✓ SSH 配置中已包含 IdentityFile${NC}"
    else
        echo -e "${YELLOW}请手动在 ~/.ssh/config 的 'Host 11.139.21.*' 块中添加:${NC}"
        echo -e "    IdentityFile ~/.ssh/${SSH_KEY_NAME}"
        echo ""
        read -p "按回车键继续..."
    fi
else
    # 添加新的配置块
    cat >> "$SSH_CONFIG" << EOF

# 集群节点配置 - 使用 ${SSH_KEY_NAME}
Host 11.139.21.*
    StrictHostKeyChecking no
    IdentityFile ~/.ssh/${SSH_KEY_NAME}
    UserKnownHostsFile=/dev/null
    LogLevel ERROR
EOF
    echo -e "${GREEN}✓ SSH 配置已创建${NC}"
fi

# 3. 分发公钥到所有节点
echo -e "\n${YELLOW}[3/4] 分发公钥到目标节点...${NC}"
echo "使用 sshpass 自动输入密码..."
echo ""

SUCCESS_COUNT=0
FAILED_NODES=()

for node in "${TARGET_NODES[@]}"; do
    echo -n "处理节点 ${node}... "

    # 使用 sshpass 和 ssh-copy-id 自动复制公钥
    if sshpass -p "${SSH_PASSWORD}" ssh-copy-id -i "${SSH_PUB_KEY_PATH}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "${SSH_USER}@${node}" &>/dev/null; then
        echo -e "${GREEN}✓${NC}"
        ((SUCCESS_COUNT++))
    else
        # 如果失败，尝试手动方式
        if sshpass -p "${SSH_PASSWORD}" ssh \
               -o StrictHostKeyChecking=no \
               -o UserKnownHostsFile=/dev/null \
               "${SSH_USER}@${node}" \
               "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && sort -u ~/.ssh/authorized_keys -o ~/.ssh/authorized_keys" \
               < "${SSH_PUB_KEY_PATH}" &>/dev/null; then
            echo -e "${GREEN}✓${NC}"
            ((SUCCESS_COUNT++))
        else
            echo -e "${RED}✗${NC}"
            FAILED_NODES+=("$node")
        fi
    fi
done

# 4. 测试连接
echo -e "\n${YELLOW}[4/4] 测试SSH免密登录...${NC}"
TEST_SUCCESS=0
TEST_FAILED=()

for node in "${TARGET_NODES[@]}"; do
    echo -n "测试节点 ${node}... "
    if ssh -i "${SSH_KEY_PATH}" \
           -o BatchMode=yes \
           -o ConnectTimeout=5 \
           -o StrictHostKeyChecking=no \
           -o UserKnownHostsFile=/dev/null \
           "${SSH_USER}@${node}" "echo 'SSH连接成功' 2>&1" &>/dev/null; then
        echo -e "${GREEN}✓${NC}"
        ((TEST_SUCCESS++))
    else
        echo -e "${RED}✗${NC}"
        TEST_FAILED+=("$node")
    fi
done

# 输出结果统计
echo -e "\n${GREEN}=== 配置完成 ===${NC}"
echo -e "公钥分发: ${SUCCESS_COUNT}/${#TARGET_NODES[@]} 成功"
echo -e "连接测试: ${TEST_SUCCESS}/${#TARGET_NODES[@]} 成功"

if [ ${#FAILED_NODES[@]} -gt 0 ]; then
    echo -e "\n${RED}分发失败的节点:${NC}"
    printf '  %s\n' "${FAILED_NODES[@]}"
fi

if [ ${#TEST_FAILED[@]} -gt 0 ]; then
    echo -e "\n${RED}测试失败的节点:${NC}"
    printf '  %s\n' "${TEST_FAILED[@]}"
    echo -e "\n${YELLOW}提示: 可以手动测试连接:${NC}"
    echo "  ssh -i ${SSH_KEY_PATH} ${SSH_USER}@<节点IP>"
fi

# 显示公钥指纹
echo -e "\n${YELLOW}SSH公钥指纹:${NC}"
ssh-keygen -lf "${SSH_PUB_KEY_PATH}"

echo -e "\n${GREEN}配置信息:${NC}"
echo "  密钥文件: ${SSH_KEY_PATH}"
echo "  公钥文件: ${SSH_PUB_KEY_PATH}"
echo "  SSH配置: ~/.ssh/config"
echo -e "\n${GREEN}使用方法:${NC}"
echo "  ssh ${SSH_USER}@<节点IP>"
