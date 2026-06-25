#!/bin/bash
#
# check_running_server.sh - 查找并显示运行中的SGLang服务
#
# Usage: ./check_running_server.sh [--detail]
#   --detail: 显示所有rank的最后10行日志
#

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="${SCRIPT_DIR}/tmp"

# 解析参数
SHOW_DETAIL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --detail)
            SHOW_DETAIL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--detail]"
            exit 1
            ;;
    esac
done

# 用于记录已处理的部署 (key: master_ip:pd_mode)
declare -A processed_deployments
found_any=false

# 查找所有可能的 nodelist 文件（包括 PD 模式）
shopt -s nullglob
nodelist_files=(${TMP_DIR}/nodelist_*)

if [[ ${#nodelist_files[@]} -eq 0 ]]; then
    echo -e "${RED}未找到任何部署信息${NC}"
    echo -e "${YELLOW}提示: 请先运行 ./run_container.sh 部署容器${NC}"
    exit 0
fi

for nodelist_file in "${nodelist_files[@]}"; do
    if [[ ! -f "$nodelist_file" ]]; then
        continue
    fi

    # 从 nodelist 文件名提取信息
    # 格式: nodelist_${MASTER_IP} 或 nodelist_${PD_MODE}_${MASTER_IP}
    nodelist_basename=$(basename "$nodelist_file")

    # 尝试匹配 PD 模式: nodelist_prefill_11.139.21.XX 或 nodelist_decode_11.139.21.XX
    if [[ "$nodelist_basename" =~ nodelist_(prefill|decode)_11\.13\.195\.([0-9]+) ]]; then
        pd_mode="${BASH_REMATCH[1]}"
        master_ip_suffix="${BASH_REMATCH[2]}"
        master_ip="11.139.21.${master_ip_suffix}"
        deployment_key="${master_ip}:${pd_mode}"
    # 尝试匹配统一模式: nodelist_11.139.21.XX
    elif [[ "$nodelist_basename" =~ nodelist_11\.13\.195\.([0-9]+) ]]; then
        pd_mode="unified"
        master_ip_suffix="${BASH_REMATCH[1]}"
        master_ip="11.139.21.${master_ip_suffix}"
        deployment_key="${master_ip}:unified"
    else
        continue
    fi

    # 如果该部署已经处理过，跳过
    if [[ -n "${processed_deployments[$deployment_key]}" ]]; then
        continue
    fi

    # 根据 PD 模式设置日志目录
    if [[ "$pd_mode" == "prefill" ]]; then
        LOG_DIR="${SCRIPT_DIR}/log/prefill"
    elif [[ "$pd_mode" == "decode" ]]; then
        LOG_DIR="${SCRIPT_DIR}/log/decode"
    else
        LOG_DIR="${SCRIPT_DIR}/log/worker"
    fi

    # 查找该部署最新的 rank0 日志文件
    latest_log=""
    latest_log_is_active=false

    # 首先查找在nodelist之后更新的日志（活跃服务器）
    for log in $(ls -t ${LOG_DIR}/*_node${master_ip_suffix}_rank0.log 2>/dev/null); do
        if [[ "$log" -nt "$nodelist_file" ]]; then
            latest_log="$log"
            latest_log_is_active=true
            break
        fi
    done

    # 如果没有活跃日志，查找最近的日志（最后一次运行）
    if [[ -z "$latest_log" ]]; then
        latest_log=$(ls -t ${LOG_DIR}/*_node${master_ip_suffix}_rank0.log 2>/dev/null | head -1)
    fi

    if [[ -z "$latest_log" ]]; then
        # 完全没有找到日志文件
        echo -e "${CYAN}==========================================${NC}"
        if [[ "$pd_mode" == "prefill" ]]; then
            echo -e "${YELLOW}[PD Prefill Server - Not Running]${NC}"
        elif [[ "$pd_mode" == "decode" ]]; then
            echo -e "${YELLOW}[PD Decode Server - Not Running]${NC}"
        else
            echo -e "${YELLOW}[Unified Server - Not Running]${NC}"
        fi
        echo -e "${CYAN}==========================================${NC}"

        # 读取 nodelist 文件，提取所有 IP 后缀
        ip_suffixes=$(awk -F'.' '{print $4}' "$nodelist_file" | grep -v '^$' | tr '\n' ',' | sed 's/,$//')
        echo -e "${YELLOW}部署节点: ${ip_suffixes}${NC}"
        echo -e "${RED}状态: 容器已部署，但从未启动过服务${NC}"
        echo -e "${YELLOW}提示: 运行 ./run_server.sh 启动服务${NC}"
        echo
        continue
    fi

    # 标记找到了服务器（活跃或历史）
    found_any=true

    # 标记该部署已处理
    processed_deployments[$deployment_key]=1

    filename=$(basename "$latest_log")

    echo -e "${CYAN}==========================================${NC}"
    if [[ "$latest_log_is_active" == true ]]; then
        # 活跃服务器
        if [[ "$pd_mode" == "prefill" ]]; then
            echo -e "${GREEN}[PD Prefill Server - Running]${NC}"
        elif [[ "$pd_mode" == "decode" ]]; then
            echo -e "${GREEN}[PD Decode Server - Running]${NC}"
        else
            echo -e "${GREEN}[Unified Server - Running]${NC}"
        fi
    else
        # 历史日志
        if [[ "$pd_mode" == "prefill" ]]; then
            echo -e "${YELLOW}[PD Prefill Server - Last Run]${NC}"
        elif [[ "$pd_mode" == "decode" ]]; then
            echo -e "${YELLOW}[PD Decode Server - Last Run]${NC}"
        else
            echo -e "${YELLOW}[Unified Server - Last Run]${NC}"
        fi
    fi
    echo -e "${GREEN}日志文件: ${latest_log}${NC}"
    echo -e "${CYAN}==========================================${NC}"

    # 提取TP/EP/DP配置
    tp=$(grep -oP 'tp.size[=: ]+\K\d+|tp_size=\K\d+' "$latest_log" | head -1)
    ep=$(grep -oP 'ep.size[=: ]+\K\d+|ep_size=\K\d+' "$latest_log" | head -1)
    dp=$(grep -oP 'dp.size[=: ]+\K\d+|dp_size=\K\d+' "$latest_log" | head -1)
    nnodes=$(grep -oP 'nnodes[=: ]+\K\d+' "$latest_log" | head -1)

    # 读取 nodelist 文件，提取所有 IP 后缀
    ip_suffixes=$(awk -F'.' '{print $4}' "$nodelist_file" | grep -v '^$' | tr '\n' ',' | sed 's/,$//')

    echo -e "${YELLOW}并行配置: TP=${tp}, EP=${ep}, DP=${dp}, Nodes=${nnodes}${NC}"
    echo -e "${YELLOW}部署节点: ${ip_suffixes}${NC}"

    if [[ "$latest_log_is_active" == false ]]; then
        echo -e "${RED}状态: 日志来自容器部署前的运行，当前服务未启动${NC}"
        echo -e "${YELLOW}提示: 运行 ./run_server.sh 启动新服务${NC}"
        echo
    else
        # 只有当服务器正在运行时才显示日志内容
        echo

        if [[ "$SHOW_DETAIL" == true ]]; then
            # 显示所有rank的日志
            # 从日志文件名中提取基础名称（时间戳+配置，不包括node和rank）
            base_name=$(echo "$filename" | sed -E 's/_node[0-9]+_rank[0-9]+\.log$//')

            # 查找所有相同配置的rank日志（按rank编号排序）
            for rank_log in $(ls ${LOG_DIR}/${base_name}_node*_rank*.log 2>/dev/null | sort -t'_' -k5 -n); do
                if [[ -f "$rank_log" ]]; then
                    rank_filename=$(basename "$rank_log")
                    rank_num=$(echo "$rank_filename" | grep -oP 'rank\K\d+')
                    node_num=$(echo "$rank_filename" | grep -oP 'node\K\d+')

                    echo -e "${BLUE}=== Rank ${rank_num} (Node 11.139.21.${node_num}) ===${NC}"
                    echo "------------------------------------------"
                    tail -10 "$rank_log"
                    echo
                fi
            done
        else
            # 只显示rank0的日志
            echo -e "${BLUE}最后20行日志:${NC}"
            echo "------------------------------------------"
            tail -20 "$latest_log"
            echo
        fi
    fi
done
