# Python 3.12 安装总结

## 安装状态

✅ **Python 3.12.7 编译安装成功**
- 安装位置：`~/sjq-workspace/python3.12`
- 可执行文件：`~/sjq-workspace/python3.12/bin/python`
- 符号链接：`python` -> `python3.12`

## 环境配置

已添加到 `~/.bashrc`：
```bash
# Python 3.12 Configuration
export PATH="$HOME/sjq-workspace/python3.12/bin:$PATH"
alias python=python3.12
alias python3=python3.12
alias pip=pip3.12
alias pip3=pip3.12
```

**激活方式**：
```bash
source ~/.bashrc
# 或者在新的shell会话中自动生效
```

## 当前问题

❌ **Python 3.12 缺少 SSL 支持**
- 原因：编译时缺少 OpenSSL 开发库
- 影响：无法使用 pip 从 PyPI 下载包
- 错误信息：`SSLError("Can't connect to HTTPS URL because the SSL module is not available.")`

## 解决方案

### 方案1：重新编译 Python 3.12（推荐，但需要管理员权限）

1. 安装 OpenSSL 开发库：
```bash
# 需要管理员权限
sudo yum install openssl-devel
```

2. 重新编译 Python：
```bash
cd ~/sjq-workspace/python-build/Python-3.12.7
make clean
./configure --prefix=/home/henry.sjq/sjq-workspace/python3.12 --enable-optimizations
make -j64
make install
```

3. 验证 SSL 支持：
```bash
~/sjq-workspace/python3.12/bin/python -c "import ssl; print(ssl.OPENSSL_VERSION)"
```

### 方案2：使用 HTTP 镜像源（临时方案）

配置 pip 使用 HTTP（不安全，不推荐生产环境）：
```bash
pip install --index-url=http://pypi.douban.com/simple/ --trusted-host pypi.douban.com pandas
```

### 方案3：手动下载 wheel 文件并安装

1. 在有网络的机器上下载：
```bash
pip download pandas numpy matplotlib seaborn plotly tqdm scipy
```

2. 传输到服务器并安装：
```bash
python3.12 -m pip install --no-index --find-links=/path/to/wheels pandas numpy matplotlib seaborn
```

### 方案4：使用容器环境（最稳定）

使用已有的 Docker/Pouch 容器，其中已安装完整的 Python 环境：
```bash
# 在容器中运行数据分析脚本
pouch exec lsw_sglang_benchmark_rank0 python3 analysis/analyze_data.py
```

## 临时解决方案

当前可以使用已编译的 Python 3.12 进行本地开发，但需要管理员权限安装 OpenSSL 开发库后重新编译才能使用 pip。

**建议**：
1. 联系管理员安装 `openssl-devel` 包
2. 或者使用容器环境进行数据分析

## 验证命令

```bash
# 设置环境变量
export PATH="$HOME/sjq-workspace/python3.12/bin:$PATH"

# 验证 Python 版本
python --version  # 应显示 Python 3.12.7

# 验证 pip（当前会失败，因为缺少 SSL）
pip --version

# 测试 SSL 支持
python -c "import ssl"  # 当前会报错
```

## 下一步行动

请选择以下方案之一：
1. 安装 openssl-devel 后重新编译（推荐）
2. 使用容器环境进行数据分析
3. 使用 HTTP 镜像源（临时方案）

---

**文档创建时间**：2025-12-23
**Python版本**：3.12.7
**安装路径**：`~/sjq-workspace/python3.12`
