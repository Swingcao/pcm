# 服务器代码同步与部署指南

本文档详细说明如何在本地与服务器之间同步 ProCoMemory 项目代码。

## 目录

1. [环境信息](#环境信息)
2. [方案一：通过 Git 同步（推荐）](#方案一通过-git-同步推荐)
3. [方案二：通过 rsync/scp 直接同步](#方案二通过-rsyncscp-直接同步)
4. [服务器环境配置](#服务器环境配置)
5. [常见问题](#常见问题)

---

## 环境信息

| 项目 | 值 |
|------|-----|
| 服务器代码目录 | `/home/20TA/zhongjian/Research-Project/` |
| GitHub 仓库地址 | `git@github.com:ZhongWishing/SEAMiLab-ProCoMem.git` |
| 仓库类型 | 私密仓库 |
| 本地已连接工具 | MobaXterm, PyCharm |

---

## 方案一：通过 Git 同步（推荐）

使用 Git 进行版本控制是最佳实践，可以追踪所有更改历史。

### 1.1 服务器端配置 Git 用户信息

首先通过 SSH 连接到服务器：

```bash
# 使用 MobaXterm 或其他终端工具连接到服务器
ssh your_username@your_server_ip
```

配置 Git 全局用户信息（使用你的 GitHub 账户信息）：

```bash
# 配置用户名（使用你的 GitHub 用户名）
git config --global user.name "ZhongWishing"

# 配置邮箱（使用你 GitHub 账户关联的邮箱）
git config --global user.email "your_email@example.com"

# 验证配置
git config --global --list
```

### 1.2 在服务器上配置 SSH Key（用于私密仓库访问）

由于是私密仓库，需要在服务器上配置 SSH key 来进行身份验证。

#### 步骤 1：检查是否已有 SSH key

```bash
ls -la ~/.ssh/
```

如果看到 `id_rsa` 和 `id_rsa.pub` 文件，说明已有 SSH key，可以跳到步骤 3。

#### 步骤 2：生成新的 SSH key

```bash
# 生成 SSH key（将邮箱替换为你的 GitHub 邮箱）
ssh-keygen -t ed25519 -C "your_email@example.com"

# 如果系统不支持 ed25519，使用 rsa：
# ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# 按提示操作：
# - 文件保存位置：直接回车使用默认路径 (~/.ssh/id_ed25519)
# - 密码短语：可以设置也可以留空直接回车
```

#### 步骤 3：启动 ssh-agent 并添加 key

```bash
# 启动 ssh-agent
eval "$(ssh-agent -s)"

# 添加私钥到 ssh-agent
ssh-add ~/.ssh/id_ed25519
# 或者如果使用 rsa：
# ssh-add ~/.ssh/id_rsa
```

#### 步骤 4：复制公钥内容

```bash
# 显示公钥内容
cat ~/.ssh/id_ed25519.pub
# 或者：
# cat ~/.ssh/id_rsa.pub
```

复制输出的全部内容（以 `ssh-ed25519` 或 `ssh-rsa` 开头的整行）。

#### 步骤 5：将公钥添加到 GitHub

1. 打开 GitHub 网站，登录你的账户
2. 点击右上角头像 → **Settings**
3. 在左侧菜单选择 **SSH and GPG keys**
4. 点击 **New SSH key**
5. 填写：
   - **Title**: 给这个 key 起个名字，如 `Server-Research`
   - **Key type**: Authentication Key
   - **Key**: 粘贴刚才复制的公钥内容
6. 点击 **Add SSH key**

#### 步骤 6：测试 SSH 连接

```bash
ssh -T git@github.com
```

首次连接会提示是否继续，输入 `yes`。成功后会显示：
```
Hi ZhongWishing! You've successfully authenticated, but GitHub does not provide shell access.
```

### 1.3 在服务器上克隆项目

```bash
# 进入目标目录
cd /home/20TA/zhongjian/Research-Project/

# 克隆私密仓库
git clone git@github.com:ZhongWishing/SEAMiLab-ProCoMem.git

# 如果要将代码直接放在当前目录（而不是子目录）：
git clone git@github.com:ZhongWishing/SEAMiLab-ProCoMem.git .
```

### 1.4 日常同步工作流程

#### 本地修改后推送到服务器

在**本地**执行：

```bash
# 1. 查看修改状态
git status

# 2. 添加所有修改
git add .

# 3. 提交修改
git commit -m "描述你的修改内容"

# 4. 推送到 GitHub
git push origin master
# 或者如果是 main 分支：
# git push origin main
```

在**服务器**上执行：

```bash
# 进入项目目录
cd /home/20TA/zhongjian/Research-Project/ProCoMemory

# 拉取最新代码
git pull origin master
# 或者：
# git pull origin main
```

#### 服务器修改后同步到本地

在**服务器**上执行：

```bash
cd /home/20TA/zhongjian/Research-Project/ProCoMemory

git add .
git commit -m "描述修改内容"
git push origin master
```

在**本地**执行：

```bash
git pull origin master
```

---

## 方案二：通过 rsync/scp 直接同步

如果不想使用 Git，可以直接使用文件传输工具同步。

### 2.1 使用 rsync（推荐）

rsync 只传输有变化的文件，效率更高。

#### 从本地同步到服务器

```bash
# 在本地执行（Windows Git Bash 或 WSL）
rsync -avz --progress \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '.git' \
    --exclude 'results' \
    --exclude '*.pyc' \
    /path/to/local/ProCoMemory/ \
    username@server_ip:/home/20TA/zhongjian/Research-Project/ProCoMemory/
```

#### 从服务器同步到本地

```bash
rsync -avz --progress \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '.git' \
    --exclude 'results' \
    username@server_ip:/home/20TA/zhongjian/Research-Project/ProCoMemory/ \
    /path/to/local/ProCoMemory/
```

### 2.2 使用 MobaXterm 图形界面

1. 在 MobaXterm 中连接到服务器
2. 左侧会显示服务器文件浏览器（SFTP）
3. 直接拖拽文件或文件夹进行上传/下载

### 2.3 使用 PyCharm 同步

1. 打开 PyCharm → **Tools** → **Deployment** → **Configuration**
2. 添加 SFTP 连接配置
3. 设置映射路径：
   - Local path: `C:\Users\33798\Desktop\Research-Project\ProCoMemory`
   - Deployment path: `/home/20TA/zhongjian/Research-Project/ProCoMemory`
4. 右键项目 → **Deployment** → **Upload to...**

---

## 服务器环境配置

### 3.1 创建 Python 虚拟环境

```bash
cd /home/20TA/zhongjian/Research-Project/ProCoMemory

# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3.2 下载 Embedding 模型到服务器

由于项目已配置使用本地模型，需要在服务器上也下载模型：

```bash
# 进入项目目录
cd /home/20TA/zhongjian/Research-Project/ProCoMemory

# 创建 models 目录
mkdir -p models

# 方法 1：使用 git lfs 克隆模型（推荐）
cd models
git lfs install
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# 方法 2：使用 Python 下载
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='./models')"
```

### 3.3 配置文件

确保 `config.yaml` 中的路径配置正确：

```yaml
embedding:
  model_id: "sentence-transformers/all-MiniLM-L6-v2"
  local_path: "./models/all-MiniLM-L6-v2"
  use_local: true
  dimension: 384

paths:
  data_dir: "./data"
  results_dir: "./results"
```

### 3.4 运行测试

```bash
# 确保虚拟环境已激活
source .venv/bin/activate

# 运行主程序测试
python main.py
```

---

## 常见问题

### Q1: SSH 连接 GitHub 失败，提示 Permission denied

**原因**：SSH key 未正确配置或未添加到 GitHub

**解决方案**：
1. 确认 SSH key 已生成：`ls ~/.ssh/`
2. 确认公钥已添加到 GitHub
3. 测试连接：`ssh -vT git@github.com`（-v 显示详细信息）

### Q2: git push 提示需要输入用户名密码

**原因**：使用了 HTTPS 而不是 SSH 方式克隆

**解决方案**：
```bash
# 查看当前远程仓库地址
git remote -v

# 如果是 https 开头，更改为 SSH 地址
git remote set-url origin git@github.com:ZhongWishing/SEAMiLab-ProCoMem.git
```

### Q3: 服务器上 git pull 时提示有本地修改冲突

**解决方案**：
```bash
# 方案 1：暂存本地修改
git stash
git pull origin master
git stash pop

# 方案 2：放弃本地修改（谨慎使用）
git checkout -- .
git pull origin master

# 方案 3：创建新分支保存本地修改
git checkout -b local-changes
git add .
git commit -m "save local changes"
git checkout master
git pull origin master
```

### Q4: 模型文件太大，同步很慢

**解决方案**：
- 将 `models/` 目录添加到 `.gitignore`
- 在服务器上单独下载模型
- 使用 rsync 的 `--exclude 'models'` 选项排除模型文件

### Q5: 服务器没有安装 git-lfs

**解决方案**：
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# CentOS/RHEL
sudo yum install git-lfs

# 或者使用 conda
conda install git-lfs
```

---

## 快速参考

### 每日工作流程（推荐）

```bash
# === 开始工作前（在本地）===
git pull origin master

# === 工作完成后（在本地）===
git add .
git commit -m "完成 xxx 功能"
git push origin master

# === 在服务器上运行 ===
ssh your_server
cd /home/20TA/zhongjian/Research-Project/ProCoMemory
git pull origin master
source .venv/bin/activate
python main.py
```

### 重要文件说明

| 文件 | 是否需要同步 | 说明 |
|------|------------|------|
| `*.py` | 是 | 源代码 |
| `config.yaml` | 是（注意路径） | 配置文件 |
| `requirements.txt` | 是 | 依赖列表 |
| `.venv/` | 否 | 虚拟环境（各自创建） |
| `models/` | 否（太大） | 模型文件（各自下载） |
| `results/` | 视情况 | 运行结果 |
| `__pycache__/` | 否 | Python 缓存 |

---

## 联系与支持

如有问题，请检查：
1. 网络连接是否正常
2. SSH key 是否正确配置
3. Git 用户信息是否正确设置
4. 服务器 Python 环境是否正确激活
