# 09 - 配置文件学习指南

本文档介绍项目中各种配置文件的作用和使用方法。

## 目录

1. [pyproject.toml](#pyprojecttoml)
2. [requirements.txt](#requirementstxt)
3. [environment.yml](#environmentyml)
4. [.gitignore](#gitignore)
5. [.pre-commit-config.yaml](#pre-commit-configyaml)

---

## pyproject.toml

### 作用

`pyproject.toml` 是 Python 项目的**统一配置文件**，符合 PEP 518/621 标准。它将项目元数据、依赖声明、工具配置整合到一个文件中。

### 项目结构

```toml
[build-system]      # 构建系统配置
[project]           # 项目元数据和依赖
[project.scripts]   # 命令行入口
[tool.*]            # 各种工具的配置
```

### 核心配置详解

#### 1. 项目元数据

```toml
[project]
name = "lingxin"                    # 包名
version = "0.2.0"                   # 语义化版本号
description = "聆心 — 实时手语识别系统"  # 项目描述
requires-python = ">=3.10"          # Python 版本要求
license = {text = "MIT"}            # 开源协议
```

#### 2. 依赖声明

```toml
[project]
dependencies = [
    "numpy>=1.24.0,<1.270",     # 版本范围：最低1.24.0，最高1.27.0
    "torch>=2.0.0,<2.12.0",     # 使用逗号分隔多个约束
    "flask>=3.0.0",             # 只设最低版本
]
```

**版本约束语法：**
| 写法 | 含义 |
|------|------|
| `>=1.0` | 大于等于1.0 |
| `<2.0` | 小于2.0 |
| `>=1.0,<2.0` | 大于等于1.0且小于2.0 |
| `~=1.0` | 兼容1.0（等价于>=1.0,<2.0） |

#### 3. 可选依赖（开发依赖）

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",    # 测试框架
    "black>=23.0.0",    # 代码格式化
    "isort>=5.12.0",    # import排序
    "flake8>=6.0.0",    # 代码检查
]
```

安装方式：`pip install -e ".[dev]"`

#### 4. 命令行入口

```toml
[project.scripts]
lingxin-collect = "tools.collect_data:main"   # 采集工具
lingxin-train = "tools.train:main"            # 训练工具
```

安装后可直接运行：
```bash
lingxin-collect
lingxin-train
```

#### 5. 工具配置

```toml
[tool.black]           # Black 格式化器配置
line-length = 100      # 行宽限制
target-version = ["py310"]

[tool.isort]           # isort 配置
profile = "black"      # 使用 Black 兼容的排序规则

[tool.pytest.ini_options]  # pytest 配置
testpaths = ["tests"]      # 测试目录
python_files = ["test_*.py"]  # 测试文件命名规则
```

---

## requirements.txt

### 作用

传统的 pip 依赖声明文件，用于快速安装项目依赖。

### 格式

```
# 注释行
numpy>=1.24.0,<1.27.0
torch>=2.0.0
flask>=3.0.0
```

### 与 pyproject.toml 的关系

| 功能 | requirements.txt | pyproject.toml |
|------|:----------------:|:--------------:|
| 声明依赖 | ✅ | ✅ |
| 版本约束 | ✅ | ✅ |
| 项目元数据 | ❌ | ✅ |
| 工具配置 | ❌ | ✅ |
| 命令行入口 | ❌ | ✅ |
| 发布到PyPI | ❌ | ✅ |

**建议：** 两个文件都保留，保持内容一致。

### 使用方式

```bash
# 安装所有依赖
pip install -r requirements.txt

# 只安装核心依赖（不安装开发工具）
pip install -r requirements.txt --no-dev
```

---

## environment.yml

### 作用

Conda 环境配置文件，用于创建可复现的开发环境。

### 结构

```yaml
name: lingxin              # 环境名称
channels:                  # 包来源渠道
  - pytorch
  - conda-forge
  - defaults
dependencies:              # 依赖列表
  - python=3.10            # Python 版本
  - pytorch>=2.0.0         # conda 包
  - pip:                   # pip 包（conda 没有的）
    - mediapipe>=0.10.33
```

### conda vs pip

| 特性 | conda | pip |
|------|-------|-----|
| 包管理 | 独立环境 | 需要 virtualenv |
| 系统依赖 | ✅ 自动处理 | ❌ 需手动安装 |
| Python版本 | ✅ 可指定 | ❌ 依赖系统 |
| 包数量 | 较少 | 更全 |

**使用方式：**

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate lingxin

# 更新环境
conda env update -f environment.yml --prune
```

---

## .gitignore

### 作用

告诉 Git 哪些文件/目录不需要版本控制。

### 项目中的配置

```gitignore
# Python 缓存
__pycache__/
*.py[cod]

# 虚拟环境
.env
.venv/

# 数据和模型（太大不提交）
data/raw/*
data/processed/*
models/*

# 日志
*.log

# IDE 配置
.vscode/
.idea/
```

### 常用规则

| 规则 | 含义 |
|------|------|
| `*.pyc` | 忽略所有 .pyc 文件 |
| `__pycache__/` | 忽略 Python 缓存目录 |
| `!data/raw/.gitkeep` | 取消忽略（保留空目录） |
| `build/` | 忽略 build 目录 |

---

## .pre-commit-config.yaml

### 作用

配置 Git 钩子，在每次提交前自动运行代码检查。

### 项目中的配置

```yaml
repos:
  - repo: https://github.com/psf/black
    hooks:
      - id: black           # 自动格式化代码
  - repo: https://github.com/pycqa/isort
    hooks:
      - id: isort           # 自动排序 import
```

### 使用方式

```bash
# 安装 pre-commit
pip install pre-commit

# 安装钩子
pre-commit install

# 手动运行所有钩子
pre-commit run --all-files
```

---

## 配置文件关系图

```
pyproject.toml ─────────────────────────────────┐
    │                                           │
    ├── dependencies ─────── requirements.txt ──┤
    │                                           │
    ├── [tool.pytest] ───── pytest 配置         │
    ├── [tool.black] ────── Black 配置          │
    ├── [tool.isort] ────── isort 配置          │
    │                                           │
    └── [project.scripts] ─ 命令行入口          │
                                                │
environment.yml ───── conda 环境（部分重叠）─────┘
                                                │
.gitignore ────────── Git 忽略规则（独立）────────┘
```

---

## 最佳实践

1. **保持一致性**：pyproject.toml 和 requirements.txt 的依赖版本要一致
2. **版本锁定**：生产环境使用 `==` 锁定版本，开发环境使用 `>=` 保持灵活
3. **分离关注点**：核心依赖和开发依赖分开声明
4. **文档化**：配置文件中添加注释说明用途
5. **定期更新**：每季度检查依赖版本，及时更新安全补丁

---

## 常见问题

### Q: 应该用 pyproject.toml 还是 requirements.txt？

**A:** 两者都用。pyproject.toml 是现代标准，requirements.txt 方便不熟悉 pip 的用户。

### Q: conda 和 pip 可以混用吗？

**A:** 可以，但建议先用 conda 安装，再用 pip 安装 conda 没有的包。

### Q: 版本号怎么选？

**A:**
- `>=1.0,<2.0`：允许小版本更新
- `>=1.0`：允许任何更新（有风险）
- `==1.0.0`：完全锁定（生产环境）

---

*最后更新：2026-06-20*
