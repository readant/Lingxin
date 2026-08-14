"""
构建脚本 — 自动生成文档与学习脚本索引

扫描 docs/ 下的分类子目录、journal/ 目录与 learning/ 教学脚本，读取每个文件的
frontmatter / docstring 元数据，生成 web/static/js/docs-index.generated.js 供前端
渲染侧边栏、卡片与搜索。

用法：
    python scripts/build_docs_index.py

规则：
- 分类由文件夹层级决定：docs/ 下目录名即类别码（guide / usage / design / dev / journal），
  frontmatter 不再声明 category
- docs/ 下仅处理 .md 文件；忽略隐藏文件与非 .md 文件
- 任意层级递归扫描，排除 README.md / INDEX.md（不挂侧边栏）
- journal/ 下日期命名（YYYY-MM-DD）文件自动归类为 "journal"，按日期倒序；
  非日期命名文件（综述）按 journal/README.md 中列出的顺序排序，未列出时按文件名兜底
- frontmatter 缺失时兜底：title 取首行 H1，order 默认 999，icon 默认 📄
- learning/ 下的教学 .py 脚本单独归类为 "learning"（学习教程），
  kind 标记为 "py" 以便前端以源码方式渲染；title 取自 docstring，order 取自文件名数字前缀
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
JOURNAL_DIR = DOCS_DIR / "journal"
LEARNING_DIR = PROJECT_ROOT / "learning"
OUTPUT_PATH = PROJECT_ROOT / "web" / "static" / "js" / "docs-index.generated.js"

EXCLUDE_FILES = {"README.md", "INDEX.md"}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
FILE_NUM_RE = re.compile(r"^(\d{1,2})_")
DOCSTRING_RE = re.compile(r'^\s*"""\s*\n(.*?)\n\s*"""', re.DOTALL)
STAGE_PREFIX_RE = re.compile(r"^第\d+阶段[:：]?\s*")

# 分类定义（键、显示名、图标、顺序即侧边栏展示顺序）
# 键与 docs/ 下的分类目录名一一对应
CATEGORIES = [
    {"key": "guide", "label": "快速入门", "icon": "🚀"},
    {"key": "usage", "label": "使用指南", "icon": "📹"},
    {"key": "design", "label": "架构与设计", "icon": "🏛️"},
    {"key": "dev", "label": "开发参考", "icon": "🛠️"},
    {"key": "learning", "label": "学习教程", "icon": "🎓"},
    {"key": "journal", "label": "项目日志", "icon": "📋"},
]
CATEGORY_ORDER = {c["key"]: i for i, c in enumerate(CATEGORIES)}


def parse_frontmatter(text):
    """解析 --- 包裹的 key: value 行，返回 dict；无 frontmatter 返回空 dict。"""
    fm = {}
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not m:
        return fm
    for line in m.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip()
    return fm


def extract_title(text):
    """从正文提取首行 H1 标题。"""
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def read_md(path):
    """读取 .md 文件，解析 frontmatter 并回填 title 兜底。"""
    text = path.read_text(encoding="utf-8")
    fm = parse_frontmatter(text)
    fm.setdefault("title", extract_title(text) or path.stem)
    fm.setdefault("order", 999)
    fm.setdefault("icon", "📄")
    fm.setdefault("description", "")
    return fm


def read_py(path):
    """读取教学 .py 脚本，从 docstring 提取 title，从文件名提取 order。"""
    text = path.read_text(encoding="utf-8")
    title = path.stem
    description = ""
    m = DOCSTRING_RE.match(text)
    if m:
        lines = m.group(1).strip().splitlines()
        first_line = lines[0].strip() if lines else ""
        # 去除「第N阶段：」前缀，如「第3阶段：MediaPipe入门」→「MediaPipe入门」
        title = STAGE_PREFIX_RE.sub("", first_line) or path.stem
        description = lines[-1].strip() if len(lines) > 1 else ""
    num_m = FILE_NUM_RE.match(path.name)
    order = int(num_m.group(1)) if num_m else 999
    return {"title": title, "order": order, "description": description}


def read_journal_order():
    """从 journal/README.md 提取综述日志推荐阅读顺序（每行一个文件名）。"""
    order_file = JOURNAL_DIR / "README.md"
    if not order_file.exists():
        return []
    order = []
    for line in order_file.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*[-*\d.)]+\s*(\S+\.md)", line)
        if m:
            order.append(m.group(1))
    return order


def scan_docs():
    """递归扫描 docs/ 下分类子目录中的文档。

    category 由文件夹层级决定：相对 docs/ 的第一级目录名即类别码。
    """
    items = []
    for md in sorted(DOCS_DIR.rglob("*.md")):
        if JOURNAL_DIR in md.parents:
            continue
        if md.name in EXCLUDE_FILES or md.name.startswith("."):
            continue
        fm = read_md(md)
        rel = md.relative_to(DOCS_DIR)
        category = rel.parts[0] if len(rel.parts) > 1 else "dev"
        items.append({
            "file": f"docs/{rel.as_posix()}",
            "title": fm["title"],
            "category": category,
            "order": fm["order"],
            "icon": fm["icon"],
            "description": fm["description"],
            "kind": "md",
        })
    return items


def scan_learning():
    """扫描 learning/ 下教学 .py 脚本，归类为 learning（学习教程）。"""
    if not LEARNING_DIR.exists():
        return []
    items = []
    for f in sorted(LEARNING_DIR.glob("*.py")):
        meta = read_py(f)
        items.append({
            "file": f"learning/{f.name}",
            "title": meta["title"],
            "category": "learning",
            "order": meta["order"],
            "icon": "🎓",
            "description": meta["description"],
            "kind": "py",
        })
    return items


def scan_journal():
    """扫描 journal/ 目录：综述在前（按 README 顺序），日期流水在后（倒序）。"""
    if not JOURNAL_DIR.exists():
        return []
    journal_order = read_journal_order()
    items = []
    for f in sorted(JOURNAL_DIR.glob("*.md")):
        if f.name in EXCLUDE_FILES or f.name.startswith("."):
            continue
        fm = read_md(f)
        items.append({
            "file": f"docs/journal/{f.name}",
            "title": fm["title"],
            "category": "journal",
            "order": fm["order"],
            "icon": fm["icon"],
            "description": fm["description"],
            "kind": "md",
            "_name": f.name,
        })

    def sort_key(item):
        name = item["_name"]
        if DATE_RE.match(name):
            # 日期流水：分类末尾，日期倒序（新在前）
            return (1, -name, item["title"])
        # 综述：按 README 顺序，未列出按文件名兜底
        idx = journal_order.index(name) if name in journal_order else 999
        return (0, idx, name)

    items.sort(key=sort_key)
    for item in items:
        item.pop("_name", None)
    return items


def collect_categories(items):
    """输出分类定义：内置 CATEGORIES 打头，扫描中发现的未知目录追加在后。"""
    cats = list(CATEGORIES)
    seen = {c["key"] for c in cats}
    for it in items:
        if it["category"] not in seen:
            seen.add(it["category"])
            cats.append({"key": it["category"], "label": it["category"], "icon": "📄"})
    return cats


def main():
    doc_items = scan_docs()
    learning_items = scan_learning()
    journal_items = scan_journal()  # 已按综述顺序 + 日期倒序排好

    # 分类文档按 (分类顺序, order, title) 排序；journal 条目保持扫描顺序拼在最后
    doc_items.sort(key=lambda i: (CATEGORY_ORDER.get(i["category"], 999), int(i["order"]), i["title"]))
    learning_items.sort(key=lambda i: (int(i["order"]), i["title"]))
    items = doc_items + learning_items + journal_items

    payload = {"categories": collect_categories(items), "items": items}
    js = "// 本文件由 scripts/build_docs_index.py 自动生成，请勿手动修改\n"
    js += "window.DOC_INDEX = " + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(js, encoding="utf-8")
    print(f"已生成 {OUTPUT_PATH.relative_to(PROJECT_ROOT)}（{len(items)} 条索引）")


if __name__ == "__main__":
    main()
