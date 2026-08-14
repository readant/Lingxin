/**
 * 文档索引渲染逻辑库
 *
 * 依赖 web/static/js/docs-index.generated.js（由 scripts/build_docs_index.py 生成），
 * 提供 window.DOC_INDEX 数据。本文件为一次性手写的渲染/搜索逻辑，页面无需重复硬编码文档清单。
 *
 * 提供：
 * - renderSidebar(container)：按分类分组渲染侧边栏
 * - getDocMeta(file) / getCategoryMeta(key)
 * - searchDocs(query)：即时匹配 title/description/category（不 fetch）
 * - fetchDocContent(file)：按需拉取 .md 全文，带缓存
 * - searchFullText(query)：对已加载正文做匹配（需先触发 fetchDocContent）
 * - getLearningDocs()：排除 journal 的文档列表（学习路径/统计用）
 */

const DocIndex = (() => {
    // 文档全文缓存：{ file: promise<text> }
    const contentCache = new Map();

    function items() {
        return (window.DOC_INDEX && window.DOC_INDEX.items) || [];
    }

    function categories() {
        return (window.DOC_INDEX && window.DOC_INDEX.categories) || [];
    }

    function getDocMeta(file) {
        return items().find(i => i.file === file);
    }

    function getCategoryMeta(key) {
        return categories().find(c => c.key === key);
    }

    /**
     * 按分类分组渲染侧边栏链接。
     * container：容器元素，生成 <div class="sidebar-section">…</div> 结构。
     * onClick：点击回调，参数为 (file)。
     */
    function renderSidebar(container, onClick) {
        container.innerHTML = '';
        categories().forEach(cat => {
            const groupItems = items().filter(i => i.category === cat.key);
            if (groupItems.length === 0) return;

            const section = document.createElement('div');
            section.className = 'sidebar-section';

            const label = document.createElement('div');
            label.className = 'sidebar-label';
            label.textContent = `${cat.icon} ${cat.label}`;
            section.appendChild(label);

            groupItems.forEach(item => {
                const btn = document.createElement('button');
                btn.className = 'sidebar-link';
                btn.dataset.file = item.file;
                btn.innerHTML = `<span class="icon">${item.icon}</span> ${item.title}`;
                btn.addEventListener('click', () => onClick(item.file));
                section.appendChild(btn);
            });

            container.appendChild(section);
        });
    }

    /** 高亮侧边栏中当前文档 */
    function highlightSidebar(file) {
        document.querySelectorAll('.sidebar-link').forEach(link => {
            link.classList.toggle('active', link.dataset.file === file);
        });
    }

    /** 即时元数据搜索：匹配 title/description/category，返回按分排序的结果 */
    function searchDocs(query) {
        const q = query.toLowerCase().trim();
        if (q.length === 0) return [];
        const results = [];
        items().forEach(item => {
            const haystacks = [
                item.title, item.description, item.category,
                (getCategoryMeta(item.category) || {}).label || '',
            ];
            const score = haystacks.reduce(
                (acc, s) => acc + ((s || '').toLowerCase().match(new RegExp(escapeRegExp(q), 'g')) || []).length,
                0
            );
            if (score > 0) results.push({ ...item, score });
        });
        return results.sort((a, b) => b.score - a.score);
    }

    /** 按需拉取文档全文（带缓存） */
    function fetchDocContent(file) {
        if (contentCache.has(file)) return contentCache.get(file);
        const p = fetch('../' + file)
            .then(resp => {
                if (!resp.ok) throw new Error('文件未找到: ' + file);
                return resp.text();
            })
            .then(text => text.replace(/^---[\s\S]*?---\n*/, ''));
        contentCache.set(file, p);
        return p;
    }

    /** 全文搜索：对已缓存/已加载的正文做匹配，返回 { file, title, preview, score } */
    async function searchFullText(query, options = {}) {
        const q = query.toLowerCase().trim();
        if (q.length === 0) return [];
        const results = [];
        const files = options.files || items().map(i => i.file);
        await Promise.all(files.map(async file => {
            try {
                const text = await fetchDocContent(file);
                if (!text) return;
                const lower = text.toLowerCase();
                const idx = lower.indexOf(q);
                if (idx >= 0) {
                    const count = (lower.match(new RegExp(escapeRegExp(q), 'g')) || []).length;
                    const meta = getDocMeta(file) || {};
                    results.push({
                        file,
                        title: meta.title || file,
                        score: 10 + count,
                        preview: text.substring(Math.max(0, idx - 30), idx + q.length + 50).replace(/\s+/g, ' '),
                    });
                }
            } catch (e) { /* 单个文件失败不影响整体 */ }
        }));
        return results.sort((a, b) => b.score - a.score);
    }

    /** 排除 journal 的文档（学习路径、统计计数用），按 (分类顺序, order) 排序 */
    function getLearningDocs() {
        const orderMap = {};
        categories().forEach((c, i) => orderMap[c.key] = i);
        return items()
            .filter(i => i.category !== 'journal')
            .sort((a, b) =>
                (orderMap[a.category] - orderMap[b.category]) ||
                (Number(a.order) - Number(b.order)) ||
                a.title.localeCompare(b.title, 'zh')
            );
    }

    function escapeRegExp(s) {
        return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    return {
        items, categories, getDocMeta, getCategoryMeta,
        renderSidebar, highlightSidebar, searchDocs,
        fetchDocContent, searchFullText, getLearningDocs,
    };
})();
