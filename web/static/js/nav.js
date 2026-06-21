/**
 * 统一导航栏组件
 * 所有页面共享，确保一致的导航体验
 */

const NavController = {
    // 当前页面标识
    getCurrentPage() {
        const path = window.location.pathname;
        if (path.includes('dashboard')) return 'dashboard';
        if (path.includes('demo')) return 'demo';
        if (path.includes('docs')) return 'docs';
        if (path.includes('resources')) return 'resources';
        if (path.includes('index_new')) return 'home';
        return 'home';
    },

    // 生成导航栏HTML
    render(options = {}) {
        const currentPage = this.getCurrentPage();
        const {
            showStatus = false,
            showThemeToggle = false,
            showExport = false,
            showOnboarding = false
        } = options;

        const navItems = [
            { id: 'home', label: '首页', icon: '🏠', href: '/' },
            { id: 'dashboard', label: '控制台', icon: '🎛️', href: '/dashboard' },
            { id: 'demo', label: '演示', icon: '🎬', href: '/demo' },
            { id: 'docs', label: '文档', icon: '📖', href: '/docs?file=docs/01-quickstart.md' },
            { id: 'resources', label: '资源', icon: '📚', href: '/resources' }
        ];

        let html = `
        <header class="nav-header">
            <div class="nav-brand">
                <img src="../assets/logo.png" alt="聆心" style="width:36px;height:36px;border-radius:8px;">
                <span class="nav-title">聆心手语识别</span>
            </div>
            <nav class="nav-links">
                ${navItems.map(item => `
                    <a href="${item.href}"
                       class="nav-item ${currentPage === item.id ? 'active' : ''}"
                       ${currentPage !== item.id ? 'target="_blank"' : ''}>
                        <span class="nav-icon">${item.icon}</span>
                        <span class="nav-label">${item.label}</span>
                    </a>
                `).join('')}
            </nav>
            <div class="nav-actions">
                ${showStatus ? `
                <div class="nav-status">
                    <span class="status-dot" id="statusDot"></span>
                    <span id="statusText">未连接</span>
                </div>
                ` : ''}
                ${showOnboarding ? `
                <button onclick="Onboarding.reset(); Onboarding.show();" class="nav-btn" title="新手引导">❓</button>
                ` : ''}
                ${showThemeToggle ? `
                <button onclick="toggleDarkMode()" class="nav-btn" id="themeToggle" title="切换主题">🌙</button>
                ` : ''}
                ${showExport ? `
                <button onclick="exportHistory()" class="nav-btn" title="导出记录">📥</button>
                ` : ''}
            </div>
        </header>`;

        return html;
    },

    // 注入导航栏到页面
    inject(targetId = 'nav-container', options = {}) {
        const target = document.getElementById(targetId);
        if (target) {
            target.innerHTML = this.render(options);
        }
    },

    // 高亮当前页面
    highlight() {
        const currentPage = this.getCurrentPage();
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
            if (item.dataset.page === currentPage) {
                item.classList.add('active');
            }
        });
    }
};

// 导航栏样式
const navStyles = `
<style>
.nav-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    height: 56px;
    background: rgba(250, 247, 242, 0.95);
    border-bottom: 1px solid #E8E0D5;
    backdrop-filter: blur(12px);
    position: sticky;
    top: 0;
    z-index: 100;
}

.nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.nav-title {
    font-family: 'Ma Shan Zheng', cursive;
    font-size: 1.1rem;
    color: #2C2420;
    letter-spacing: 0.05em;
}

.nav-links {
    display: flex;
    align-items: center;
    gap: 8px;
}

.nav-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    border-radius: 8px;
    text-decoration: none;
    color: #6B5E54;
    font-size: 0.9rem;
    transition: all 0.2s;
    border: 1px solid transparent;
}

.nav-item:hover {
    background: #F5F0E8;
    color: #B8860B;
}

.nav-item.active {
    background: rgba(184, 134, 11, 0.1);
    color: #B8860B;
    border-color: rgba(184, 134, 11, 0.2);
    font-weight: 500;
}

.nav-icon {
    font-size: 1rem;
}

.nav-actions {
    display: flex;
    align-items: center;
    gap: 12px;
}

.nav-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    background: #F5F0E8;
    border-radius: 20px;
    font-size: 0.8rem;
    color: #9B8B7A;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #C17F59;
}

.status-dot.online {
    background: #5A8A6A;
    box-shadow: 0 0 8px rgba(90, 138, 106, 0.5);
}

.nav-btn {
    padding: 8px 12px;
    background: #F5F0E8;
    border: 1px solid #E8E0D5;
    border-radius: 8px;
    cursor: pointer;
    font-size: 1rem;
    transition: all 0.2s;
}

.nav-btn:hover {
    background: #E8E0D5;
}

/* 响应式 */
@media (max-width: 768px) {
    .nav-header {
        padding: 0 16px;
    }
    .nav-label {
        display: none;
    }
    .nav-item {
        padding: 8px 12px;
    }
}
</style>
`;

// 自动注入样式
if (!document.getElementById('nav-styles')) {
    const styleEl = document.createElement('div');
    styleEl.id = 'nav-styles';
    styleEl.innerHTML = navStyles;
    document.head.appendChild(styleEl);
}
