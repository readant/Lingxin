/**
 * Lingxin Onboarding - 新手引导流程
 * 首次访问时显示分步引导，帮助用户完成首次识别
 */

const Onboarding = {
    STORAGE_KEY: 'lingxin_onboarding_done',
    currentStep: 0,
    steps: [
        {
            title: '👋 欢迎使用聆心手语识别系统',
            description: '让我们用3步帮你完成第一次手语识别',
            action: null
        },
        {
            title: '🔌 第1步：连接后端服务',
            description: '正在检查API服务状态...',
            action: 'checkApi'
        },
        {
            title: '📷 第2步：开启摄像头',
            description: '点击下方按钮授权摄像头访问',
            action: 'startCamera'
        },
        {
            title: '✋ 第3步：开始识别',
            description: '将手放在摄像头前，系统会自动识别手语',
            action: 'waitResult'
        },
        {
            title: '🎉 引导完成！',
            description: '你已成功完成首次识别，现在可以自由使用了',
            action: null
        }
    ],

    init() {
        const done = localStorage.getItem(this.STORAGE_KEY);
        if (!done) {
            this.show();
        }
    },

    show() {
        const modal = document.createElement('div');
        modal.id = 'onboarding-modal';
        modal.className = 'fixed inset-0 z-[9999] flex items-center justify-center bg-black/70 backdrop-blur-sm';
        modal.innerHTML = this.renderStep();
        document.body.appendChild(modal);
    },

    renderStep() {
        const step = this.steps[this.currentStep];
        return `
            <div class="bg-surface-card rounded-2xl border border-white/10 p-8 max-w-md w-full mx-4 shadow-2xl">
                <div class="text-center mb-6">
                    <div class="text-sm text-gray-500 mb-2">步骤 ${this.currentStep + 1} / ${this.steps.length}</div>
                    <div class="w-full h-1 bg-white/10 rounded-full overflow-hidden">
                        <div class="h-full bg-brand rounded-full transition-all duration-500"
                             style="width: ${((this.currentStep + 1) / this.steps.length) * 100}%"></div>
                    </div>
                </div>

                <div class="text-center mb-8">
                    <h2 class="text-xl font-bold mb-3">${step.title}</h2>
                    <p class="text-gray-400">${step.description}</p>
                </div>

                <div id="onboarding-status" class="text-center mb-6 min-h-[40px]">
                    ${this.renderStepContent()}
                </div>

                <div class="flex gap-3">
                    <button onclick="Onboarding.skip()"
                            class="flex-1 px-4 py-3 bg-white/5 hover:bg-white/10 rounded-lg text-gray-400 transition-colors">
                        跳过引导
                    </button>
                    <button id="onboarding-btn" onclick="Onboarding.next()"
                            class="flex-1 px-4 py-3 bg-brand hover:bg-brand-light rounded-lg font-medium transition-colors">
                        ${this.getButtonText()}
                    </button>
                </div>
            </div>
        `;
    },

    renderStepContent() {
        const step = this.steps[this.currentStep];

        if (step.action === 'checkApi') {
            return `
                <div class="inline-flex items-center gap-2 text-gray-400">
                    <div class="w-5 h-5 border-2 border-brand border-t-transparent rounded-full animate-spin"></div>
                    <span>正在连接...</span>
                </div>
            `;
        }

        if (step.action === 'startCamera') {
            return `
                <div class="inline-flex items-center gap-2 text-gray-400">
                    <span>📷 等待开启摄像头</span>
                </div>
            `;
        }

        if (step.action === 'waitResult') {
            return `
                <div class="text-center">
                    <div class="text-4xl mb-4" id="onboarding-prediction">--</div>
                    <div class="text-sm text-gray-500">请将手放在摄像头前</div>
                </div>
            `;
        }

        return '';
    },

    getButtonText() {
        const step = this.steps[this.currentStep];
        if (step.action === 'startCamera') return '开启摄像头';
        if (step.action === 'checkApi') return '检查连接';
        if (step.action === 'waitResult') return '完成';
        if (this.currentStep === this.steps.length - 1) return '开始使用';
        return '下一步';
    },

    async next() {
        const step = this.steps[this.currentStep];

        if (step.action === 'checkApi') {
            const btn = document.getElementById('onboarding-btn');
            btn.disabled = true;
            btn.textContent = '检查中...';

            const status = document.getElementById('onboarding-status');
            try {
                const r = await fetch('/api/health');
                const d = await r.json();
                if (d.status === 'ok') {
                    status.innerHTML = `
                        <div class="inline-flex items-center gap-2 text-green-400">
                            <span>✓</span>
                            <span>API服务已连接 (${d.model_type || '未加载模型'})</span>
                        </div>
                    `;
                    btn.disabled = false;
                    setTimeout(() => this.nextStep(), 1000);
                    return;
                }
            } catch (e) {
                status.innerHTML = `
                    <div class="inline-flex items-center gap-2 text-red-400">
                        <span>✗</span>
                        <span>API服务未连接，请先运行: python api/app.py</span>
                    </div>
                `;
                btn.disabled = false;
                btn.textContent = '重试';
                return;
            }
        }

        if (step.action === 'startCamera') {
            try {
                await App.toggleCamera();
                this.nextStep();
            } catch (e) {
                document.getElementById('onboarding-status').innerHTML = `
                    <div class="inline-flex items-center gap-2 text-red-400">
                        <span>✗</span>
                        <span>摄像头启动失败: ${e.message}</span>
                    </div>
                `;
            }
            return;
        }

        if (step.action === 'waitResult') {
            this.finish();
            return;
        }

        this.nextStep();
    },

    nextStep() {
        if (this.currentStep < this.steps.length - 1) {
            this.currentStep++;
            const modal = document.getElementById('onboarding-modal');
            if (modal) {
                modal.innerHTML = this.renderStep();
            }
        }
    },

    skip() {
        this.finish();
    },

    finish() {
        localStorage.setItem(this.STORAGE_KEY, 'true');
        const modal = document.getElementById('onboarding-modal');
        if (modal) modal.remove();
    },

    reset() {
        localStorage.removeItem(this.STORAGE_KEY);
    }
};

// Auto-init when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => Onboarding.init());
} else {
    Onboarding.init();
}

window.Onboarding = Onboarding;
