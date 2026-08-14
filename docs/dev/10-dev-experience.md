---
title: 开发经验与踩坑记录
order: 4
icon: 🛠️
description: 工程实现角度的架构经验、特征技巧、关键踩坑与 Web 端经验
---

# 10 — 开发经验与踩坑记录

> 本文汇总本项目在开发过程中沉淀的工程经验与踩坑记录，供后续开发和二次贡献者参考。
> 与 [07-常见问题](07-faq.md)（面向使用者的排查）不同，本文侧重**从工程实现角度**复盘「为什么这样设计」「哪里容易出错」。

## 一、架构与设计经验

### 1.1 单一数据源原则（配置 + 常量分离）

项目将「可变配置」与「不可变常量」拆成两个模块，避免混用导致语义混乱：

| 模块 | 内容 | 变更频率 |
|------|------|----------|
| `src/config.py` | 路径、超参数、设备、采集参数 | 运行期/部署期可变 |
| `src/constants.py` | 关键点连接、特征维度、序列参数 | 业务规则，几乎不变 |

**经验**：路径与超参数集中在 `ProjectConfig` 单例，所有模块从 `config` 取值，彻底消除硬编码。新增模型时只需在 `classifier_models` / `deep_learning_models` 元组登记，`get_model_path` 会自动分派 `.pkl` / `.pth` 后缀——这是「字典/元组映射替代 if-else 分支」的典型应用，避免新增类型时漏改判断逻辑。

### 1.2 模板方法统一训练流程

`BaseModel` 用模板方法把训练循环、早停、学习率调度、最佳模型保存收敛到一个入口 `train_model`，子类只需实现 `forward` 与可选的 `_get_criterion` / `_get_optimizer` 钩子。

**经验**：当多个模型共享「训练-验证-保存」骨架时，用模板方法而非复制代码。所有深度学习模型的训练行为保持一致（早停、LR 调度、最佳权重回滚），这是后续评估结果可横向对比的前提。

### 1.3 按人员划分数据集，杜绝数据泄露

`split_by_person` / `auto_split_persons` 保证**同一人的样本不会跨集合**。

**经验**：手语等动作识别任务中，同一个人不同动作的相似度远高于跨人相似度。若随机切分，模型可能学到「这是某个人」而非「这是某个词」，导致评估虚高。按人员划分是这类任务的标准基线。

### 1.4 设备自动检测

`device='auto'` 时运行时探测 CUDA，模型 `to_device('auto')` 同理。这样脚本在无 GPU 机器上也能直接跑，避免硬编码 `cuda`。

---

## 二、数据与特征工程经验

### 2.1 特征相对化（以手腕为原点）

`FeatureExtractor` 把 21 个关键点坐标减去手腕坐标，实现位置不变性；再补充手指长度与关节角度，构成 71 维特征。

**踩坑**：角度计算用点积反余弦时，数值误差可能使余弦值略超 `[-1, 1]`，导致 `arccos` 返回 NaN。必须 `np.clip(cosine, -1, 1)` 并给分母加 `1e-8` 防除零。

```python
cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
angle = np.arccos(np.clip(cosine_angle, -1, 1))
```

### 2.2 数据增强：2D 与 3D 输入必须分治

`KeypointAugmenter.__call__` 区分输入维度：
- **3D** `(frames, landmarks, coords)`：可做平移/缩放/噪声/遮挡/时间扭曲
- **2D** `(frames, features)`：71 维特征无法按「关键点×3」分解，只做平移/噪声/时间扭曲

**踩坑**：若想当然把 2D 输入按 63 维坐标拆解去缩放/遮挡，会破坏后 8 维标量特征（手指长度、角度）。代码里 `_translate` 对 2D 输入只平移前 63 维、保留标量特征，就是这个原因。

### 2.3 序列对齐：补零 vs 截断

`load_sequence_data` 对短序列尾部补零到 `max_length`，长序列截断。补零发生在**时序末尾**而非开头，是为了保持动作起始相位。

**经验**：序列长度取 `max_sequence_length=30` 是平衡信息量与计算量的折中；`min_sequence_length=15` 用于采集质量过滤。补零方向要一致，否则引入虚假的时间偏移。

---

## 三、关键踩坑记录

### 3.1 模块导入路径问题

**现象**：`ModuleNotFoundError: No module named 'src'`

**原因**：工具脚本在 `tools/` 下，直接运行时项目根目录不在 `sys.path`。

**解决**：脚本开头插入根目录：

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

**经验**：所有 `tools/` 脚本统一做路径引导，保证「从任意目录运行」都可靠。本项目通过 `pyproject.toml` 的 CLI 入口点（`lingxin-train` 等）进一步规避了裸脚本依赖问题。

### 3.2 MediaPipe 双版本 API 冲突

**现象**：`AttributeError: module 'mediapipe' has no attribute 'solutions'`（或反向的 `no attribute 'tasks'`）

**原因**：MediaPipe 新旧 API 并存且互斥：
- 旧版 `mp.solutions.hands`（< 0.10.33）
- 新版 `mp.tasks.vision`（>= 0.10.33）

**解决**：本项目统一升级到 Task API（>= 0.10.33）。**教训**：升级大版本后务必用真实代码冒烟测试，README 里明确锁定版本下限，避免新老 API 混装。

### 3.3 OpenCV 中文显示乱码

**现象**：`cv2.putText()` 绘制中文显示为 `???`。

**原因**：OpenCV 默认字体不含中文字形。

**解决**：走 PIL 往返（cv2→PIL→cv2）。**踩坑**：采集工具对**静态文本**逐帧做 PIL 往返会造成明显性能开销——应改为 RGBA 预渲染一次 + alpha 合成，动态文本再增量更新。这也是帧率优化的关键点之一。

### 3.4 OpenCV `waitKey` 窗口泄露导致卡顿（重点坑）

**现象**：预览回放返回主界面后帧率明显下降。

**根因**：`waitKey` 是全局事件泵，会处理**所有已创建窗口**的消息。回放打开了 `Preview` 窗口但未销毁，导致每帧 `waitKey(1)` 做两倍事件处理；叠加摄像头缓冲区积压，表现为掉帧。

**解决**：
1. `try/finally` 确保回放结束必调 `cv2.destroyWindow('Preview')` + 数次 `waitKey(1)` 完成底层清理
2. 预览返回后立即 `_flush_capture` 清空积压帧

**经验**：OpenCV GUI 程序里，「打开窗口必销毁」「无限 `waitKey(0)` 有死锁风险」是两条铁律。用 `waitKey(33)` 循环 + 窗口关闭检测替代 `waitKey(0)`，可避免窗口失焦时主线程永久阻塞。

### 3.5 训练过拟合 / Loss 不下降

典型表现与对策：

| 现象 | 原因 | 对策 |
|------|------|------|
| 训练高、验证低 | 过拟合 | 数据增强、Dropout、L2、减小模型 |
| loss 为 NaN | 数据含 NaN / 学习率过大 | 清洗数据、StandardScaler、降学习率 |
| 只输出一个类别 | 类别不平衡 / 输出层维度不匹配 | 类别权重、过采样、检查 `num_classes` |

**经验**：小数据集上先用几十条样本验证模型能过拟合（train_acc 接近 100%），再上全量——能快速暴露模型/数据管线的结构性错误。

### 3.6 「运行 Python 文件」点了没反应（重点坑）

**现象**：点击编辑器右上角「运行 Python 文件」，终端只出现 `(lingxin-gpu) PS E:\Projects0\Lingxin>` 提示符，**没有任何命令被发出**。此前正常，某次之后突然失效。不报错、无日志，表现像是"根本没点"。

**排查路径（按证据链）**：
1. 排除解释器本身——用 lingxin-gpu 直接 `import` 项目模块、创建 Flask app 均正常，workspace storage 里解释器记忆也正确。
2. 看扩展宿主日志 `%APPDATA%\Code\logs\<会话>\window1\exthost\exthost.log`：其他扩展（claude-code、turbo-console-log、LiveServer 等）都有 `_doActivateExtension` 激活记录，唯独 `ms-python.python` 从头到尾**零记录** → 问题在扩展激活链路，不在解释器、不在项目代码。
3. 查扩展安装时间戳（`%USERPROFILE%\.vscode\extensions\extensions.json` 的 `updated` 字段）：`ms-python.python` 2026-05-23 装后从未更新；`ms-python.vscode-python-envs` 1.36.0 是 2026-06-19 更新过 → 嫌疑锁定在新改动的环境管理插件。

**根因**：`ms-python.vscode-python-envs`（Python Environment Manager）1.36.0 干扰了 `ms-python.python` 的激活，导致运行命令链路断裂。**卸载该插件即恢复**。

**经验**：
- 「运行 Python 文件」由 Python 扩展负责。症状是**静默无反应**而非报错时，优先怀疑扩展激活链路，而不是反复改解释器/终端配置。
- `exthost.log` 里有没有目标扩展的激活记录，是判断"扩展是否正常加载"的第一手证据，比瞎试配置高效得多。
- 环境管理类插件与 Python 主扩展职责高度重叠，出问题时可先卸载环境管理插件（python-envs）验证，这是最干净的单变量实验。

### 3.7 终端自动激活 conda 环境的弯路（排查干扰项）

**现象**：希望进入项目目录时终端自动激活 `lingxin-gpu`，多次方案失败。

**尝试过的方案（均不理想）**：
1. `python-envs.terminal.autoActivationType`（shellStartup）——python-envs 1.36.0 忽略该配置，无效。
2. `terminal.integrated.automationProfile.windows` 改成 `powershell`——这会让 Python 扩展的**自动化终端**也走 PowerShell 分支，对"运行"问题无济于事，还引入了配置变量的干扰。
3. 改 PowerShell profile 加目录判断自动激活——有效，但属于绕行方案，且在后续排查「运行无反应」时成为干扰变量，最终被回滚。

**经验**：
- 排查一个问题时**一次只改一个变量**。本次「运行无反应」的排查中，自动激活方案与插件问题叠加，一度把 `automationProfile` 误判为元凶，白走一轮。
- VSCode 扩展的配置项名看着合理未必生效（尤其新扩展），以日志证据为准，不要以配置文档为准。
- 最终结论：环境激活交给「状态栏解释器 + 项目 `.vscode/settings.json` 的 `python.defaultInterpreterPath`」，终端自动激活属于锦上添花，不值得为它引入 profile 级别的全局改动。

---

## 四、工程化经验

### 4.1 依赖单一真相源：`pyproject.toml`

`pyproject.toml` 是**唯一真相源**：声明依赖（`[project.dependencies]`）与 CLI 入口（`[project.scripts]`）。不再维护 `requirements.txt` 与 `environment.yml`（二者已删除）；仅保留 `environment-gpu.yml` 作为 GPU 环境重建手册（CUDA 12.8 + PyTorch + 清华镜像，针对 RTX 5060 Blackwell sm_120 配置），其 pip 段与 `pyproject.toml` 保持手动一致。

**经验**：多份依赖清单必然漂移——pip 装到 A、import 查 B。收敛为一份 `pyproject.toml` 后，安装/重建/新增依赖都只有一处入口。详见 [11-Python 环境与导入规范](11-python-env-and-import-guide.md) 的「依赖收敛」一节。

### 4.2 早停 + 学习率调度 + 最佳权重回滚

`EarlyStopping` 以「负数验证损失」作为单调分数，配合 `ReduceLROnPlateau`，训练结束用 `load_best` 回滚到最优权重而非最后一步。

**经验**：训练结束时「最后一个 epoch 的模型」往往不是最优，必须回滚到 `best_epoch` 再保存，否则线上模型被欠调优的末轮权重拖累。

### 4.3 日志系统统一

用 `src/utils/logger.py` 的 `get_logger` 替代散落 `print()`，便于定位问题与异步调试。深度学习模块均通过 `self.logger` 输出训练进度。

---

## 五、Web 端经验

### 5.1 文档在线查看器（`web/docs.html`）

- 前端用 `marked` v15 解析 Markdown，`link` 渲染器拦截 `.md` 内链转成 `loadDoc` 调用，实现站内无刷新跳转
- **踩坑**：marked v15 的 Renderer 方法接收 **token 对象**而非多个参数。旧版 `link(href, title, text)` 写法需改为 `link(token)`，否则链接渲染失效。

### 5.2 索引自动生成，告别三处硬编码

早期版本中，新增一篇文档需要同步 `web/docs.html` 的侧边栏按钮、`DOC_FILES` 搜索数组、`DOC_LABELS` 标签三处硬编码，漏改其一会导致「能点开但搜不到」的不一致。

现在改为：文档写好 YAML frontmatter 后运行 `python scripts/build_docs_index.py`，自动生成 `web/static/js/docs-index.generated.js` 统一驱动侧边栏、卡片与搜索。

### 5.3 CORS 限制

直接用浏览器打开 `web/index.html` 无法 `fetch` Markdown（跨域限制）。需在项目根目录起静态服务：`python -m http.server 8080`，再访问 `http://localhost:8080`。

---

## 六、开发流程建议（给贡献者）

1. **小步提交 + pre-commit**：项目已配置 `.pre-commit-config.yaml`（trailing-whitespace、commitizen 规范提交），提交前自动校验。
2. **改配置先看 `config.py`**：任何路径/超参数变动先查是否有现成配置项，避免二次硬编码。
3. **新增模型**：继承 `BaseModel` 实现 `forward`，在 `config.all_models` 登记，复用统一训练流程，勿重复造训练循环。
4. **采集新数据**：遵守 `data/raw/collected/{word}/{person}_{index:03d}.npy` 命名，预处理才能正确解析人员并按人员划分。
5. **新增文档**：写好 frontmatter（title/category/order/icon/description）后运行构建脚本生成索引，再提交。

---

*最后更新：2026-08-15*
