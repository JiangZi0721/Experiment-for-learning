# WhiteBox Transformer 教学系统

一个面向 Transformer 初学者的交互式教学系统。网页端以 10 个重新编排、重新表述的模块讲清 Transformer，并将原始课件截图和口语讲稿保留为可展开参考；Python 端保留可在本地运行的张量级 Encoder-Decoder 推演。

## 技术选择

- **HTML / CSS / 原生 JavaScript**：课程站是纯静态应用，没有为单页课程站引入 React、构建链或后端。
- **KaTeX CDN + 文本回退**：用于可选公式排版；加载失败时课程正文仍可阅读。
- **Python 标准库 + 可选 Rich**：现有白盒程序以确定性小矩阵演示 Embedding、注意力、掩码、交叉注意力与生成。
- **Vercel**：只部署网页静态资源；Python CLI 不属于 Vercel 运行时。

## 课程使用

### 主题切换

顶部“主题”下拉菜单可选择浅色或深色，默认浅色。选择同时作用于课程、透视公式、矩阵、注意力热图和概率分布，切换章节不会重置主题；刷新页面后保留上次选择。浏览器禁用本地存储时仍可切换，但无法跨刷新保存。

实现使用 `web/theme.js` 保存 `transformer-theme` 设置，并通过根元素的 `data-theme` 与 `web/theme.css` 统一颜色。原页面存在硬编码的深色公式背景、浅色矩阵文字，以及课程专用浅色样式，因此只改背景变量不足以修复；主题样式同时覆盖这些组件，但不改变热力图的数值背景与计算逻辑。本次仅调整本地网页，不继续部署。

从项目根目录启动静态服务器：

```powershell
cd F:\LearningNotes\Transformer
python -m http.server 8765
```

访问 `http://localhost:8765/web/`，点击“完整课程”。每个模块包含：

1. 重写后的书面教学正文。
2. 首次出现时就地展示的概念解释。
3. 公式、结构和常见误解。
4. 跳转到相关的“透视”实验，以及可展开的原始截图和讲稿参考。

“透视”中的数值为固定的 **教学玩具模型**，用于看清 Q/K/V、Softmax、Mask、Encoder Memory 和自回归计算，不能被解释为训练得到的翻译模型。

## 本地 Python 程序

```powershell
python main.py
python run_web.py
python whitebox_transformer/main.py --mode full
```

`main.py` 是根目录快捷入口，`whitebox_transformer/` 保存实际的白盒计算模块。网页使用同一组教学概念，但不依赖本地 Python 服务。

## 内容来源与结构

```text
Transformer_Full_Instructional_Course.md   原始 30 讲口语材料参考
Transformer_Architecture_Master.md         架构补充解释的唯一来源
images/slide_*.jpg                         30 张原始课程截图
web/index.html                             网页入口
web/app.js                                 原有张量级透视与自回归演播
web/course-v2.js                           10 个课程模块、概念卡和上下文透视入口
web/style.css                              网页样式
whitebox_transformer/                      本地 Python 白盒程序
docs/superpowers/specs/                    已确认的设计与实施计划
```

课程主体由 `web/course-v2.js` 以 10 个模块维护。原始 30 讲 Markdown 与截图是材料依据，不再作为默认阅读正文。必须通过 HTTP 服务访问；直接双击 `web/index.html` 会导致引用素材无法稳定加载。

## 完整性校验

```powershell
powershell -ExecutionPolicy Bypass -File verify-content.ps1
```

校验会确认 30 张原始材料截图和课程入口存在。

## Vercel 部署

部署根目录，因为网页要读取根目录的两份课程 Markdown 和 `images/`：

```powershell
npm i -g vercel
vercel login
cd F:\LearningNotes\Transformer
vercel link --yes --project transformer-learning-system
vercel deploy --prod --yes
```

若访客看到 Vercel 登录墙，在 Vercel 项目的 `Settings` -> `Deployment Protection` 关闭 Vercel Authentication。Vercel 只能托管网页端；Python 程序由 GitHub 仓库提供给本地运行。

## 已发现的问题与处理

| 问题 | 处理 |
| --- | --- |
| 原网页只覆盖少量概念，课程正文被省略 | 重构为 10 个递进模块，正文以初学者可读的书面教材表达。 |
| 原始讲稿口语化且存在转写风险 | 原始材料降为可展开参考；主课程按准确概念与公式重新组织。 |
| “透视”与课程阅读脱节 | 每个模块提供与当前主题匹配的透视入口，端到端演播保持不变。 |
| 固定玩具矩阵可能被误解为真实模型结果 | 所有课程透视入口明确标注其教学性质。 |
| 网页和 Python 端部署边界不清 | README 和 `vercel.json` 明确 Vercel 仅部署网页。 |
| 项目此前没有 Git 元数据 | 已初始化 Git 仓库；推送前必须先获取并合并远端历史，禁止强推。 |
