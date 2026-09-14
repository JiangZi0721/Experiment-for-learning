(() => {
  const courseUrl = "../Transformer_Full_Instructional_Course.md";
  const architectureUrl = "../Transformer_Architecture_Master.md";
  const lensByLesson = [
    "端到端自回归演播", "端到端自回归演播", "核心理论与证明", "核心理论与证明",
    "编码器透视", "编码器透视", "编码器透视", "编码器透视", "编码器透视", "编码器透视",
    "编码器透视", "编码器透视", "编码器透视", "编码器透视", "编码器透视", "编码器透视",
    "核心理论与证明", "编码器透视", "核心理论与证明", "解码器透视", "解码器透视", "解码器透视",
    "解码器透视", "解码器透视", "解码器透视", "解码器透视", "核心理论与证明", "核心理论与证明",
    "核心理论与证明", "核心理论与证明"
  ];

  const explanations = [
    ["先建立问题地图", "Transformer 是处理序列到序列任务的架构；后续章节会逐层解释它怎样把一串输入变成一串输出。", "先区分任务目标与某个具体模型：这里讲的是计算结构，不是某个聊天产品。"],
    ["理解 Seq2Seq 的输入输出", "编码器读取源序列，解码器在已有输出的条件下逐个产生目标 token。", "训练时可并行看到完整目标序列，推理时必须自回归，这两个场景不能混为一谈。"],
    ["把注意力放回应用", "同一套序列建模机制可以用于翻译、语音、摘要与视觉 token；输入 token 的含义会变，计算骨架不变。", "应用广泛不等于无需理解 Q、K、V；所有应用仍依赖这些基本计算。"],
    ["为什么需要 Self-Attention", "RNN 的时间依赖阻碍并行，CNN 的单层感受野是局部的；注意力让任意两个 token 一步交互。", "注意力不是免费：序列很长时注意力矩阵会带来二次复杂度。"],
    ["从 token 到向量", "模型实际处理的是向量序列。每个位置都能以 Query 主动向整个序列检索相关信息。", "全局可见不代表所有 token 获得相同权重。"],
    ["Q、K、V 的角色", "Query 表示当前位置要找什么，Key 表示可被匹配的索引，Value 是最终被汇聚的内容。", "Q、K、V 是不同线性投影，不是同一向量改三个名字。"],
    ["缩放点积", "QK^T 产生相关性得分；除以 sqrt(d_k) 让 Softmax 不因维度增大而过度饱和。", "缩放不是为了把所有分数限制在 0 到 1；归一化是 Softmax 的工作。"],
    ["Softmax 汇聚", "Softmax 把每行得分变成和为 1 的权重，再对 Value 作加权和。", "权重和为 1 不意味着每个 Value 的贡献相等。"],
    ["矩阵并行", "把每个 token 的 q、k、v 堆叠成矩阵，就能用一次矩阵乘法同时计算所有位置的关系。", "并行计算不改变每个位置的数学定义。"],
    ["读懂注意力矩阵", "分数矩阵的行对应 Query 位置，列对应 Key 位置；读取行才能知道某个当前位置在关注谁。", "矩阵写法的转置方向取决于行向量或列向量约定，必须同时检查形状。"],
    ["从权重到输出", "Attention 权重矩阵乘 V，得到与输入长度相同、但已融合上下文的新表示。", "输出不是某一个 token 的复制，而是 Value 的加权组合。"],
    ["多头的目的", "不同头在不同子空间进行关系检索，允许模型同时学习不同的依赖模式。", "多头不是把同一个注意力结果简单重复多次。"],
    ["多头并行", "总维度 D 被切为 h 个 d_k=D/h 的子空间，因此注意力主计算量仍是 O(N^2D)。", "头数增加不保证效果必然更好；每头维度太小也会受限。"],
    ["拼接与输出投影", "各头输出拼接回 D 维，再通过 W^O 让子空间的信息重新混合。", "Concat 后仍需线性投影，不能把拼接误认为最终表示。"],
    ["位置是必要信息", "纯注意力对输入排列不敏感，因此必须加入位置编码。", "位置编码不是额外 token，而是加到每个 token 表示上的同维向量。"],
    ["正余弦位置编码", "不同频率的 sin/cos 让位置具有可比较、可外推的连续表示。", "正余弦编码不是唯一方案；可学习位置嵌入也是常用设计。"],
    ["相加而非拼接", "相加保持模型维度不变，后续线性层仍能分离词义和位置信息。", "相加并不表示位置信息消失；它在向量各维中持续参与计算。"],
    ["编码器层", "Encoder 由自注意力、残差归一化和逐位置 FFN 组成，输出为供解码器读取的 memory。", "残差不是跳过计算，而是把输入和子层输出相加。"],
    ["LayerNorm", "LayerNorm 在每个 token 的特征维度上归一化，适合长度和 batch 会变化的序列。", "它不是在时间维或整个 batch 上做统计。"],
    ["解码器与生成", "Decoder 先处理已生成的目标前缀，再读取 Encoder memory，最后预测下一个 token。", "解码器不是 Encoder 的简单镜像：它额外需要因果约束和交叉注意力。"],
    ["因果掩码", "把未来位置的分数设为极小值，Softmax 后这些位置权重严格为零。", "掩码不是在输出后删除结果，而是在注意力归一化之前阻止信息泄漏。"],
    ["交叉注意力", "Decoder 状态提供 Q，Encoder memory 提供 K 和 V；输出长度由目标序列决定。", "交叉注意力不是把两段序列直接拼接。"],
    ["观察头的分工", "热力图能显示某个教学示例中每个 Query 对 Key 的权重分布。", "单个热力图只能解释当前固定示例，不能概括所有训练模型。"],
    ["Teacher Forcing", "训练时以真实前缀作为 Decoder 输入，让所有目标位置可以并行计算损失。", "Teacher forcing 并不等同于推理过程。"],
    ["Exposure Bias", "训练见到真实前缀、推理见到自身预测前缀，两者分布差异会累积错误。", "它是训练和推理条件不同带来的问题，不是 Softmax 的数值错误。"],
    ["解码策略", "贪婪搜索每步选最大概率；束搜索保留若干候选序列以比较整体概率。", "束宽增加会增加计算，也未必总让生成更符合人的偏好。"],
    ["Self-Attention 与 CNN", "注意力可看作数据依赖的自适应感受野，而卷积通常使用固定局部核。", "两者并非互斥，现代模型也会组合注意力与卷积思想。"],
    ["Self-Attention 与 RNN", "注意力缩短任意 token 间的信息路径并支持并行，但长序列成本仍需认真处理。", "Transformer 不是所有序列任务的无条件最优解。"],
    ["ViT", "把图像分块当作 token，Transformer 的序列计算骨架可以迁移到视觉。", "视觉应用仍要处理二维结构和归纳偏置，不能只照搬文本设置。"],
    ["回到整体", "把输入表示、注意力、多头、编码器、掩码、交叉注意力和生成串成一个完整闭环。", "记住名词不等于掌握；应回到透视实验验证每一步的张量形状和作用。"]
  ];

  let lessons = [];
  let architecture = "";

  const escapeHtml = value => value.replace(/[&<>\"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[char]));
  const parseLessons = markdown => markdown.split(/(?=## 🖼️ 第 \d+ 讲：)/).filter(part => /^## 🖼️ 第 \d+ 讲：/.test(part)).map((part, index) => {
    const title = (part.match(/^## 🖼️ 第 \d+ 讲：(.*)$/m) || ["", `第 ${index + 1} 讲`])[1].trim();
    const image = (part.match(/!\[[^\]]*\]\((images\/[^)]+)\)/) || [])[1] || "";
    return { index, title, image, source: part.replace(/!\[[^\]]*\]\(images\/[^)]+\)\n?/, "").trim() };
  });
  const renderSource = source => escapeHtml(source).replace(/^### (.*)$/gm, "<h4>$1</h4>").replace(/^> (.*)$/gm, "<p class=\"course-note\">$1</p>").replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>").replace(/\n\n/g, "</p><p>").replace(/\n/g, "<br>");

  function renderLesson(index) {
    const lesson = lessons[index];
    const [heading, explanation, misconception] = explanations[index];
    const sidebar = document.getElementById("step-nav-list");
    document.getElementById("sidebar-category-title").textContent = "📚 完整课程目录";
    sidebar.innerHTML = lessons.map((item, i) => `<li><button class="step-item ${i === index ? "active" : ""}" data-course-lesson="${i}" aria-current="${i === index ? "page" : "false"}"><span class="step-num">${String(i + 1).padStart(2, "0")}</span><span>${item.title}</span></button></li>`).join("");
    sidebar.querySelectorAll("[data-course-lesson]").forEach(button => button.addEventListener("click", () => renderLesson(Number(button.dataset.courseLesson))));
    document.getElementById("main-content-view").innerHTML = `
      <article class="course-workspace">
        <p class="eyebrow">第 ${index + 1} / 30 讲 · ${lensByLesson[index]}</p>
        <h1>${lesson.title}</h1>
        <section class="course-section"><h2>原文讲义</h2>${lesson.image ? `<img class="course-slide" src="../${lesson.image}" alt="第 ${index + 1} 讲原始课程截图">` : ""}<div class="course-source"><p>${renderSource(lesson.source)}</p></div></section>
        <section class="course-section"><h2>分层解释：${heading}</h2><p>${explanation}</p><h3>常见误解</h3><p>${misconception}</p><details><summary>查看相关架构笔记全文</summary><div class="course-source"><p>${renderSource(architecture)}</p></div></details></section>
        <section class="course-section lens-callout"><h2>对应透视实验</h2><p>这是固定数值的教学玩具模型，不是训练权重。切换到“${lensByLesson[index]}”后，可逐步查看公式、张量形状和注意力热力图。</p><button class="btn-nav primary" data-open-lens="${lensByLesson[index]}">打开透视实验</button></section>
        <section class="course-section"><h2>本讲检查</h2><p>${heading}：请用自己的话解释本讲的核心机制，并回到透视实验核对其输入、输出和约束条件。</p></section>
      </article>`;
    document.querySelector("[data-open-lens]").addEventListener("click", event => {
      const label = event.currentTarget.dataset.openLens;
      const tab = [...document.querySelectorAll(".nav-tab[data-tab]")].find(button => button.textContent.includes(label));
      if (tab) tab.click();
    });
  }

  async function openCourse() {
    const content = document.getElementById("main-content-view");
    content.innerHTML = "<p class=\"course-loading\">正在加载 30 讲完整原文…</p>";
    try {
      const [course, master] = await Promise.all([fetch(courseUrl).then(response => response.text()), fetch(architectureUrl).then(response => response.text())]);
      lessons = parseLessons(course);
      architecture = master;
      if (lessons.length !== 30) throw new Error(`课程解析失败：预期 30 讲，实际 ${lessons.length} 讲。`);
      renderLesson(0);
    } catch (error) {
      content.innerHTML = `<section class="section-card"><h2>课程原文无法加载</h2><p>${escapeHtml(error.message)}</p><p>请从项目根目录运行 <code>python -m http.server 8765</code>，再访问 <code>/web/</code>；直接双击 HTML 会被浏览器阻止读取课程文件。</p></section>`;
    }
  }

  window.addEventListener("DOMContentLoaded", () => document.getElementById("course-tab").addEventListener("click", openCourse));
})();
