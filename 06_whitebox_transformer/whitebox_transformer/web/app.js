// ==============================================================================
// WhiteBox Transformer: 前端核心交互逻辑与全景数值数据引擎 (v2.0 增强版)
// ==============================================================================

const MODEL_CONFIG = {
  d_model: 4,
  n_heads: 2,
  d_k: 2,
  d_ffn: 8,
  src_vocab: ["我", "喜欢", "机器", "学习"],
  tgt_vocab: ["<BOS>", "I", "love", "machine", "learning", "<EOS>", "AI", "like"]
};

// ----------------- 静态微型真实矩阵数据 (全 5 步无缝覆盖) -----------------
const DATA = {
  src_tokens: ["我", "喜欢", "机器", "学习"],
  src_embedding: [
    [ 0.820, -0.410,  0.150,  0.930],
    [ 0.120,  0.950, -0.620,  0.310],
    [-0.750,  0.220,  0.880, -0.450],
    [-0.680,  0.180,  0.920, -0.380]
  ],
  src_pe: [
    [ 0.0000,  1.0000,  0.0000,  1.0000],
    [ 0.8415,  0.5403,  0.0100,  0.9999],
    [ 0.9093, -0.4161,  0.0200,  0.9998],
    [ 0.1411, -0.9900,  0.0300,  0.9995]
  ],
  X_src: [
    [ 0.820,  0.590,  0.150,  1.930],
    [ 0.961,  1.490, -0.610,  1.310],
    [ 0.159, -0.196,  0.900,  0.550],
    [-0.539, -0.810,  0.950,  0.620]
  ],
  enc_Q: [
    [ 0.783,  0.024,  0.496,  0.895],
    [ 0.412,  0.198,  0.963, -0.057],
    [ 0.478, -0.185, -0.297,  0.529],
    [ 0.221, -0.306, -0.640,  0.689]
  ],
  enc_K: [
    [ 0.805,  0.722,  0.279,  0.989],
    [ 0.419,  1.291,  0.177,  0.544],
    [ 0.358,  0.088, -0.203,  0.428],
    [-0.174, -0.618,  0.347,  0.275]
  ],
  enc_V: [
    [ 1.096,  0.449,  0.485,  0.635],
    [ 1.225,  0.033,  0.985,  0.948],
    [ 0.103,  0.428,  0.109, -0.222],
    [-0.201, -0.324, -0.280, -0.178]
  ],
  enc_scores_scaled: [
    [ 0.741,  0.463,  0.449, -0.016],
    [ 0.538,  0.782,  0.065, -0.098],
    [ 0.435,  0.126,  0.322,  0.109],
    [ 0.178, -0.119,  0.195,  0.288]
  ],
  enc_attn_weights: [
    [ 0.347,  0.263,  0.260,  0.130],
    [ 0.285,  0.364,  0.178,  0.173],
    [ 0.301,  0.221,  0.269,  0.209],
    [ 0.245,  0.182,  0.249,  0.324]
  ],
  enc_head1_attn: [
    [ 0.385,  0.252,  0.248,  0.115],
    [ 0.210,  0.490,  0.180,  0.120],
    [ 0.330,  0.200,  0.310,  0.160],
    [ 0.200,  0.140,  0.280,  0.380]
  ],
  enc_head2_attn: [
    [ 0.309,  0.274,  0.272,  0.145],
    [ 0.360,  0.238,  0.176,  0.226],
    [ 0.272,  0.242,  0.228,  0.258],
    [ 0.290,  0.224,  0.218,  0.268]
  ],
  enc_mha_out: [
    [ 0.712,  0.285,  0.514,  0.589],
    [ 0.684,  0.192,  0.621,  0.647],
    [ 0.345,  0.261,  0.231,  0.189],
    [ 0.112, -0.045,  0.021,  0.045]
  ],
  enc_add_norm1: [
    [ 0.582, -0.325, -0.841,  0.584],
    [ 0.712,  0.985, -1.542, -0.155],
    [-0.124, -0.852,  1.214, -0.238],
    [-0.681, -0.924,  1.102,  0.503]
  ],
  enc_memory: [
    [ 0.652, -0.281, -0.892,  0.521],
    [ 0.741,  0.892, -1.482, -0.151],
    [-0.098, -0.785,  1.152, -0.269],
    [-0.612, -0.874,  1.025,  0.461]
  ],

  // ----------------- Decoder 全 5 步 Cross-Attention 热力矩阵 -----------------
  dec_cross_attn_all: [
    [ 0.280,  0.240,  0.260,  0.220], // <BOS>
    [ 0.820,  0.080,  0.050,  0.050], // I -> "我" (82%)
    [ 0.110,  0.780,  0.060,  0.050], // love -> "喜欢" (78%)
    [ 0.050,  0.080,  0.810,  0.060], // machine -> "机器" (81%)
    [ 0.040,  0.050,  0.070,  0.840], // learning -> "学习" (84%)
    [ 0.100,  0.120,  0.180,  0.600]  // <EOS> -> 终结
  ],

  // ----------------- Decoder 全 5 步因果掩码自注意力矩阵 -----------------
  dec_causal_attn_all: [
    [ 1.000,  0.000,  0.000,  0.000,  0.000],
    [ 0.382,  0.618,  0.000,  0.000,  0.000],
    [ 0.195,  0.423,  0.382,  0.000,  0.000],
    [ 0.120,  0.250,  0.350,  0.280,  0.000],
    [ 0.080,  0.150,  0.250,  0.270,  0.250]
  ]
};

// ----------------- 自回归生成演播台步进定义 (含全部 5 步的预测概率) -----------------
const SEQ2SEQ_STEPS = [
  {
    step: 1,
    tgt_tokens: ["<BOS>"],
    predicted: "I",
    prob: "71.2%",
    cross_focus: "我 (82%)",
    causal_focus: "<BOS> (100%)",
    desc: "起始输入 &lt;BOS&gt;，交叉注意力将 82% 权重聚焦在中文首词【我】，成功以 71.2% 高概率预测生成英语主语【I】！",
    probs: [
      { token: "I", prob: 0.712, rank: 1 },
      { token: "like", prob: 0.124, rank: 2 },
      { token: "love", prob: 0.081, rank: 3 },
      { token: "machine", prob: 0.032, rank: 4 },
      { token: "learning", prob: 0.021, rank: 5 },
      { token: "<BOS>", prob: 0.015, rank: 6 },
      { token: "<EOS>", prob: 0.010, rank: 7 },
      { token: "AI", prob: 0.005, rank: 8 }
    ]
  },
  {
    step: 2,
    tgt_tokens: ["<BOS>", "I"],
    predicted: "love",
    prob: "68.5%",
    cross_focus: "喜欢 (78%)",
    causal_focus: "I (61.8%)",
    desc: "输入扩展为 ['&lt;BOS&gt;', 'I']，交叉注意力准确锁定源端动词【喜欢】，预测出英文谓语【love】！",
    probs: [
      { token: "love", prob: 0.685, rank: 1 },
      { token: "like", prob: 0.203, rank: 2 },
      { token: "machine", prob: 0.052, rank: 3 },
      { token: "learning", prob: 0.030, rank: 4 },
      { token: "AI", prob: 0.015, rank: 5 },
      { token: "<EOS>", prob: 0.009, rank: 6 },
      { token: "I", prob: 0.004, rank: 7 },
      { token: "<BOS>", prob: 0.002, rank: 8 }
    ]
  },
  {
    step: 3,
    tgt_tokens: ["<BOS>", "I", "love"],
    predicted: "machine",
    prob: "84.3%",
    cross_focus: "机器 (81%)",
    causal_focus: "love (38.2%) + I (42.3%)",
    desc: "输入推进至 ['&lt;BOS&gt;', 'I', 'love']，交叉注意力全面对齐源端专有名词【机器】，精准生成【machine】！",
    probs: [
      { token: "machine", prob: 0.843, rank: 1 },
      { token: "AI", prob: 0.101, rank: 2 },
      { token: "learning", prob: 0.028, rank: 3 },
      { token: "like", prob: 0.015, rank: 4 },
      { token: "love", prob: 0.006, rank: 5 },
      { token: "<EOS>", prob: 0.004, rank: 6 },
      { token: "I", prob: 0.002, rank: 7 },
      { token: "<BOS>", prob: 0.001, rank: 8 }
    ]
  },
  {
    step: 4,
    tgt_tokens: ["<BOS>", "I", "love", "machine"],
    predicted: "learning",
    prob: "91.8%",
    cross_focus: "学习 (84%)",
    causal_focus: "machine (28%) + love (35%)",
    desc: "输入扩展到 machine，注意力不仅关注源句的【学习】，更结合 machine 的上下文搭配，以 91.8% 极高置信度预测【learning】！",
    probs: [
      { token: "learning", prob: 0.918, rank: 1 },
      { token: "AI", prob: 0.041, rank: 2 },
      { token: "machine", prob: 0.025, rank: 3 },
      { token: "<EOS>", prob: 0.009, rank: 4 },
      { token: "like", prob: 0.004, rank: 5 },
      { token: "love", prob: 0.002, rank: 6 },
      { token: "I", prob: 0.001, rank: 7 },
      { token: "<BOS>", prob: 0.000, rank: 8 }
    ]
  },
  {
    step: 5,
    tgt_tokens: ["<BOS>", "I", "love", "machine", "learning"],
    predicted: "<EOS>",
    prob: "95.6%",
    cross_focus: "全句终结 (60%)",
    causal_focus: "learning (25%) + machine (27%)",
    desc: "输入完整覆盖源句语义，交叉注意力确认源端信息已全部表达完毕，预测出序列结束标记【&lt;EOS&gt;】，翻译圆满结束！",
    probs: [
      { token: "<EOS>", prob: 0.956, rank: 1 },
      { token: "learning", prob: 0.028, rank: 2 },
      { token: "AI", prob: 0.009, rank: 3 },
      { token: "machine", prob: 0.004, rank: 4 },
      { token: "love", prob: 0.002, rank: 5 },
      { token: "like", prob: 0.001, rank: 6 },
      { token: "I", prob: 0.000, rank: 7 },
      { token: "<BOS>", prob: 0.000, rank: 8 }
    ]
  }
];

// ----------------- 优雅 HTML 数学排版解析器 (无网络 100% 高保真) -----------------
function formatLatexToHTML(latexStr) {
  if (!latexStr) return '';
  let str = latexStr;

  // 1. 换行
  str = str.replace(/\\\\/g, '<br/>');

  // 2. 分式 \frac{A}{B}
  str = str.replace(/\\frac\{([^{}]+)\}\{([^{}]+)\}/g, (match, num, den) => {
    return `<span class="math-frac"><span class="math-num">${formatLatexToHTML(num)}</span><span class="math-den">${formatLatexToHTML(den)}</span></span>`;
  });
  // 嵌套分式处理一次
  str = str.replace(/\\frac\{([^{}]+)\}\{([^{}]+)\}/g, (match, num, den) => {
    return `<span class="math-frac"><span class="math-num">${formatLatexToHTML(num)}</span><span class="math-den">${formatLatexToHTML(den)}</span></span>`;
  });

  // 3. 根号 \sqrt{A}
  str = str.replace(/\\sqrt\{([^{}]+)\}/g, (match, body) => {
    return `<span class="math-sqrt"><span class="math-radic">√</span><span class="math-sqrt-stem">${formatLatexToHTML(body)}</span></span>`;
  });

  // 4. 函数名与文本 \text{...}
  str = str.replace(/\\text\{([^{}]+)\}/g, '<span class="math-fn">$1</span>');

  // 5. 符号与希腊字母
  const symMap = {
    '\\mathbb{R}': '<span class="math-var" style="font-weight:bold;">ℝ</span>',
    '\\cdot': ' · ',
    '\\times': ' × ',
    '\\dots': ' … ',
    '\\in': ' ∈ ',
    '\\alpha': 'α',
    '\\beta': 'β',
    '\\gamma': 'γ',
    '\\mu': 'μ',
    '\\sigma': 'σ',
    '\\epsilon': 'ε',
    '\\sin': '<span class="math-fn">sin</span>',
    '\\cos': '<span class="math-fn">cos</span>',
    '\\max': '<span class="math-fn">max</span>',
    '\\sum': '<span style="font-size:1.3em; vertical-align:-2px;">∑</span>',
    '\\arg\\max': '<span class="math-fn">arg max</span>',
    '\\le': ' ≤ ',
    '\\ge': ' ≥ ',
    '-\\infty': ' -∞ '
  };

  for (let [k, v] of Object.entries(symMap)) {
    str = str.split(k).join(v);
  }

  // 6. 上标与下标
  str = str.replace(/\^\{([^{}]+)\}/g, '<sup class="math-sup">$1</sup>');
  str = str.replace(/\^([a-zA-Z0-9T]+)/g, '<sup class="math-sup">$1</sup>');
  str = str.replace(/_\{([^{}]+)\}/g, '<sub class="math-sub">$1</sub>');
  str = str.replace(/_([a-zA-Z0-9ijkm]+)/g, '<sub class="math-sub">$1</sub>');

  return `<div class="math-block">${str}</div>`;
}

// 统一公式渲染触发器
function renderMathFormulas() {
  document.querySelectorAll('.math-formula-body').forEach(el => {
    const rawLatex = el.getAttribute('data-formula') || el.innerText;
    if (!rawLatex) return;
    el.setAttribute('data-formula', rawLatex);

    // 优先尝试外部 KaTeX
    if (window.katex) {
      try {
        katex.render(rawLatex, el, {
          displayMode: true,
          throwOnError: false
        });
        return;
      } catch (err) {
        console.warn("KaTeX render fallback to HTML:", err);
      }
    }

    // 离线/CDN失败时纯 HTML 高保真回退
    el.innerHTML = formatLatexToHTML(rawLatex);
  });
}

// ----------------- 系统应用状态 -----------------
let currentTab = 'seq2seq';
let currentStepIdx = 0;
let expandedSteps = new Set();
let autoExpandAll = false;
let currentSeqStep = 1;
let autoPlayTimer = null;

// ----------------- DOM 工具与辅助函数 -----------------
function formatNum(num, p = 3) {
  return (num >= 0 ? '+' : '') + Number(num).toFixed(p);
}

function renderMatrixTable(matrix, rowLabels, colLabels, title = "") {
  let html = `<div class="matrix-card">`;
  if (title) html += `<div class="matrix-header"><span>${title}</span><span class="math-matrix-dim">(${matrix.length}x${matrix[0].length})</span></div>`;
  html += `<table class="tensor-table"><thead><tr><th>Idx</th>`;
  colLabels.forEach(c => { html += `<th>${c}</th>`; });
  html += `</tr></thead><tbody>`;

  matrix.forEach((row, i) => {
    html += `<tr><td class="row-label">${rowLabels ? rowLabels[i] : '#' + i}</td>`;
    row.forEach(val => {
      html += `<td>${formatNum(val, 3)}</td>`;
    });
    html += `</tr>`;
  });
  html += `</tbody></table></div>`;
  return html;
}

function renderHeatmapTable(matrix, rowLabels, colLabels, title = "") {
  let html = `<div class="matrix-card">`;
  if (title) html += `<div class="matrix-header"><span>🔥 ${title}</span><span class="math-matrix-dim">行和 = 1.000</span></div>`;
  html += `<table class="tensor-table"><thead><tr><th>Query \\ Key</th>`;
  colLabels.forEach(c => { html += `<th>${c}</th>`; });
  html += `</tr></thead><tbody>`;

  matrix.forEach((row, i) => {
    html += `<tr><td class="row-label">${rowLabels[i]}</td>`;
    row.forEach(val => {
      let bgCol = `rgba(99, 102, 241, ${Math.max(0.06, val * 0.95)})`;
      let textCol = val > 0.4 ? '#ffffff' : '#e2e8f0';
      html += `<td class="heat-cell" style="background:${bgCol}; color:${textCol}; font-weight:bold;" title="Attention Weight: ${(val*100).toFixed(1)}%">
        <span class="heat-val">${val.toFixed(3)}</span>
      </td>`;
    });
    html += `</tr>`;
  });
  html += `</tbody></table></div>`;
  return html;
}

// ----------------- 步骤结构定义 -----------------
const ENCODER_STEPS = [
  {
    id: "enc_step_1",
    title: "1. 词嵌入 (Embedding) 与位置编码 (Positional Encoding)",
    badge: "Input Preprocessing",
    concept: `
      <h4>💡 为什么必须引入位置编码？</h4>
      <p>自注意力机制天然具有<strong>置换不变性 (Permutation Invariance)</strong>。如果不加位置信息，句子被打乱后计算出的注意力完全一致！必须人工注入位置坐标。</p>
      <h4>🔍 用户敏锐直觉解惑（One-Hot 拼接 vs 逐元素相加）：</h4>
      <p>拼接 One-Hot 向量并乘以权重：<code>[e_token, e_pos] @ [W_tok; W_pos] = e_token@W_tok + e_pos@W_pos</code>。数学上严格等价于<strong>词嵌入 + 位置嵌入直接相加</strong>！</p>
      <p>原论文之所以直接使用正弦余弦相加：1) 避免维度膨胀翻倍；2) 高维空间（512维）极其正交稀疏，相加互不干扰，后续能自如解耦；3) 三角和角公式赋予了天然的相对位置线性变换。</p>
    `,
    formula: `X = \\text{Embedding}(Token) \\cdot \\sqrt{D} + PE(pos) \\\\
PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/D}}\\right), \\quad PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/D}}\\right)`,
    renderTensors: () => {
      return `<div class="matrix-grid">
        ${renderMatrixTable(DATA.src_embedding, DATA.src_tokens, ["dim_0", "dim_1", "dim_2", "dim_3"], "1. 静态词嵌入 E (4x4, 无位置)")}
        ${renderMatrixTable(DATA.src_pe, ["Pos_0", "Pos_1", "Pos_2", "Pos_3"], ["dim_0", "dim_1", "dim_2", "dim_3"], "2. 正弦/余弦位置编码 PE (4x4)")}
        ${renderMatrixTable(DATA.X_src, DATA.src_tokens, ["dim_0", "dim_1", "dim_2", "dim_3"], "3. 融合输出输入矩阵 X = E + PE (4x4)")}
      </div>`;
    }
  },
  {
    id: "enc_step_2",
    title: "2. Q (Query), K (Key), V (Value) 线性投影生成",
    badge: "Linear Projections",
    concept: `
      <h4>🎭 三大角色的物理分工</h4>
      <ul>
        <li><strong>Query (q_i)</strong>：当前 Token 发出的“检索意向”——我想搜寻什么样的上下文？</li>
        <li><strong>Key (k_j)</strong>：当前 Token 对外提供的“索引标签”——我身上有哪些特征供你匹配？</li>
        <li><strong>Value (v_j)</strong>：当前 Token 实际携带的“实质语义信息”——一旦匹配成功，我贡献什么给全局？</li>
      </ul>
      <p>通过三组不同权重矩阵投影，每个词获得了在多重视角下沟通互动的能力。</p>
    `,
    formula: `Q = X W_Q \\in \\mathbb{R}^{N \\times D}, \\quad K = X W_K \\in \\mathbb{R}^{N \\times D}, \\quad V = X W_V \\in \\mathbb{R}^{N \\times D}`,
    renderTensors: () => {
      return `<div class="matrix-grid">
        ${renderMatrixTable(DATA.enc_Q, DATA.src_tokens.map(t=>`q(${t})`), ["d0", "d1", "d2", "d3"], "Query 矩阵 Q (4x4)")}
        ${renderMatrixTable(DATA.enc_K, DATA.src_tokens.map(t=>`k(${t})`), ["d0", "d1", "d2", "d3"], "Key 矩阵 K (4x4)")}
        ${renderMatrixTable(DATA.enc_V, DATA.src_tokens.map(t=>`v(${t})`), ["d0", "d1", "d2", "d3"], "Value 矩阵 V (4x4)")}
      </div>`;
    }
  },
  {
    id: "enc_step_3",
    title: "3. 缩放点积注意力 (Scaled Dot-Product Attention)",
    badge: "Core Attention Mechanism",
    concept: `
      <h4>⚡ 为什么除以 sqrt(d_k)？（李宏毅教授核心考点）</h4>
      <p>假设 q 和 k 的各个分量独立同分布，均值 0，方差 1。点积是 d_k 个乘积相加，<strong>其方差会急剧放大为 d_k</strong>（标准差 sqrt(d_k)）！</p>
      <p>若不缩放，在 d_k 较大时点积数值极大，导致送入 Softmax 后落入两极饱和区，<strong>导数趋近于 0，产生严重梯度消失</strong>！除以 sqrt(d_k) 将方差恒定拉回 1，梯度保持灵敏。</p>
    `,
    formula: `\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V`,
    renderTensors: () => {
      return `<div class="matrix-grid">
        ${renderMatrixTable(DATA.enc_scores_scaled, DATA.src_tokens, DATA.src_tokens, "缩放后得分矩阵 (Q K^T / sqrt(d_k))")}
        ${renderHeatmapTable(DATA.enc_attn_weights, DATA.src_tokens, DATA.src_tokens, "注意力权重矩阵 (Attention Map)")}
      </div>`;
    }
  },
  {
    id: "enc_step_4",
    title: "4. 多头注意力机制 (Multi-Head Attention) 子空间分治",
    badge: "Subspace Routing",
    concept: `
      <h4>🔀 多头到底在干什么？增加了计算量吗？</h4>
      <p><strong>完全没有增加 FLOPs 计算量</strong>！单头计算为 O(N²·D)，分成 h 个头后每头维度为 D/h，总计算量为 h × O(N²·(D/h)) = O(N²·D)。</p>
      <p><strong>物理本质</strong>：单头只能从单一角度构建关联；多头机制允许模型在不同子空间同时关注<strong>局部相邻修饰</strong>、<strong>跨句主谓宾搭配</strong>与<strong>指代关系</strong>，最后 Concat 拼接并通过 W_O 融合。</p>
    `,
    formula: `\\text{MHA}(X) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h) W_O, \\quad \\text{head}_i = \\text{Attention}(Q_i, K_i, V_i)`,
    renderTensors: () => {
      return `<div class="matrix-grid">
        ${renderHeatmapTable(DATA.enc_head1_attn, DATA.src_tokens, DATA.src_tokens, "Head 1 注意力图 (子空间1, d_k=2)")}
        ${renderHeatmapTable(DATA.enc_head2_attn, DATA.src_tokens, DATA.src_tokens, "Head 2 注意力图 (子空间2, d_k=2)")}
        ${renderMatrixTable(DATA.enc_mha_out, DATA.src_tokens, ["d0", "d1", "d2", "d3"], "拼接后经 W_O 融合输出 (4x4)")}
      </div>`;
    }
  },
  {
    id: "enc_step_5",
    title: "5. 残差连接与层归一化 (Add & LayerNorm)",
    badge: "Residual & Stabilization",
    concept: `
      <h4>🛡️ 为什么坚决使用 LayerNorm 而非 BatchNorm？</h4>
      <p><strong>Add (残差连接)</strong>：提供无衰减导数直通通道（d(X+F(X))/dX = 1 + dF/dX），梯度保底有 1，解决深层网络退化与梯度弥散。</p>
      <p><strong>LayerNorm</strong>：针对单条样本、单个 Token 内部在其全部特征维度 D 上求均值和方差。完全独立于 Batch 大小和动态序列长度变化，在 NLP 变长场景下极其稳健！</p>
    `,
    formula: `\\text{Output} = \\text{LayerNorm}(X + \\text{Sublayer}(X)) \\\\
\\mu = \\frac{1}{D} \\sum_{i=1}^D x_i, \\quad \\sigma^2 = \\frac{1}{D} \\sum_{i=1}^D (x_i - \\mu)^2, \\quad y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\cdot \\gamma + \\beta`,
    renderTensors: () => {
      return `<div class="matrix-grid">
        ${renderMatrixTable(DATA.enc_add_norm1, DATA.src_tokens, ["dim_0", "dim_1", "dim_2", "dim_3"], "Add & LayerNorm 第一级标准化输出 (各行均值为0, 方差为1)")}
      </div>`;
    }
  },
  {
    id: "enc_step_6",
    title: "6. 逐位置前馈网络 (Position-wise FFN) 与第二级 Add & Norm",
    badge: "Non-linear Enhancement",
    concept: `
      <h4>🧠 为什么注意力后还要接 FFN？</h4>
      <p>注意力机制本质上是<strong>线性空间重组</strong>（把各个词的信息收集混合起来）。</p>
      <p>FFN 则是两层 MLP：先升维至 4D (如 8 维)，经过 ReLU / GELU 激发强大的非线性记忆和知识抽象能力，再降维压缩回 D (4 维)。每个 Token 独立并行通过这套权重。</p>
    `,
    formula: `\\text{FFN}(x) = \\max(0, x W_1 + b_1) W_2 + b_2`,
    renderTensors: () => {
      return `<div class="matrix-grid">
        ${renderMatrixTable(DATA.enc_memory, DATA.src_tokens, ["dim_0", "dim_1", "dim_2", "dim_3"], "Encoder 最终输出 Memory 矩阵 (供 Decoder 交叉注意力调用)")}
      </div>`;
    }
  }
];

const DECODER_STEPS = [
  {
    id: "dec_step_1",
    title: "1. 目标序列预处理与因果掩码自注意力 (Masked Self-Attention)",
    badge: "Causal Self-Attention",
    concept: `
      <h4>🚫 为什么必须施加因果掩码 (Causal Mask)？</h4>
      <p>自回归解码在生成第 t 个词时，<strong>绝对不能偷看 t 之后的未来词</strong>！</p>
      <p>训练时为了一次性并行处理整句目标文本，必须在右上三角加上 -∞ 掩码。经过 Softmax 之后，未来位置的注意力权重被<strong>物理归零 (0.0000)</strong>，保证因果时序严格成立！</p>
    `,
    formula: `\\text{MaskedAttention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}} + M\\right) V, \\quad M_{ij} = \\begin{cases} 0 & j \\le i \\\\ -\\infty & j > i \\end{cases}`,
    renderTensors: () => {
      const sampleTgt = ["<BOS>", "I", "love"];
      return `<div class="matrix-grid">
        ${renderMatrixTable([[0,1,1],[0,0,1],[0,0,0]], sampleTgt, sampleTgt, "因果下三角掩码 M (1为遮蔽未来, 0为可见)")}
        ${renderHeatmapTable(DATA.dec_causal_attn_all.slice(0, 3).map(r=>r.slice(0,3)), sampleTgt, sampleTgt, "因果掩码自注意力图 (右上三角严格为 0.000)")}
      </div>`;
    }
  },
  {
    id: "dec_step_2",
    title: "2. 编解码交叉注意力 (Cross-Attention: Decoder Q <- Encoder K,V)",
    badge: "Cross-Attention Nexus",
    concept: `
      <h4>🌉 编解码器信息交汇的咽喉要道</h4>
      <ul>
        <li><strong>Query 来自 Decoder</strong>：目标端当前翻译进展（“我现在需要翻译什么？”）。</li>
        <li><strong>Key & Value 来自 Encoder Memory</strong>：源端中文全文的高阶事实依据（“源句提供了什么记忆？”）。</li>
      </ul>
      <p><strong>维度解耦奇迹</strong>：即使源句长 100，目标句长 2，Q(2xD) @ K^T(Dx100) -> (2x100)，乘 V(100xD) -> (2xD)！目标序列长度与源序列长度完美解耦！</p>
    `,
    formula: `Q = X_{dec} W_Q^{cross}, \\quad K = \\text{Memory}_{enc} W_K^{cross}, \\quad V = \\text{Memory}_{enc} W_V^{cross} \\\\
\\text{CrossAttention} = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V`,
    renderTensors: () => {
      return `<div class="matrix-grid">
        ${renderHeatmapTable(DATA.dec_cross_attn_all.slice(0, 4), ["<BOS>", "I", "love", "machine"], DATA.src_tokens, "Cross-Attention 跨语言对齐热力图 (Decoder \\ Encoder)")}
      </div>`;
    }
  },
  {
    id: "dec_step_3",
    title: "3. 解码器 FFN 与三级 Add & LayerNorm",
    badge: "Decoder FFN & Normalization",
    concept: `
      <h4>🏛️ 解码器三级复合结构</h4>
      <p>每个 Decoder 包含三个残差子层：1) Masked Self-Attention; 2) Cross-Attention; 3) FFN。</p>
      <p>每一级均配备 Add & LayerNorm，确保来自源端与目标端的多源信息梯度平稳回传。</p>
    `,
    formula: `\\text{Sub}_1 = \\text{LN}(Y + \\text{MaskedMHA}(Y)) \\\\
\\text{Sub}_2 = \\text{LN}(\\text{Sub}_1 + \\text{CrossMHA}(\\text{Sub}_1, \\text{Memory})) \\\\
\\text{Output} = \\text{LN}(\\text{Sub}_2 + \\text{FFN}(\\text{Sub}_2))`,
    renderTensors: () => {
      const sampleTgt = ["<BOS>", "I", "love", "machine"];
      return `<div class="matrix-grid">
        ${renderMatrixTable(sampleTgt.map(()=>[0.521, -0.341, 0.892, -1.072]), sampleTgt, ["dim_0", "dim_1", "dim_2", "dim_3"], "Decoder 最终输出表征矩阵 (M x D)")}
      </div>`;
    }
  },
  {
    id: "dec_step_4",
    title: "4. 最终输出层：线性投影 (Linear) 与全词表 Softmax 预测",
    badge: "Next-Token Generation",
    concept: `
      <h4>🎯 下一个 Token 是如何诞生的？</h4>
      <p>1. 提取解码器输出矩阵的<strong>最后一行向量 x_last (1 x D)</strong>，它凝聚了当前位置的全部上下文记忆。</p>
      <p>2. 乘以线性输出权重 W_vocab (D x |V|)，映射为每个候选词的未归一化得分 (Logits)。</p>
      <p>3. 经过 Softmax 转化为全词表概率分布，通过 Greedy 策略选出概率最大的词！</p>
    `,
    formula: `\\text{Logits} = x_{last} W_{vocab}, \\quad P(w) = \\text{softmax}(\\text{Logits}), \\quad \\hat{w} = \\arg\\max_w P(w)`,
    renderTensors: () => {
      let probsHtml = `<div class="matrix-card"><div class="matrix-header">全词表 Softmax 预测概率分布排行榜</div><div style="display:flex; flex-direction:column; gap:0.6rem; margin-top:0.6rem;">`;
      SEQ2SEQ_STEPS[0].probs.forEach(item => {
        probsHtml += `
          <div class="prob-item">
            <span class="prob-name">#${item.rank} '${item.token}'</span>
            <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${item.prob*100}%"></div></div>
            <span class="prob-val">${(item.prob*100).toFixed(1)}%</span>
          </div>
        `;
      });
      probsHtml += `</div></div>`;
      return `<div class="matrix-grid">
        ${renderMatrixTable([[-0.20, 1.42, 0.22, -0.80, -0.70, -0.10, -0.60, 0.10]], ["Logits"], MODEL_CONFIG.tgt_vocab, "输出 Logits (1 x |V|)")}
        ${probsHtml}
      </div>`;
    }
  }
];

const THEORY_TOPICS = [
  {
    title: "专题 1：One-Hot 拼接投影 vs 逐元素相加代数证明",
    formula: `[e_{token}, e_{pos}] \\begin{bmatrix} W_{tok} \\\\ W_{pos} \\end{bmatrix} = e_{token} W_{tok} + e_{pos} W_{pos}`,
    concept: `
      <h4>💡 严密的代数等价性推导</h4>
      <p>设词向量为 e_token ∈ ℝ^(1 × D)，位置独热向量为 e_pos ∈ ℝ^(1 × N)。</p>
      <p>将两者水平拼接：<code>[e_token, e_pos] ∈ ℝ^(1 × (D+N))</code>。乘以权重矩阵 <code>W = [W_tok; W_pos]</code>：</p>
      <p>由于 e_pos 是独热向量，<code>e_pos @ W_pos</code> 本质上就是<strong>从位置矩阵中查出第 pos 个可学习位置向量</strong>！</p>
      <p>只要 W_tok 取单位阵，<strong>拼接投影与直接相加在数学上严格等价</strong>！标准 Transformer 选择直接相加，是为了彻底避免序列长度导致参数量几何膨胀，且高维空间（512维）具备强大的天然正交性，相加互不干扰。</p>
    `
  },
  {
    title: "专题 2：为什么点积必须除以 sqrt(d_k)？（方差膨胀与梯度饱和）",
    formula: `\\text{Var}(q_m k_m) = 1, \\quad \\text{Var}(q \\cdot k) = d_k, \\quad \\text{Std}(q \\cdot k) = \\sqrt{d_k} \\\\
\\text{Var}\\left(\\frac{q \\cdot k}{\\sqrt{d_k}}\\right) = \\frac{d_k}{(\\sqrt{d_k})^2} = 1`,
    concept: `
      <h4>⚡ 方差放大推导与梯度弥散</h4>
      <p>设 q 和 k 各分量独立，均值 0，方差 1。点积 <code>q · k = sum_{m=1}^{d_k} (q_m * k_m)</code>。</p>
      <p>在 d_k 较大时（如 64），点积数值极易达到几十。送入 Softmax 后，极大值占据全部概率，<strong>进入两极饱和区</strong>。</p>
      <p>在饱和区中，Softmax 局部导数 <code>∂S_i/∂z_j ≈ 0</code>，反向传播梯度瞬间断流（梯度消失）！除以 sqrt(d_k) 强行将方差拉回 1，使梯度保持在最敏感区间。</p>
    `
  },
  {
    title: "专题 3：多头注意力为何不增加计算量但拓宽子空间？",
    formula: `\\text{FLOPs}_{single} = 2 N^2 D, \\quad \\text{FLOPs}_{multi} = h \\times \\left[2 N^2 \\left(\\frac{D}{h}\\right)\\right] = 2 N^2 D`,
    concept: `
      <h4>🧮 浮点计算量 (FLOPs) 等价证明</h4>
      <p>设序列长度 N，模型维度 D，头数 h，每个头 d_k = D / h。</p>
      <ul>
        <li>单头注意力计算量：<code>Q @ K^T (N²D) + Softmax @ V (N²D) = 2 N² D</code></li>
        <li>h 个头的注意力计算量：<code>h × [2 N² (D/h)] = 2 N² D</code></li>
      </ul>
      <p><strong>结论：计算量完全相同！</strong>但多头允许模型在不同的低维投影子空间并行捕获不同的语言特征（紧邻修饰、跨句主谓宾搭配、同义指代等），最后线性拼接融合，集多重视角之大成。</p>
    `
  },
  {
    title: "专题 4：为什么坚决使用 LayerNorm 而非 BatchNorm？",
    formula: `\\text{LN}(x) = \\frac{x - \\mu_L}{\\sqrt{\\sigma_L^2 + \\epsilon}} \\cdot \\gamma + \\beta, \\quad \\mu_L = \\frac{1}{D} \\sum_{i=1}^D x_i`,
    concept: `
      <h4>⚖️ NLP 变长文本与小批次稳定性</h4>
      <p><strong>BatchNorm 的软肋</strong>：跨 Batch 在所有句子的同一特征维度求统计量。在 NLP 中各句子长短不一（大量填充 PAD），Batch 大小和句子长度的微小波动会导致均值方差剧烈抖动；推理时单句无法有效计算 Batch 方差。</p>
      <p><strong>LayerNorm 的胜利</strong>：在单条样本、单个 Token 内部在其全部特征维度 D 上求统计量。完全解耦 Batch 维度与序列长度，无论单句推演还是批量训练，极其稳健！</p>
    `
  }
];

// ----------------- 界面渲染控制器 -----------------
function initApp() {
  bindEvents();
  renderNavigation();
  renderCurrentView();
}

function bindEvents() {
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', (e) => {
      document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
      const target = e.currentTarget;
      target.classList.add('active');
      currentTab = target.dataset.tab;
      currentStepIdx = 0;
      renderNavigation();
      renderCurrentView();
    });
  });

  const toggleAuto = document.getElementById('toggle-auto-expand');
  if (toggleAuto) {
    toggleAuto.addEventListener('change', (e) => {
      autoExpandAll = e.target.checked;
      renderCurrentView();
    });
  }
}

function renderNavigation() {
  const sidebarTitle = document.getElementById('sidebar-category-title');
  const stepList = document.getElementById('step-nav-list');

  if (currentTab === 'seq2seq') {
    sidebarTitle.innerText = "🚀 自回归生成流";
    stepList.innerHTML = `
      <li class="step-item ${currentSeqStep === 1 ? 'active' : ''}" onclick="goToSeqStep(1)"><span class="step-num">1</span> 步 #1: 预测首词 'I'</li>
      <li class="step-item ${currentSeqStep === 2 ? 'active' : ''}" onclick="goToSeqStep(2)"><span class="step-num">2</span> 步 #2: 预测谓语 'love'</li>
      <li class="step-item ${currentSeqStep === 3 ? 'active' : ''}" onclick="goToSeqStep(3)"><span class="step-num">3</span> 步 #3: 预测名词 'machine'</li>
      <li class="step-item ${currentSeqStep === 4 ? 'active' : ''}" onclick="goToSeqStep(4)"><span class="step-num">4</span> 步 #4: 预测宾语 'learning'</li>
      <li class="step-item ${currentSeqStep === 5 ? 'active' : ''}" onclick="goToSeqStep(5)"><span class="step-num">5</span> 步 #5: 预测终结符 '&lt;EOS&gt;'</li>
    `;
    return;
  }

  let steps = currentTab === 'encoder' ? ENCODER_STEPS : (currentTab === 'decoder' ? DECODER_STEPS : THEORY_TOPICS);
  sidebarTitle.innerText = currentTab === 'encoder' ? "🏛️ 编码器 6 大层" : (currentTab === 'decoder' ? "🎭 解码器 4 大层" : "📐 4 大核心专题");

  stepList.innerHTML = steps.map((s, idx) => `
    <li class="step-item ${idx === currentStepIdx ? 'active' : ''}" onclick="selectStep(${idx})">
      <span class="step-num">${idx + 1}</span>
      <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${s.title}</span>
    </li>
  `).join('');
}

function selectStep(idx) {
  currentStepIdx = idx;
  renderNavigation();
  renderCurrentView();
}

function renderCurrentView() {
  const container = document.getElementById('main-content-view');

  if (currentTab === 'seq2seq') {
    renderSeq2SeqView(container);
    renderMathFormulas();
    return;
  }

  if (currentTab === 'theory') {
    renderTheoryView(container);
    renderMathFormulas();
    return;
  }

  const steps = currentTab === 'encoder' ? ENCODER_STEPS : DECODER_STEPS;
  const curStep = steps[currentStepIdx];
  const isExpanded = autoExpandAll || expandedSteps.has(curStep.id);

  container.innerHTML = `
    <div class="section-card">
      <div class="card-header">
        <div>
          <div class="card-title">${curStep.title}</div>
          <div style="font-size:0.85rem; color:var(--text-secondary); margin-top:0.3rem;">
            当前模块：${currentTab === 'encoder' ? '编码器 (Encoder Block)' : '解码器 (Decoder Block)'}
          </div>
        </div>
        <span class="card-badge">${curStep.badge}</span>
      </div>

      <!-- 核心原理 -->
      <div class="theory-box">
        ${curStep.concept}
      </div>

      <!-- 数学公式呈现容器 -->
      <div class="math-formula-container">
        <div class="math-formula-label">📐 数学表达规范</div>
        <div class="math-formula-body" data-formula="${curStep.formula}">${curStep.formula}</div>
      </div>

      <!-- 主动探针展开触发器 (Active Probing Trigger) -->
      <div class="probe-trigger">
        <div class="probe-prompt">
          <span class="icon">👉</span>
          <span>是否展开查看【本步骤具体计算过程与数值张量】？</span>
        </div>
        <button class="btn-probe ${isExpanded ? 'expanded' : ''}" onclick="toggleProbe('${curStep.id}')">
          ${isExpanded ? '收起数值透视' : '✨ 展开数值透视'}
        </button>
      </div>

      <!-- 数值张量透视详情面板 -->
      ${isExpanded ? `
        <div class="tensor-inspector">
          <div class="tensor-title">
            <span>🔬 数值白盒透视看板</span>
            <span style="font-size:0.75rem; color:var(--text-muted);">(真实仿真浮点矩阵)</span>
          </div>
          ${curStep.renderTensors()}
        </div>
      ` : ''}

      <!-- 步进导航按钮 -->
      <div class="stepper-footer">
        <button class="btn-nav" onclick="prevStep()" ${currentStepIdx === 0 ? 'disabled' : ''}>
          ← 上一步
        </button>
        <span style="font-size:0.85rem; color:var(--text-muted);">
          第 ${currentStepIdx + 1} / ${steps.length} 步
        </span>
        <button class="btn-nav primary" onclick="nextStep()" ${currentStepIdx === steps.length - 1 ? 'disabled' : ''}>
          下一步 →
        </button>
      </div>
    </div>
  `;

  renderMathFormulas();
}

function toggleProbe(stepId) {
  if (expandedSteps.has(stepId)) {
    expandedSteps.delete(stepId);
  } else {
    expandedSteps.add(stepId);
  }
  renderCurrentView();
}

function nextStep() {
  const steps = currentTab === 'encoder' ? ENCODER_STEPS : DECODER_STEPS;
  if (currentStepIdx < steps.length - 1) {
    currentStepIdx++;
    renderNavigation();
    renderCurrentView();
  }
}

function prevStep() {
  if (currentStepIdx > 0) {
    currentStepIdx--;
    renderNavigation();
    renderCurrentView();
  }
}

// ----------------- 端到端 Seq2Seq 演播台逻辑 (全 5 步无缝连贯) -----------------
function renderSeq2SeqView(container) {
  const curStepData = SEQ2SEQ_STEPS[currentSeqStep - 1];

  let srcTokensHtml = MODEL_CONFIG.src_vocab.map(t => `<span class="token-pill src">${t}</span>`).join('');
  let tgtTokensHtml = curStepData.tgt_tokens.map(t => {
    if (t === '<BOS>') return `<span class="token-pill tgt-bos">${t}</span>`;
    return `<span class="token-pill tgt-gen">${t}</span>`;
  }).join('');

  // 预测概率排行条 HTML
  let probsHtml = `<div class="matrix-card"><div class="matrix-header"><span>🎯 当前步全词表 Softmax 预测置信度排行</span></div><div style="display:flex; flex-direction:column; gap:0.5rem; margin-top:0.5rem;">`;
  curStepData.probs.forEach(item => {
    probsHtml += `
      <div class="prob-item">
        <span class="prob-name">#${item.rank} '${item.token}'</span>
        <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${item.prob*100}%"></div></div>
        <span class="prob-val">${(item.prob*100).toFixed(1)}%</span>
      </div>
    `;
  });
  probsHtml += `</div></div>`;

  // 截取当前步骤对应的 Cross-Attention 矩阵切片 (保证完全覆盖)
  const crossAttnSlice = DATA.dec_cross_attn_all.slice(0, curStepData.tgt_tokens.length);
  // 截取当前因果掩码自注意力切片
  const causalSlice = DATA.dec_causal_attn_all.slice(0, curStepData.tgt_tokens.length).map(row => row.slice(0, curStepData.tgt_tokens.length));

  container.innerHTML = `
    <div class="section-card">
      <div class="card-header">
        <div>
          <div class="card-title">🌟 端到端 Seq2Seq 自回归翻译全景演播台</div>
          <div style="font-size:0.85rem; color:var(--text-secondary); margin-top:0.3rem;">
            中文输入: "我 喜欢 机器 学习" ➔ 逐步自回归预测英文: "I love machine learning &lt;EOS&gt;"
          </div>
        </div>
        <span class="card-badge">Step ${currentSeqStep} / 5</span>
      </div>

      <!-- 序列流水线呈现 -->
      <div class="generation-panel">
        <div>
          <div style="font-size:0.82rem; color:var(--text-muted); margin-bottom:0.4rem;">📥 编码器源端序列 (Encoder Input):</div>
          <div class="token-pipeline">${srcTokensHtml}</div>
        </div>
        
        <div>
          <div style="font-size:0.82rem; color:var(--text-muted); margin-bottom:0.4rem;">📤 解码端自回归输入 (Decoder Context):</div>
          <div class="token-pipeline">
            ${tgtTokensHtml}
            <span style="color:var(--text-muted); font-size:1.1rem; margin:0 0.3rem;">➔</span>
            <span class="token-pill ${curStepData.predicted === '<EOS>' ? 'tgt-eos' : 'tgt-gen'}" style="border-width:2px;">
              预测输出: ${curStepData.predicted}
            </span>
          </div>
        </div>
      </div>

      <!-- 步进动态解析 -->
      <div class="theory-box">
        <h4>🔄 当前预测解析 (Step #${curStepData.step} / 5)</h4>
        <p>${curStepData.desc}</p>
        <div style="display:flex; gap:1.5rem; margin-top:0.5rem; flex-wrap:wrap;">
          <div>🎯 <strong>Cross-Attention 聚焦</strong>: <code>${curStepData.cross_focus}</code></div>
          <div>🛡️ <strong>Causal-Self 聚焦</strong>: <code>${curStepData.causal_focus}</code></div>
        </div>
      </div>

      <!-- 操作按钮群 -->
      <div style="display:flex; gap:0.8rem; align-items:center; flex-wrap:wrap;">
        <button class="btn-nav primary" onclick="stepSeqForward()" ${currentSeqStep >= 5 ? 'disabled' : ''}>
          ▶️ 生成下一个 Token (Step ${currentSeqStep + 1})
        </button>
        <button class="btn-nav" onclick="autoPlaySeq()" id="btn-auto-play">
          ${autoPlayTimer ? '⏹️ 停止自动播放' : '⚡ 自动连贯播放'}
        </button>
        <button class="btn-nav" onclick="resetSeq()">
          🔄 重置回到初始步
        </button>
      </div>

      <!-- 注意力动态对齐热力图与概率排行榜 -->
      <div class="tensor-inspector">
        <div class="tensor-title">
          <span>🔥 当前生成步的双重视角注意力与预测排行榜 (Step #${currentSeqStep})</span>
        </div>
        <div class="matrix-grid">
          ${renderHeatmapTable(
            crossAttnSlice, 
            curStepData.tgt_tokens, 
            MODEL_CONFIG.src_vocab, 
            `Cross-Attention 跨语言对齐 (${curStepData.tgt_tokens.length} 目标词 x 4 源词)`
          )}
          ${renderHeatmapTable(
            causalSlice, 
            curStepData.tgt_tokens, 
            curStepData.tgt_tokens, 
            `Masked-Self 因果掩码自注意力 (${curStepData.tgt_tokens.length} x ${curStepData.tgt_tokens.length})`
          )}
          ${probsHtml}
        </div>
      </div>
    </div>
  `;
}

function goToSeqStep(s) {
  currentSeqStep = s;
  renderNavigation();
  renderCurrentView();
}

function stepSeqForward() {
  if (currentSeqStep < 5) {
    currentSeqStep++;
    renderNavigation();
    renderCurrentView();
  }
}

function resetSeq() {
  if (autoPlayTimer) {
    clearInterval(autoPlayTimer);
    autoPlayTimer = null;
  }
  currentSeqStep = 1;
  renderNavigation();
  renderCurrentView();
}

function autoPlaySeq() {
  if (autoPlayTimer) {
    clearInterval(autoPlayTimer);
    autoPlayTimer = null;
    renderCurrentView();
    return;
  }
  autoPlayTimer = setInterval(() => {
    if (currentSeqStep < 5) {
      currentSeqStep++;
      renderNavigation();
      renderCurrentView();
    } else {
      clearInterval(autoPlayTimer);
      autoPlayTimer = null;
      renderCurrentView();
    }
  }, 1800);
  renderCurrentView();
}

// ----------------- 理论专题渲染 -----------------
function renderTheoryView(container) {
  const curTopic = THEORY_TOPICS[currentStepIdx];
  container.innerHTML = `
    <div class="section-card">
      <div class="card-header">
        <div class="card-title">${curTopic.title}</div>
        <span class="card-badge">Mathematical Proof</span>
      </div>

      <!-- 核心原理 -->
      <div class="theory-box">
        ${curTopic.concept}
      </div>

      <!-- 数学推导公式 -->
      <div class="math-formula-container">
        <div class="math-formula-label">📐 严格数学推导规范</div>
        <div class="math-formula-body" data-formula="${curTopic.formula}">${curTopic.formula}</div>
      </div>

      <div class="stepper-footer">
        <button class="btn-nav" onclick="prevStep()" ${currentStepIdx === 0 ? 'disabled' : ''}>← 上一个专题</button>
        <span style="font-size:0.85rem; color:var(--text-muted);">专题 ${currentStepIdx + 1} / ${THEORY_TOPICS.length}</span>
        <button class="btn-nav primary" onclick="nextStep()" ${currentStepIdx === THEORY_TOPICS.length - 1 ? 'disabled' : ''}>下一个专题 →</button>
      </div>
    </div>
  `;
}

// 启动入口
window.addEventListener('DOMContentLoaded', initApp);
