# 旋转位置编码 (RoPE) 数学原理与外推性

## 1. 绝对位置编码与相对位置编码的融合
传统 Absolute Embedding 简单将位置向量与 Token 嵌入相加，缺乏相对距离的几何不变性。

## 2. 旋转矩阵的复数推导
RoPE 将二维向量视为复数，通过正交旋转矩阵对 Query 和 Key 进行旋转：
$$R_{\Theta, m}^d = \text{diag}\left(R_{\theta_1, m}, R_{\theta_2, m}, ..., R_{\theta_{d/2}, m}\right)$$
其中内积满足：
$$\langle R_m q, R_n k \rangle = g(q, k, m-n)$$
两个 Token 的注意力得分只取决于它们在序列中的**相对距离 $m-n$**，天然具备极强的长上下文外推（Length Extrapolation）能力。
