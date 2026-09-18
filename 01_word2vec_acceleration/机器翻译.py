from collections import Counter
import math


def compute_ngrams(tokens, n):
    """计算 n-gram 列表"""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def transparent_bleu(reference: str, candidate: str, max_n: int = 4, weights=None):
    """透视版 BLEU 计算函数：分步打印全部中间变量与状态"""
    if weights is None:
        weights = [1.0 / max_n] * max_n

    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)

    print(f"\n{'='*70}")
    print(f"【BLEU 全过程透视分析】")
    print(f"参考译文 (Reference): '{reference}'")
    print(f"候选译文 (Candidate): '{candidate}'")
    print(f"{'='*70}")

    # -------------------------------------------------------------
    # 步骤 1：长度统计与 Brevity Penalty (BP) 计算
    # -------------------------------------------------------------
    print(f"\n[步骤 1: 长度与短句惩罚 (Brevity Penalty, BP)]")
    print(f"  - 参考译文有效长度 (r): {ref_len} tokens -> {ref_tokens}")
    print(f"  - 候选译文有效长度 (c): {cand_len} tokens -> {cand_tokens}")

    if cand_len == 0:
        print("  - 候选译文为空，BLEU 得分为 0.0")
        return 0.0

    if cand_len <= ref_len:
        ratio = ref_len / cand_len
        exponent = 1.0 - ratio
        bp = math.exp(exponent)
        print(f"  - 条件判断: c <= r ({cand_len} <= {ref_len})，触发长度惩罚")
        print(f"  - BP 计算公式: exp(1 - r/c) = exp(1 - {ref_len}/{cand_len}) = exp({exponent:.4f})")
        print(f"  - 简短惩罚因子 (BP): {bp:.6f}")
    else:
        bp = 1.0
        print(f"  - 条件判断: c > r ({cand_len} > {ref_len})，无短句惩罚")
        print(f"  - 简短惩罚因子 (BP): {bp:.6f}")

    # -------------------------------------------------------------
    # 步骤 2：逐阶 N-gram 截断匹配精确率 (Modified Precision)
    # -------------------------------------------------------------
    print(f"\n[步骤 2: 逐阶 N-gram 匹配与截断精确率 (Modified Precision)]")
    precisions = []

    for n in range(1, max_n + 1):
        print(f"\n  --- 阶数: {n}-gram ---")
        cand_ngram_list = compute_ngrams(cand_tokens, n)
        ref_ngram_list = compute_ngrams(ref_tokens, n)

        cand_ngrams = Counter(cand_ngram_list)
        ref_ngrams = Counter(ref_ngram_list)

        total_cand_ngrams = len(cand_ngram_list)
        print(f"    * 候选生成总数 (分母): {total_cand_ngrams}")

        if total_cand_ngrams == 0:
            print(f"    * 候选生成数量为 0，{n}-gram 精确率 p_{n} = 0.0000")
            precisions.append(0.0)
            continue

        clipped_matches = 0
        print(f"    * 逐项比对明细:")
        for ngram, cand_count in cand_ngrams.items():
            ref_count = ref_ngrams.get(ngram, 0)
            # 关键截断逻辑：min(候选出现频次, 参考出现频次)
            match_val = min(cand_count, ref_count)
            clipped_matches += match_val
            ngram_str = " ".join(ngram)
            match_flag = "✓" if match_val > 0 else "✗"
            print(f"        [{match_flag}] '{ngram_str}': 候选频次={cand_count}, 参考频次={ref_count} -> 截断计入: min({cand_count}, {ref_count}) = {match_val}")

        p_n = clipped_matches / total_cand_ngrams
        precisions.append(p_n)
        print(f"    * {n}-gram 截断精确率 p_{n} = {clipped_matches} / {total_cand_ngrams} = {p_n:.6f}")

    # -------------------------------------------------------------
    # 步骤 3：对数加权几何平均 (Geometric Mean)
    # -------------------------------------------------------------
    print(f"\n[步骤 3: 几何平均汇总 (Geometric Mean)]")
    print(f"  - 各阶精确率: {[round(p, 6) for p in precisions]}")
    print(f"  - 各阶分配权重: {weights}")

    log_sum = 0.0
    zero_precision = False
    log_details = []

    for i, (p, w) in enumerate(zip(precisions, weights), 1):
        if p == 0:
            zero_precision = True
            log_val = -math.inf
            log_details.append(f"w_{i}*log(p_{i}) = {w} * log(0) = -inf")
        else:
            log_val = math.log(p)
            log_sum += w * log_val
            log_details.append(f"w_{i}*log(p_{i}) = {w:.2f} * log({p:.6f}) = {w * log_val:.6f}")

    for detail in log_details:
        print(f"    * {detail}")

    if zero_precision:
        geo_mean = 0.0
        print(f"  - 存在阶数精确率为 0，几何平均值退化为: 0.0000")
    else:
        geo_mean = math.exp(log_sum)
        print(f"  - 加权对数和: {log_sum:.6f}")
        print(f"  - 几何平均值 exp(sum(w_n * log(p_n))): {geo_mean:.6f}")

    # -------------------------------------------------------------
    # 步骤 4：最终得分结算
    # -------------------------------------------------------------
    bleu_score = bp * geo_mean
    print(f"\n[步骤 4: 最终 BLEU 计算]")
    print(f"  - 公式: BLEU = BP * 几何平均值")
    print(f"  - 计算: {bp:.6f} * {geo_mean:.6f} = {bleu_score:.6f}")
    print(f"{'='*70}\n")

    return bleu_score


# ============================================
# 典型案例测试
# ============================================
if __name__ == "__main__":
    ref_text = "the cat sat on the mat"

    # 案例 1：完全匹配
    transparent_bleu(ref_text, "the cat sat on the mat")

    # 案例 2：短句且发生截断（展示频次截断 min(count, ref_count) 与 BP 惩罚）
    transparent_bleu(ref_text, "the the the")

    # 案例 3：语序全乱（展示 1-gram 命中但高阶 n-gram 崩塌）
    transparent_bleu(ref_text, "mat the on sat cat the")