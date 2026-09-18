import numpy as np
import pandas as pd

def build_co_occurrence_matrix(text, window_size=2):
    """
    根据输入的一段话生成共现矩阵。
    """
    # 尝试使用 jieba 进行中文分词。如果没有安装，则默认按空格或字符分割
    try:
        import jieba
        words = list(jieba.cut(text))
        # 过滤掉空字符或标点（这里仅过滤空白字符作为示例）
        words = [w for w in words if w.strip() and w not in ["，", "。", "！", "？", ","]]
    except ImportError:
        print("提示: 未安装 jieba 库，将使用默认的空格/字符分割方式。可以通过 'pip install jieba' 安装。")
        words = text.split() if ' ' in text else list(text)
    
    # 提取词表并建立词到索引的映射
    vocab = list(set(words))
    print(f"词汇表: {vocab}\n")
    # 排序词汇表，保证每次运行输出的矩阵顺序一致
    vocab.sort() 
    print(f"排序后的词汇表: {vocab}\n")
    word_to_id = {word: i for i, word in enumerate(vocab)}
    print(f"词到索引映射: {word_to_id}\n")
    # 初始化共现矩阵 (全 0)
    matrix = np.zeros((len(vocab), len(vocab)), dtype=int)
    
    # 填充共现矩阵
    for i, target_word in enumerate(words):
        target_id = word_to_id[target_word]
        
        # 确定滑动窗口的边界
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        
        for j in range(start, end):
            if i != j:
                context_word = words[j]
                context_id = word_to_id[context_word]
                matrix[target_id][context_id] += 1
                
    # 使用 pandas DataFrame 使矩阵输出更美观
    df = pd.DataFrame(matrix, index=vocab, columns=vocab)
    return df

if __name__ == "__main__":
    sample_text = "you say goodbye and I say hello"
    print(f"输入文本: {sample_text}\n")
    
    # 设置窗口大小，例如 window_size=1 表示看目标词的前后各1个词
    matrix_df = build_co_occurrence_matrix(sample_text, window_size=1)
    
    print("共现矩阵:")
    print(matrix_df)
