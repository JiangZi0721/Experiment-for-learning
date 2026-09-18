import numpy as np
def into_corpus(text):
    text = text.lower()
    words = text.split(" ")
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word

def create_co_matrix(corpus, window_size):
    text_size = len(corpus)
    size = len(set(corpus))
    co_matrix = np.zeros((size,size), dtype=np.int32)
    for idx,word in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx - i
            right_idx = idx + i
            if left_idx >= 0:
                left_word = corpus[left_idx]
                co_matrix[word, left_word] += 1
            if right_idx < text_size:
                right_word = corpus[right_idx]
                co_matrix[word, right_word] += 1
    print(corpus)
    return co_matrix

if __name__ == "__main__": 
    text = ["you", "say", "goodbye", "and", "I", "say", "hello","."]
    corpus, word_to_id, id_to_word = into_corpus(" ".join(text))
    co_matrix = create_co_matrix(corpus, window_size=1)
    print("Corpus:", corpus)
    print("Co-occurrence matrix:\n", co_matrix)
