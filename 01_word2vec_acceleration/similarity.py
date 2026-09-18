import consin
import my_co_matrix
import numpy as np

text = "you say goodbye and I say hello."
corpus, word_to_id, id_to_word = my_co_matrix.into_corpus(text)
co_matrix = my_co_matrix.create_co_matrix(corpus, window_size=1)
def most_similar(query, corpus=corpus, co_matrix=co_matrix, eps=1e-8):
    similarity = {word: consin.cos_similarity(co_matrix[word_to_id[query]], co_matrix[word_to_id[word]]) for word in word_to_id if word != query}
    return sorted(similarity.items(), key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    query = "you"
    print(f"与 '{query}' 最相似的词:")
    for word, sim in most_similar(query):
        print(f"{word}: {sim:.4f}")