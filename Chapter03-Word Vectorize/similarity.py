import gensim

if __name__ == '__main__':
    w2v = gensim.models.KeyedVectors.load_word2vec_format('../models/word2vec.txt')
    print(w2v.most_similar(positive=['apple']))
