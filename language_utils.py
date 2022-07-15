import nltk
import gensim
import numpy as np
from gensim.parsing.preprocessing import remove_stopwords

def normalization_text(text):
    return ' '.join(gensim.utils.simple_preprocess(text))

# def filter(text):
#     return remove_stopwords(text)

def tokenize(text):
    return list(gensim.utils.tokenize(text))

def convert_to_vec(tokenized_sentence,all_words):
    bag = np.zeros(len(all_words),dtype=np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx]=1.0
    return bag

# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
# bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]

# print(convert_to_vec(sentence,words))
# test_text = 'How long does delivery take?'
# print(tokenize(normalization_text(test_text)))
# print(tokenize(test_text))