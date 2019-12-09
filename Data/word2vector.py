import logging
import itertools
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import os
import time


# train model
def train_w2v(text_file, mode_file):
    start_time = time.time()
    w2v = Word2Vec(LineSentence(text_file), workers=4, min_count=5, max_vocab_size=50000, size=128)
    w2v.save(mode_file)
    print(time.time() - start_time)


if __name__ == "__main__":
    abspath = os.path.abspath("../")
    text_File = os.path.join(abspath, "Data", "corpus_TextSet.txt")
    model_File = os.path.join(abspath,  "Data",  "w2v.model")

    train_w2v(text_File, model_File)
