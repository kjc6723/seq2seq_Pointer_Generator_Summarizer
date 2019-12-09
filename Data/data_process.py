import pandas as pd
import os, re
import numpy as numpy
import jieba
import tensorflow as tf


def seg_line(line):
    tokens = jieba.cut(str(line), cut_all=False)
    words = []
    for i in tokens:
        words.append(i)
    return " ".join(words)


def split_data(df):
    for i in df.keys():
        for j in df.index:
            tmp = df[i][j]
            tmp = re.findall('[\u4e00-\u9fa5a-zA-Z0-9-]+', str(tmp))
            tmp = "".join(tmp)
            df[i][j] = seg_line(tmp)
    df.dropna(axis=0, how='any', inplace=True)
    df.reset_index(inplace=True, drop=True)


if __name__ == "__main__":
    abspath = os.path.abspath("./")
    testText_file = os.path.join(abspath, "AutoMaster_TestSet.csv")
    trainText_file = os.path.join(abspath, "AutoMaster_TrainSet.csv")
    testText_saveFile = os.path.join(abspath, "AutoMaster_TestSet_save.csv")
    trainText_saveFile = os.path.join(abspath, "AutoMaster_TrainSet_save.csv")
    vocab_file = os.path.join(abspath, "vocab_File.csv")

    stopWord_File = os.path.join(abspath, "stopword.txt")
    corpusText_File = os.path.join(abspath, "corpus_TextSet.txt")
    """
    train = pd.read_csv(trainText_file)
    split_data(train)
    train['input'] = train['QID'] + ' ' + train['Brand'] + ' ' + train['Model'] + ' ' + train['Question'] + ' ' + train[
        'Dialogue']
    train.drop(['QID', 'Brand', 'Model', 'Question', 'Dialogue'], axis=1, inplace=True)
    train.to_csv(trainText_saveFile, index=False, encoding='utf-8')

    test = pd.read_csv(testText_file)
    split_data(test)
    test['input'] = test['QID'] + ' ' + test['Brand'] + ' ' + test['Model'] + ' ' + test['Question'] + ' ' + test[
        'Dialogue']
    test.drop(['QID', 'Brand', 'Model', 'Question', 'Dialogue'], axis=1, inplace=True)
    test.to_csv(testText_saveFile, index=False, encoding='utf-8')
    """
    train_save = pd.read_csv(trainText_saveFile)
    source = [str(d) for d in train_save['input'].values.tolist()]
    target = [str(d) for d in train_save['Report'].values.tolist()]

    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=' ', oov_token="unk")
    tokenizer.fit_on_texts(source + target)
    vocab = []
    with open(vocab_file, "w", encoding='utf-8') as f:
        for (k, v) in sorted(tokenizer.word_index.items(), key=lambda x: x[1]):
            f.write(str(k) + "," + str(v) + "\n")
