# seq2seq_Pointer_Generator_Summarizer
A deep learning neural network for abstractive deep summarization
This is a project of generating abstractive summerization from Chinese conversation. The funny conversation is between customers and car technicians, with 80000+ samples for training and testing and 20000 samples for prediction.

Everything is classic and built with tensorflow 2.0, word embedding is pretrained by word2vec, and seq2seq includes Gru as encoder, Bahdanau attention and unidirection Gru as decoder. The model also embrace pointer generator network and coverage loss to deal with oov and repeating. ref. arXiv:1704.04368v2. Prediction implements beam search.

The data pipline is somehow typical for Chinese, purge data - segment - tokenize - batch. However it's tricky to deal with long conversation and to add special token to word2vec model. A tfidf filter is used. Special tokens is added to the w2v model by retraining the model.

Files like original dataset, segment dataset, w2v model are also provided for immediate test. Note the embedding matrix file is too large to upload.

Any comments are welcomed and good luck.

The pointer generator is a deep neural network built for abstractive summarizations. For more informations on this model, https://arxiv.org/pdf/1704.04368

With my collaborator Kevin Sylla , we re-made this model in tensorflow2.0 for our project. This neural net will be our baseline model. We will do some experiments with this model, and propose a new architecture based on this one.

In this project, you can:

train models
test 
evaluate 
