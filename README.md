# seq2seq_Pointer_Generator_Summarizer
A deep learning neural network for abstractive deep summarization
project:
This is a project of generating abstractive summerization from Chinese conversation. The funny conversation is between customers and car technicians, with 80000+ samples for training and testing and 20000 samples for prediction.
The data pipline is somehow typical for Chinese, purge data - segment - tokenize - batch. However it's tricky to deal with long conversation and to add special token to word2vec model.  Special tokens is added to the w2v model by retraining the model.

Files like original dataset, segment dataset, w2v model are also provided for immediate test. Note the embedding matrix file is too large to upload.

In this project, you can:
train models,test or evaluate 
model:
Everything is classic and built with tensorflow 2.0, word embedding is pretrained by word2vec, and seq2seq includes Gru as encoder, Bahdanau attention and unidirection Gru as decoder. The model also embrace pointer generator network and coverage loss to deal with oov and repeating. ref. arXiv:1704.04368v2. Prediction implements beam search.



This neural net will be our baseline model. I will do some experiments with this model, and propose a new architecture based on this one.


