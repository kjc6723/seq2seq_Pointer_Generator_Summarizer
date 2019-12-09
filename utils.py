import tensorflow as tf
import logging
from gensim.models import Word2Vec
import numpy as np
import os


def define_logger(log_file):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    # get TF logger
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)


def get_embedding(max_features, embed_size,  vocab, model_file):
    """
    give vocab ,max_features,embed_size,return embedding
    Args: vocab ,max_features,embed_size
    output:embedding
    """
    # load model
    abspath = os.path.abspath("./")
    model_path = os.path.join(abspath, model_file)
    w2v_model = Word2Vec.load(model_path)
    # init embedding
    embedding = np.random.uniform(-1, 1, (max_features, embed_size))
    # gen embedding
    for word, id in vocab.word2id.items():
        if id <= max_features:
            if word in w2v_model.wv.vocab:
                embedding[id] = w2v_model.wv[word]

    return embedding


def _calc_final_dist(_enc_batch_extend_vocab, vocab_dists, attn_dists, p_gens, batch_oov_len, vocab_size, batch_size):
  """Calculate the final distribution, for the pointer-generator model
  Args:
  vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
  attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays
  Returns:
  final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
  """
  # Multiply vocab dists by p_gen and attention dists by (1-p_gen)

  vocab_dists = [p_gen * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
  attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(p_gens, attn_dists)]

  # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
  extended_vsize = vocab_size + batch_oov_len  # the maximum (over the batch) size of the extended vocabulary
  extra_zeros = tf.zeros((batch_size, batch_oov_len))
  vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in
                            vocab_dists]  # list length max_dec_steps of shape (batch_size, extended_vsize)

  # Project the values in the attention distributions onto the appropriate entries in the final distributions
  # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
  # This is done for each decoder timestep.
  # This is fiddly; we use tf.scatter_nd to do the projection
  batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
  batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
  attn_len = tf.shape(_enc_batch_extend_vocab)[1]  # number of states we attend over
  batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
  indices = tf.stack((batch_nums, _enc_batch_extend_vocab), axis=2)  # shape (batch_size, enc_t, 2)
  shape = [batch_size, extended_vsize]
  attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in
                            attn_dists]  # list length max_dec_steps (batch_size, extended_vsize)

  # Add the vocab distributions and the copy distributions together to get the final distributions
  # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
  # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
  final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
                 zip(vocab_dists_extended, attn_dists_projected)]
  return final_dists
