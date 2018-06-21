from tensorlayer.layers import *
import tensorlayer as tl

###============= model
from model_config import emb_dim


def model(encode_seqs, decode_seqs, xvocab_size, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        # for chatbot, you can use the same embedding layer,
        # for translation, you may want to use 2 seperated embedding layers
        with tf.variable_scope("embedding") as vs:
            net_encode = EmbeddingInputlayer(
                inputs=encode_seqs,
                vocabulary_size=xvocab_size,
                embedding_size=emb_dim,
                name='seq_embedding')
            vs.reuse_variables()
            tl.layers.set_name_reuse(True)  # remove if TL version == 1.8.0+
            net_decode = EmbeddingInputlayer(
                inputs=decode_seqs,
                vocabulary_size=xvocab_size,
                embedding_size=emb_dim,
                name='seq_embedding')
        net_rnn = Seq2Seq(net_encode, net_decode,
                          cell_fn=tf.contrib.rnn.BasicLSTMCell,
                          n_hidden=emb_dim,
                          initializer=tf.random_uniform_initializer(-0.1, 0.1),
                          encode_sequence_length=retrieve_seq_length_op2(encode_seqs),
                          decode_sequence_length=retrieve_seq_length_op2(decode_seqs),
                          initial_state_encode=None,
                          dropout=(0.5 if is_train else None),
                          n_layer=3,
                          return_seq_2d=True,
                          name='seq2seq')
        net_out = DenseLayer(net_rnn, n_units=xvocab_size, act=tf.identity, name='output')
    return net_out, net_rnn
