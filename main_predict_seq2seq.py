import tensorlayer as tl
import tensorflow as tf

import model_config
import model_seq2seq
from data.my_data import data

metadata, idx_q, idx_a = data.load_data(PATH='./data/my_data/')  # My-data

(trainX, trainY), (testX, testY), (validX, validY) = data.split_dataset(idx_q, idx_a)

trainX = trainX.tolist()
trainY = trainY.tolist()
testX = testX.tolist()
testY = testY.tolist()
validX = validX.tolist()
validY = validY.tolist()

trainX = tl.prepro.remove_pad_sequences(trainX)
trainY = tl.prepro.remove_pad_sequences(trainY)
testX = tl.prepro.remove_pad_sequences(testX)
testY = tl.prepro.remove_pad_sequences(testY)
validX = tl.prepro.remove_pad_sequences(validX)
validY = tl.prepro.remove_pad_sequences(validY)

###============= parameters
xseq_len = len(trainX)  # .shape[-1]
yseq_len = len(trainY)  # .shape[-1]
assert xseq_len == yseq_len
n_step = int(xseq_len / model_config.batch_size)
xvocab_size = len(metadata['idx2w'])  # 8002 (0~8001)
w2idx = metadata['w2idx']  # dict  word 2 index
idx2w = metadata['idx2w']  # list index 2 word

unk_id = w2idx['unk']  # 1
pad_id = w2idx['_']  # 0

start_id = xvocab_size  # 8002
end_id = xvocab_size + 1  # 8003

w2idx.update({'start_id': start_id})
w2idx.update({'end_id': end_id})
idx2w = idx2w + ['start_id', 'end_id']

xvocab_size = xvocab_size + 2
#  NTT / End data processing

# Blueprint model
# model for training
encode_seqs = tf.placeholder(dtype=tf.int64, shape=[model_config.batch_size, None], name="encode_seqs")
decode_seqs = tf.placeholder(dtype=tf.int64, shape=[model_config.batch_size, None], name="decode_seqs")
target_seqs = tf.placeholder(dtype=tf.int64, shape=[model_config.batch_size, None], name="target_seqs")
target_mask = tf.placeholder(dtype=tf.int64, shape=[model_config.batch_size, None],
                             name="target_mask")  # tl.prepro.sequences_get_mask()
net_out, _ = model_seq2seq.model(encode_seqs, decode_seqs, xvocab_size, is_train=True, reuse=False)
# model for inferencing -> predict
encode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs")
decode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_seqs")
net, net_rnn = model_seq2seq.model(encode_seqs2, decode_seqs2, xvocab_size, is_train=False, reuse=True)
y = tf.nn.softmax(net.outputs)

# Load model trained
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
tl.layers.initialize_global_variables(sess)
tl.files.load_and_assign_npz(sess=sess, name='n.npz', network=net)

# NTT TODO test train
seeds = [u"Sư phạm kỹ thuật thành phố hồ chí minh ở đâu", u'Địa chỉ Trường_Đại học Sư phạm Kỹ_Thuật nằm ở đâu', u'SPKT nằm ở đâu']
for seed in seeds:
    print("Query >", seed)
    seed_id = [w2idx[w] if (w in w2idx) else w2idx[model_config.UNK] for w in seed.split(' ')]
    # for _ in range(5):  # 1 Query --> 5 Reply
    # 1. encode, get state
    state = sess.run(net_rnn.final_state_encode,
                     {encode_seqs2: [seed_id]})
    # 2. decode, feed start_id, get first word
    #   ref https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py
    o, state = sess.run([y, net_rnn.final_state_decode],
                        {net_rnn.initial_state_decode: state,
                         decode_seqs2: [[start_id]]})
    w_id = tl.nlp.sample_top(o[0], top_k=3)
    w = idx2w[w_id]
    # 3. decode, feed state iteratively
    sentence = [w]
    for _ in range(30):  # max sentence length
        o, state = sess.run([y, net_rnn.final_state_decode],
                            {net_rnn.initial_state_decode: state,
                             decode_seqs2: [[w_id]]})
        w_id = tl.nlp.sample_top(o[0], top_k=2)
        w = idx2w[w_id]
        if w_id == end_id:
            break
        sentence = sentence + [w]
    print(" >", ' '.join(sentence))
    #\End for _ in range(5):
