#-*- coding: utf-8 -*-
import numpy as np
import sys
import getopt
import jieba
import logging
import tensorflow as tf
import time


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    #filename='test.log',
                    #filemode='w'
                    )

MAX_SIZE = 350

def gen_words_vec(file):
    words_vec = {}
    np.random.seed(0)
    words_vec["unk"] = np.random.rand(64)
    f = open(file, "r") 
    for line in f.readlines():
        line = line.strip()
        line_decode = line.encode('utf-8').decode("utf-8")
        line_split = line_decode.split('\t')
        key = line_split[1]
        vec = line_split[3]
        words_vec[key] = vec
    f.close()
    return words_vec

def words_array(lineList, words_vec):
    wordsArray = []
    has_cnt = 0
    need_cnt = 0
    
    for w in lineList:
        try:
            w_vec = words_vec[w]
        except Keyrror:
            w_vec = words_vec['unk']
        wordsArray.append(eval(w_vec))
        has_cnt += 1
        if has_cnt >= 350:
            break
    
    need_cnt = MAX_SIZE - has_cnt

    # 补齐0的操作
    if need_cnt > 0:
        for i in range(need_cnt):
            wordsArray.append(([0.0] * 64))
    
    return wordsArray, has_cnt

# 构造篇章的向量数据
def make_data(words_vec, flag, file_path):
    docs = []
    seq_steps = []
    labels = []
    if flag == 1:
        label = [1, 0]
    else:
        label = [0, 1]
    f = open(file_path, 'r')
    for line in f.readlines():
        line = line.strip()
        line_decode = line.encode('utf-8').decode("utf-8")
        seg_list = list(jieba.cut(line_decode, cut_all=True))
        wordsArray, has_cnt = words_array(seg_list, words_vec)
        docs.append(wordsArray)
        seq_steps.append(has_cnt)
        labels.append(label)
    return docs, seq_steps, labels


def gen_data(pos_docs, pos_seq_steps, pos_label, neg_docs, neg_seq_steps, neg_label):
    rand_sample = []
    data = []
    steps = []
    labels = []
    for i in range(len(pos_docs)):
        rand_sample.append([pos_docs[i], pos_seq_steps[i], pos_label[i]])
    for i in range(len(neg_docs)):
        rand_sample.append([neg_docs[i], neg_seq_steps[i], neg_label[i]])

    for i in range(len(rand_sample)):
        data.append(rand_sample[i][0])
        steps.append(rand_sample[i][1])
        labels.append(rand_sample[i][2])

    data = np.array(data)
    steps = np.array(steps)
    labels = np.array(labels)
    
    return data, steps, labels


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("usage: python3 make_sentence_vector.py --file1 wordscnt")

    opts, args = getopt.getopt(sys.argv[1:], "ho:", ["help", "file1="])
    for opt, val in opts:
        if opt == "--file1":
            file = val
    words_vec = gen_words_vec(file)

    file1 = 'data/B/Pos-train.txt'
    file2 = 'data/B/Neg-train.txt'
    
    begin_time = time.time()
    pos_docs, pos_seq_steps, pos_label = make_data(words_vec, 1, file1)
    neg_docs, neg_seq_steps, neg_label = make_data(words_vec, 0, file2)
    
    train_data, train_steps, train_labels = gen_data(pos_docs, pos_seq_steps, pos_label, neg_docs, neg_seq_steps, neg_label)
    
    end_time = time.time()

    logging.info('train_data.shape %s', train_data.shape)
    logging.info('train_steps.shape %s', train_steps.shape)
    logging.info('train_labels.shape %s', train_labels.shape)

    logging.info('use time is %s', (end_time - begin_time))

# lstm神经网络开始定义
    num_nodes = 128
    batch_size = 10
    output_size = 2
    # 词的embeding维度
    dimsh = 64

    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, MAX_SIZE, dimsh))
        tf_train_steps = tf.placeholder(tf.int32, shape=(batch_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, output_size))

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_nodes, state_is_tuple=True)

        w1 = tf.Variable(tf.truncated_normal([num_nodes, num_nodes // 2], stddev=0.1))
        b1 = tf.Variable(tf.truncated_normal([num_nodes // 2], stddev=0.1))

        w2 = tf.Variable(tf.truncated_normal([num_nodes // 2, 2], stddev=0.1))
        b2 = tf.Variable(tf.truncated_normal([2], stddev=0.1))

        def model(dataset, steps):
            outputs, last_states = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                        dtype=tf.float32,
                                                        sequence_length=steps,
                                                        inputs=dataset)
            hidden = last_states[-1]
            hidden = tf.matmul(hidden, w1) + b1
            logits = tf.matmul(hidden, w2) + b2
            return logits

        train_logits = model(tf_train_dataset, tf_train_steps)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=train_logits))
        
        optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        
        
# 喂数据，开始训练网络
    num_epochs = 20
    summary_freq = 5
    
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        
        # 模型保存函数
        saver = tf.train.Saver()
        
        mean_loss = 0
        for step in range(num_epochs):
            offset = (step * batch_size) % (len(train_labels) - batch_size)
            feed_dict = {tf_train_dataset: train_data[offset: offset + batch_size],
                          tf_train_steps: train_steps[offset: offset + batch_size],
                          tf_train_labels: train_labels[offset: offset + batch_size]
                        }
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)

            mean_loss += l
            if step >0 and step % summary_freq == 0:
                mean_loss = mean_loss / summary_freq
                saver.save(session, './my-model', global_step=step)

                logging.info("The step is: %d"%(step))
                logging.info("In train data,the loss is:%.4f"%(mean_loss))

    over_time = time.time()
    logging.info("train time use is %s", (over_time - end_time))
        
                            
