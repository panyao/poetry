import os
import numpy as np
import tensorflow as tf
import random

from preprocess import preprocess_poems
from preprocess import generate_batch 
from model import rnn_model


checkpoints_dir = os.path.abspath('./checkpoints/')
poetry_file ='data/poetry.txt'
model_prefix = 'poem'
start_token = '['
end_token = ']'

batch_size = 64
learning_rate = 0.005


def to_word(predict, vocabs):
    predict = predict[0]
    max_prob = max(predict)
    threshold = np.random.uniform(0.1,0.2)*max_prob
    true_idx = np.argmax(predict)
    cnt = 0
    while(True):
        idx = random.randint(0,2000-1)
        #print('cnt:',cnt,' probi:',predict[true_idx],' true_idx:',true_idx,' w:',vocabs[true_idx])
        #print('threshold:',threshold,' pred_idx:',idx,' prob:',predict[idx],' w:',vocabs[idx])
        if(predict[idx]>=threshold):
            print(vocabs[idx])
            return vocabs[idx]
        cnt = cnt + 1

def gen_poem(begin_word):
    
    input_data = tf.placeholder(tf.int32, [1, None])
    poems_vector, word_int_map, vocabularies = preprocess_poems()
    
    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=learning_rate)
    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
        saver.restore(sess, checkpoint)

        x = np.array([list(map(word_int_map.get, start_token))])

        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        if begin_word:
            word = begin_word
        else:
            word = to_word(predict, vocabularies)
        poem = ''
        cnt = 0
        while word != end_token:
            poem += word
            x = np.zeros((1, 1))
            x[0, 0] = word_int_map[word]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, vocabularies)
            cnt = cnt + 1
            if cnt > 80:
                break 
        return poem

def print_poem(poem):
    sentence = poem.split('。')
    for s in sentence:
        print(s)
        

tf.reset_default_graph()
print_poem(gen_poem("雨"))

