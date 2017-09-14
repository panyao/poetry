import os
import numpy as np
import tensorflow as tf
from preprocess import preprocess_poems
from preprocess import generate_batch 
from model import rnn_model

checkpoints_dir = os.path.abspath('./checkpoints/')
poetry_file ='data/poetry.txt'
model_prefix = 'poem'
start_token = '['
end_token = ']'

batch_size = 64
epochs = 51
learning_rate = 0.005

        
def run_training():
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    poems_vector, word_to_int, vocabularies = preprocess_poems()
    batches_inputs, batches_outputs = generate_batch(poems_vector, word_to_int)
    
    input_data = tf.placeholder(tf.int32, [batch_size, None])
    output_targets = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:

        sess.run(init_op)

        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("[INFO] restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('[INFO] start training...')
        
        for epoch in range(start_epoch, epochs):
            n = 0
            n_chunk = len(poems_vector) // batch_size
            loss_array = []
            for batch in range(n_chunk):
                loss, _, _ = sess.run([
                    end_points['total_loss'],
                    end_points['last_state'],
                    end_points['train_op']
                ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                n += 1  
                loss_array.append(loss)
                
            if epoch % 6 == 0:
                saver.save(sess, os.path.join(checkpoints_dir, model_prefix), global_step=epoch)
                
            print('[INFO] Epoch: %d , training loss: %.6f' % (epoch, np.mean(loss_array)))
        

        
tf.reset_default_graph() 
run_training()
 
