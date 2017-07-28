import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import math
import numpy as np
import datetime
from feeder_old import feeder

possible_phases  = ['training', 'validation', 'test', 'conversion']
phase            = possible_phases[0]
n_steps          = 1072                 # The max number of frames
n_inputs         = 49                   # Width of each frame
n_neurons        = [64, 128, 256, 256, 128, 64]
n_layers         = 6
learning_rate    = 0.0001
keep_probability = 0.9
n_epoches        = 50
batch_size       = 2
check_step       = 1
save_step        = check_step * 1
validation_step  = check_step * 5

data_path            = '/Users/ChlorophyII/SPIA/code/SPIA-2017-data/'
train_data_path      = data_path+'bdl2slt/'
validation_data_path = data_path+'bdl2slt_validation/'
test_data_path       = data_path+'bdl2slt_test/'
conversion_data_path = data_path+'bdl2slt_conversion/'
checkpoint_path      = data_path+'checkpoints/'
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# Model

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
Y = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
W = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
keep_prob = tf.placeholder(tf.float32)
global_step = tf.Variable(0, trainable=False)

# Cells

def lstm_cell(n):
    return tf.contrib.rnn.BasicLSTMCell(n)
initializer = tf.contrib.layers.xavier_initializer()
with tf.variable_scope('forward', initializer=initializer):
    cells_fw = [lstm_cell(n_neurons[n]) for n in range(n_layers)]
    cells_fw = [rnn.DropoutWrapper(cell, input_keep_prob=keep_prob) for cell in cells_fw]

with tf.variable_scope('backward', initializer=initializer):
    cells_bw = [lstm_cell(n_neurons[n]) for n in range(n_layers)]
    cells_bw = [rnn.DropoutWrapper(cell, input_keep_prob=keep_prob) for cell in cells_bw]

# Layers

outputs = X

for n in range(n_layers):
    with tf.variable_scope('BiRNN'+str(n), initializer=initializer):
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cells_fw[n], cells_bw[n], outputs, dtype=tf.float32)
    outputs = tf.concat(outputs, 2)

outputs = tf.layers.dense(outputs, n_inputs)

# Loss function

loss_factor = 10 * math.sqrt(2) / math.log(10)
loss = tf.multiply(tf.divide(tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.multiply(tf.squared_difference(outputs, Y), W), axis=2)+1e-20)), tf.divide(tf.reduce_sum(W), n_inputs)), loss_factor)

# Optimizer

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss, global_step=global_step)

def load_data(phase):
    print "Loading data..."
    if phase == 'training' or phase == 'validation':
        global training_feeder
        training_feeder = feeder(batch_size, train_data_path, n_steps, n_inputs, phase='training')
        print "\"training_feeder\" is ready."
        global validation_feeder
        validation_feeder = feeder(-1, validation_data_path, n_steps, n_inputs, phase='validation')
        print "\"validation_feeder\" is ready."
    elif phase == 'test':
        global test_feeder
        test_feeder = feeder(-1, validation_data_path, n_steps, n_inputs, phase='test')
        print "\"test_feeder\" is ready."
    elif phase == 'conversion':
        global conversion_feeder
        conversion_feeder = feeder(-1, conversion_data_path, n_steps, n_inputs, phase='conversion')
        print "\"conversion_feeder\" is ready."
    else:
        assert False, "\"phase\" is incorrect."
    print "Data loaded."

def print_step(step, epoch, loss, validation_loss=-1):
    if validation_loss == -1:
        print (str(datetime.datetime.utcnow())+ \
               ' step: {0:<4} epoch: {1: <3} average loss: {2: <5}'.format(step, epoch, loss))
    else:
        print (str(datetime.datetime.utcnow())+ \
               ' step: {0:<4} epoch: {1: <3} average loss: {2: <5} validation loss: {3: <5}'.format(step, epoch, loss, validation_loss))

config=tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
#with tf.Session() as sess:
    load_data(phase)
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print ("Loading model parameters from %s" % ckpt.model_checkpoint_path)
        tf.reset_default_graph()
        print "PHASE", phase
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+".meta", clear_devices=True)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print ("Model parameters loaded.")
    else:
        print ("Create model with brand new parameters.")
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables())
        sess.run(init)

    if phase == 'training':
        print "In training"
        loss_sum = 0
        step = global_step.eval()
        epoch = 1 + (step * batch_size - 1) / training_feeder.n_data # -1 to fix the error
        print "Start training..."
        while epoch < n_epoches:
            X_batch, Y_batch, weights, batch_filenames = training_feeder.get_batch()
            _, step_loss, step, results = sess.run([training_op, loss, global_step, outputs], feed_dict={X:X_batch, Y:Y_batch, W:weights, keep_prob:keep_probability})
            loss_sum = loss_sum+step_loss

            if step % check_step == 0:
                epoch = 1 + (step * batch_size - 1) / training_feeder.n_data # -1 to fix the error
                if step % validation_step == 0:
                    X_batch, Y_batch, weights, batch_filenames = validation_feeder.get_batch()
                    validation_loss = loss.eval(feed_dict={X:X_batch, Y:Y_batch, W:weights, keep_prob:1})
                    print_step(step, epoch, loss_sum/check_step, validation_loss)
                else:
                    print_step(step, epoch, loss_sum/check_step)
                loss_sum = 0
            if step % save_step == 0:
                saver.save(sess, checkpoint_path+'vc', global_step=global_step)
                print ("Parameters saved.")
    elif phase == 'test':
        print "Start testing..."
        X_batch, Y_batch, weights, batch_filenames = test_feeder.get_batch()
        test_loss = loss.eval(feed_dict={X:X_batch, Y:Y_batch, W:weights, keep_prob:1})
        print "Test loss: {0}".format(test_loss)
    elif phase == 'conversion':
        print "Start converting..."
        X_batch, Y_batch, weights, batch_filenames = conversion_feeder.get_batch()
        conversion_result = outputs.eval(feed_dict={X:X_batch, Y:Y_batch, W:weights, keep_prob:1})
        conversion_feeder.save_outputs(conversion_result, batch_filenames)
        print "Converted data saved at:"
        print conversion_data_path
    else:
        print "WTF"
        assert False, "\"phase\" is incorrect."
