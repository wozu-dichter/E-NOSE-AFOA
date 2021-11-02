import tensorflow as tf
import numpy as np
import time
from datetime import datetime
import math
import os
from PIL import Image

# retention_time = 900

#创建文件夹
def mkdir(path):
    # 引入模块
    import os
 
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path) 
        print (path+' path created.')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print (path+' path has existed.')
        return False

def average_gradients(tower_grads):
    average_grads=[]
    for grad_and_vars in zip(*tower_grads):
        grads=[]
        for g,_ in grad_and_vars:
            expend_g=tf.expand_dims(g,0)
            grads.append(expend_g)
        grad=tf.concat(grads,0)
        grad=tf.reduce_mean(grad,0)
        v=grad_and_vars[0][1]
        grad_and_var=(grad,v)
        average_grads.append(grad_and_var)
    return average_grads

def lstm_cell(lstm_size, keep_prob_rnn):
    lstm_basic = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    drop_cell = tf.contrib.rnn.DropoutWrapper(lstm_basic, keep_prob_rnn)
    return drop_cell

def dynamicRNN(x, seqlen, weights, biases, cell, n_classes):

    # 输入x的形状： (batch_size, max_seq_len, n_input)
    # 输入seqlen的形状：(batch_size, )
    
    # normalize
    # normalized_x = tf.nn.l2_normalize(x)

    # 定义一个lstm_cell，隐层的大小为n_hidden（之前的参数）
    # initializer = tf.random_uniform_initializer(-1, 1)      

    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # dropout
    # lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, keep_prob_rnn)

    # stacked_lstm_cell = tf.nn.rnn_cell.MultiRNNCell( \
    #     [tf.nn.rnn_cell.DropoutWrapper(lstm_cell, keep_prob) \
    #     for _ in range(number_of_layers)])

    # stacked_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * number_of_layers)

    # 使用tf.nn.dynamic_rnn展开时间维度
    # 此外sequence_length=seqlen也很重要，它告诉TensorFlow每一个序列应该运行多少步
    # outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32,
    #                             sequence_length=seqlen)    
    outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32)
    logits = fully_connected(states[NUM_OF_LAYERS-1][1],n_classes,activation_fn=None)
    
    # outputs的形状为(batch_size, max_seq_len, n_hidden)
    # 如果有疑问可以参考上一章内容

    # 我们希望的是取出与序列长度相对应的输出。如一个序列长度为10，我们就应该取出第10个输出
    # 但是TensorFlow不支持直接对outputs进行索引，因此我们用下面的方法来做：

    # batch_size = tf.shape(outputs)[0]
    # print(tf.shape(batch_size))
    # print(tf.shape(seqlen))
    # 得到每一个序列真正的index
    index_1 = tf.range(0, batch_size) * seq_max_len + (output_time_1 - 1)
    index_2 = tf.range(0, batch_size) * seq_max_len + (output_time_2 - 1)

    # index_2 = tf.range(0, batch_size) * seq_max_len + (seqlen/tf.float32(2) - 1)
    # # index_3 = tf.range(0, batch_size) * seq_max_len + (seqlen/4 - 1)
    outputs_1_1 = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index_1)
    outputs_1_2 = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index_2)
    outputs_2 = tf.concat([outputs_1_1,outputs_1_2],axis=1)
    multi_time_point_outputs = tf.matmul(outputs_2, weights['out']) + biases['out']

    # 给最后的输出
    # return tf.matmul(outputs_1, weights['out']) + biases['out']
    return outputs, logits, multi_time_point_outputs, states

def get_record_parser(retention_time,num_transform,num_sensor):
    def parse(example):
        features = tf.parse_single_example(
                   example,features={
                                    'img':tf.FixedLenFeature([], tf.string),
                                    'label':tf.FixedLenFeature([], tf.int64), # for RNN and CNN
                                    'time_label':tf.FixedLenFeature([retention_time], tf.int64) # for RNN time level label                                    
                                    }
                    )
        image = tf.reshape(tf.decode_raw(features['img'],tf.float64),[retention_time*num_transform*num_sensor])
        label = features['label']
        time_label = features['time_label']

        return image, label, time_label
    return parse
 
def get_batch_dataset(record_file, parser, batch_size, shuffle):
    num_threads = tf.constant(5, dtype=tf.int32)
    # num_parallel_calls use multi-thread
    # shuffle size: 1000
    # repeat() is for infinite repeating dataset, when repeat(2) amount to epoch=2
    dataset = tf.data.TFRecordDataset(record_file).map(parser, num_parallel_calls=num_threads).shuffle(shuffle).repeat().batch(batch_size)
    return dataset

def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
                                 # initializer=tf.contrib.layers.variance_scaling_initializer())
                                 # initializer=tf.contrib.layers.xavier_initializer())


        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation

def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[n_in, n_out],
                                 dtype=tf.float32, 
                                 # initializer=tf.contrib.layers.xavier_initializer())
                                 initializer=tf.contrib.layers.variance_scaling_initializer())

        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
        return activation, kernel

def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)

def data_normalize(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]

# def data_normalize(data):
#     #mean, std = data.mean(), data.std()
#     mean, variance = tf.nn.moments(data)
#     data = (data - mean) / variance
#     return data


def inference_op(input_op, keep_prob):
    p = []
    # assume input_op shape is 224x224x3
    
    # tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    k = 1.0
    n = 8
    alpha = 0.001 / 9.0 
    beta = 0.75
    # data normalization
    
    normalized_input_op = tf.nn.l2_normalize(input_op)

    # VGG like network

    # block 1 -- outputs 650x6x3
    conv1_1 = conv_op(normalized_input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    pool1 = mpool_op(conv1_1,   name="pool1",   kh=2, kw=2, dh=2, dw=2)
    #norm1 = tf.nn.lrn(pool1, n/2, bias=k, alpha=alpha, beta=beta)


    # block 2 -- outputs 56x56x128
    conv2_1 = conv_op(pool1,    name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool2 = mpool_op(conv2_1,   name="pool2",   kh=2, kw=2, dh=2, dw=2)
    #norm2 = tf.nn.lrn(pool2, n/2, bias=k, alpha=alpha, beta=beta)


    # block 3 -- outputs 28x28x256
    conv3_1 = conv_op(pool2,    name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1,  name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_2,   name="pool3",   kh=2, kw=2, dh=2, dw=2)
    #norm3 = tf.nn.lrn(pool3, n/2, bias=k, alpha=alpha, beta=beta)


    # block 4 -- outputs 14x14x512
    # conv4_1 = conv_op(norm3,    name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # conv4_2 = conv_op(conv4_1,  name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # pool4 = mpool_op(conv4_2,   name="pool4",   kh=2, kw=2, dh=2, dw=2)
    # norm4 = tf.nn.lrn(pool4, n/2, bias=k, alpha=alpha, beta=beta)


    # # block 5 -- outputs 7x7x512
    # conv5_1 = conv_op(norm4,    name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # conv5_2 = conv_op(conv5_1,  name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # pool5 = mpool_op(conv5_2,   name="pool5",   kh=2, kw=2, dw=2, dh=2)
    # norm5 = tf.nn.lrn(pool5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


    # flatten
    shp = pool3.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool3, [-1, flattened_shape], name="resh1")

    # fully connected
    fc6 = fc_op(resh1, name="fc6", n_out=1024, p=p)
    #n_out=4096
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    fc7 = fc_op(fc6_drop, name="fc7", n_out=1024, p=p)
    #n_out=4096
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    logits = fc_op(fc7_drop, name="fc8", n_out=5, p=p)
    softmax = tf.nn.softmax(logits)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, logits, predictions

    
def inference_op_one_dimension_nn(input_op, keep_prob):
    p = []
    one_dimension_input = []
    logits = []
    # assume input_op shape is 224x224x3
    
    # tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    k = 1.0
    n = 8
    alpha = 0.001 / 9.0 
    beta = 0.75
    # data normalization
    
    normalized_input_op = tf.nn.l2_normalize(input_op)
    # print(normalized_input_op[:,:,1,:].shape)
    #print(normalized_input_op)
    # for i in range(input_op.shape[2]):
    #     print(normalized_input_op[:,:,i,:])
    #     one_dimension_input = np.append(one_dimension_input, normalized_input_op[:,:,i,:])
    #     print(one_dimension_input.shape)
    # print('test...')
    # print(one_dimension_input)
    one_dimension_input = tf.reshape(normalized_input_op, [input_op.shape[0]*input_op.shape[2], 650, 1, 3])
    #print(one_dimension_input.shape)

    #normalized_input_op = data_normalize(input_op)

    # VGG like network

    # block 1 -- outputs 650x6x3
    conv1_1 = conv_op(one_dimension_input, name="conv1_1", kh=3, kw=1, n_out=64, dh=1, dw=1, p=p)
    pool1 = mpool_op(conv1_1,   name="pool1",   kh=2, kw=1, dh=2, dw=1)
    #norm1 = tf.nn.lrn(pool1, n/2, bias=k, alpha=alpha, beta=beta)


    # block 2 -- outputs 56x56x128
    conv2_1 = conv_op(pool1,    name="conv2_1", kh=3, kw=1, n_out=128, dh=1, dw=1, p=p)
    pool2 = mpool_op(conv2_1,   name="pool2",   kh=2, kw=1, dh=2, dw=1)
    #norm2 = tf.nn.lrn(pool2, n/2, bias=k, alpha=alpha, beta=beta)


    # block 3 -- outputs 28x28x256
    conv3_1 = conv_op(pool2,    name="conv3_1", kh=3, kw=1, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1,  name="conv3_2", kh=3, kw=1, n_out=256, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_2,   name="pool3",   kh=2, kw=1, dh=2, dw=1)
    #norm3 = tf.nn.lrn(pool3, n/2, bias=k, alpha=alpha, beta=beta)


    # block 4 -- outputs 14x14x512
    # conv4_1 = conv_op(norm3,    name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # conv4_2 = conv_op(conv4_1,  name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # pool4 = mpool_op(conv4_2,   name="pool4",   kh=2, kw=2, dh=2, dw=2)
    # norm4 = tf.nn.lrn(pool4, n/2, bias=k, alpha=alpha, beta=beta)


    # # block 5 -- outputs 7x7x512
    # conv5_1 = conv_op(norm4,    name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # conv5_2 = conv_op(conv5_1,  name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # pool5 = mpool_op(conv5_2,   name="pool5",   kh=2, kw=2, dw=2, dh=2)
    # norm5 = tf.nn.lrn(pool5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


    # flatten
    shp = pool3.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool3, [-1, flattened_shape], name="resh1")

    # fully connected
    fc6 = fc_op(resh1, name="fc6", n_out=1024, p=p)
    #n_out=4096
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    fc7 = fc_op(fc6_drop, name="fc7", n_out=1024, p=p)
    #n_out=4096
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    logits = fc_op(fc7_drop, name="fc8", n_out=5, p=p)
    softmax = tf.nn.softmax(logits)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, logits, predictions    



def inference_op_one_dimension_nn_650_6(input_op, keep_prob, n_class):
    p = []
    one_dimension_input = []
    logits = []
    # assume input_op shape is 224x224x3
    
    # tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    k = 1.0
    n = 8
    alpha = 0.001 / 9.0 
    beta = 0.75
    # data normalization
    
    # normalized_input_op = tf.nn.l2_normalize(input_op)


    # normalized_input_op = input_op



    # print(normalized_input_op[:,:,1,:].shape)
    #print(normalized_input_op)
    # for i in range(input_op.shape[2]):
    #     print(normalized_input_op[:,:,i,:])
    #     one_dimension_input = np.append(one_dimension_input, normalized_input_op[:,:,i,:])
    #     print(one_dimension_input.shape)
    # print('test...')
    # print(one_dimension_input)
    #one_dimension_input = tf.reshape(normalized_input_op, [input_op.shape[0]*input_op.shape[2], 650, 1, 3])
    #print(one_dimension_input.shape)

    #normalized_input_op = data_normalize(input_op)

    # VGG like network

    # block 1 -- outputs 650x6x3
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=1, n_out=64, dh=1, dw=1, p=p)
    pool1 = mpool_op(conv1_1,   name="pool1",   kh=2, kw=1, dh=2, dw=1)
    norm1 = tf.nn.lrn(pool1, n/2, bias=k, alpha=alpha, beta=beta)


    # block 2 -- outputs 56x56x128
    # conv2_1 = conv_op(norm1,    name="conv2_1", kh=3, kw=1, n_out=128, dh=1, dw=1, p=p)
    # pool2 = mpool_op(conv2_1,   name="pool2",   kh=2, kw=1, dh=2, dw=1)
    # norm2 = tf.nn.lrn(pool2, n/2, bias=k, alpha=alpha, beta=beta)


    # block 3 -- outputs 28x28x256
    # conv3_1 = conv_op(norm2,    name="conv3_1", kh=3, kw=1, n_out=256, dh=1, dw=1, p=p)
    # conv3_2 = conv_op(conv3_1,  name="conv3_2", kh=3, kw=1, n_out=256, dh=1, dw=1, p=p)
    # pool3 = mpool_op(conv3_2,   name="pool3",   kh=2, kw=1, dh=2, dw=1)
    # norm3 = tf.nn.lrn(pool3, n/2, bias=k, alpha=alpha, beta=beta)


    # # block 4 -- outputs 14x14x512
    # conv4_1 = conv_op(norm3,    name="conv4_1", kh=3, kw=1, n_out=512, dh=1, dw=1, p=p)
    # conv4_2 = conv_op(conv4_1,  name="conv4_2", kh=3, kw=1, n_out=512, dh=1, dw=1, p=p)
    # pool4 = mpool_op(conv4_2,   name="pool4",   kh=2, kw=1, dh=2, dw=1)
    # norm4 = tf.nn.lrn(pool4, n/2, bias=k, alpha=alpha, beta=beta)


    # # block 5 -- outputs 7x7x512
    # conv5_1 = conv_op(norm4,    name="conv5_1", kh=3, kw=1, n_out=512, dh=1, dw=1, p=p)
    # conv5_2 = conv_op(conv5_1,  name="conv5_2", kh=3, kw=1, n_out=512, dh=1, dw=1, p=p)
    # pool5 = mpool_op(conv5_2,   name="pool5",   kh=2, kw=1, dh=1, dw=1)
    # # # norm5 = tf.nn.lrn(pool5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    # norm5 = tf.nn.lrn(pool5, n/2, bias=k, alpha=alpha, beta=beta)



    # flatten
    shp = norm1.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(norm1, [-1, flattened_shape], name="resh1")

    # fully connected
    fc6, fc6_kernel = fc_op(resh1, name="fc6", n_out=1024, p=p)
    #n_out=4096
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    fc7, fc7_kernel = fc_op(fc6_drop, name="fc7", n_out=1024, p=p)
    #n_out=4096
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    logits, logits_kernel = fc_op(fc7_drop, name="fc8", n_out=n_class, p=p)
    softmax = tf.nn.softmax(logits)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, logits, input_op, conv1_1, pool1, norm1, resh1, fc6, fc6_drop, fc7, fc7_drop, logits_kernel



def time_tensorflow_run(session, target, feed, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print ('%s: step %d, duration = %.3f' %
                       (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
           (datetime.now(), info_string, num_batches, mn, sd))


def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


def loss(logits, labels):
#      """Add L2Loss to all the trainable variables.
#      Add summary for "Loss" and "Loss/avg".
#      Args:
#        logits: Logits from inference().
#        labels: Labels from distorted_inputs or inputs(). 1-D tensor
#                of shape [batch_size]
#      Returns:
#        Loss tensor of type float.
#      """
#      # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
  
###