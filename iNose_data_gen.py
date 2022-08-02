import tensorflow as tf
import os
from __init__one_dimension_nn import mkdir, get_record_parser, get_batch_dataset


def iNose_data_gen(time_step, num_train_data, num_test_data, num_sensor, num_transform, file_path):

    # gain data
    record_file_train = file_path + os.sep + 'train_iNose_data.tfrecords'
    record_file_test = file_path + os.sep + 'test_iNose_data.tfrecords'
    parser = get_record_parser(time_step, num_sensor, num_transform)
    dataset_train = get_batch_dataset(record_file_train, parser, num_train_data, shuffle=num_train_data)
    dataset_test = get_batch_dataset(record_file_test, parser, num_test_data, shuffle=num_test_data)

    iterator_train = dataset_train.make_one_shot_iterator()
    next_element_train = iterator_train.get_next()
    iterator_test = dataset_test.make_one_shot_iterator()
    next_element_test = iterator_test.get_next()         

    images_train_raw, labels_train, time_labels_train = next_element_train
    images_test_raw, labels_test, time_labels_test = next_element_test
    images_train = tf.reshape(images_train_raw, [num_train_data, time_step, num_sensor*num_transform])
    images_test = tf.reshape(images_test_raw, [num_test_data, time_step, num_sensor*num_transform])

    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        images_train_load, labels_train_load, time_labels_train_load, images_test_load, labels_test_load, time_labels_test_load = sess.run([images_train, labels_train, time_labels_train, images_test, labels_test, time_labels_test])

    return images_train_load, labels_train_load, time_labels_train_load, images_test_load, labels_test_load, time_labels_test_load

def iNose_data_gen_train_data_only(time_step, num_train_data, num_sensor, num_transform, file_path):

    # gain data
    record_file_train = os.getcwd() + os.sep + file_path + os.sep + 'train_iNose_data.tfrecords'
    parser = get_record_parser(time_step, num_sensor, num_transform)
    dataset_train = get_batch_dataset(record_file_train, parser, num_train_data, shuffle=num_train_data)

    iterator_train = dataset_train.make_one_shot_iterator()
    next_element_train = iterator_train.get_next()
         

    images_train_raw, labels_train, time_labels_train = next_element_train

    images_train = tf.reshape(images_train_raw, [num_train_data, time_step, num_sensor*num_transform])

    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        images_train_load, labels_train_load = sess.run([images_train, labels_train])
    return images_train_load, labels_train_load

