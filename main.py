from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, concatenate
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam, Adadelta
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from models import resnet_v2
from utils import lr_schedule
import numpy as np
import os
from iNose_data_gen import iNose_data_gen
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
import time
from keras.utils.vis_utils import plot_model
import argparse
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


parser = argparse.ArgumentParser(description="Keras e-nose signals Training.")
parser.add_argument('--attention-mode', type=str, default='cbam2',
                    choices=['se','cbam1','cbam2','cbam3','cbam4','cbam5','None'],
                    help='attention mode (default: cbam_block_channel_first)')
parser.add_argument('--dataset', type=str, default='e-nose',
                    choices=['e-nose'],
                    help='dataset name (default: e-nose)')
parser.add_argument('--lstm-hidden', type=int, default=128,
                    help='lstm hidden layers')
# # training hyper params
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--batch-size', type=int, default=128,
                    metavar='N', help='input batch size for \
                            training (default: 128)')
parser.add_argument('--classes-num', type=int, default=6,
                    metavar='N', help='classes number for \
                            training (default: 6)')
parser.add_argument('--res-first-filters', type=int, default=16,
                    metavar='N', help='resnet first filters for \
                            training (default: 16)')
parser.add_argument('--pool-size', type=int, default=8,
                    metavar='N', help='average pooling size after LSTM\
                             (default: 8)')
parser.add_argument('--stage-num', type=int, default=3,
                    metavar='N', help='stage num for res-like network\
                             (default: 3)')
parser.add_argument('--dropout', type=float, default=0.1,
                    metavar='N', help='dropout of LSTM input\
                             (default: 0.1)')
parser.add_argument('--r-dropout', type=float, default=0.2,
                    metavar='N', help='dropout of LSTM output\
                             (default: 0.2)')
parser.add_argument('--d-dropout', type=float, default=0.1,
                    metavar='N', help='dropout of dense output\
                             (default: 0.1)')
parser.add_argument('--sensor', type=int, default=0,
                    metavar='N', help='choosed sensor nunber, 0 for all 6 sensors\
                             (default: 0)')
parser.add_argument('--seq-len', type=int, default=779,
                    metavar='N', help='setting sensor response signal length, 300 - 779\
                             (default: 779)')
args = parser.parse_args()

# Training parameters
GPU_NUM = 2
batch_size = args.batch_size
epochs = args.epochs
data_augmentation = False
num_classes = args.classes_num
# subtract_pixel_mean = True  # Subtracting pixel mean improves accuracy
subtract_pixel_mean = False  # Subtracting pixel mean improves accuracy
base_model = 'resnet20'
LSTM_layers = args.lstm_hidden  
return_sequences = False

# Choose what attention_module to use: cbam_block / se_block / None
if args.attention_mode == 'se':
    attention_module = 'se_block'
    cbam_mode = 1
elif args.attention_mode == 'cbam1':
    attention_module = 'cbam_block'
    cbam_mode = 1
elif args.attention_mode == 'cbam2':
    attention_module = 'cbam_block'
    cbam_mode = 2
elif args.attention_mode == 'cbam3':
    attention_module = 'cbam_block'
    cbam_mode = 3
elif args.attention_mode == 'cbam4':
    attention_module = 'cbam_block'
    cbam_mode = 4
elif args.attention_mode == 'cbam5':
    attention_module = 'cbam_block'
    cbam_mode = 5
else:
    attention_module = None
    cbam_mode = 1

# cbam mode 1: cbam_block_parallel mode        
# cbam mode 2: cbam_block_channel_first mode       
# cbam mode 3: cbam_block_spatial_first mode       
# cbam mode 4: channel_attention_module_only mode       
# cbam mode 5: spatial_attention_module_only mode

if attention_module == 'cbam_block':
	if cbam_mode==1:
	    mode_type='cbam_block_parallel'
	elif cbam_mode==2:
	    mode_type='cbam_block_channel_first'
	elif cbam_mode==3:
	    mode_type='cbam_block_spatial_first'
	elif cbam_mode==4:
	    mode_type='channel_attention_module_only'
	elif cbam_mode==5:
	    mode_type='spatial_attention_module_only'
	else:
	    cbam_mode_type='cbam_mode should be 1 to 5!'
elif attention_module == 'se_block':
	mode_type='se_block enabled.'
else:
    mode_type='without attention.'

model_type = base_model if attention_module==None else base_model+'_'+attention_module

# Load the e-nose data
retention_time = 779
if args.dataset == 'e-nose':
    test_examples_num = 840
    train_examples_num = 16800
    tfrecord_file = '/home1/xxxx/data/e-nose/xxxx'

# Dataset parameters
num_sensor = 6
num_transform = 5 # Number of channels extended by data transformation

start_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
x_train, y_train, time_label_train, x_test, y_test, time_label_test = iNose_data_gen(time_step=retention_time, num_train_data=train_examples_num, num_test_data=test_examples_num, num_sensor=num_sensor, num_transform=num_transform, file_path=tfrecord_file)

print('Loaded data shapes:')
print('x_train shape: ',x_train.shape)
print('y_train shape: ',y_train.shape)
print('x_test shape: ',x_test.shape)
print('y_test shape: ',y_test.shape)
print('time_label_train shape: ',time_label_train.shape)
print('time_label_test shape: ',time_label_test.shape)

choosed_sensors = args.sensor
seq_len = args.seq_len

# Choose sensor
if choosed_sensors!=0:
    x_train = x_train[:,:,5*(choosed_sensors-1):5*choosed_sensors]
    x_test = x_test[:,:,5*(choosed_sensors-1):5*choosed_sensors]

x_train = x_train[:,:seq_len,:]
x_test = x_test[:,:seq_len,:]

print('Loaded extracted data shapes:')
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Input image dimensions
input_shape = x_train.shape[1:]
print('input_shape :',input_shape)

depth = 20 # For ResNet, specify the depth (e.g. ResNet50: depth=50)
parallel_model = resnet_v4.resnet_v2_original(input_shape=input_shape, depth=depth, attention_module=attention_module, \
    cbam_mode=cbam_mode, lstm_layers=LSTM_layers,return_sequences=return_sequences,num_filters_in=args.res_first_filters, \
    pool_size=args.pool_size,stages_num=args.stage_num,dropout=args.dropout,recurrent_dropout=args.r_dropout,dense_dropout=args.d_dropout)   

parallel_model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])

parallel_model.summary()
print(model_type)
print(mode_type)

sess = K.get_session()
sess.run(tf.global_variables_initializer())
# Prepare model model saving directory
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'iNose_%s_model.{epoch:03d}_sensor%d_len%d.h5' % (model_type, args.sensor, args.seq_len)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# Run training, with or without data augmentation
if not data_augmentation:
    print('Not using data augmentation.')
    parallel_model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow()
    parallel_model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)
# Score trained model
scores = parallel_model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print('Dataset: ',tfrecord_file)
print('Start time: ',start_time)
print('Batch size = : ',batch_size)
print('network parameters: ')
print('return_sequences = : ',return_sequences)
print('LSTM layers = : ',LSTM_layers)
print('attention_module = : ',args.attention_mode)
print('res_first_filters = : ',args.res_first_filters)
print('stage_num = : ',args.stage_num)
print('pool_size = : ',args.pool_size)
print('dropout = : ',args.dropout)
print('recurrent dropout = : ',args.r_dropout)
print('dense dropout = : ',args.d_dropout)
