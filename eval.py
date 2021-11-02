from __future__ import print_function
import keras
from keras.models import Model 
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
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
from keras.models import load_model
import pdb

import matplotlib.pyplot as plt
from numpy import var
from scipy import interpolate
import torch.nn.functional as F
import torch

# Seed value (can actually be different for each attribution step)
seed_value= 0
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)

parser = argparse.ArgumentParser(description="Keras iNose signals Training.")
parser.add_argument('--attention-mode', type=str, default='cbam2',
                    choices=['se','cbam1','cbam2','cbam3','cbam4','cbam5','None'],
                    help='attention mode (default: cbam_block_channel_first)')
parser.add_argument('--dataset', type=str, default='inose-35-batch3-3P3N',
                    choices=['inose-31', 'inose-10-old', 'inose-26','inose-10-new','inose-16', 'inose-10-old-0','inose-35-batch3-3P3N','inose-35-batch3-P2N','inose-35-batch3-N2P','inose-35-batch3-3P3N-zero'],
                    help='dataset name (default: inose-v1)')
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
parser.add_argument('--model-path', type=str, default=None,
                        help='put the path to saved model')
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

retention_time = 779

if args.dataset == 'e-nose':
    test_examples_num = 840
    train_examples_num = 16800
    tfrecord_file = '/home/xxxx/data/e-nose/xxxx'

num_sensor = 6
num_transform = 5 # 数据变换扩展的数量

start_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

x_train, y_train, time_label_train, x_test, y_test, time_label_test = iNose_data_gen(time_step=retention_time, num_train_data=train_examples_num, num_test_data=test_examples_num, num_sensor=num_sensor, num_transform=num_transform, file_path=tfrecord_file)

print('Loaded data shapes:')
print('x_train shape: ',x_train.shape)
print('y_train shape: ',y_train.shape)
print('x_test shape: ',x_test.shape)
print('y_test shape: ',y_test.shape)
print('time_label_train shape: ',time_label_train.shape)
print('time_label_test shape: ',time_label_test.shape)

# choose features
if args.sensor!=0:
    start_feature=args.sensor-1
    end_feature=start_feature+1
elif args.sensor==0:
    start_feature=0
    end_feature=6
num_choosed_feature=end_feature- start_feature
x_train = x_train.reshape(train_examples_num,retention_time,num_sensor,num_transform)
x_test = x_test.reshape(test_examples_num,retention_time,num_sensor,num_transform)

x_train = x_train[:,:,start_feature:end_feature,:]
x_test = x_test[:,:,start_feature:end_feature,:]

x_train = x_train.reshape(train_examples_num,retention_time,num_choosed_feature*num_transform)
x_test = x_test.reshape(test_examples_num,retention_time,num_choosed_feature*num_transform)

seq_len = args.seq_len
x_train = x_train[:,:seq_len,:]
x_test = x_test[:,:seq_len,:]

print('Loaded extracted data shapes:')
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# load model
loaded_model = load_model('./saved_models/'+args.model_path)
loaded_model.summary()

# predict
result = loaded_model.predict(x_test, batch_size=120, verbose=1)

time_attention_12_layer_model = Model(inputs=loaded_model.input, outputs=loaded_model.get_layer('multiply_12').output)
time_attention_12_output = time_attention_12_layer_model.predict(x_test)

result=np.argmax(result,axis=1)
y_test=np.argmax(y_test,axis=1)
errors=np.argwhere(result!=y_test)
print('errors num: ',len(errors))
print('predict acc = ',1-len(errors)/test_examples_num)

target_names = ['atmosphere', 'Liquor1', 'Liquor2', 'Liquor3', 'Liquor4', 'Liquor5']
print(classification_report(y_test, result, target_names=target_names, digits=3))
print(confusion_matrix(y_test, result, labels=range(args.classes_num)))