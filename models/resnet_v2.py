"""
ResNet v2
This is a revised implementation from Cifar10 ResNet example in Keras:
(https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py)
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
"""

from __future__ import print_function
import keras
from keras.layers import Dense, Conv1D, BatchNormalization, Activation, Dropout, Permute, Concatenate, Reshape
from keras.layers import AveragePooling1D, Input, Flatten, Lambda, Add, average, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from models.attention_module import attach_attention_module
from keras.layers.recurrent import LSTM, GRU
from models.layer_utils import AttentionLSTM
from models.attention_lstm import AttentionLSTM_t
import pdb
from keras import initializers




def slice(x,index):
        return x[:,:,index]

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """1D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv1D number of filters
        kernel_size (int): Conv1D square kernel dimensions
        strides (int): Conv1D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv1D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer=initializers.he_normal(seed=None),
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v2_original(input_shape, depth, num_classes=6, attention_module=None, cbam_mode=1,lstm_layers=128, \
                       return_sequences=False,num_filters_in=16,pool_size=8,stages_num=3,dropout=0.1, \
                       recurrent_dropout=0.2,dense_dropout=0.1):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    # num_filters_in = 32
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)

    print('shape of inputs: ',inputs.shape)

    inputs_1=Lambda(lambda x:x[:,:,0:5])(inputs)

    # v2 performs Conv1D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    print('output shape of first conv: ',x.shape)

    # Instantiate the stack of residual units
    for stage in range(stages_num):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample
                    num_filters_in = int(num_filters_in/2)
                num_filters_out = num_filters_in * 4
                
                     

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            print('output shape of layer 1: ',y.shape)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            print('output shape of layer 2: ',y.shape)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            print('output shape of layer 3: ',y.shape)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            # attention_module
            if attention_module is not None:
                print('shape before CBAM: ',y.shape)
                y = attach_attention_module(y, attention_module, cbam_mode)
                print('shape after CBAM: ',y.shape)
                
            x = keras.layers.add([x, y])
            print('output shape of layer: ',x.shape)
        num_filters_in = num_filters_out
        if stage == 0:
          stage1_output = x
          print('output shape of stage1: ',stage1_output.shape)
        elif stage == 1:
          stage2_output = x
          print('output shape of stage2: ',stage2_output.shape)


    print('output shape of ResNet: ',x.shape)
    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x1 = GlobalAveragePooling1D()(x)
    RNN_in = AveragePooling1D(pool_size=pool_size)(x)
    print('output shape of Pooling: ',x.shape)
    y3 = LSTM(lstm_layers,return_state=False, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=return_sequences)(RNN_in)

    if return_sequences==True:
      y = Flatten()(y)

      print('Flatten output shape: ',y.shape)
    
    x = Concatenate(axis=1)([y3,x1])
    x = Dense(512,activation='relu')(x)
    x = Dropout(dense_dropout)(x)
    x = Dense(512,activation='relu')(x)
    x = Dropout(dense_dropout)(x)

    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer=initializers.he_normal(seed=None))(x)
    print('outputs shape: ',outputs.shape)

    # Instantiate model.
    print(inputs)
    print(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    return model
