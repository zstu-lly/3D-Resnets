from keras.layers import Dropout, GlobalAveragePooling3D, Input, Dense, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, AveragePooling2D, AveragePooling3D, Flatten, BatchNormalization, ReLU, TimeDistributed
from keras.models import Model
import keras
import tensorflow as tf
from keras.applications.xception import Xception
from keras import layers
from config import num_classes, img_size, clip_length


def bottleneck_residual(x, out_filter, strides, activate_before_residual=False, inflate=False):
    orig_x = x
    # a
    if inflate:
        x = Conv3D(out_filter // 4, (3, 1, 1), strides=strides, padding='same')(x)
    else:
        x = Conv3D(out_filter // 4, (1, 1, 1), strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # b
    if orig_x.get_shape().as_list()[-1] != out_filter and out_filter != 256:
        x = Conv3D(out_filter // 4, (1, 3, 3), strides=(1, 2, 2), padding='same')(x)
    else:
        x = Conv3D(out_filter // 4, (1, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # c
    x = Conv3D(out_filter, (1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # when channels change, shortcut
    if orig_x.get_shape().as_list()[-1] != out_filter and out_filter != 256:
        orig_x = Conv3D(out_filter, (1, 1, 1), strides=(1, 2, 2), padding='same')(orig_x)
    else:
        orig_x = Conv3D(out_filter, (1, 1, 1), strides=(1, 1, 1), padding='same')(orig_x)
    orig_x = BatchNormalization()(orig_x)
    orig_x = ReLU()(orig_x)

    x = keras.layers.add([orig_x, x])
    x = ReLU()(x)

    return x


def build_3d_net(model_net, num_classes, clip_length, img_size):
    video_input = Input(shape=(clip_length, img_size, img_size, 3))
    encoded_frame_sequence = TimeDistributed(model_net)(video_input)
    print(encoded_frame_sequence.shape)
    resnet = Conv3D(64, (5, 7, 7), strides=(1, 1, 1), padding='same', name='3d_resnet_3a_1_3x3x3')(encoded_frame_sequence)
    resnet = BatchNormalization()(resnet)
    resnet = ReLU()(resnet)

    block_num = [3, 4, 6, 3]
    # res2
    resnet = bottleneck_residual(resnet, 256, strides=(1, 1, 1), inflate=True)
    for _ in range(1, block_num[0]):
        resnet = bottleneck_residual(resnet, 256, strides=(1, 1, 1), activate_before_residual=False, inflate=True)

    resnet = MaxPooling3D(pool_size=(3, 1, 1), strides=(2, 1, 1), padding='same')(resnet)

    # res3
    resnet = bottleneck_residual(resnet, 512, strides=(1, 1, 1), activate_before_residual=False, inflate=True)
    for _ in range(1, block_num[1]):
        if _ % 2:
            resnet = bottleneck_residual(resnet, 512, strides=(1, 1, 1), activate_before_residual=False, inflate=False)
        else:
            resnet = bottleneck_residual(resnet, 512, strides=(1, 1, 1), activate_before_residual=False, inflate=True)

    # res4
    resnet = bottleneck_residual(resnet, 1024, strides=(1, 1, 1), activate_before_residual=False, inflate=True)
    for _ in range(1, block_num[1]):
        if _ % 2:
            resnet = bottleneck_residual(resnet, 1024, strides=(1, 1, 1), activate_before_residual=False, inflate=False)
        else:
            resnet = bottleneck_residual(resnet, 1024, strides=(1, 1, 1), activate_before_residual=False, inflate=True)

    resnet = MaxPooling3D(pool_size=(3, 1, 1), strides=(2, 1, 1), padding='same')(resnet)

    # res5    全连接层原文中用的1024
    resnet = bottleneck_residual(resnet, 1024, strides=(1, 1, 1), activate_before_residual=False, inflate=True)
    for _ in range(1, block_num[1]):
        if _ % 2:
            resnet = bottleneck_residual(resnet, 1024, strides=(1, 1, 1), activate_before_residual=False, inflate=True)
        else:
            resnet = bottleneck_residual(resnet, 1024, strides=(1, 1, 1), activate_before_residual=False, inflate=False)

    print(resnet.shape)
    resnet = GlobalAveragePooling3D()(resnet)    # ECO作者用的GAP
    print(resnet.shape)
    resnet = Dropout(0.5)(resnet)
    # 354个类别
    predictions = Dense(num_classes, activation='softmax')(resnet)
    print(predictions.shape)
    # 最终模型
    with tf.device('/cpu:0'):
        InceptionV3_Resnet3D = Model(inputs=video_input, outputs=predictions)
    print(InceptionV3_Resnet3D.summary())
    return InceptionV3_Resnet3D


def build_2d_net(img_size):
    # 输出是96*28*28
    base_model = Xception(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))    # default input shape is (299, 299, 3)

    x = layers.SeparableConv2D(96, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv1')(base_model.get_layer('block4_sepconv1_act').output)
    x = layers.BatchNormalization(name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)

    model = Model(inputs=base_model.input, outputs=x)
    print(model.summary())
    return model


if __name__ == '__main__':
    model_2d = build_2d_net(img_size)
    model_3d = build_3d_net(model_2d, num_classes, clip_length, img_size)

