import tensorflow as tf
from cvnn.layers import ComplexConv2D, ComplexConv2DTranspose, ComplexMaxPooling2D, ComplexUpSampling2D
#from cvnn import ComplexBatchNormalization
from cvnn.activations import crelu, zrelu, modrelu, cart_leaky_relu
from typing import Tuple

DEFAULT_ACTIVATION = 'crelu'

def get_activation(activation: str):
    
    activations = {'crelu': crelu,
                   'zrelu': zrelu,
                   'modrelu': modrelu,
                   'cleakyrelu':cart_leaky_relu}
    if activation is not None:
        activation = activations[activation]
    return activation

def conv(n_filters: int,
         kernel_size: Tuple = (3,3), 
         strides: Tuple = (1,1),
         activation: str = DEFAULT_ACTIVATION,
         padding: str = 'same',
         data_format: str = 'channels_last'):
    
    return ComplexConv2D(n_filters, kernel_size, activation=get_activation(activation), 
                         strides=strides, padding=padding, data_format=data_format)


def down_conv(n_filters: int, 
         kernel_size: Tuple= (3,3),
         down_strides: Tuple= (2,2),
         activation: str = DEFAULT_ACTIVATION,
         padding: str = 'same',
         data_format: str = 'channels_last'):
    return ComplexConv2D(n_filters, kernel_size, activation=get_activation(activation), 
                         strides=down_strides, padding=padding, data_format=data_format)

def pool(pool_size: Tuple= (2,2),
         padding: str = 'same',
         data_format: str = 'channels_last'):
    return ComplexMaxPooling2D(pool_size=pool_size, padding=padding, data_format=data_format)

def up_conv(n_filters: int,
           kernel_size: Tuple = (3,3),
           up_strides: Tuple = (2,2), 
           activation: str = DEFAULT_ACTIVATION,
           padding: str = 'same',
           data_format: str = 'channels_last'):
    return ComplexConv2DTranspose(n_filters, kernel_size, activation=get_activation(activation), 
                                  strides=up_strides, padding=padding, data_format=data_format)

def upsample(size: Tuple = (2,2),
             padding: str = 'same',
             data_format: str = 'channels_last'):
    return ComplexUpSampling2D(size=size, padding=padding, data_format=data_format)
            

def create_basic_model(img_h, img_w):
    input_img = tf.keras.Input(shape=(img_h, img_w, 1), dtype='complex64') 
    
    down_stack = [conv(32), 
                  down_conv(32),
                  conv(64),
                  down_conv(64),
                  conv(128),
                  down_conv(128)]
    up_stack = [conv(128),
                up_conv(128),
                conv(64),
                up_conv(64),
                conv(32),
                up_conv(32)]
    last = ComplexConv2D(1, (3,3), padding='same', data_format='channels_last')
    
    x = input_img
    for layer in down_stack:
        x = layer(x)
            
    for layer in up_stack:
        x = layer(x)
        
    decoded = last(x)
    model = tf.keras.Model(input_img, decoded)
    return model

def create_unet_model(img_h, img_w):
    input_img = tf.keras.Input(shape=(img_h, img_w, 1), dtype='complex64') 
    
    down_stack = [conv(32), 
                  down_conv(32),
                  conv(64),
                  down_conv(64),
                  conv(128),
                  down_conv(128)]
    up_stack = [conv(128),
                up_conv(128),
                conv(64),
                up_conv(64),
                conv(32),
                up_conv(32)]
    
    last = ComplexConv2D(1, (3,3), padding='same', data_format='channels_last')
    concat = tf.keras.layers.Concatenate()
    
    x = input_img
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
        
    skips = reversed(skips)
    
    # Upsampling and establishing the skip connections
    for indx, (up, skip) in enumerate(zip(up_stack, skips)):
        x = up(x)
        if (indx + 1) % 2 == 0: 
            print(x, skip)
            x = concat([x, skip])
                  
    decoded = last(x)
    model = tf.keras.Model(input_img, decoded)
    return model



def create_resnet_model(img_h, img_w):
    input_img = tf.keras.Input(shape=(img_h, img_w, 1), dtype='complex64') 
    
    down_stack = [conv(32), 
                  down_conv(32),
                  conv(64),
                  down_conv(64),
                  conv(128),
                  down_conv(128)]
    up_stack = [conv(128),
                up_conv(128),
                conv(64),
                up_conv(64),
                conv(32),
                up_conv(32)]
    
    last = ComplexConv2D(1, (3,3), padding='same', data_format='channels_last')
    concat = tf.keras.layers.Concatenate()
    
    x = input_img
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
        
    skips = reversed(skips)
    
    # Upsampling and establishing the skip connections
    for indx, (up, skip) in enumerate(zip(up_stack, skips)):
        x = up(x)
        x = x - skip
        #if (indx + 1) % 2 == 0: 
        #    print(x, skip)
        #    x = concat([x, skip])
                  
    decoded = last(x)
    model = tf.keras.Model(input_img, decoded)
    return model

def full_summary(layer):
    
    #check if this layer has layers
    if hasattr(layer, 'layers'):
        print('summary for ' + layer.name)
        layer.summary()

        for l in layer.layers:
            full_summary(l)
    
