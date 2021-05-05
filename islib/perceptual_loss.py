from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras import Model
import tensorflow as tf


"""layer_names = ['block1_pool',
               'block2_pool',
               'block3_pool',
               ]"""

layer_names = ['conv_7b_ac']

"""vgg = VGG16(include_top=False, weights='imagenet', input_shape=(160, 160, 3))
vgg.trainable = False
outputs = [vgg.get_layer(name).output for name in layer_names_2]
vgg = Model([vgg.input], outputs)
vgg.summary()"""

resnet = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(160, 160, 3))
resnet.trainable = False
outputs = [resnet.get_layer(name).output for name in layer_names]
resnet = Model([resnet.input], outputs)
resnet.summary()


def perceptual(true, pred):
    true = tf.expand_dims(true, 0)
    pred = tf.expand_dims(pred, 0)
    y1 = resnet(preprocess_input(true))
    x1 = resnet(preprocess_input(pred))
    names = [0]
    erro = tf.add_n([tf.reduce_mean(tf.square(y1[name] - x1[name]))
                     for name in names])
    return erro
