from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed, \
    Conv3D, Conv2D, Conv1D, Flatten, MaxPooling1D, MaxPooling3D, MaxPooling2D, \
    GlobalAveragePooling2D
from keras.layers import GRU, Bidirectional, BatchNormalization
from keras.layers import Input, ELU, Lambda
from keras.layers import Reshape
from keras.optimizers import Adam, Nadam
from keras import backend as keras_backend
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras_contrib.applications import resnet
from keras_contrib.applications.resnet import ResNet152, ResNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.vgg16 import VGG16
from keras_contrib.layers import advanced_activations
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, Concatenate, ReLU, LeakyReLU
import tensorflow as tf

from dataset import *
from train_utils import *
from metrics import MacroF1Score


def elu(x, alpha=0.05):
    return K.elu(x, alpha)


def create_model_resnet50_plain(dataset: Dataset, input_shape, dropout=0.3, datatype: DataType = DataType.train):
    """
    Function creating the model's graph in Keras.
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance
    """
    base_input = Input(shape=input_shape)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=base_input, pooling=None)
    x = GlobalAveragePooling2D()(base_model.layers[-1].output)
    # x = BatchNormalization()(x)
    # if datatype != DataType.test:
    #     x = Dropout(dropout)(x)
    # x = BatchNormalization()(x)
    if datatype != DataType.test:
        x = Dropout(dropout)(x)
    x = Dense(dataset.class_num, activation='softmax')(x)
    model = Model(inputs=[base_model.input], outputs=[x])
    model.summary()
    return model


def create_model_resnet152_plain(dataset: Dataset, input_shape, dropout=0.5, datatype: DataType = DataType.train):
    """
    Function creating the model's graph in Keras.
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance
    """
    x_input = Input(name='my_input', shape=input_shape)
    x = x_input
    x = Conv2D(name='my_conf', padding='same', filters=3, kernel_size=(2, 2))(x)
    # x = MaxPooling2D(name='my_max_pool', pool_size=(2, 2), strides=2, padding='same')(x)
    x = BatchNormalization(name='my_batch')(x)

    base_input = Input(shape=(IMAGE_SIZE//2, IMAGE_SIZE//2, 3))
    base_model = ResNet((IMAGE_SIZE//2, IMAGE_SIZE//2, 3), block='bottleneck', repetitions=[3, 8, 36, 3], include_top=False)
    base_model.load_weights('resnet152_weights_tf.h5')
    # base_model.summary()
    base_model.layers.pop(0)

    x_classify = base_model(x)
    x_classify = GlobalAveragePooling2D()(x_classify)
    if datatype != DataType.test:
        x_classify = Dropout(dropout)(x_classify)
    x_classify = Dense(dataset.class_num, activation='softmax')(x_classify)
    model = Model(inputs=[x_input], outputs=[x_classify])
    model.summary()
    return model


def create_model_inceptionresnetv2_plain(dataset: Dataset, input_shape, dropout=0.5, datatype: DataType = DataType.train):
    """
    Function creating the model's graph in Keras.
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance
    """
    base_input = Input(shape=input_shape)
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=base_input, pooling=None)
    x = GlobalAveragePooling2D()(base_model.layers[-1].output)
    if datatype != DataType.test:
        x = Dropout(dropout)(x)
    x = Dense(dataset.class_num, activation='softmax')(x)
    model = Model(inputs=[base_model.input], outputs=[x])
    model.summary()
    return model


def create_model_mobilenet(dataset: Dataset, input_shape, dropout=0.5, datatype: DataType = DataType.train):
    """
    Function creating the model's graph in Keras.
    loss: 0.8077 - f1_score: 0.4574 - val_loss: 0.8119 - val_f1_score: 0.4471
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance
    """
    base_input = Input(shape=input_shape)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=base_input, pooling=None)
    x = GlobalAveragePooling2D()(base_model.layers[-1].output)
    if datatype != DataType.test:
        x = Dropout(dropout)(x)
    x = Dense(dataset.class_num, activation='softmax')(x)
    model = Model(inputs=[base_model.input], outputs=[x])
    model.summary()
    return model


def create_model_giim(dataset: Dataset, input_shape, dropout=0.5, datatype: DataType = DataType.train):
    model = Sequential()

    # picking vgg16 as pretrained (base) model https://keras.io/applications/#vgg16
    conv_base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in conv_base.layers:
        layer.trainable = False

    # maybe unfreeze last layer
    conv_base.layers[-2].trainable = True

    model.add(conv_base)
    model.add(Flatten())
    model.add(Dropout(0.33))
    model.add(Dense(48, activation='relu'))  # 64
    model.add(Dropout(0.33))
    model.add(Dense(48, activation='relu'))  # 48
    model.add(Dropout(0.33))
    model.add(Dense(dataset.class_num, activation='softmax'))
    return model


def max_average_precision(y_true, y_pred):
    result, update = tf.metrics.average_precision_at_k(tf.cast(y_true, tf.int64), tf.cast(y_pred, tf.float32), 5)
    return result


def build_model(model: Model, model_filename: str = None, learning_rate=0.0001):
    if model_filename and os.path.exists(model_filename):
        print(f"load weights: file={model_filename}")
        model.load_weights(model_filename)
    opt = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
    model.compile(# optimizer=keras.optimizers.Adadelta(),
                  optimizer=opt,
                  loss=keras.losses.categorical_crossentropy,
                  # loss=max_average_precision,
                  metrics=['acc', ],)
    return model
