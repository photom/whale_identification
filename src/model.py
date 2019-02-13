import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed, \
    Conv3D, Conv2D, Conv1D, Flatten, MaxPooling1D, MaxPooling3D, MaxPooling2D, \
    GlobalAveragePooling2D, Layer, GlobalMaxPooling2D, AveragePooling2D
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
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, \
    Concatenate, ReLU, LeakyReLU
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

    base_input = Input(shape=(IMAGE_SIZE // 2, IMAGE_SIZE // 2, 3))
    base_model = ResNet((IMAGE_SIZE // 2, IMAGE_SIZE // 2, 3), block='bottleneck', repetitions=[3, 8, 36, 3],
                        include_top=False)
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


def create_model_inceptionresnetv2_plain(dataset: Dataset, input_shape, dropout=0.5,
                                         datatype: DataType = DataType.train):
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


def create_model_siamese_resnet(dataset: Dataset, input_shape, dropout=0.5, datatype: DataType = DataType.train):
    """
    https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d
        Model architecture
    """

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    base_input = Input(shape=input_shape)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=base_input, pooling=None)
    x = GlobalAveragePooling2D()(base_model.layers[-1].output)
    if datatype != DataType.test:
        x = Dropout(dropout)(x)
    x = Dense(4096, activation='sigmoid')(x)
    trunk_model = Model(inputs=[base_input], outputs=[x])
    # Generate the encodings (feature vectors) for the two images
    encoded_l = trunk_model(left_input)
    encoded_r = trunk_model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    l1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    l1_distance = l1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid')(l1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
    siamese_net.summary()
    # return the model
    return siamese_net


def create_model_siamese(dataset: Dataset, input_shape, dropout=0.5, datatype: DataType = DataType.train):
    """
    https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d
        Model architecture
    """

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu'))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid'))

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid')(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # return the model
    return siamese_net


class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        a, p, n = inputs
        p_dist = K.sum(K.square(a - p), axis=-1)
        n_dist = K.sum(K.square(a - n), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = K.cast(self.triplet_loss(inputs), 'float32')
        self.add_loss(loss)
        return loss


def build_triplet_trunk_model(input_tensor, dropout=0.5, datatype: DataType = DataType.train,
                              layer_name='anchor'):
    # Convolutional Neural Network
    # base_input = Input(shape=input_shape)
    # base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor, pooling=None)
    base_model = ResNet50(weights=None, include_top=False, input_tensor=input_tensor, pooling=None)

    x = GlobalAveragePooling2D()(base_model.layers[-1].output)
    # x = GlobalMaxPooling2D()(base_model.output)
    if datatype != DataType.test:
        x = Dropout(dropout)(x)
    # x = Dense(4096)(x)
    # x = BatchNormalization()(x)
    # x = Dense(1024, activation='sigmoid')(x)
    # x = Dense(50)(x)
    # x = BatchNormalization()(x)
    # x = Lambda(lambda x_input: K.l2_normalize(x_input, axis=1))(x)
    # x = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), data_format='channels_first')(base_model.output)
    # x = Flatten()(x)
    # if datatype != DataType.test:
    #     x = Dropout(dropout)(x)
    x = Dense(256, name='dense_layer')(x)
    # L2 normalization
    x = Lambda(lambda xi: K.l2_normalize(xi, axis=1))(x)

    tmp_model = Model(input_tensor, x)
    for idx, layer in enumerate(tmp_model.layers):
        layer.trainable = True
        # layer.name = f"{layer_name}_{idx}"
    return tmp_model


def create_model_triplet_loss(dataset: Dataset, input_shape, dropout=0.5, datatype: DataType = DataType.train):
    # loss: 1.5538e-05 - val_loss: 0.0204 lr=0.0000001 alpha=0.005
    anchor_input = Input(shape=input_shape)
    positive_input = Input(shape=input_shape)
    negative_input = Input(shape=input_shape)
    base_input = Input(shape=input_shape)
    trunk_model = build_triplet_trunk_model(base_input, dropout, datatype)
    anchor_embedding = trunk_model(anchor_input)
    positive_embedding = trunk_model(positive_input)
    negative_embedding = trunk_model(negative_input)
    # distance_func = Lambda(lambda tensors: K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims=True),
    #                        name='distance')
    # triplet_loss_func = Lambda(lambda loss: K.maximum(loss[0] - loss[1] + alpha, 0.0),
    #                            name='triplet_loss')
    # p_distance = distance_func([anchor_embedding, positive_embedding])
    # n_distance = distance_func([anchor_embedding, negative_embedding])
    # final_loss = triplet_loss_func([p_distance, n_distance])
    # xa_inp = Input(shape=trunk_model.output_shape[1:])
    # xp_inp = Input(shape=trunk_model.output_shape[1:])
    # xn_inp = Input(shape=trunk_model.output_shape[1:])

    # final_loss = triplet_loss_func([xa_inp, xp_inp, xn_inp])

    # head_model = Model([xa_inp, xp_inp, xn_inp], final_loss)
    # model = Model([anchor_input, positive_input, negative_input],
    #               head_model([anchor_embedding, positive_embedding, negative_embedding]))
    # BPR loss
    # final_loss = 1.0 - K.sigmoid(
    #     K.sum(anchor_embedding * positive_embedding, axis=-1, keepdims=True) -
    #     K.sum(anchor_embedding * negative_embedding, axis=-1, keepdims=True))
    # bpr_triplet_loss_func = Lambda(lambda x_input: bpr_triplet_loss(x_input), name='loss')
    # final_loss = bpr_triplet_loss_func([anchor_embedding, positive_embedding, negative_embedding])

    triplet_loss_func = Lambda(lambda x_input: calc_triplet_loss(x_input), name='triplet_loss')
    final_loss = triplet_loss_func([anchor_embedding, positive_embedding, negative_embedding])
    model = Model([anchor_input, positive_input, negative_input], final_loss)

    for idx, layer in enumerate(model.layers):
        layer.trainable = True
    # triplet_loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([embedding_a, embedding_p, embedding_n])
    # model = Model([anchor_input, positive_input, negative_input], triplet_loss_layer)
    model.submodel = trunk_model
    model.summary()
    return model


def calc_triplet_loss(x, alpha=0.01):
    anchor, positive, negative = x

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    # print(f"loss: {K.shape(loss)}")
    return loss


def triplet_loss(y_pred, alpha=0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    anchor, positive, negative = y_pred

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=-1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=-1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)

    return loss


def bpr_triplet_loss(X):
    """
    https://nanx.me/blog/post/triplet-loss-r-keras/
    :param X:
    :return:
    """
    user_latent, negative_item_latent, positive_item_latent = X
    # BPR loss
    loss = 1.0 - K.sigmoid(
        K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
        K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))
    return loss


def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)


def build_inference_model(weight_path: str, dataset: Dataset, input_shape,
                          dropout=0.5, datatype: DataType = DataType.train):
    anchor_input = Input(shape=input_shape)
    positive_input = Input(shape=input_shape)
    negative_input = Input(shape=input_shape)
    anchor_model = build_triplet_trunk_model(anchor_input, dropout, datatype, layer_name='anchor')
    positive_model = build_triplet_trunk_model(positive_input, dropout, datatype, layer_name='positive')
    negative_model = build_triplet_trunk_model(negative_input, dropout, datatype, layer_name='negative')
    anchor_embedding = anchor_model.output
    positive_embedding = positive_model.output
    negative_embedding = negative_model.output

    bpr_triplet_loss_func = Lambda(lambda x_input: bpr_triplet_loss(x_input), name='loss')

    final_loss = bpr_triplet_loss_func([anchor_embedding, positive_embedding, negative_embedding])
    model = Model(inputs=[anchor_input, positive_input, negative_input],
                  outputs=final_loss)
    model.load_weights(weight_path)

    inference_model = Model(inputs=[anchor_model.get_input_at(0)], outputs=[anchor_model.get_output_at(0)])
    print(inference_model.summary())

    return inference_model


def build_anchor_model(weight_path: str, input_shape,
                       dropout=0.5, datatype: DataType = DataType.train):
    anchor_input = Input(shape=input_shape)
    positive_input = Input(shape=input_shape)
    negative_input = Input(shape=input_shape)
    anchor_model = build_triplet_trunk_model(anchor_input, dropout, datatype, layer_name='anchor')
    positive_model = build_triplet_trunk_model(positive_input, dropout, datatype, layer_name='positive')
    negative_model = build_triplet_trunk_model(negative_input, dropout, datatype, layer_name='negative')
    anchor_embedding = anchor_model.output
    positive_embedding = positive_model.output
    negative_embedding = negative_model.output

    bpr_triplet_loss_func = Lambda(lambda x_input: bpr_triplet_loss(x_input), name='loss')

    final_loss = bpr_triplet_loss_func([anchor_embedding, positive_embedding, negative_embedding])
    model = Model(inputs=[anchor_input, positive_input, negative_input],
                  outputs=final_loss)
    if os.path.exists(weight_path):
        model.load_weights(weight_path)

    inference_model = Model(inputs=[anchor_model.get_input_at(0)], outputs=[anchor_model.get_output_at(0)])
    # print(inference_model.summary())
    return anchor_input, inference_model


def build_model(model: Model, model_filename: str = None, learning_rate=0.000005):
    if model_filename and os.path.exists(model_filename):
        print(f"load weights: file={model_filename}")
        model.load_weights(model_filename)
    submodel_filename = f"{model_filename}.submodel"
    if submodel_filename and os.path.exists(submodel_filename):
        print(f"load weights: file={submodel_filename}")
        model.submodel.load_weights(submodel_filename)

    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
    model.compile(
        # optimizer=keras.optimizers.Adadelta(),
        optimizer=opt,
        loss=identity_loss,
        # loss=keras.losses.mean_absolute_error,
        # loss=keras.losses.binary_crossentropy,
        # loss=keras.losses.categorical_crossentropy,
        # loss=max_average_precision,
        # metrics=['acc'], )
    )

    return model
