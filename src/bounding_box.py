# https://www.kaggle.com/martinpiotte/bounding-box-model/notebook

from PIL import Image as pil_image
from PIL.ImageDraw import Draw
from os.path import isfile
import numpy as np
from scipy.ndimage import affine_transform
from keras.preprocessing.image import img_to_array
import random
from keras.utils import Sequence
from keras import backend as K
from numpy.linalg import inv as mat_inv
from keras.engine.topology import Input
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Model


# Define useful constants
img_shape = (128, 128, 1)
anisotropy = 2.15


def expand_path(p):
    if isfile('train/' + p): return 'train/' + p
    if isfile('test/' + p): return 'test/' + p
    if isfile('whale_train/' + p): return 'whale_train/' + p
    if isfile('whale_test/' + p): return 'whale_test/' + p
    return p


def read_raw_image(p):
    return pil_image.open(expand_path(p))


def draw_dot(draw, x, y):
    draw.ellipse(((x - 5, y - 5), (x + 5, y + 5)), fill='red', outline='red')


def draw_dots(draw, coordinates):
    for x, y in coordinates: draw_dot(draw, x, y)


def bounding_rectangle(list):
    x0, y0 = list[0]
    x1, y1 = x0, y0
    for x, y in list[1:]:
        x0 = min(x0, x)
        y0 = min(y0, y)
        x1 = max(x1, x)
        y1 = max(y1, y)
    return x0, y0, x1, y1


# Read an image as black&white numpy array
def read_array(p):
    img = read_raw_image(p).convert('L')
    return img_to_array(img)


def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    rotation = np.deg2rad(rotation)
    shear = np.deg2rad(shear)
    rotation_matrix = np.array(
        [[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
    shear_matrix = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix = np.array([[1.0 / height_zoom, 0, 0], [0, 1.0 / width_zoom, 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))


# Compute the coordinate transformation required to center the pictures, padding as required.
def center_transform(affine, input_shape):
    hi, wi = float(input_shape[0]), float(input_shape[1])
    ho, wo = float(img_shape[0]), float(img_shape[1])
    top, left, bottom, right = 0, 0, hi, wi
    if wi / hi / anisotropy < wo / ho:  # input image too narrow, extend width
        w = hi * wo / ho * anisotropy
        left = (wi - w) / 2
        right = left + w
    else:  # input image too wide, extend height
        h = wi * ho / wo / anisotropy
        top = (hi - h) / 2
        bottom = top + h
    center_matrix = np.array([[1, 0, -ho / 2], [0, 1, -wo / 2], [0, 0, 1]])
    scale_matrix = np.array([[(bottom - top) / ho, 0, 0], [0, (right - left) / wo, 0], [0, 0, 1]])
    decenter_matrix = np.array([[1, 0, hi / 2], [0, 1, wi / 2], [0, 0, 1]])
    return np.dot(np.dot(decenter_matrix, scale_matrix), np.dot(affine, center_matrix))


# Apply an affine transformation to an image represented as a numpy array.
def transform_img(x, affine):
    matrix = affine[:2, :2]
    offset = affine[:2, 2]
    x = np.moveaxis(x, -1, 0)
    channels = [affine_transform(channel, matrix, offset, output_shape=img_shape[:-1], order=1,
                                 mode='constant', cval=np.average(channel)) for channel in x]
    return np.moveaxis(np.stack(channels, axis=0), 0, -1)


# Read an image for validation, i.e. without data augmentation.
def read_for_validation(p):
    x = read_array(p)
    t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t = center_transform(t, x.shape)
    x = transform_img(x, t)
    x -= np.mean(x, keepdims=True)
    x /= np.std(x, keepdims=True) + K.epsilon()
    return x, t


# Read an image for validation, i.e. without data augmentation.
def normalize(x):
    x -= np.mean(x, keepdims=True)
    x /= np.std(x, keepdims=True) + K.epsilon()
    return x


# Read an image for training, i.e. including a random affine transformation
def read_for_training(p):
    x = read_array(p)
    t = build_transform(
        random.uniform(-5, 5),
        random.uniform(-5, 5),
        random.uniform(0.9, 1.0),
        random.uniform(0.9, 1.0),
        random.uniform(-0.05 * img_shape[0], 0.05 * img_shape[0]),
        random.uniform(-0.05 * img_shape[1], 0.05 * img_shape[1]))
    t = center_transform(t, x.shape)
    x = transform_img(x, t)
    x -= np.mean(x, keepdims=True)
    x /= np.std(x, keepdims=True) + K.epsilon()
    return x, t


# Transform corrdinates according to the provided affine transformation
def coord_transform(list, trans):
    result = []
    for x, y in list:
        y, x, _ = trans.dot([y, x, 1]).astype(np.int)
        result.append((x, y))
    return result


def build_bounding_box_model(with_dropout=True):
    kwargs = {'activation': 'relu', 'padding': 'same'}
    conv_drop = 0.2
    dense_drop = 0.5
    inp = Input(shape=img_shape)

    x = inp

    x = Conv2D(64, (9, 9), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    h = MaxPooling2D(pool_size=(1, int(x.shape[2])))(x)
    h = Flatten()(h)
    if with_dropout: h = Dropout(dense_drop)(h)
    h = Dense(16, activation='relu')(h)

    v = MaxPooling2D(pool_size=(int(x.shape[1]), 1))(x)
    v = Flatten()(v)
    if with_dropout: v = Dropout(dense_drop)(v)
    v = Dense(16, activation='relu')(v)

    x = Concatenate()([h, v])
    if with_dropout: x = Dropout(0.5)(x)
    x = Dense(4, activation='linear')(x)
    return Model(inp, x)


class TrainingData(Sequence):
    def __init__(self, batch_size=32):
        super(TrainingData, self).__init__()
        self.batch_size = batch_size

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(len(train), start + self.batch_size)
        size = end - start
        a = np.zeros((size,) + img_shape, dtype=K.floatx())
        b = np.zeros((size, 4), dtype=K.floatx())
        for i, (p, coords) in enumerate(train[start:end]):
            img, trans = read_for_training(p)
            coords = coord_transform(coords, mat_inv(trans))
            x0, y0, x1, y1 = bounding_rectangle(coords)
            a[i, :, :, :] = img
            b[i, 0] = x0
            b[i, 1] = y0
            b[i, 2] = x1
            b[i, 3] = y1
        return a, b

    def __len__(self):
        return (len(train) + self.batch_size - 1) // self.batch_size
