import os
import pickle
import sys
import pathlib
from pathlib import Path
import csv
from enum import Enum
from typing import Union

import keras
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from iterstrat.ml_stratifiers import RepeatedMultilabelStratifiedKFold
from skmultilearn import model_selection
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator

RANDOM_NUM = 77777
IMAGE_BASE_DIR = '../whale_identification_dataset'
TRAIN_DIRS = [f"{IMAGE_BASE_DIR}/train", ]
TRAIN_ANSWER_FILES = ['train.csv', ]
TEST_DIR = IMAGE_BASE_DIR + '/test'
# dataset ratio
TRAIN_RATIO = 0.9
VALIDATE_RATIO = 0.1
TEST_RATIO = 0.05

IMAGE_SIZE = 256
IMAGE_MINMAX_MAP = {}
BATCH_SIZE = 5
EPOCHS = 100


IMAGE_GEN = ImageDataGenerator(
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True)


class DataType(Enum):
    train = 1
    validate = 2
    test = 3


class LabelOneHotEncoder(object):
    def __init__(self):
        self.ohe = OneHotEncoder()
        self.le = LabelEncoder()

    def fit_transform(self, x):
        features = self.le.fit_transform(x)
        return self.ohe.fit_transform(features.reshape(-1, 1))

    def transform(self, x):
        return self.ohe.transform(self.la.transform(x.reshape(-1, 1)))

    def inverse_tranform(self, x):
        return self.le.inverse_transform(self.ohe.inverse_tranform(x))

    def inverse_labels(self, x):
        return self.le.inverse_transform(x)


class Dataset(Callback):
    def __init__(self, data_list: np.array, ids: set):
        super(Dataset, self).__init__()
        self.data_list = data_list
        self.y_list = None
        self.class_num = len(ids)
        self.train_index_list = np.array([])
        self.validate_index_list = np.array([])
        self.train_counter = 0
        self.validate_counter = 0
        self.index_generator = None
        self.label_onehot_encoder = None
        self.answers = np.array([])

    def on_train_begin(self, logs=None):
        self.set_index_list()

    def on_epoch_end(self, epoch, logs=None):
        self.set_index_list()

    def set_index_list(self):
        try:
            train_index, validate_index = next(self.index_generator)
        except StopIteration:
            sss = StratifiedKFold(n_splits=10, shuffle=True, random_state=np.random.randint(0, RANDOM_NUM))
            self.index_generator = sss.split(np.zeros(self.answers.shape[0]), self.answers)
            train_index, validate_index = next(self.index_generator)
        # print(f"train_index:{train_index} validate_index:{validate_index}")
        self.train_index_list = train_index
        self.validate_index_list = validate_index
        self.train_counter = 0
        self.validate_counter = 0
        print(f"reset dataset. train_dataset:{len(train_index)} validate_dataset:{len(validate_index)}")

    def increment_train(self):
        self.train_counter = (self.train_counter + 1) % len(self.train_index_list)

    def increment_validate(self):
        self.validate_counter = (self.validate_counter + 1) % len(self.validate_index_list)

    def next_train_data(self):
        index = self.train_index_list[self.train_counter]
        self.increment_train()
        # print(f"next_train_data index:{index}")
        return self.data_list[index]

    def next_validate_data(self):
        index = self.validate_index_list[self.validate_counter]
        self.increment_validate()
        # print(f"next_validate_data index:{index}")
        return self.data_list[index]


class TestDataset:
    def __init__(self, data_list: list):
        self.data_list = data_list


class DataUnit:
    def __init__(self, filename: str, answer: Union[str, None], source_dir: str):
        self.filename = filename
        self.answer = answer
        self.source_dir = source_dir
        self.encoded_answer = None


def load_raw_data():
    data_list = []
    ids = set()
    answers = []

    for idx, train_answer_file in enumerate(TRAIN_ANSWER_FILES):
        source_dir = TRAIN_DIRS[idx]
        with open(train_answer_file, 'r') as f:
            csv_reader = csv.reader(f)
            # skip header
            next(csv_reader)
            # read lines
            for row in csv_reader:
                if len(row) != 2:
                    continue
                filename = row[0]
                answer = row[1].strip()
                # print(f"filename:{filename} answer:{answer}")
                ids.add(answer)
                data_unit = DataUnit(filename, answer, source_dir)
                data_list.append(data_unit)
                answers.append(answer)

    # complement single class for stratified K-fold
    single_classes = []
    for data_unit in data_list:
        x_list = []
        x = generate_input(data_unit)
        x_list.append(x)
        if answers.count(data_unit.answer) == 1:
            single_classes.append(data_unit)
            answers.append(data_unit.answer)
            x_list.append(x)
        x_list = np.array(x_list)
        x_list = x_list.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 3))
        IMAGE_GEN.fit(x_list)
    data_list = data_list + single_classes

    # set up one hot encoder
    lohe = LabelOneHotEncoder()
    y_list = lohe.fit_transform(answers).toarray()

    dataset = Dataset(np.array(data_list), ids)
    # dataset = Dataset(np.array(data_list), ids, x_list)
    for data_unit, y_encoded in zip(dataset.data_list, y_list):
        data_unit.encoded_answer = y_encoded
        # data_unit.encoded_answer = np.array(list(dataset.encoded_map[data_unit.answer]))
        # print(y_encoded)
    # dataset.data_list = np.array(data_list + data_list)
    dataset.y_list = y_list
    dataset.answers = np.array(answers)
    # dataset.y_list = y_list
    # print(f"y_list:{dataset.y_list.shape} {dataset.y_list}")
    dataset.label_onehot_encoder = lohe
    # classes, y_indices = np.unique(dataset.y_list, return_inverse=True)
    # n_classes = classes.shape[0]
    # class_counts = np.bincount(y_indices)
    # print(f"data_list:{dataset.data_list.shape} y_list:{dataset.y_list.shape} class_counts:{class_counts} y_indices:{y_indices} classes:{classes}")
    sss = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=RANDOM_NUM)
    dataset.index_generator = sss.split(np.zeros(dataset.answers.shape[0]), dataset.answers)
    # print(f"encoded_answer{data_unit.encoded_answer}")
    return dataset


def load_test_data():
    data_list = []
    for filename in os.listdir(TEST_DIR):
        data_list.append(DataUnit(filename, None, TEST_DIR))
    return TestDataset(data_list)


def normalize_image(x: np.array, data_unit: DataUnit):
    x = np.array(x)
    if data_unit.filename in IMAGE_MINMAX_MAP:
        min_val, max_val = IMAGE_MINMAX_MAP[data_unit.filename]
    else:
        min_val, max_val = float(x.min()), float(x.max())
        IMAGE_MINMAX_MAP[data_unit.filename] = (min_val, max_val)
    if max_val > min_val + np.finfo(float).eps:
        x = (x - min_val) / (max_val - min_val)
    else:
        print(f"too small difference. file={data_unit.filename}")
        x = x / 255.0
    x = np.stack(x, axis=1)
    return x


def create_xy(dataset: Dataset, datatype: DataType):
    # sample_num = len(dataset.data_list)
    # train_num = int(sample_num * TRAIN_RATIO)
    # validate_num = int(sample_num * VALIDATE_RATIO)
    # if datatype == DataType.train:
    #     index = np.random.randint(0, train_num)
    # elif datatype == DataType.validate:
    #     index = np.random.randint(train_num, train_num + validate_num)
    # else:
    #     raise RuntimeError(f"invalid data type. type={datatype}")
    # print(f"type:{str(datatype)} index:{index}")
    # data_unit = dataset.data_list[index]
    # print("create_xy")
    if datatype == DataType.train:
        data_unit = dataset.next_train_data()
    elif datatype == DataType.validate:
        data_unit = dataset.next_validate_data()
    else:
        raise RuntimeError(f"invalid data type. type={datatype}")
    # print(f"create_xy data_unit:{data_unit.answer}")
    x = generate_input(data_unit)
    # x = normalize_image(x, data_unit)
    y = data_unit.encoded_answer
    # print(f"create_training_sample x:{x.shape} y:{y.shape}")
    return x, y


def make_square(img, min_size=10, fill_color=(0, 0, 0, 0)):
    x, y = img.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(img, ((size - x) // 2, (size - y) // 2))
    # del img
    return new_im


def generate_input(data_unit: DataUnit):
    file_path = Path(data_unit.source_dir, data_unit.filename)
    img = Image.open(str(file_path))
    # img = make_square(img)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.LANCZOS)
    # img = make_square(img, IMAGE_SIZE)
    img = np.array(img)
    # x.append(img)
    if img.ndim == 2:  # imported BW picture and converting to "dumb RGB"
        img = np.tile(img, (3, 1, 1)).transpose((1, 2, 0))
    # merge and normalize
    x = img
    return x


def create_unit_dataset(data_unit: DataUnit):
    x = generate_input(data_unit)
    x = x.astype("float32")
    return IMAGE_GEN.standardize(x.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3))


def create_dataset(dataset: Dataset, num: int, datatype: DataType = DataType.train):
    train_dataset_x = []
    train_dataset_y = []
    for i in range(num):
        x, y = create_xy(dataset, datatype)
        train_dataset_x.append(x)
        train_dataset_y.append(y)
    train_dataset_x = np.array(train_dataset_x)
    train_dataset_y = np.array(train_dataset_y)
    return train_dataset_x, train_dataset_y


def load_dataset(filename):
    with open(filename, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        train_dataset = pickle.load(f)
    return train_dataset
