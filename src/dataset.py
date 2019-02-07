import os
import pickle
import sys
import pathlib
from pathlib import Path
import csv
from enum import Enum
from typing import Union
import uuid
import copy
import threading

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

IMAGE_SIZE = 200
IMAGE_MINMAX_MAP = {}
BATCH_SIZE = 5
EPOCHS = 30
NEW_LABEL = 'new_whale'

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
    def __init__(self, data_list: np.array, class_map: dict):
        super(Dataset, self).__init__()
        self.data_list = data_list
        self.class_map = class_map
        self.train_class_map = {}
        self.validate_class_map = {}
        self.classes = np.array(list(class_map.keys()))
        self.class_num = len(self.classes)
        self.train_index_list = np.array([])
        self.validate_index_list = np.array([])
        self.train_counter = 0
        self.validate_counter = 0
        self.index_generator = None
        self.label_onehot_encoder = None
        self.lock = threading.Lock()

    def on_train_begin(self, logs=None):
        sss = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=RANDOM_NUM)
        answers = np.array([data_unit.answer for data_unit in self.data_list])
        self.lock.acquire()
        try:
            self.index_generator = sss.split(np.zeros(answers.shape[0]), answers)
            self.set_index_list()
        finally:
            self.lock.release()

    def on_epoch_end(self, epoch, logs=None):
        self.lock.acquire()
        try:
            self.set_index_list()
        finally:
            self.lock.release()

    def set_index_list(self):
        # initialize index
        try:
            train_index, validate_index = next(self.index_generator)
        except StopIteration:
            sss = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=np.random.randint(0, RANDOM_NUM))
            answers = np.array([data_unit.answer for data_unit in self.data_list])
            self.index_generator = sss.split(np.zeros(answers.shape[0]), answers)
            train_index, validate_index = next(self.index_generator)
        # print(f"train_index:{train_index} validate_index:{validate_index}")
        self.train_index_list = train_index
        self.validate_index_list = validate_index
        print(f"reset dataset. train_dataset:{len(train_index)} validate_dataset:{len(validate_index)}")
        self.train_counter = 0
        self.validate_counter = 0

        # initialize class_map
        self.train_class_map = {}
        self.validate_class_map = {}
        train_list = [self.data_list[i].uuid for i in train_index]
        validate_list = [self.data_list[i].uuid for i in validate_index]
        for klass, data_list in self.class_map.items():
            self.train_class_map[klass] = []
            self.validate_class_map[klass] = []
            for data_unit in data_list:
                if data_unit.uuid in train_list:
                    self.train_class_map[klass].append(data_unit)
                if data_unit.uuid in validate_list:
                    self.validate_class_map[klass].append(data_unit)
        # for klass, data_list in self.class_map.items():
        #     print(
        #         f"class:{klass} train_class_map:{len(self.train_class_map[klass])} validate_class_map:{len(self.validate_class_map[klass])}")

    def increment_train(self):
        self.train_counter = (self.train_counter + 1) % len(self.train_index_list)

    def increment_validate(self):
        self.validate_counter = (self.validate_counter + 1) % len(self.validate_index_list)

    def next_train_data(self):
        self.lock.acquire()
        try:
            self.increment_train()
            index = self.train_index_list[self.train_counter]
            data_unit = self.data_list[index]
            # wait until set_index_list finished
            while data_unit.answer not in list(self.train_class_map.keys()):
                self.increment_train()
                index = self.train_index_list[self.train_counter]
                data_unit = self.data_list[index]
            # print(f"next_train_data index:{index}")
            return data_unit
        finally:
            self.lock.release()

    def next_validate_data(self):
        self.lock.acquire()
        try:
            self.increment_validate()
            index = self.validate_index_list[self.validate_counter]
            data_unit = self.data_list[index]
            # wait until set_index_list finished
            while data_unit.answer not in list(self.validate_class_map.keys()):
                self.increment_validate()
                index = self.validate_index_list[self.validate_counter]
                data_unit = self.data_list[index]
                if data_unit.answer not in list(self.validate_class_map.keys()):
                    print(f"klass not included in validate map. class:{data_unit.answer} validate_map:{list(self.validate_class_map.keys())}")
            return data_unit
        finally:
            self.lock.release()


class TestDataset:
    def __init__(self, data_list: list):
        self.data_list = data_list


class DataUnit:
    def __init__(self, filename: str, answer: Union[str, None], source_dir: str):
        self.filename = filename
        self.answer = answer
        self.source_dir = source_dir
        self.uuid = str(uuid.uuid1())


def load_raw_data():
    data_list = []
    class_map = {}
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
                answers.append(answer)
                # print(f"filename:{filename} answer:{answer}")
                data_unit = DataUnit(filename, answer, source_dir)
                data_list.append(data_unit)
                if answer not in class_map:
                    class_map[answer] = []
                class_map[answer].append(data_unit)

    # complement single class for stratified K-fold
    single_classes = []
    for data_unit in data_list:
        if answers.count(data_unit.answer) == 1:
            cloned = copy.deepcopy(data_unit)
            cloned.uuid = str(uuid.uuid1())
            single_classes.append(cloned)
            class_map[data_unit.answer].append(cloned)
            answers.append(cloned.answer)
    data_list = data_list + single_classes
    dataset = Dataset(np.array(data_list), class_map)
    uuids = [data_unit.uuid for data_unit in dataset.data_list]
    for uuuid in uuids:
        if uuids.count(uuuid) > 1:
            print(f"uuid: {uuuid} count:{uuids.count(uuuid)}")
            assert(uuids.count(uuuid) <= 1)
    # set up one hot encoder
    lohe = LabelOneHotEncoder()
    # y_list = lohe.fit_transform(answers).toarray()
    lohe.fit_transform(answers)
    dataset.label_onehot_encoder = lohe
    # print(f"data_list:{dataset.data_list.shape} y_list:{dataset.y_list.shape} class_counts:{class_counts} y_indices:{y_indices} classes:{classes}")
    return dataset


def load_test_data():
    data_list = []
    for filename in os.listdir(TEST_DIR):
        data_list.append(DataUnit(filename, None, TEST_DIR))
    return TestDataset(data_list)


def fit_image_generator(dataset: Dataset, test_dataset: TestDataset):
    # for data_list in [test_dataset.data_list, dataset.data_list]:
    for data_list in [dataset.data_list]:
        for data_unit in data_list:
            x = generate_input(data_unit)
            x_list = np.array([x]).reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 3))
            IMAGE_GEN.fit(x_list)


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
        raise RuntimeError(f"invalid data type. type={str(datatype)}")
    # print(f"create_xy data_unit:{data_unit.answer}")
    x = generate_input(data_unit)
    # x = normalize_image(x, data_unit)
    y = dataset.classes.tolist().index(data_unit.answer)
    # print(f"create_training_sample x:{x.shape} y:{y.shape}")
    return x, y, data_unit


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
    return img


def create_unit_dataset(data_unit: DataUnit):
    x = generate_input(data_unit)
    x = x.astype("float32")
    return IMAGE_GEN.standardize(x.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3))


def load_dataset(filename):
    with open(filename, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        train_dataset = pickle.load(f)
    return train_dataset
