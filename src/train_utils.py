import string
import random
import sys
import pickle
import pathlib
import time

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa

# sys.path.append(pathlib.Path(__file__).parent)
from dataset import *

seq = iaa.Sequential([
    iaa.OneOf([
        iaa.Fliplr(0.5),  # horizontal flips
        # iaa.Flipud(0.5),  # vertically flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.5))
                      ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=True),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-60, 60),
            shear=(-8, 8),
        )
    ])], random_order=True)


def create_callbacks(dataset: Dataset, name_weights, patience_lr=10, patience_es=150):
    # mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    mcp_save = AllModelCheckpoint(name_weights)
    # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_lr, verbose=1, min_delta=1e-4, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience_es, verbose=1, mode='auto')
    # return [early_stopping, mcp_save, reduce_lr_loss]
    # return [f1metrics, early_stopping, mcp_save]
    # return [early_stopping, mcp_save, dataset]
    return [mcp_save, dataset]


def load_dataset(filename):
    with open(filename, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        train_dataset = pickle.load(f)
    return train_dataset


def next_triplet_dataset(dataset: Dataset, batch_size: int, datatype: DataType):
    # for klass, data_list in dataset.class_map.items():
    #     print(
    #         f"class:{klass} datatype:{datatype} train_class_map:{len(dataset.train_class_map[klass])} validate_class_map:{len(dataset.validate_class_map[klass])}")
    while True:
        anchor_input = []
        positive_input = []
        negative_input = []

        # setup batch dataset
        for i in range(batch_size):
            # make anchor
            x, y, data_unit, index = create_xy(dataset, datatype)
            anchor_input.append(x)
            # make positive
            if datatype == DataType.train:
                positive_list = dataset.train_class_map[data_unit.answer]
            elif datatype == DataType.validate:
                while data_unit.answer not in dataset.validate_class_map:
                    print(f"failed to find label in validate dataset. label={data_unit.answer} map:{list(dataset.validate_class_map.keys())}")
                    time.sleep(3)
                    x, y, data_unit, index = create_xy(dataset, datatype)
                positive_list = dataset.validate_class_map[data_unit.answer]
            else:
                raise RuntimeError(f"invalid type:{str(datatype)}")
            positive = np.random.choice(positive_list)
            # Make sure it's not comparing to itself
            if len(positive_list) > 1:
                while data_unit.uuid == positive.uuid:
                    positive = np.random.choice(positive_list)
            x_positive = create_unit_dataset(dataset, positive)
            # y_positive = dataset.classes.tolist().index(positive.answer)
            positive_input.append(x_positive)

            # make negative
            candidate_indices = np.argsort(dataset.score[index])
            # print(f"candidate_indices:{candidate_indices} sorted_score:{np.sort(dataset.score[index])}")
            for candidate_index in candidate_indices:
                if datatype == DataType.train and candidate_index not in dataset.train_index_list:
                    continue
                elif datatype == DataType.validate and candidate_index not in dataset.validate_index_list:
                    continue
                negative = dataset.data_list[candidate_index]
                if negative.answer == data_unit.answer:
                    continue
                else:
                    # set picked up data selected last next time during one epoch.
                    dataset.score[:, candidate_index] = float('inf')
                    break
            else:
                raise RuntimeError(f"no candidate could be found.")
            x_negative = create_unit_dataset(dataset, negative)
            negative_input.append(x_negative)
            # y_negative = dataset.classes.tolist().index(negative.answer)

        yield [np.array(anchor_input).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DIM),
               np.array(positive_input).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DIM),
               np.array(negative_input).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DIM)], np.ones(batch_size)


def next_martine_dataset(dataset: Dataset, batch_size: int, datatype: DataType):
    # for klass, data_list in dataset.class_map.items():
    #     print(
    #         f"class:{klass} datatype:{datatype} train_class_map:{len(dataset.train_class_map[klass])} validate_class_map:{len(dataset.validate_class_map[klass])}")
    while True:
        compared_input = []
        compare_to_input = []
        y = []
        # setup batch dataset
        for i in range(batch_size // 2):
            # make anchor
            x, _, data_unit, index = create_xy(dataset, datatype)
            compared_input.append(x)
            # make positive
            if datatype == DataType.train:
                positive_list = dataset.train_class_map[data_unit.answer]
            elif datatype == DataType.validate:
                while data_unit.answer not in dataset.validate_class_map:
                    print(f"failed to find label in validate dataset. label={data_unit.answer} map:{list(dataset.validate_class_map.keys())}")
                    time.sleep(3)
                    x, _, data_unit, index = create_xy(dataset, datatype)
                positive_list = dataset.validate_class_map[data_unit.answer]
            else:
                raise RuntimeError(f"invalid type:{str(datatype)}")
            positive = np.random.choice(positive_list)
            # Make sure it's not comparing to itself
            if len(positive_list) > 1:
                while data_unit.uuid == positive.uuid:
                    positive = np.random.choice(positive_list)
            x_positive = create_unit_dataset(dataset, positive)
            # y_positive = dataset.classes.tolist().index(positive.answer)
            compare_to_input.append(x_positive)
            y.append(1.0)

            # make negative
            compared_input.append(x)
            candidate_indices = np.argsort(dataset.score[index])
            # print(f"candidate_indices:{candidate_indices} sorted_score:{np.sort(dataset.score[index])}")
            for candidate_index in candidate_indices:
                if datatype == DataType.train and candidate_index not in dataset.train_index_list:
                    continue
                elif datatype == DataType.validate and candidate_index not in dataset.validate_index_list:
                    continue
                negative = dataset.data_list[candidate_index]
                if negative.answer == data_unit.answer:
                    continue
                else:
                    # set picked up data selected last next time during one epoch.
                    dataset.score[:, candidate_index] = float('inf')
                    break
            else:
                raise RuntimeError(f"no candidate could be found.")
            x_negative = create_unit_dataset(dataset, negative)
            compare_to_input.append(x_negative)
            y.append(0)
            # y_negative = dataset.classes.tolist().index(negative.answer)

        yield [np.array(compared_input).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DIM),
               np.array(compare_to_input).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DIM)], y


def next_simple_dataset(dataset: Dataset, batch_size: int, datatype: DataType):
    """ Obtain a batch of training data
    """
    while True:
        x_batch = []
        y_batch = []
        for i in range(batch_size):
            x, y, data_unit, index = create_xy(dataset, datatype)
            x_batch.append(x)
            y_batch.append(y)
        x_batch, y_batch = np.array(x_batch), np.array(y_batch)
        # print(f"x:{x.shape}")
        # image_gen.fit(x, augment=True)
        # print(f"next_dataset x_list:{np.array(x).shape} y_data:{np.array(y).shape}")
        if datatype != DataType.test:
            x_batch = seq.augment_images(x_batch).astype("float32")
            yield bounding_box.normalize(x_batch), y_batch
            # yield IMAGE_GEN.standardize(x_batch), y_batch
            # x_batch, y_batch = next(IMAGE_GEN.flow(x_batch, y_batch, batch_size=batch_size))
            # print(f"next_dataset x_list:{x_batch} y_data:{y_batch}")
            # yield x_batch, y_batch
            # do_augment = np.random.randint(BATCH_SIZE)
            # if do_augment:
            #     yield seq.augment_images(x), y
            # else:
            #     yield x, y
        else:
            yield x_batch, y_batch


def train_model(model: Model, dataset: Dataset, model_filename: str,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,):
    callbacks = create_callbacks(dataset, model_filename)
    dataset.model = model
    answers = [data_unit.answer for data_unit in dataset.data_list]
    new_whale_num = answers.count(NEW_LABEL)
    sample_num = len(answers) - new_whale_num
    # sample_num = len(answers)
    train_num = (sample_num * TRAIN_RATIO)
    validate_num = (sample_num * VALIDATE_RATIO)
    steps_per_epoch = train_num // batch_size
    # steps_per_epoch = 50
    validation_steps = validate_num // batch_size
    print(f"new_whale_num:{new_whale_num} sample_num:{sample_num} train_num:{train_num} validate_num:{validate_num}")

    model.fit_generator(generator=next_martine_dataset(dataset, batch_size, DataType.train),
                        epochs=epochs,
                        validation_data=next_martine_dataset(dataset, batch_size, DataType.validate),
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        callbacks=callbacks, verbose=1)


class AllModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(AllModelCheckpoint, self).__init__(filepath, monitor, verbose,
                                                 save_best_only, save_weights_only,
                                                 mode, period)

    def on_epoch_end(self, epoch, logs=None):
        super(AllModelCheckpoint, self).on_epoch_end(epoch, logs)
        # logs = logs or {}
        # current = logs.get(self.monitor)
        # if self.monitor_op(current, self.best):
        self.model.submodel.save(f"{self.filepath}.submodel")
