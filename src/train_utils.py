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
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-60, 60),
            shear=(-8, 8)
        )
    ])], random_order=True)


def create_callbacks(dataset: Dataset, name_weights, patience_lr=10, patience_es=150):
    mcp_save = ModelCheckpoint(name_weights,
                               save_best_only=True, monitor='val_loss', mode='min')
    # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_lr, verbose=1, min_delta=1e-4, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience_es, verbose=1, mode='auto')
    # return [early_stopping, mcp_save, reduce_lr_loss]
    # return [f1metrics, early_stopping, mcp_save]
    return [early_stopping, mcp_save, dataset]
    # return [mcp_save, dataset]


def load_dataset(filename):
    with open(filename, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        train_dataset = pickle.load(f)
    return train_dataset


def get_batch(batch_size, s="train"):
    """
    Create batch of n pairs, half same class, half different class
    """
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h = X.shape
    # randomly sample several classes to use in the batch
    categories = rng.choice(n_classes, size=(batch_size,), replace=False)

    # initialize 2 empty arrays for the input image batch
    pairs = [np.zeros((batch_size, h, w, 1)) for i in range(2)]

    # initialize vector for the targets
    targets = np.zeros((batch_size,))

    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size // 2:] = 1
    for i in range(batch_size):
        category = categories[i]
        idx_1 = rng.randint(0, n_examples)
        pairs[0][i, :, :, :] = X[category, idx_1].reshape(w, h, 1)
        idx_2 = rng.randint(0, n_examples)

        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category
        else:
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + rng.randint(1, n_classes)) % n_classes

        pairs[1][i, :, :, :] = X[category_2, idx_2].reshape(w, h, 1)

    return pairs, targets


def next_dataset(dataset: Dataset, batch_size: int, datatype: DataType):
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
            x, y, data_unit = create_xy(dataset, datatype)
            anchor_input.append(x)
            # make positive
            if datatype == DataType.train:
                positive_list = dataset.train_class_map[data_unit.answer]
            elif datatype == DataType.validate:
                while data_unit.answer not in dataset.validate_class_map:
                    print(f"failed to find label in validate dataset. label={data_unit.answer} map:{list(dataset.validate_class_map.keys())}")
                    time.sleep(3)
                    x, y, data_unit = create_xy(dataset, datatype)
                positive_list = dataset.validate_class_map[data_unit.answer]
            else:
                raise RuntimeError(f"invalid type:{str(datatype)}")
            positive = np.random.choice(positive_list)
            # Make sure it's not comparing to itself
            if len(positive_list) > 1:
                while data_unit.uuid == positive.uuid:
                    positive = np.random.choice(positive_list)
            x_positive = create_unit_dataset(positive)
            # y_positive = dataset.classes.tolist().index(positive.answer)
            positive_input.append(x_positive)

            # make negative
            negative = None
            negative_class = data_unit.answer
            # Make sure it's not comparing to itself
            while data_unit.answer == negative_class:
                if datatype == DataType.train:
                    compare_to_index = np.random.choice(dataset.train_index_list)
                elif datatype == DataType.validate:
                    compare_to_index = np.random.choice(dataset.validate_index_list)
                else:
                    raise RuntimeError(f"invalid datatype:{str(datatype)}")
                negative = dataset.data_list[compare_to_index]
                negative_class = negative.answer
            x_negative = create_unit_dataset(negative)
            negative_input.append(x_negative)
            # y_negative = dataset.classes.tolist().index(negative.answer)

        yield [np.array(anchor_input).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3),
               np.array(positive_input).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3),
               np.array(negative_input).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)], np.ones(batch_size)


def next_dataset_old(dataset: Dataset, batch_size: int, datatype: DataType):
    """ Obtain a batch of training data
    """
    while True:
        yield get_batch(dataset, batch_size, datatype)
        # x_batch = []
        # y_batch = []
        # for i in range(batch_size):
        #     x, y = create_xy(dataset, datatype)
        #     x_batch.append(x)
        #     y_batch.append(y)
        # x_batch, y_batch = np.array(x_batch), np.array(y_batch)
        # print(f"x:{x.shape}")
        # image_gen.fit(x, augment=True)
        # print(f"next_dataset x_list:{np.array(x).shape} y_data:{np.array(y).shape}")
        # if datatype != DataType.test:
        # x_batch = seq.augment_images(x_batch).astype("float32")
        # yield IMAGE_GEN.standardize(x_batch), y_batch
        # x_batch, y_batch = next(IMAGE_GEN.flow(x_batch, y_batch, batch_size=batch_size))
        # print(f"next_dataset x_list:{x_batch} y_data:{y_batch}")
        # yield x_batch, y_batch
        # do_augment = np.random.randint(BATCH_SIZE)
        # if do_augment:
        #     yield seq.augment_images(x), y
        # else:
        #     yield x, y
        # else:
        # yield x_batch, y_batch


def train_model(model: Model, dataset: Dataset, model_filename: str,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS, ):
    callbacks = create_callbacks(dataset, model_filename)

    answers = [data_unit.answer for data_unit in dataset.data_list]
    new_whale_num = answers.count(NEW_LABEL)
    sample_num = len(answers) - new_whale_num
    train_num = (sample_num * TRAIN_RATIO)
    validate_num = (sample_num * VALIDATE_RATIO)
    steps_per_epoch = train_num // batch_size
    validation_steps = validate_num // batch_size
    print(f"new_whale_num:{new_whale_num} sample_num:{sample_num} train_num:{train_num} validate_num:{validate_num}")

    model.fit_generator(generator=next_dataset(dataset, batch_size, DataType.train),
                        epochs=epochs,
                        validation_data=next_dataset(dataset, batch_size, DataType.validate),
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        callbacks=callbacks, verbose=1)
    # model.fit_generator(IMAGE_GEN.flow(dataset.x_list, dataset.y_list, batch_size=batch_size),
    #                     steps_per_epoch=dataset.x_list.shape[0] // epochs,
    #                     epochs=epochs,
    #                     callbacks=callbacks,
    #                     verbose=1)


def predict(data_unit: DataUnit, model: Model):
    x = create_unit_dataset(data_unit)
    # print(f"x:{x.shape}")
    # predict
    result = model.predict(np.array([x]))
    predicted = np.round(result)
    return predicted


def eval_model(model: Model, dataset: TestDataset):
    sample_num = len(dataset.data_list)
    test_num = int(sample_num * VALIDATE_RATIO)
    test_num = test_num

    metrics = MacroF1Score()
    score = None
    result = None
    num_step = test_num // BATCH_SIZE
    counter = 0
    for x, y in next_dataset(dataset, BATCH_SIZE, DataType.test):
        if counter >= num_step:
            break
        else:
            counter += 1
        result = model.predict(x)
        predicted = np.round(result)
        score = metrics(y, predicted)
        result = K.eval(score)

    print(f"counter={counter} macro_f1_score={result}")
    # tf.keras.backend.clear_session()
