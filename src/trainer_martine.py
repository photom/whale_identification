#!/usr/bin/env python

import mpl_toolkits # import before pathlib
import sys
import pathlib
import gc
from typing import Optional

from sklearn.neighbors import NearestNeighbors
from tensorflow import set_random_seed

# sys.path.append(pathlib.Path(__file__).parent)
from train_utils import *
from model import *
from dataset import *

np.random.seed(RANDOM_NUM)
set_random_seed(RANDOM_NUM)

OUTPUT_FILE = 'test_dataset_prediction.txt'

# BASE_MODEL = 'vgg19'
# BASE_MODEL = 'incepstionresnetv2'
# BASE_MODEL = 'resnet50'
# BASE_MODEL = 'resnet152'
# BASE_MODEL = 'adams'
# BASE_MODEL = 'michel'
# BASE_MODEL = 'mobilenet'
# BASE_MODEL = 'local'
# BASE_MODEL = 'giim'
# BASE_MODEL = 'siamese'
# BASE_MODEL = 'triplet_loss'
BASE_MODEL = 'martine'
if BASE_MODEL == 'resnet50':
    create_model = create_model_resnet50_plain
elif BASE_MODEL == 'resnet152':
    create_model = create_model_resnet152_plain
elif BASE_MODEL == 'incepstionresnetv2':
    create_model = create_model_inceptionresnetv2_plain
elif BASE_MODEL == 'mobilenet':
    create_model = create_model_mobilenet
elif BASE_MODEL == 'giim':
    create_model = create_model_giim
elif BASE_MODEL == 'siamese':
    create_model = create_model_siamese
elif BASE_MODEL == 'siamese_resnet':
    create_model = create_model_siamese_resnet
elif BASE_MODEL == 'triplet_loss':
    create_model = create_model_triplet_loss
elif BASE_MODEL == 'martine':
    create_model = create_martine_model
else:
    raise Exception("unimplemented model")


def test(dataset: Optional[Dataset], model: Optional[Model]):
    if dataset is None:
        dataset = load_raw_data()
    test_dataset = load_test_data()
    weight_param_path = f"model/{BASE_MODEL}.weights.best.hdf5"
    if model is None:
        model = create_model(dataset=dataset, input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_DIM))
        model = build_model(model, weight_param_path)
    train_preds = []
    train_data_list = []
    for data_unit in dataset.data_list:
        if data_unit.answer == NEW_LABEL:
            continue
        x = create_unit_dataset(dataset, data_unit)
        predicts = model.submodel.predict([x])
        predicts = predicts.tolist()
        train_preds += predicts
        train_data_list.append(data_unit)
    train_preds = np.array(train_preds)
    print(f"train_preds:{train_preds.shape}")
    print(f"train_preds:{train_preds}")

    test_preds = []
    test_data_list = []
    for data_unit in test_dataset.data_list:
        x = create_unit_dataset(dataset, data_unit)
        predicts = model.submodel.predict(x)
        predicts = predicts.tolist()
        test_preds += predicts
        test_data_list.append(data_unit)
    test_preds = np.array(test_preds)
    print(f"test_preds: {test_preds.shape}")
    print(f"test_preds:{test_preds}")

    neigh = NearestNeighbors(n_neighbors=6)
    neigh.fit(train_preds)
    # distances, neighbors = neigh.kneighbors(train_preds)
    # print(distances, neighbors)
    distances_test, neighbors_test = neigh.kneighbors(test_preds)
    distances_test, neighbors_test = distances_test.tolist(), neighbors_test.tolist()

    df = pd.DataFrame([], columns=['Image', 'Id'])
    for data_unit, distance, neighbour_ in zip(test_data_list, distances_test, neighbors_test):
        sample_result = []
        sample_classes = []
        for d, n in zip(distance, neighbour_):
            train_data_unit = train_data_list[n]
            sample_classes.append(train_data_unit.answer)
            sample_result.append((train_data_unit.answer, d))

        if NEW_LABEL not in sample_classes:
            # pbr
            # sample_result.append((NEW_LABEL, 0.0002))
            # alpha
            sample_result.append((NEW_LABEL, 27))
        sample_result.sort(key=lambda x: x[1])
        print(f"sample:{sample_result}")
        sample_result = sample_result[:5]
        pred_str = " ".join([x[0] for x in sample_result])
        df = df.append(pd.DataFrame([[data_unit.filename, pred_str]], columns=['Image', 'Id']),
                       ignore_index=True)
    df.to_csv(OUTPUT_FILE, index=False)


def main():
    dataset = load_raw_data()

    print(f"class_num:{dataset.class_num}")
    weight_param_path = f"model/{BASE_MODEL}.weights.best.hdf5"
    model = create_model(dataset=dataset, input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_DIM))
    model = build_model(model, weight_param_path)
    # model = create_martine_model()
    for i in range(0, 1):
        print(f"num:{i}. start train")
        train_model(model, dataset, weight_param_path)
    model.save(weight_param_path)
    test(dataset, model)


if __name__ == "__main__":
    main()
