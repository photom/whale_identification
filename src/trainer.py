#!/usr/bin/env python

import mpl_toolkits # import before pathlib
import sys
import pathlib
import gc

from tensorflow import set_random_seed

# sys.path.append(pathlib.Path(__file__).parent)
from train_utils import *
from model import *
from dataset import *

np.random.seed(RANDOM_NUM)
set_random_seed(RANDOM_NUM)

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
BASE_MODEL = 'triplet_loss'
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
else:
    raise Exception("unimplemented model")


def main():
    # Load audio segments using pydub
    dataset = load_raw_data()

    print(f"class_num:{dataset.class_num}")
    weight_param_path = f"model/{BASE_MODEL}.weights.best.hdf5"
    model = create_model(dataset=dataset, input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_DIM))
    model = build_model(model, weight_param_path)
    for i in range(0, 1):
        print(f"num:{i}. start train")
        train_model(model, dataset, weight_param_path)
    model.save(weight_param_path)


if __name__ == "__main__":
    main()
