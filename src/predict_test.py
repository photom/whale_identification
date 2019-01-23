import mpl_toolkits  # import before pathlib
import sys
from pathlib import Path

from tensorflow import set_random_seed

sys.path.append(Path(__file__).parent)
from model import *
from dataset import *
from metrics import *

np.random.seed(19)
set_random_seed(19)

OUTPUT_FILE = 'test_dataset_prediction.txt'
BASE_MODEL = 'resnet50'
# BASE_MODEL = 'vgg11'
# BASE_MODEL = 'incepstionresnetv2'
# BASE_MODEL = 'adams'
# BASE_MODEL = 'michel'
# BASE_MODEL = 'giim'
if BASE_MODEL == 'resnet50':
    create_model = create_model_resnet50_plain
elif BASE_MODEL == 'vgg19':
    create_model = create_model_vgg19_plain
elif BASE_MODEL == 'incepstionresnetv2':
    create_model = create_model_inceptionresnetv2_plain
elif BASE_MODEL == 'giim':
    create_model = create_model_giim
else:
    raise Exception("unimplemented model")


def predict(data_unit: DataUnit, test_model: Model):
    x = create_unit_dataset(data_unit)
    y_pred = test_model.predict(x.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3))
    # result = test_model.predict(np.array(x))
    # y_pred = np.array(y_pred[0])
    print(f"{y_pred}")
    # sorted_indices = np.array(result[0]).argpartition(-5)[-5:]
    # sorted_indices = np.argsort(-result)[:5]
    sorted_indices = np.argsort(y_pred)[0][::-1][:5]
    print(sorted_indices)
    print(np.array(y_pred[0])[sorted_indices])
    return list(sorted_indices)


def main():
    dataset = load_raw_data()
    test_model = create_model(dataset, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), datatype=DataType.test)
    test_dataset = load_test_data()
    weight_param_path = f"model/{BASE_MODEL}.weights.best.hdf5"
    test_model.load_weights(weight_param_path)
    content = "Image,Id\n"
    for i in range(len(test_dataset.data_list)):
        data_unit = test_dataset.data_list[i]
        y_pred = predict(data_unit, test_model)
        decoded = dataset.label_onehot_encoder.inverse_labels(y_pred)
        # decoded = [decode_map[idx] for idx in predicted]
        joined = " ".join(decoded)
        print(f"i:{i} y_pred:{y_pred} {joined} ")
        content += f"{data_unit.filename},{joined}\n"

    with open(OUTPUT_FILE, 'w') as f:
        f.write(content)


if __name__ == "__main__":
    main()
