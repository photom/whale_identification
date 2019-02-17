import mpl_toolkits  # import before pathlib
import sys
from pathlib import Path

from tensorflow import set_random_seed
from sklearn.neighbors import NearestNeighbors

sys.path.append(Path(__file__).parent)
from model import *
from dataset import *
from metrics import *

np.random.seed(RANDOM_NUM)
set_random_seed(RANDOM_NUM)

OUTPUT_FILE = 'test_dataset_prediction.txt'
# BASE_MODEL = 'resnet50'
# BASE_MODEL = 'vgg11'
# BASE_MODEL = 'incepstionresnetv2'
# BASE_MODEL = 'adams'
# BASE_MODEL = 'michel'
# BASE_MODEL = 'giim'
# BASE_MODEL = 'siamese_resnet'
BASE_MODEL = 'triplet_loss'
if BASE_MODEL == 'resnet50':
    create_model = create_model_resnet50_plain
elif BASE_MODEL == 'incepstionresnetv2':
    create_model = create_model_inceptionresnetv2_plain
elif BASE_MODEL == 'giim':
    create_model = create_model_giim
elif BASE_MODEL == 'siamese_resnet':
    create_model = create_model_siamese_resnet
elif BASE_MODEL == 'triplet_loss':
    create_model = create_model_triplet_loss
else:
    raise Exception("unimplemented model")


def predict(data_unit: DataUnit, dataset: Dataset, test_model: Model):
    x = create_unit_dataset(dataset, data_unit)
    prediction_results = []
    for label, data_list in dataset.class_map.items():
        if label == NEW_LABEL:
            continue
        x_compare_to = []
        x_compared = []
        if len(data_list) > 10:
            compare_to_data_list = np.random.choice(data_list, 10)
        else:
            compare_to_data_list = data_list
        for compare_to_data_unit in compare_to_data_list:
            x_compared.append(x)
            x_compare_to.append(create_unit_dataset(dataset, compare_to_data_unit))
        x_input = [np.array(x_compared).reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 3)),
                   np.array(x_compare_to).reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 3))]
        y_pred = test_model.predict(x_input, batch_size=len(compare_to_data_list))
        y_pred = np.sort(y_pred[0])
        if len(y_pred) > 2:
            y_pred = y_pred[1:-1]
        prediction_results.append((label, y_pred.mean()))

    print(f"{prediction_results}")
    sorted(prediction_results, key=lambda tup: (-tup[1], tup[0]))
    results = [result[0] for result in prediction_results[:5]]
    if prediction_results[4][1] < 0.5:
        results[4] = NEW_LABEL
    return results


def main():
    dataset = load_raw_data()
    test_dataset = load_test_data()
    weight_param_path = f"model/{BASE_MODEL}.weights.best.hdf5"
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
            sample_result.append((NEW_LABEL, 0.175))
        sample_result.sort(key=lambda x: x[1])
        print(f"sample:{sample_result}")
        sample_result = sample_result[:5]
        pred_str = " ".join([x[0] for x in sample_result])
        df = df.append(pd.DataFrame([[data_unit.filename, pred_str]], columns=['Image', 'Id']),
                       ignore_index=True)
    df.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
