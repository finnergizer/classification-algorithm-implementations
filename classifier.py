import scipy.io
import random
class Classifier(object):

    def __init__(self):
        return


# mat = scipy.io.loadmat("./observed/classify_d99_k50_saved1.mat")
mat = scipy.io.loadmat("./observed/classify_d3_k2_saved1.mat")


def load_split_data(file_name="./observed/classify_d3_k2_saved1.mat", test_percentage=0.3):
    data = scipy.io.loadmat(file_name)
    train = {}
    test = {}
    for cls in ['class_1', 'class_2']:
        matrix = data[cls].tolist()
        train[cls] = [[] for f in matrix]
        test[cls] = [[] for f in matrix]
        pop_count = int(len(matrix[0])*test_percentage)
        for i in range(pop_count):
            pop_index = random.choice(range(len(matrix[0])))
            for idx, feature_vec in enumerate(matrix):
                test[cls][idx] = feature_vec.pop(pop_index)
        train[cls] = matrix
    return {"train":train, "test":test}

data = load_split_data(test_percentage=0.7)
