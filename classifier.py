import scipy.io
import random
import util
import math
from scipy import optimize

import numpy as np

class NBClassifier(object):

    def __init__(self):
        return

    def load_split_data(self, file_name="./observed/classify_d3_k2_saved1.mat", test_percentage=0.3):
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
                    test[cls][idx].append(feature_vec.pop(pop_index))
            train[cls] = matrix
        return {"train":train, "test":test}

    # Calculate the mean and stdev of each feature in each class
    def train(self, training_dataset):
        feature_summary = {"class_1":[], "class_2":[]}
        for cls, feature_matrix in training_dataset.iteritems():
            for feature in feature_matrix:
                feature_summary[cls].append((util.mean(feature), util.stdev(feature)))
        return feature_summary

    def train_kde(self, training_dataset):
        feature_summary = {"class_1":[], "class_2":[]}
        for cls, feature_matrix in training_dataset.iteritems():
            for feature in feature_matrix:
                feature_summary[cls].append((feature, util.stdev(feature)))
        return feature_summary

    def calc_class_probability(self, feature_vec, training_params):
        class_probs = {}
        for cls, params in training_params.iteritems():
            class_probs[cls] = 1
            for feature_idx, feature_params in enumerate(params):
                mean, stdev = feature_params
                class_probs[cls] *= util.calc_probability(feature_vec[feature_idx], mean, stdev)
        return class_probs

    def calc_class_probability_kde(self, feature_vec, training_params):
        class_probs = {}
        for cls, params in training_params.iteritems():
            class_probs[cls] = 1
            for feature_idx, feature_params in enumerate(params):
                means, stdev = feature_params
                class_probs[cls] *= util.calc_probability_kde(feature_vec[feature_idx], means, stdev)
        return class_probs

    def classify(self, feature_vec, training_params):
        probabilities = self.calc_class_probability(feature_vec, training_params)
        best_class, best_prob = None, -1
        for cls, prob in probabilities.iteritems():
            if best_class is None or prob > best_prob:
                best_class = cls
                best_prob = prob
        return best_class

    def classify_kde(self, feature_vec, training_params):
        probabilities = self.calc_class_probability_kde(feature_vec, training_params)
        best_class, best_prob = None, -1
        for cls, prob in probabilities.iteritems():
            if best_class is None or prob > best_prob:
                best_class = cls
                best_prob = prob
        return best_class

    def test_kde(self, test_dataset, training_params):
        correct, incorrect = 0, 0
        for cls, feature_matrix in test_dataset.iteritems():
            for observation in range(len(feature_matrix[0])):
                features = []
                for feature in range(len(feature_matrix)):
                    features.append(feature_matrix[feature][observation])
                if cls == self.classify_kde(features, training_params):
                    correct += 1
                else:
                    incorrect += 1
        return (correct, incorrect)

    def test(self, test_dataset, training_params):
        correct, incorrect = 0, 0
        for cls, feature_matrix in test_dataset.iteritems():
            for observation in range(len(feature_matrix[0])):
                features = []
                for feature in range(len(feature_matrix)):
                    features.append(feature_matrix[feature][observation])
                if cls == self.classify(features, training_params):
                    correct += 1
                else:
                    incorrect += 1
        return (correct, incorrect)




# data = load_split_data("./observed/classify_d99_k50_saved1.mat")
# training_params = train(data['train'])
# prob = calc_class_probability([0, 0,0], training_params)
# print(classify([0, 0,0], training_params))
# test_results = test(data['test'], training_params)
# print test_results

# data = scipy.io.loadmat("./observed/classify_d3_k2_saved1.mat")


# clf = Classifier()
# clf.clf_with_svm()


class LogisticRegressionClassifier():

    def __init__(self):
        self.data = None;
        self.training_data = None
        self.theta = None;
        return

    def load_split_data(self, file_name="./observed/classify_d3_k2_saved1.mat", test_percentage=0.3):
        data = scipy.io.loadmat(file_name)
        train = {}
        test = {}
        for cls in ['class_1', 'class_2']:
            matrix = data[cls].tolist()
            train[cls] = [[] for f in matrix]
            test[cls] = [[] for f in matrix]
            pop_count = int(len(matrix[0])*test_percentage)
            for i in range(pop_count):
                # pop_index = random.choice(range(len(matrix[0])))
                pop_index = 0
                for idx, feature_vec in enumerate(matrix):
                    test[cls][idx].append(feature_vec.pop(pop_index))
            train[cls] = matrix
        return {"train":train, "test":test}

    def transform_training_data(self, split_data):
        training_data = []
        for cls in ['class_1', 'class_2']:
            for i in range(len(split_data['train'][cls][0])):
                training_data.append((util.column(split_data['train'][cls], i), cls))
        return training_data


    def __cost(self, theta, x, y):
        return (-y*(math.log(self.hypothesis(theta, x))) - (1-y)*math.log(1+1e-10-self.hypothesis(theta, x)))

    def hypothesis(self, theta, x):
        return 1/(1+math.exp(-(np.dot(np.transpose(theta), x))))

    def J(self, theta):
        sum = 0
        for entry in self.training_data:
            features, cls = entry
            sum += self.__cost(theta, features, self.class_to_boolean(cls))
        return sum/float(len(self.training_data))

    def class_to_boolean(self, cls):
        if cls == "class_1":
            return 0
        elif cls == "class_2":
            return 1



    def train(self, fname):
        self.data = self.load_split_data(fname)
        self.training_data = self.transform_training_data(self.data)
        self.theta = optimize.fmin(self.J, x0=[1]*len(self.training_data[0][0]))
        return

    def test(self):
        correct, incorrect = 0, 0
        for item in np.transpose(self.data['test']['class_1']):
            if self.hypothesis(self.theta, item) > 0.5:
                incorrect += 1
            else:
                correct += 1
        for item in np.transpose(self.data['test']['class_2']):
            if self.hypothesis(self.theta, item) < 0.5:
                incorrect += 1
            else:
                correct += 1
        test_results = (correct, incorrect)
        print "Correct: {correct}, Incorrect: {incorrect}, Accuracy: {accuracy:0.2f}".format(
            correct=test_results[0], incorrect=test_results[1], accuracy=test_results[0]/float(sum(test_results)))
        return

lrc = LogisticRegressionClassifier()
# data = lrc.load_split_data()
# lrc.data = lrc.transform_training_data(data)
# xopt = optimize.fmin(lrc.J, x0=[1]*len(lrc.data[0][0]))

# lrc.train("./observed/classify_d99_k50_saved2.mat")
# lrc.test()


class KNearestNeighbourClassifier():
    def __init__(self, data):
        self.data = data
        self.training_data = None
        return

    def train(self, data):
        training_data = []
        for cls in ["class_1", "class_2"]:
            for feature in np.transpose(data['train'][cls]):
                training_data.append((feature, cls))
        return training_data

    def get_nearest_neighbhours(self, training_data, test_instance, k_neighbours=None):
        items_with_distances = []
        for item in training_data:
            dist = util.euclidean_distance(item[0], test_instance)
            items_with_distances.append({"item":item, "distance":dist})
        items_with_distances = sorted(items_with_distances, key=lambda k:k["distance"])
        if k_neighbours != None:
            return items_with_distances[:k_neighbours]
        else:
            return items_with_distances

    def compute_class(self, test_instance, training_data, k_neighbours=51):
        class1, class2 = 0,0
        neighbours=self.get_nearest_neighbhours(training_data, test_instance, k_neighbours)
        for n in neighbours:
            if n["item"][1] == "class_1":
                class1 += 1
            elif n["item"][1] == "class_2":
                class2 += 1
        return "class_1" if class1 > class2 else "class_2"

    def test(self, training_data):

        incorrect, correct = 0, 0
        for item in np.transpose(self.data['test']['class_1']):
            if self.compute_class(item, training_data) == "class_1":
                correct += 1
            else:
                incorrect += 1
        for item in np.transpose(self.data['test']['class_2']):
            if self.compute_class(item, training_data) == "class_2":
                correct += 1
            else:
                incorrect += 1
        test_results = (correct, incorrect)
        # print "Correct: {correct}, Incorrect: {incorrect}, Accuracy: {accuracy:0.2f}".format(
        #     correct=test_results[0], incorrect=test_results[1], accuracy=test_results[0]/float(sum(test_results)))
        return test_results



# data = util.load_split_data()
# knn_classifier = KNNClassifier(data)
# training_data = knn_classifier.train(data)
# knn_classifier.test(training_data)


