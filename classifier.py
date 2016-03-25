import scipy.io
import random
import util
import math
from scipy import optimize

import numpy as np


class NBClassifier(object):
    def __init__(self):
        self.training_params = None

    # Calculate the mean and stdev of each feature in each class
    def train(self, training_dataset):
        feature_summary = {"class_1": [], "class_2": []}
        for cls, feature_matrix in training_dataset.iteritems():
            for feature in feature_matrix:
                feature_summary[cls].append((util.mean(feature), util.stdev(feature)))
        self.training_params = feature_summary
        return feature_summary

    def calc_class_probability(self, feature_vec, training_params=None):
        if training_params is None:
            if self.training_params is None:
                raise ValueError("NB Model has not been fit with any training params, nor provided any to classify")
            else:
                training_params = self.training_params
        if len(feature_vec) != len(training_params['class_1']):
            raise ValueError(
                "Training paramaters for every feature are required (i.e. They must be of the same length)")

        class_probs = {}
        for cls, params in training_params.iteritems():
            class_probs[cls] = 1
            for feature_idx, feature_params in enumerate(params):
                mean, stdev = feature_params
                class_probs[cls] *= util.calc_probability_gaussian(feature_vec[feature_idx], mean, stdev)
        return class_probs

    def classify(self, feature_vec, training_params=None):
        probabilities = self.calc_class_probability(feature_vec, training_params)
        best_class, best_prob = None, -1
        for cls, prob in probabilities.iteritems():
            if best_class is None or prob > best_prob:
                best_class = cls
                best_prob = prob
        return best_class

    def test(self, test_dataset, training_params=None):
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


class KernelDensityEstimateNaiveBayesClassifier():
    def __init__(self):
        self.training_params = None

    def train(self, training_dataset):
        feature_summary = {"class_1": [], "class_2": []}
        for cls, feature_matrix in training_dataset.iteritems():
            for feature in feature_matrix:
                feature_summary[cls].append((feature, util.stdev(feature)))
        return feature_summary

    def calc_class_probability(self, feature_vec, training_params):
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


class LogisticRegressionClassifier():
    def __init__(self):
        self.training_data = None
        self.theta = None;

    def load_split_data(self, file_name="./observed/classify_d3_k2_saved1.mat", test_percentage=0.3):
        data = scipy.io.loadmat(file_name)
        train = {}
        test = {}
        for cls in ['class_1', 'class_2']:
            matrix = data[cls].tolist()
            train[cls] = [[] for f in matrix]
            test[cls] = [[] for f in matrix]
            pop_count = int(len(matrix[0]) * test_percentage)
            for i in range(pop_count):
                # pop_index = random.choice(range(len(matrix[0])))
                pop_index = 0
                for idx, feature_vec in enumerate(matrix):
                    test[cls][idx].append(feature_vec.pop(pop_index))
            train[cls] = matrix
        return {"train": train, "test": test}

    def transform_training_data(self, split_data):
        training_data = []
        for cls in ['class_1', 'class_2']:
            for i in range(len(split_data['train'][cls][0])):
                training_data.append((util.column(split_data['train'][cls], i), cls))
        return training_data

    def __class_to_boolean(self, cls):
        if cls == "class_1":
            return 0
        elif cls == "class_2":
            return 1

    def __cost(self, theta, x, y):
        """

        :param theta:
        :param x:
        :param y:
        :return:
        """
        return (-y * (math.log(self.hypothesis(theta, x))) - (1 - y) * math.log(1 + 1e-10 - self.hypothesis(theta, x)))

    def hypothesis(self, x, theta=None):
        return 1 / (1 + math.exp(-(np.dot(np.transpose(theta), x))))

    def J(self, theta):
        """

        :param theta:
        :return:
        """
        sum = 0
        for entry in self.training_data:
            features, cls = entry
            sum += self.__cost(theta, features, self.__class_to_boolean(cls))
        return sum / float(len(self.training_data))

    def train(self, fname):
        self.data = self.load_split_data(fname)
        self.training_data = self.transform_training_data(self.data)
        self.theta = optimize.fmin(self.J, x0=[1] * len(self.training_data[0][0]))
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
        # print "Correct: {correct}, Incorrect: {incorrect}, Accuracy: {accuracy:0.2f}".format(
        #     correct=test_results[0], incorrect=test_results[1], accuracy=test_results[0] / float(sum(test_results)))
        return test_results


# lrc = LogisticRegressionClassifier()


# data = lrc.load_split_data()
# lrc.data = lrc.transform_training_data(data)
# xopt = optimize.fmin(lrc.J, x0=[1]*len(lrc.data[0][0]))

# lrc.train("./observed/classify_d99_k50_saved2.mat")
# lrc.test()


class KNearestNeighbourClassifier():
    def __init__(self):
        self.training_data = None

    def train(self, data):
        training_data = []
        for cls in ["class_1", "class_2"]:
            for feature in np.transpose(data['train'][cls]):
                training_data.append((feature, cls))
        self.training_data = training_data
        return training_data

    def get_nearest_neighbhours(self, test_instance, training_data=None, k_neighbours=None):
        if k_neighbours is None:
            k_neighbours = 51

        if training_data is None:
            if self.training_data is None:
                raise ValueError("KNN Model has not been given with any training data to compute neighbours")
            else:
                training_data = self.training_data

        items_with_distances = []
        for item in training_data:
            dist = util.euclidean_distance(item[0], test_instance)
            items_with_distances.append({"item": item, "distance": dist})
        items_with_distances = sorted(items_with_distances, key=lambda k: k["distance"])
        if k_neighbours != None:
            return items_with_distances[:k_neighbours]
        else:
            return items_with_distances

    def compute_class(self, test_instance, training_data=None, k_neighbours=None):
        class1, class2 = 0, 0
        neighbours = self.get_nearest_neighbhours(test_instance, training_data, k_neighbours)
        for n in neighbours:
            if n["item"][1] == "class_1":
                class1 += 1
            elif n["item"][1] == "class_2":
                class2 += 1
        return "class_1" if class1 > class2 else "class_2"

    def test(self, test_data, training_data=None, k_neighbours=None):
        incorrect, correct = 0, 0
        for cls in test_data:
            print cls
            for item in np.transpose(test_data[cls]):
                if self.compute_class(item, training_data) == cls:
                    correct += 1
                else:
                    incorrect += 1
        test_results = (correct, incorrect)
        return test_results

# data = util.load_split_data()
# knn_classifier = KNNClassifier(data)
# training_data = knn_classifier.train(data)
# knn_classifier.test(training_data)
