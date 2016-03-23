import scipy.io
import random
import util
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
import math
from scipy import optimize

import numpy as np

class Classifier(object):

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

    def clf_with_svm(self):
        # clf = svm.SVC()
        # clf = BernoulliNB()
        # clf = DecisionTreeClassifier()
        clf = linear_model.LogisticRegression()
        data = self.load_split_data("./observed/classify_d99_k50_saved2.mat")
        X = []
        y = []
        for cls in ['class_1', 'class_2']:
            for i in range(len(data['train'][cls][0])):
                sample = []
                for j in range(len(data['train'][cls])):
                    sample.append(data['train'][cls][j][i])
                sample = np.array(sample)
                X.append(sample)
                y.append(cls)
        X=np.array(X)
        y=np.array(y)
        print "Fitting..."
        clf.fit(X, y)
        correct, incorrect = 0, 0
        for cls in ['class_1', 'class_2']:
            for i in range(len(data['test'][cls][0])):
                sample = []
                for j in range(len(data['test'][cls])):
                    sample.append(data['test'][cls][j][i])
                sample = [sample]
                prediction = clf.predict(sample)
                if prediction == cls:
                    correct += 1
                else:
                    incorrect +=1
        test_results = (correct, incorrect)
        print "Correct: {correct}, Incorrect: {incorrect}, Accuracy: {accuracy:0.2f}".format(
            correct=test_results[0], incorrect=test_results[1], accuracy=test_results[0]/float(sum(test_results)))



# data = load_split_data("./observed/classify_d99_k50_saved1.mat")
# training_params = train(data['train'])
# prob = calc_class_probability([0, 0,0], training_params)
# print(classify([0, 0,0], training_params))
# test_results = test(data['test'], training_params)
# print test_results

# data = scipy.io.loadmat("./observed/classify_d3_k2_saved1.mat")


# clf = Classifier()
# clf.clf_with_svm()


class LRClassifier():

    def __init__(self):
        self.data = None;
        self.training_data = None
        self.theta = None;
        return

    def load_split_data(self, file_name="./observed/classify_d3_k2_saved1.mat", test_percentage=0.1):
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
                # pop_index = 0
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
        return (-y*(math.log(self.hypothesis(theta, x))) - (1-y)*math.log(1-self.hypothesis(theta, x)))

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

lrc = LRClassifier()
# data = lrc.load_split_data()
# lrc.data = lrc.transform_training_data(data)
# xopt = optimize.fmin(lrc.J, x0=[1]*len(lrc.data[0][0]))

lrc.train("./observed/classify_d5_k3_saved2.mat")
lrc.test()

