import scipy.io
import random
import math
import os
import util

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



    def calc_class_probability(self, feature_vec, training_params):
        class_probs = {}
        for cls, params in training_params.iteritems():
            class_probs[cls] = 1
            for feature_idx, feature_params in enumerate(params):
                mean, stdev = feature_params
                class_probs[cls] *= util.calc_probability(feature_vec[feature_idx], mean, stdev)
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

def run():
    for name in os.listdir("./observed"):
        print "Training and testing with dataset {0}...".format(name)
        classifier = Classifier()
        data = classifier.load_split_data("./observed/{0}".format(name))
        training_params = classifier.train(data["train"])
        test_results = classifier.test(data["test"], training_params)
        print "Correct: {correct}, Incorrect: {incorrect}, Accuracy: {accuracy:0.2f}".format(
            correct=test_results[0], incorrect=test_results[1], accuracy=test_results[0]/float(sum(test_results)))


run()



# data = load_split_data("./observed/classify_d99_k50_saved1.mat")
# training_params = train(data['train'])
# prob = calc_class_probability([0, 0,0], training_params)
# print(classify([0, 0,0], training_params))
# test_results = test(data['test'], training_params)
# print test_results




