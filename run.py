from classifier import Classifier
import os

def run():
    for name in os.listdir("./observed"):
        print "Training and testing with dataset {0}...".format(name)
        classifier = Classifier()
        data = classifier.load_split_data("./observed/{0}".format(name), 0.25)
        training_params = classifier.train(data["train"])
        test_results = classifier.test(data["test"], training_params)
        print "Correct: {correct}, Incorrect: {incorrect}, Accuracy: {accuracy:0.2f}".format(
            correct=test_results[0], incorrect=test_results[1], accuracy=test_results[0]/float(sum(test_results)))

def run_kde():
    for name in os.listdir("./observed"):
        print "Training and testing with dataset {0}...".format(name)
        classifier = Classifier()
        data = classifier.load_split_data("./observed/{0}".format(name), 0.25)
        training_params = classifier.train_kde(data["train"])
        test_results = classifier.test_kde(data["test"], training_params)
        print "Correct: {correct}, Incorrect: {incorrect}, Accuracy: {accuracy:0.2f}".format(
            correct=test_results[0], incorrect=test_results[1], accuracy=test_results[0]/float(sum(test_results)))

run()
# run_kde()