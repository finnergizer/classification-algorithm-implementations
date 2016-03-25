from classifier import NBClassifier, KNearestNeighbourClassifier, KernelDensityEstimateNaiveBayesClassifier, LogisticRegressionClassifier
import os
import util

def run_nb():
    for name in os.listdir("./observed"):
        print "Training and testing with dataset {0}...".format(name)
        data = util.load_split_data("./observed/{0}".format(name))
        classifier = NBClassifier()
        training_params = classifier.train(data["train"])
        test_results = classifier.test(data["test"], training_params)
        util.print_report(data, test_results)

def run_kde():
    for name in os.listdir("./observed"):
        print "Training and testing with dataset {0}...".format(name)
        classifier = KernelDensityEstimateNaiveBayesClassifier()
        data = util.load_split_data("./observed/{0}".format(name))
        training_params = classifier.train(data["train"])
        test_results = classifier.test(data["test"], training_params)
        util.print_report(data, test_results)

def run_lr():
    for name in os.listdir("./observed"):
        print "Training and testing with dataset {0}...".format(name)
        classifier = LogisticRegressionClassifier()
        data = util.load_split_data("./observed/{0}".format(name))
        training_params = classifier.train(data["train"])
        test_results = classifier.test_kde(data["test"], training_params)
        util.print_report(data, test_results)

def run_knn():
    for name in os.listdir("./observed"):
        print "Training and testing with dataset {0}...".format(name)
        data = util.load_split_data("./observed/{0}".format(name))
        knn_classifier = KNearestNeighbourClassifier()
        training_data = knn_classifier.train(data)
        test_results = knn_classifier.test(data['test'], training_data)
        util.print_report(data, test_results)
# run()
# run_kde()

run_knn()
# run_nb()