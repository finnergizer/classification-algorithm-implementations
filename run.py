from classifier import NBClassifier, KNearestNeighbourClassifier, KernelDensityEstimateNaiveBayesClassifier, LogisticRegressionClassifier
import os
import util

def run_nb():
    report = util.Report("naive-bayes-classifier-test")
    for name in os.listdir("./observed"):
        print "Training and testing with dataset {0}...".format(name)
        data = util.load_split_data("./observed/{0}".format(name))
        classifier = NBClassifier()
        training_params = classifier.train(data["train"])
        test_results = classifier.test(data["test"], training_params)
        # util.print_report(data, test_results)
        report.add_row(name,data,test_results)
    report.write_to_results_csv()

def run_kde():
    report = util.Report("kernel-density-estimate-naive-bayes-classifier-test")
    for name in os.listdir("./observed"):
        print "Training and testing with dataset {0}...".format(name)
        classifier = KernelDensityEstimateNaiveBayesClassifier()
        data = util.load_split_data("./observed/{0}".format(name))
        training_params = classifier.train(data["train"])
        test_results = classifier.test(data["test"], training_params)
        # util.print_report(data, test_results)
        report.add_row(name,data,test_results)
    report.write_to_results_csv()

def run_lr():
    report = util.Report("logistic-regression-classifier-test")
    for name in os.listdir("./observed"):
        print "Training and testing with dataset {0}...".format(name)
        classifier = LogisticRegressionClassifier()
        data = util.load_split_data("./observed/{0}".format(name))
        training_params = classifier.train(data["train"])
        test_results = classifier.test(data["test"])
        # util.print_report(data, test_results)
        report.add_row(name,data,test_results)
    report.write_to_results_csv()

def run_knn():
    report = util.Report("k-nearest-neighbour-classifier-test")
    for name in os.listdir("./observed"):
        print "Training and testing with dataset {0}...".format(name)
        data = util.load_split_data("./observed/{0}".format(name))
        knn_classifier = KNearestNeighbourClassifier()
        training_data = knn_classifier.train(data)
        test_results = knn_classifier.test(data['test'], training_data)
        # util.print_report(data, test_results)
        report.add_row(name,data,test_results)
    report.write_to_results_csv()

run_nb()
run_kde()
run_lr()
run_knn()