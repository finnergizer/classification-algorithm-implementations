from classifier import NBClassifier, KNearestNeighbourClassifier, KernelDensityEstimateNaiveBayesClassifier, \
    LogisticRegressionClassifier
import os
import util
import sys


def run_nb():
    report = util.Report("naive-bayes-classifier-test")
    for name in os.listdir("./observed"):
        print "Training and testing Gaussian Naive Bayes Classifier with dataset {0}...".format(name)
        data = util.load_split_data("./observed/{0}".format(name))
        classifier = NBClassifier()
        training_params = classifier.train(data["train"])
        test_results = classifier.test(data["test"], training_params)
        # util.print_report(data, test_results)
        report.add_row(name, data, test_results)
    report.write_to_results_csv()


def run_kde():
    report = util.Report("kernel-density-estimate-naive-bayes-classifier-test")
    for name in os.listdir("./observed"):
        print "Training and testing Kernel Density Estimate Naive Bayes Classifier with dataset {0}...".format(name)
        classifier = KernelDensityEstimateNaiveBayesClassifier()
        data = util.load_split_data("./observed/{0}".format(name))
        training_params = classifier.train(data["train"])
        test_results = classifier.test(data["test"], training_params)
        # util.print_report(data, test_results)
        report.add_row(name, data, test_results)
    report.write_to_results_csv()


def run_lr():
    report = util.Report("logistic-regression-classifier-test")
    for name in os.listdir("./observed"):
        print "Training and testing Logistic Regression Classifier with dataset {0}...".format(name)
        classifier = LogisticRegressionClassifier()
        data = util.load_split_data("./observed/{0}".format(name))
        training_params = classifier.train(data["train"])
        test_results = classifier.test(data["test"])
        # util.print_report(data, test_results)
        report.add_row(name, data, test_results)
    report.write_to_results_csv()


def run_knn():
    report = util.Report("k-nearest-neighbour-classifier-test")
    for name in os.listdir("./observed"):
        print "Training and testing K Nearest Neighbours Classifier with dataset {0}...".format(name)
        data = util.load_split_data("./observed/{0}".format(name))
        knn_classifier = KNearestNeighbourClassifier()
        training_data = knn_classifier.train(data)
        test_results = knn_classifier.test(data['test'], training_data)
        # util.print_report(data, test_results)
        report.add_row(name, data, test_results)
    report.write_to_results_csv()


try:
    if sys.argv[1] == "--gnb":
        run_nb()
    elif sys.argv[1] == "--knn":
        run_knn()
    elif sys.argv[1] == "--lr":
        run_lr()
    elif sys.argv[1] == "--kde":
        run_knn()
    elif sys.argv[1] == "--all":
        run_knn()
        run_nb()
        run_kde()
        run_lr()
    else:
        raise Exception()
except Exception as e:
    print "Please run using --gnb, --knn, --lr, or --kde to run the Gaussian Naive Bayes, K-Nearest-Neighbour," \
          "Logistic Regression, or Kernel Density Estimate Naive Bayes, respectively. Use --all to run " \
          "training and testing for all classifiers and print results."
