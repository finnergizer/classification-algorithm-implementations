import math
import scipy
import csv
import random
import scipy.io
try:
    from tabulate import tabulate
except Exception:
    pass

# Basic statistical calculations given python lists of values
def mean(nums):
    return sum(nums)/float(len(nums))

def variance(nums):
    av = mean(nums)
    return sum([pow(x-av, 2) for x in nums])/float(len(nums))

def stdev(nums):
    return math.sqrt(variance(nums))


def calc_probability_gaussian(x, mean, stdev):
    """
    Probability density function of a Gaussian distribution with mean and stdev, evaluated at x
    :param x: the value to evaluate the PDF at
    :param mean: the mean of the Gaussian distribution
    :param stdev: the stdev of the Gaussian distribtuion
    :return: Gaussian distribution evaluated at x
    """
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calc_probability_kde(x, means, stdev):
    """

    :param x:
    :param means:
    :param stdev:
    :return:
    """
    val = 0
    for ui in means:
        exponent = math.exp(-(math.pow(x-ui,2)/(2*math.pow(stdev,2))))
        val += (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
    return val/float(len(means))

def column(matrix, i):
    """

    :param matrix: A 2D list (matrix) with rows being features, columns representing each item
    :param i: Which column in the matrix to return as a list
    :return: The feature vector for observation i in the matrix (i.e. a list containing the values of all features for
    observation i)
    """
    return [row[i] for row in matrix]

def load_split_data(file_name="./observed/classify_d3_k2_saved1.mat", test_percentage=0.2, rand=False):
    """

    :param file_name: The .mat file to load into a dictionary of python lists/matrices with rows as features, columns
    as items/observations
    :param test_percentage: The fraction of data that will be stripped and used as test data (default 0.2 for 4:1 ratio)
    :param random: Randomly select from the observations for each class, or select the first n observations based on the
    test_percentage provided
    :return: dictionary of the form
     {"train":{"class_1":[d*n matrix of d features, n observations to train with for class_1], "class_2": [d*n matrix]
      "test" :{"class_1":[d*n matrix of d features, n observations to test with for class_1], "class_2": [d*n matrix]}
    """
    data = scipy.io.loadmat(file_name)
    train, test = {}, {}
    for cls in ['class_1', 'class_2']:
        matrix = data[cls].tolist()
        train[cls] = [[] for f in matrix]
        test[cls] = [[] for f in matrix]
        # How many column vectors (observations) should we pop from the matrix
        pop_count = int(len(matrix[0])*test_percentage)
        for i in range(pop_count):
            pop_index = random.choice(range(len(matrix[0]))) if rand else 0
            for idx, feature_vec in enumerate(matrix):
                test[cls][idx].append(feature_vec.pop(pop_index))
        train[cls] = matrix
    return {"train":train, "test":test}


def euclidean_distance(x1, x2):
    """
    Calculates the euclidian distance between two vectors x1, x2
    :param x1: first vector to compare
    :param x2: second vector to compare
    :return: a real valued distance between the two vectors
    """
    if len(x1) != len(x2):
        raise ValueError("x1, x2 must be of the same length")
    distance = 0
    for i in range(len(x1)):
        distance += math.pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)

class Report():
    """
    A class to write the results for a given classifier to CSV and also pretty print them to stdout
    using tabulate
    """
    def __init__(self, report_name):
        self.report_name = report_name
        self.headers = ["Dataset Name", "Total Items in Dataset", "# Of Training Items", "# Of Test Items",
                        "Correct Predictions", "Incorrect Predictions", "Accuracy" ]
        self.csv_array = [self.headers]


    def add_row(self, dataset_name, data, test_results):
        accuracy = "{:.5f}".format(test_results[0]/float(sum(test_results)))
        items_trained = len(data["train"]["class_1"][0]) + len(data["train"]["class_2"][0])
        items_tested = len(data["test"]["class_1"][0]) + len(data["test"]["class_2"][0])
        total_items = items_trained + items_tested
        row = [dataset_name, total_items, items_trained, items_tested, test_results[0], test_results[1], accuracy]
        self.csv_array.append(row)

    def write_to_results_csv(self, file_name=None):
        if file_name is None:
            file_name = "./results/" + self.report_name + ".csv"
        with open(file_name, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in self.csv_array:
                writer.writerow(row)
        print "CSV File with results available at {:s}".format(file_name)
        self.print_csv(file_name)

    def print_csv(self, name):
        with open(name) as f:
            csv_reader = csv.reader(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_list = []
            for row in csv_reader:
                csv_list.append(row)
            try:
                print tabulate(csv_list[1:], headers=self.headers)
            except Exception:
                pass
