import math
from scipy import optimize
import numpy as np
import util

from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

def clf_with_sklearn():
        clf = svm.SVC(kernel='linear')
        # clf = BernoulliNB()
        # clf = DecisionTreeClassifier()
        # clf = linear_model.LogisticRegression()
        data = util.load_split_data("./observed/classify_d99_k50_saved2.mat")
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

def fh(x,y):
    return math.pow(x[0],y[0])

xmin = optimize.fmin(func=fh, x0=1, a)


# data = load_split_data("./observed/classify_d99_k50_saved1.mat")
# training_params = train(data['train'])
# prob = calc_class_probability([0, 0,0], training_params)
# print(classify([0, 0,0], training_params))
# test_results = test(data['test'], training_params)
# print test_results

# data = scipy.io.loadmat("./observed/classify_d3_k2_saved1.mat")

# clf = Classifier()
# clf.clf_with_svm()
