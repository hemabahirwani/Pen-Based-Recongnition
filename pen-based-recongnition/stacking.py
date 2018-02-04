'''
Description: This python script implements Stacked Generalization 
Author 1: Hema Bahirwani (hgb1348)
Author 2: Navneet Sinha (nxs9384)
Author 3: Vatsala Singh (vs2080)
'''
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import tree, svm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix
import csv
from numpy import genfromtxt
'''
This method loads the training data from the csv file
'''
def load_train_data():
    X_Train = genfromtxt('training.csv', delimiter=',')
    X_Train = X_Train[:, :-1]
    print(X_Train.shape)
    Y_Train = []
    with open('training.csv', newline='\n') as csvfile:
        # Prepare the CSV parser
        my_file_reader = csv.reader(csvfile, delimiter=',')
        for row in my_file_reader:
            Y_Train.append(int(row[-1]))
    Y_Train = np.array(Y_Train)
    return X_Train, Y_Train
'''
This method loads the testing data from the csv file
'''
def load_test_data():
    X_Test = genfromtxt('test.csv', delimiter=',')
    X_Test = X_Test[:, :-1]
    Y_Test = []
    with open('test.csv', newline='\n') as csvfile:
        # Prepare the CSV parser
        my_file_reader = csv.reader(csvfile, delimiter=',')
        for row in my_file_reader:
            Y_Test.append(int(row[-1]))
    Y_Test = np.array(Y_Test)
    return X_Test, Y_Test
'''
This method implements the Random Forest Classifier
'''
def randomForest(data_train, data_test, labels_train, labels_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(data_train, labels_train)
    res = clf.predict(data_test)
    res = np.array(res)
    return res, clf
'''
This method implements the Decision Tree Classifier
'''
def decisionTree(data_train, data_test, labels_train, labels_test):
    clf = tree.DecisionTreeClassifier()
    clf.fit(data_train, labels_train)
    res = clf.predict(data_test)
    res = np.array(res)
    return res, clf
'''
This method implements the Multinomial Logistic Regression
'''
def logistic(newTrainingSet, labels_test):
    log = LogisticRegression()
    log.fit(newTrainingSet, labels_test)
    return log
'''
This method implements the Stacked Generalization algorithm
'''
def stacking():
    X_Train, Y_Train = load_train_data()
    data_train, data_test, labels_train, labels_test = train_test_split(X_Train,
                                                                        Y_Train, test_size=0.20,
                                                                        random_state=42)
    rfPred, rf = randomForest(data_train, data_test, labels_train, labels_test)
    naivePred, nb = decisionTree(data_train, data_test, labels_train, labels_test)
    newTrainingSet = np.column_stack((rfPred, naivePred))
    log = logistic(newTrainingSet, labels_test)
    X_Test, Y_Test = load_test_data()
    rfTestPred = rf.predict(X_Test)
    nbTestPred = nb.predict(X_Test)
    logTestData = np.column_stack((rfTestPred, nbTestPred))
    res = log.predict(logTestData)
    print(accuracy_score(res, Y_Test))
if __name__ == "__main__":
    stacking()


