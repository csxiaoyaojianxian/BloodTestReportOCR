#coding = utf-8
import pickle
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import train_test_split


def extract(filename):
    X = np.loadtxt(filename, skiprows= 1,delimiter=',', usecols=(3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28))
    y = np.loadtxt(filename, dtype='string', skiprows= 1,delimiter=',', usecols=(1,))
    for i in range(len(y)):
        if y[i] == '\xc4\xd0':
            y[i] = 1
        else:
            y[i] = 0
    return X,y

def split_test(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    return X_train, X_test, y_train, y_test

def save_model(model,name):
    pickle.dump(model, open(str(name)+'.pkl', 'w'))

def load_model(name):
    model = pickle.load(open(str(name)+'.pkl'))
    return model

if __name__ == "__main__":
    X, y = extract('train.csv')
    X_train, X_test, y_train, y_test = split_test(X, y)
    clf = svm.SVC(kernel='linear', gamma=0.7, C = 1.0).fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    print metrics.classification_report(y_test, y_predicted)
    print
    print "test_accuracy_score"
    print metrics.accuracy_score(y_test, y_predicted)
    save_model(clf,'sex')

    X, y =extract('predict.csv')
    clf2 = load_model('sex')
    y2_predicted = clf2.predict(X)
    print "accuracy_score"
    print metrics.accuracy_score(y, y2_predicted)



