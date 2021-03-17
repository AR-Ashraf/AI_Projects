import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import pickle
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)
'''
best = 0

for _ in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size= 0.2)
    clf = svm.SVC(kernel="linear")

    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_predict)
    print(acc)

    if acc > best:
        best = acc
        with open("breast_cancer.pickle", "wb") as f:
            pickle.dump(clf, f)'''

pickle_in = open("SVM/breast_cancer.pickle", "rb")
clf = pickle.load(pickle_in)

classes = ['malignant', 'benign']
y_predict = clf.predict(x_test)
for z in range(len(y_predict)):
    print(classes[int(y_predict[z])])
