import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import pickle

data = pd.read_csv("KNN/car.data")

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
classs = le.fit_transform(list(data["class"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))

predict = "class"
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(classs)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
'''
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

    model = KNeighborsClassifier(n_neighbors= 7)
    model.fit(x_train, y_train)
    acc = model.score(x_test,y_test)
    print(acc)
    if acc>best:
        best = acc
        with open("KNN/carmodel.pickle", "wb") as f:
            pickle.dump(model, f)'''
pickle_in = open("KNN/carmodel.pickle", "rb")
model = pickle.load(pickle_in)

predicted = model.predict(x_test)
name = ["unacc", "acc", "good", "vgood"]

for z in range(len(predicted)):
    print("Predicted: " , name[predicted[z]] , "Data: " , x_test[z], "Actual: " , name[y_test[z]])