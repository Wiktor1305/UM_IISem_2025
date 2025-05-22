from sklearn.datasets import load_wine
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas
import numpy

###NA PODSTAWIE ZADANIA 1
print(f"NA PODSTAWIE ZADANIA 1")

X, y = load_wine(return_X_y=True, as_frame=True)
#wczytanie zbioru danych wine z biblioteki scikit-learn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=45)
#podzial na zbior treningowy i testowy

classifiers = {
    "decision tree": DecisionTreeClassifier(random_state=45),
    "logistic regression": LogisticRegression(max_iter=1000, random_state=45),
    "kNN": KNeighborsClassifier(n_neighbors=3),
    "perceptron": Perceptron(random_state=45, max_iter=1500),}
#lista klasyfikatorow

for clf_name, clf in classifiers.items():
    if clf_name == "logistic regression" or clf_name == "perceptron":
        pipeline = Pipeline([('scaler', StandardScaler()), (clf_name, clf)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        #dodanie standaryzacji dla regresji logistycznej i perceptronu
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        #trenowanie klasyfikatora bez potoku
#trenowanie i ocena klasyfikatorow


    print(f"#{clf_name}")
    print("macierz pomylek:")
    print(confusion_matrix(y_test, y_pred))
    print("raport klasyfikacji:")
    print(classification_report(y_test, y_pred))
    print(f"dokladnosc: {accuracy_score(y_test, y_pred):.4f}")
    print("\n")
    #wyniki

###NA PODSTAWIE ZADANIA 2

print(f"NA PODSTAWIE ZADANIA 2")

kdd = fetch_kddcup99(return_X_y=False, percent10=True)
X = pandas.DataFrame(kdd.data)
y = kdd.target
#wczytywanie danych

y_binary = numpy.where(y == b'normal.', 1, 0)
#konwersja etykiet na problem binarny

numeric_cols = []
for i in range(X.shape[1]):
    try:
        pandas.to_numeric(X.iloc[:, i])
        numeric_cols.append(i)
    except ValueError:
        continue
X_numeric = X.iloc[:, numeric_cols].astype(float)
#wybor kolumn numerycznych

X_train, X_test, y_train, y_test = train_test_split(X_numeric, y_binary, test_size=0.6, random_state=46)
#podzial na zbior treningowy i testowy

classifiers = {
    "decision tree": DecisionTreeClassifier(random_state=46),
    "logistic regression": LogisticRegression(max_iter=1500, random_state=46),
    "kNN": KNeighborsClassifier(n_neighbors=3),
    "perceptron": Perceptron(random_state=46, max_iter=1500),}
#lista klasyfikatorow

for clf_name, clf in classifiers.items():
    if clf_name == "logistic regression" or clf_name == "perceptron":
        pipeline = Pipeline([('scaler', StandardScaler()), (clf_name, clf)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        #dodanie standaryzacji dla regresji logistycznej i perceptronu
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        #trenowanie klasyfikatora bez potoku
#trenowanie i ocena klasyfikatorow

    print(f"#{clf_name}")
    print("macierz pomylek:")
    print(confusion_matrix(y_test, y_pred))
    print("raport klasyfikacji:")
    print(classification_report(y_test, y_pred))
    print(f"dokladnosc: {accuracy_score(y_test, y_pred):.4f}")
    print("\n")
    #wyniki

###Komentarz: na podstawie zadania 1 (zbior danych "wine") najlepsza okazala sie regresja logistyczna osiagajac wartosc
#97,75% oraz wysokie metryki, precision, recall i f1 score. Jesli chodzi o zadanie 2 (na podstawie zbioru danych kkdcup99)
#to wygrywa decision tree (99,94%).