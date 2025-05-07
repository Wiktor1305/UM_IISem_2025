from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy
import pandas

kdd = fetch_kddcup99(return_X_y=False, percent10=True)
X = pandas.DataFrame(kdd.data)
y = kdd.target
#wczytanie i przygotowanie danych

y_binary = numpy.where(y == b'normal.', 1, 0)
#konwersja etykiet na binarny problem

numeric_cols = []
for i in range(X.shape[1]):
    try:
        pandas.to_numeric(X.iloc[:, i])
        numeric_cols.append(i)
    except ValueError:
        continue
#proba konwersji kolumny na float

X_numeric = X.iloc[:, numeric_cols].astype(float)
#usunienie kolumn nieliczbowych (szukanie i zachowanie tylko kolumn numerycznych)

X_train, X_test, y_train, y_test = train_test_split(X_numeric, y_binary, test_size=0.6, random_state=46)
#podzial na zbior treningowy i testowy

perceptron = Perceptron(random_state=46, max_iter=1500)
perceptron.fit(X_train, y_train)
#trenowanie perceptronu

accuracy = perceptron.score(X_test, y_test)
print(f"dokladnosc perceptronu: {accuracy:.4f}")
#wyliczenie dokladnosci

print("wagi perceptronu:")
print(perceptron.coef_)
print("obciazenie (bias):")
print(perceptron.intercept_)
#wyswietlanie wag i obciazenia

pipeline = Pipeline([('scaler', StandardScaler()), ('perceptron', Perceptron(random_state=46, max_iter=1500))])
#utworzenie potoku z standardscaler

pipeline.fit(X_train, y_train)
accuracy_pipeline = pipeline.score(X_test, y_test)
print(f"dokladnosc perceptronu ze standaryzacja: {accuracy_pipeline:.4f}")