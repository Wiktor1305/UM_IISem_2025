from sklearn.datasets import load_wine
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy
import matplotlib.pyplot as matplt
from matplotlib.colors import ListedColormap

X, y = load_wine(return_X_y=True, as_frame=True)
#wczytanie zbioru danych wine z biblioteki scikit-learn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=45)
#podzial na zbior treningowy i testowy

y_train_0 = (y_train == 0).astype(int)
y_test_0 = (y_test == 0).astype(int)

perceptron = Perceptron(random_state=45)
perceptron.fit(X_train, y_train_0)
#trenowanie perceptronu rozpoznajacego klase 0

accuracy = perceptron.score(X_test, y_test_0)
print(f"dokladnosc perecptronu: {accuracy:.4f}")
#wyliczenie dokladnosci na zbiorze testowym

print("wagi perceptronu:")
print(perceptron.coef_)
print("obciazenie (bias):")
print(perceptron.intercept_)
#wyswietlenie wagi i obciazenia

pipeline = Pipeline([('scaler', StandardScaler()), ('perceptron', Perceptron(random_state=45))])
#utworzenie potoku pipeline z standardscaler i perceptronem

pipeline.fit(X_train, y_train_0)
accuracy_pipeline = pipeline.score(X_test, y_test_0)
print(f"dokladnosc perceptronu ze standaryzacja: {accuracy_pipeline:.4f}")
#trenowaine potoku i wyliczenie dokladnosci perceptronu

#ZADANIE DODATKOWE(*) G

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 0).astype(int)
#zaladowanie zbioru danych iris z wybraymi zmiennymi: petal length i petal width

per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)
#trenowanie perceptronu

w = per_clf.coef_[0]
b = per_clf.intercept_
a = -w[0] / w[1]
c = -b / w[1]
#wyznaczenie linii decyzyjnej

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_range = numpy.linspace(x_min, x_max, 100)
#osie

decision_boundary = a * x_range + c

xx, yy = numpy.meshgrid(numpy.linspace(x_min, x_max, 200), numpy.linspace(y_min, y_max, 200))
grid = numpy.c_[xx.ravel(), yy.ravel()]
predicted = per_clf.predict(grid).reshape(xx.shape)
#siatka

matplt.figure(figsize=(10, 6))
custom_cmap = ListedColormap(['#fafab0', '#9898ff'])  # Kolory tła
matplt.contourf(xx, yy, predicted, cmap=custom_cmap, alpha=0.3)
#wykres

matplt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='klasa 0 (Iris setosa)', edgecolor='k')
matplt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='pozostale klasy', edgecolor='k')
matplt.scatter(X[y == 2, 0], X[y == 2, 1], color='red', edgecolor='k')
#obiekty

matplt.plot(x_range, decision_boundary, "k-", linewidth=2, label='granica decyzyjna')

matplt.xlabel("dlugosc platka (cm)")
matplt.ylabel("szerokosc platka (cm)")
matplt.title("wizualizacja klasyfikacji perceptronu")
matplt.legend()
matplt.grid()
matplt.show()
#legenda wykresu