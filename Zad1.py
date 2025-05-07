from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

