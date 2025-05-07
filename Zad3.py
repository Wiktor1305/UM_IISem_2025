from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy
import pandas
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

kdd = fetch_kddcup99(percent10=True, return_X_y=False)
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

X_numeric = X.iloc[:, numeric_cols].astype(float)
#wybor kolumn numerycznych

X_train, X_test, y_train, y_test = train_test_split(X_numeric, y_binary, test_size=0.6, random_state=46)
#podzial na zbior treningowy i testowy

mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1500, random_state=46, early_stopping=True))])
#MLP z 1 warstwa ukryta (10 neuronow)

mlp_pipeline.fit(X_train, y_train)
#trenowanie MLP

mlp_accuracy = mlp_pipeline.score(X_test, y_test)
print(f"dokladnosc MLP (1 warstwa ukryta, 10 neuronow): {mlp_accuracy:.4f}")
#ocena dokladnosci

perceptron_pipeline = Pipeline([('scaler', StandardScaler()), ('perceptron', Perceptron(random_state=46, max_iter=1500))])

perceptron_pipeline.fit(X_train, y_train)
perceptron_accuracy = perceptron_pipeline.score(X_test, y_test)

print(f"dokladnosc prostego perceptronu: {perceptron_accuracy:.4f}")
print(f"roznica dokładnosci: {mlp_accuracy - perceptron_accuracy:.4f}")
#porownanie z prostym perceptronem

print("raport klasyfikacji dla MLP:")
print(classification_report(y_test, mlp_pipeline.predict(X_test)))
#dodatkowa analiza

#czy perceptron wielowarstwowy z jedna warstwa ukryta zawierajaca 10 neuronow znaczaco zwieksza dokladnosc klasyfikatora?
#jak widac nie bo w tym przypadku tylko o 0.0025