from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas
import numpy

kdd = fetch_kddcup99(return_X_y=False, percent10=True)
X = pandas.DataFrame(kdd.data)
y = kdd.target
#wczytanie danych bez usuwania kolumn nieliczbowych

y_binary = numpy.where(y == b'normal.', 1, 0)
#konwersja etykiet na problem binarny

X_sampled = X.sample(frac=0.02, random_state=42)
y_sampled = y_binary[X_sampled.index]
#zmniejszenie rozmiaru danych do 2% - bo mi komputer probowal odleciec :)

categorical_cols = X_sampled.select_dtypes(include=['object']).columns
numeric_cols = X_sampled.select_dtypes(include=['float64', 'int64']).columns
#identyfikacja kolumn kategorycznych i numerycznych

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols), ('num', 'passthrough', numeric_cols)])
#przeksztalcenia danych: Ordinal Encoding

classifiers = {
    "decision tree": Pipeline([('preprocessor', preprocessor), ('model', DecisionTreeClassifier(random_state=46))]),
    "logistic regression": Pipeline([('preprocessor', preprocessor), ('scaler', StandardScaler(with_mean=False)), ('model', LogisticRegression(solver='saga', max_iter=1500, random_state=46))]),
    "kNN": Pipeline([('preprocessor', preprocessor), ('scaler', StandardScaler(with_mean=False)), ('model', KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree'))]),
    "perceptron": Pipeline([('preprocessor', preprocessor), ('scaler', StandardScaler(with_mean=False)), ('model', Perceptron(random_state=46, max_iter=1500))]),
    "MLP neural network": Pipeline([('preprocessor', preprocessor), ('scaler', StandardScaler(with_mean=False)), ('model', MLPClassifier(hidden_layer_sizes=(10,), max_iter=1500, random_state=46))])}
#lista klasyfikatorow

X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.6, random_state=46)
#podzial danych na zbior treningowy i testowy

for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"#{clf_name}")
    print("macierz pomylek:")
    print(confusion_matrix(y_test, y_pred))
    print("raport klasyfikacji:")
    print(classification_report(y_test, y_pred))
    print(f"dokladnosc: {accuracy_score(y_test, y_pred):.4f}\n")
#trenowanie i ocena klasyfikatorow

#ciezko to podsumowac, poniewaz uzyto tutaj tylko 2% danych - inaczej komputer nie dawal rady...