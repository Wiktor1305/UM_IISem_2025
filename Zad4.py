from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

X, y = fetch_openml('liver-disorders', as_frame=True, return_X_y=True)
#wczytanie zbioru danych liver-disorders

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=47)
#podzial na zbior treningowy i testowy

mlp = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1600, random_state=47)
mlp.fit(X_train, y_train)
#trenowanie mlpregressor

mlp_mae = mean_absolute_error(y_test, mlp.predict(X_test))
print(f"mae mlpregressor: {mlp_mae:.4f}")
#obliczenie mean absolute error na zbiorze testowym

print("wagi warstwy ukrytej:")
print(mlp.coefs_[0])
print("wagi warstwy wyjsciowej:")
print(mlp.coefs_[1])
print("obciazenia:")
print(mlp.intercepts_)
#wyswietlenie wag i obciazenia

mlp_pipeline = Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(hidden_layer_sizes=(10,), activation='relu', max_iter=1600, random_state=47))])
mlp_pipeline.fit(X_train, y_train)
mlp_scaled_mae = mean_absolute_error(y_test, mlp_pipeline.predict(X_test))
print(f"mae mlpRegressor ze standaryzacja: {mlp_scaled_mae:.4f}")
#potok ze standardscaler

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_mae = mean_absolute_error(y_test, lr.predict(X_test))
print(f"mae linearregression: {lr_mae:.4f}")
#porownanie z linearregression

lr_pipeline = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])

lr_pipeline.fit(X_train, y_train)
lr_scaled_mae = mean_absolute_error(y_test, lr_pipeline.predict(X_test))
print(f"mae linearregression ze standaryzacja: {lr_scaled_mae:.4f}")
#potok z linearregression dla porownania

if mlp_scaled_mae < lr_scaled_mae:
    print("mlpregressor przewiduje lepiej niz linearregression")
else:
    print("linearregression przewiduje lepiej niz mlpregressor")
#ocena ktora metoda jest lepsza