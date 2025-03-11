import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from codecarbon import EmissionsTracker  # Importar CodeCarbon

# 1. Iniciar el tracker de CodeCarbon
tracker = EmissionsTracker()
tracker.start()

# 2. Cargar los datos procesados (características de usuarios)
X_train = pd.read_pickle("data1m/x_train_alpha(0.045).pkl")

# Verificar columnas en X_train
print('Columnas en X_train:', X_train.columns)

# 3. Cargar los ratings (etiquetas)
ratings = pd.read_csv('C:/Users/xpati/Documents/TFG/ml-1m/ratings.dat', sep='::', engine='python', 
                      names=['UID', 'MID', 'rate', 'time'], encoding='latin-1')
y = ratings['rate']  # Etiquetas (ratings)

# 4. Cargar características de películas (si las tienes)
movies = pd.read_csv('C:/Users/xpati/Documents/TFG/ml-1m/movies.dat', sep='::', engine='python', 
                     names=['MID', 'title', 'genres'], encoding='latin-1')
# Procesar características de películas (one-hot encoding para géneros)
movies['genres'] = movies['genres'].str.split('|')
genres_encoded = movies['genres'].explode().str.get_dummies().groupby(level=0).sum()
movies = pd.concat([movies, genres_encoded], axis=1)
movies = movies.drop(['title', 'genres'], axis=1)

# 5. Combinar características de usuarios y películas
data = pd.merge(ratings, movies, on='MID')  # Combinar ratings y películas

# Verificar columnas en data
print('Columnas en data:', data.columns)

# 6. Combinar con características de usuarios
# Asegúrate de que 'UID' esté en ambos DataFrames
if 'UID' in X_train.columns and 'UID' in data.columns:
    data = pd.merge(data, X_train, on='UID')
else:
    raise KeyError("La columna 'UID' no está presente en ambos DataFrames.")

# 7. Dividir los datos en entrenamiento y prueba
X = data.drop('rate', axis=1)  # Características
y = data['rate']               # Etiquetas (ratings)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Entrenar un modelo híbrido (por ejemplo, Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
tracker.start_task("Entrenamiento del modelo")  # Iniciar medición para el entrenamiento
model.fit(X_train, y_train)
tracker.stop_task()  # Detener medición para el entrenamiento

# 9. Predecir en el conjunto de prueba
tracker.start_task("Predicción del modelo")  # Iniciar medición para la predicción
y_pred = model.predict(X_test)
tracker.stop_task()  # Detener medición para la predicción

# 10. Evaluar el modelo
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

# 11. Detener el tracker de CodeCarbon y obtener resultados
emissions: float = tracker.stop()
print(f"Emisiones de CO₂: {emissions} kg")