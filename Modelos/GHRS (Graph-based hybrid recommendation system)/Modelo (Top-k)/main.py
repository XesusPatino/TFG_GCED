import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from codecarbon import EmissionsTracker
import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder # Import LabelEncoder
from collections import defaultdict
import random

# Importaciones alternativas
Model = tf.keras.models.Model
Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Concatenate = tf.keras.layers.Concatenate

# Configuración de rutas y directorios
result_path = "results"
os.makedirs(result_path, exist_ok=True)
os.makedirs(f"{result_path}/emissions_reports", exist_ok=True)
os.makedirs(f"{result_path}/emissions_plots", exist_ok=True)

config = {
    'embedding_dim': 128,
    'hidden_dim': 128,
    'batch_size': 1024,
    'learning_rate': 1e-3,
    'epochs': 50,
    'display_step': 1,
    'test_size': 0.2,
    'random_state': 42,
    'top_k_values': [5, 10, 20]  # Valores de K para las métricas Top-K
}

# Clase para seguimiento de emisiones por época
class EmissionsPerEpochTracker:
    def __init__(self, result_path, model_name="LRML"):
        self.result_path = result_path
        self.model_name = model_name
        self.epoch_emissions = []
        self.cumulative_emissions = []
        self.epoch_rmse = []
        self.epoch_loss = []
        self.epoch_val_loss = []
        self.epoch_recall = []
        self.epoch_ndcg = []
        self.total_emissions = 0.0
        self.trackers = {}
        
        # Crear directorio para emisiones
        os.makedirs(f"{result_path}/emissions_reports", exist_ok=True)
        os.makedirs(f"{result_path}/emissions_plots", exist_ok=True)
    
    def start_epoch(self, epoch):
        # Crear un tracker con un nombre único basado en timestamp
        timestamp = int(time.time())
        tracker_name = f"{self.model_name}_epoch{epoch}_{timestamp}"
        
        self.trackers[epoch] = EmissionsTracker(
            project_name=tracker_name,
            output_dir=f"{self.result_path}/emissions_reports",
            save_to_file=True,
            log_level="error",
            save_to_api=False
        )
        
        try:
            self.trackers[epoch].start()
        except Exception as e:
            print(f"Advertencia: No se pudo iniciar el tracker para la época {epoch}: {e}")
            self.trackers[epoch] = None
    
    def end_epoch(self, epoch, loss, val_loss=None, val_rmse=None, recall=None, ndcg=None):
        try:
            epoch_co2 = 0.0
            if epoch in self.trackers and self.trackers[epoch]:
                try:
                    epoch_co2 = self.trackers[epoch].stop() or 0.0
                except Exception as e:
                    print(f"Advertencia: Error al detener el tracker para la época {epoch}: {e}")
                    epoch_co2 = 0.0
            
            # Acumular emisiones totales
            self.total_emissions += epoch_co2
            
            # Guardar datos de esta época
            self.epoch_emissions.append(epoch_co2)
            self.cumulative_emissions.append(self.total_emissions)
            self.epoch_loss.append(loss)
            
            if val_loss is not None:
                self.epoch_val_loss.append(val_loss)
                
            if val_rmse is not None:
                self.epoch_rmse.append(val_rmse)

            if recall is not None:
                self.epoch_recall.append(recall)

            if ndcg is not None:
                self.epoch_ndcg.append(ndcg)
            
            print(f"Epoch {epoch+1} - Emisiones: {epoch_co2:.8f} kg, Acumulado: {self.total_emissions:.8f} kg")
            print(f"Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}")
            if recall is not None and ndcg is not None:
                print(f"Recall@{config['top_k_values'][0]}: {recall:.4f}, NDCG@{config['top_k_values'][0]}: {ndcg:.4f}")
            
        except Exception as e:
            print(f"Error al medir emisiones en época {epoch}: {e}")
    
    def end_training(self, final_rmse, final_recall=None, final_ndcg=None):
        try:
            # Asegurarse de que todos los trackers estén detenidos
            for epoch, tracker in self.trackers.items():
                if tracker is not None:
                    try:
                        tracker.stop()
                    except:
                        pass
            
            # Crear dataframe con todos los datos
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            epochs_range = range(1, len(self.epoch_emissions) + 1)
            
            df = pd.DataFrame({
                'epoch': epochs_range,
                'epoch_emissions_kg': self.epoch_emissions,
                'cumulative_emissions_kg': self.cumulative_emissions,
                'loss': self.epoch_loss,
                'val_loss': self.epoch_val_loss if self.epoch_val_loss else [None] * len(self.epoch_emissions),
                'rmse': self.epoch_rmse if self.epoch_rmse else [None] * len(self.epoch_emissions),
                'recall': self.epoch_recall if self.epoch_recall else [None] * len(self.epoch_emissions),
                'ndcg': self.epoch_ndcg if self.epoch_ndcg else [None] * len(self.epoch_emissions)
            })
            
            # Crear carpetas si no existen
            os.makedirs(f"{self.result_path}/emissions_reports", exist_ok=True)
            os.makedirs(f"{self.result_path}/emissions_plots", exist_ok=True)
            
            emissions_file = f'{self.result_path}/emissions_reports/emissions_metrics_{self.model_name}_{timestamp}.csv'
            df.to_csv(emissions_file, index=False)
            print(f"Métricas de emisiones guardadas en: {emissions_file}")
            
            # Graficar las relaciones
            self.plot_emissions_vs_metrics(epochs_range, timestamp, final_rmse, final_recall, final_ndcg)
            
        except Exception as e:
            print(f"Error al generar gráficos de emisiones: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_emissions_vs_metrics(self, epochs_range, timestamp, final_rmse=None, final_recall=None, final_ndcg=None):
        """Genera gráficos para emisiones vs métricas"""
        
        try:
            # 1. Gráfico combinado: Emisiones por época y acumulativas
            plt.figure(figsize=(12, 10))
            
            plt.subplot(2, 2, 1)
            plt.plot(epochs_range, self.epoch_emissions, 'r-', marker='x')
            plt.title('Emisiones por Época')
            plt.xlabel('Época')
            plt.ylabel('CO₂ Emissions (kg)')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.plot(epochs_range, self.cumulative_emissions, 'r-', marker='o')
            plt.title('Emisiones Acumuladas por Época')
            plt.xlabel('Época')
            plt.ylabel('CO₂ Emissions (kg)')
            plt.grid(True, alpha=0.3)
            
            if self.epoch_loss:
                plt.subplot(2, 2, 3)
                plt.plot(epochs_range, self.epoch_loss, 'g-', marker='o', label='Train Loss')
                if self.epoch_val_loss:
                    plt.plot(epochs_range, self.epoch_val_loss, 'b-', marker='x', label='Val Loss')
                plt.title('Loss por Época')
                plt.xlabel('Época')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            if self.epoch_rmse:
                plt.subplot(2, 2, 4)
                plt.plot(epochs_range, self.epoch_rmse, 'b-', marker='o')
                plt.title('RMSE por Época')
                plt.xlabel('Época')
                plt.ylabel('RMSE')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            file_path = f'{self.result_path}/emissions_plots/metrics_by_epoch_{self.model_name}_{timestamp}.png'
            plt.savefig(file_path)
            plt.close()
            print(f"Gráfico guardado en: {file_path}")
            
            if self.epoch_rmse:
                # 2. RMSE vs Emisiones acumuladas
                plt.figure(figsize=(10, 6))
                plt.plot(self.cumulative_emissions, self.epoch_rmse, 'b-', marker='o')
                
                # Añadir etiquetas con el número de época
                for i, (emissions, rmse) in enumerate(zip(self.cumulative_emissions, self.epoch_rmse)):
                    plt.annotate(f"{i+1}", (emissions, rmse), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9)
                    
                plt.xlabel('Emisiones de CO₂ acumuladas (kg)')
                plt.ylabel('RMSE')
                plt.title('Relación entre Emisiones Acumuladas y RMSE')
                plt.grid(True, alpha=0.3)
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_rmse_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path)
                plt.close()
                print(f"Gráfico guardado en: {file_path}")
                
                # 3. Scatter plot de rendimiento frente a emisiones acumulativas
                plt.figure(figsize=(10, 6))
                
                # Ajustar tamaño de los puntos según la época
                sizes = [(i+1)*20 for i in range(len(self.cumulative_emissions))]
                
                scatter = plt.scatter(self.epoch_rmse, self.cumulative_emissions, 
                            color='blue', marker='o', s=sizes, alpha=0.7)
                
                # Añadir etiquetas de época
                for i, (rmse, em) in enumerate(zip(self.epoch_rmse, self.cumulative_emissions)):
                    plt.annotate(f"{i+1}", (rmse, em), textcoords="offset points", 
                                xytext=(0,5), ha='center', fontsize=9)
                
                plt.ylabel('Emisiones de CO₂ acumuladas (kg)')
                plt.xlabel('RMSE')
                plt.title('Relación entre RMSE y Emisiones Acumuladas')
                plt.grid(True, alpha=0.3)
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_performance_scatter_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path)
                plt.close()
                print(f"Gráfico guardado en: {file_path}")
        except Exception as e:
            print(f"Error al generar los gráficos: {e}")
            import traceback
            traceback.print_exc()

# Configuración de rutas y directorios
result_path = "results"
os.makedirs(result_path, exist_ok=True)

# 1. Cargar los datos procesados (que ya contienen características de grafos)
X_users = pd.read_pickle("data1m/x_train_alpha(0.045).pkl")

# Verificar columnas en X_users
print('Columnas en X_users:', X_users.columns)

# 2. Cargar los ratings (etiquetas)
ratings = pd.read_csv('C:/Users/xpati/Documents/TFG/ml-1m/ratings.dat', sep='::', engine='python', 
                      names=['UID', 'MID', 'rate', 'time'], encoding='latin-1')
y = ratings['rate']  # Etiquetas (ratings)

# 2. Cargar características de películas (si las tienes)
movies = pd.read_csv('C:/Users/xpati/Documents/TFG/ml-1m/movies.dat', sep='::', engine='python', 
                     names=['MID', 'title', 'genres'], encoding='latin-1')
# Procesar características de películas (one-hot encoding para géneros)
movies['genres'] = movies['genres'].str.split('|')
genres_encoded = movies['genres'].explode().str.get_dummies().groupby(level=0).sum()
movies = pd.concat([movies, genres_encoded], axis=1)
movies = movies.drop(['title', 'genres'], axis=1)

# 3. Combinar características de usuarios y películas
data = pd.merge(ratings, movies, on='MID')  # Combinar ratings y películas

# Verificar columnas en data
print('Columnas en data:', data.columns)

# 3. Combinar con características de usuarios
# Asegúrate de que 'UID' esté en ambos DataFrames
if 'UID' in X_users.columns and 'UID' in data.columns:
    data = pd.merge(data, X_users, on='UID')
else:
    raise KeyError("La columna 'UID' no está presente en ambos DataFrames.")

# 4. Label encoding de user y movie IDs
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

data['UID'] = user_encoder.fit_transform(data['UID'])
data['MID'] = movie_encoder.fit_transform(data['MID'])

# Obtener número de usuarios y películas únicas
n_users = len(user_encoder.classes_)
n_movies = len(movie_encoder.classes_)

print(f'Number of users: {n_users}')
print(f'Number of movies: {n_movies}')

# 5. Dividir los datos en entrenamiento y prueba
X = data.drop('rate', axis=1)  # Características
y = data['rate']               # Etiquetas (ratings)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Normalizar características numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convertir a tensores de TensorFlow
X_train_scaled = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)
X_test_scaled = tf.convert_to_tensor(X_test_scaled, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train.values, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test.values, dtype=tf.float32)

# 7. Construir modelo híbrido con TensorFlow/Keras
input_layer = Input(shape=(X_train_scaled.shape[1],))

# Ramas específicas para tipos de características si quieres un modelo realmente híbrido
graph_features = Dense(64, activation='relu')(input_layer)
graph_features = Dropout(0.2)(graph_features)

user_features = Dense(32, activation='relu')(input_layer)
user_features = Dropout(0.2)(user_features)

movie_features = Dense(32, activation='relu')(input_layer)
movie_features = Dropout(0.2)(movie_features)

# Combinar las ramas
combined = Concatenate()([graph_features, user_features, movie_features])
combined = Dense(64, activation='relu')(combined)
combined = Dropout(0.3)(combined)
combined = Dense(32, activation='relu')(combined)

# Capa de salida
output = Dense(1)(combined)

# Crear y compilar el modelo
model = Model(inputs=input_layer, outputs=output)
model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['mae'])

# 8. Preparar el tracking de emisiones por época
emissions_tracker = EmissionsPerEpochTracker(result_path)

# 9. Métricas Top-K
def calculate_topk_metrics(model, X, y, user_ids, item_ids, k=5):
    """Calcula Recall@K y NDCG@K."""

    # Hacer predicciones para todos los pares usuario-ítem en X
    y_pred = model.predict(X).flatten()

    # Crear un DataFrame con las predicciones, los valores reales, UIDs y MIDs
    df = pd.DataFrame({'UID': user_ids, 'MID': item_ids, 'y_true': y.numpy(), 'y_pred': y_pred})

    # Agrupar por usuario y ordenar las predicciones
    df = df.groupby('UID').apply(lambda x: x.sort_values('y_pred', ascending=False))

    # Calcular Recall@K y NDCG@K para cada usuario
    recall_sum = 0
    ndcg_sum = 0
    num_users = len(df.index.levels[0])  # Número de usuarios únicos

    for user_id in df.index.levels[0]:
        user_data = df.loc[user_id]
        
        # Convertir y_true a un conjunto de elementos relevantes (rating >= 4)
        relevant_items = set(user_data[user_data['y_true'] >= 4]['MID'].values)
        
        # Obtener los top-k MIDs predichos
        top_k_items = user_data['MID'].values[:k]
        
        # Calcular hits
        hits = len(set(top_k_items) & relevant_items)
        
        # Calcular Recall@K
        recall = hits / len(relevant_items) if relevant_items else 0.0
        recall_sum += recall
        
        # Calcular NDCG@K
        dcg = 0.0
        for i, item in enumerate(top_k_items):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)
        
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_sum += ndcg

    # Promediar Recall@K y NDCG@K sobre todos los usuarios
    recall = recall_sum / num_users if num_users > 0 else 0.0
    ndcg = ndcg_sum / num_users if num_users > 0 else 0.0

    return recall, ndcg

# 10. Entrenar el modelo
print("Iniciando entrenamiento del modelo...")
batch_size = config['batch_size']
epochs = config['epochs']

# Extraer user_ids y item_ids del conjunto de prueba
user_ids_test = X_test['UID'].values
item_ids_test = X_test['MID'].values

# Mejorar la historia del entrenamiento
history = {"loss": [], "val_loss": [], "val_rmse": [], "recall": [], "ndcg": []}

for epoch in range(epochs):
    emissions_tracker.start_epoch(epoch)
    
    # Entrenar una época completa
    history_callback = model.fit(
        X_train_scaled, y_train,
        batch_size=batch_size,
        epochs=1,
        verbose=0,
        validation_data=(X_test_scaled, y_test)
    )
    
    # Obtener métricas de la época
    train_loss = history_callback.history['loss'][0]
    val_loss = history_callback.history['val_loss'][0]
    
    # Guardar métricas en el historial
    history["loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    
    # Calcular RMSE en el conjunto de validación
    y_pred_val = model.predict(X_test_scaled, verbose=0)
    val_rmse = mean_squared_error(y_test.numpy(), y_pred_val.flatten(), squared=False)
    history["val_rmse"].append(val_rmse)
    
    # Calcular Recall@K y NDCG@K en el conjunto de validación
    recall, ndcg = calculate_topk_metrics(model, X_test_scaled, y_test, user_ids_test, item_ids_test, k=config['top_k_values'][0])
    history["recall"].append(recall)
    history["ndcg"].append(ndcg)

    # Registrar métricas y emisiones
    emissions_tracker.end_epoch(
        epoch,
        train_loss,
        val_loss=val_loss,
        val_rmse=val_rmse,
        recall=recall,
        ndcg=ndcg
    )
    
    # Mostrar progreso
    if (epoch + 1) % config['display_step'] == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}")
        print(f"Recall@{config['top_k_values'][0]}: {recall:.4f}, NDCG@{config['top_k_values'][0]}: {ndcg:.4f}")

# 11. Evaluar el modelo final
print("Realizando predicciones finales...")
y_pred = model.predict(X_test_scaled, verbose=0)

# 12. Evaluar el modelo con métricas finales
rmse = mean_squared_error(y_test.numpy(), y_pred.flatten(), squared=False)
mae = mean_absolute_error(y_test.numpy(), y_pred.flatten())
r2 = r2_score(y_test.numpy(), y_pred.flatten())

print(f"RMSE final: {rmse}")
print(f"MAE final: {mae}")
print(f"R² final: {r2}")

# 13. Calcular métricas de ranking finales
final_metrics = {}
for k in config['top_k_values']:
    recall, ndcg = calculate_topk_metrics(model, X_test_scaled, y_test, user_ids_test, item_ids_test, k=k)
    final_metrics[f'recall@{k}'] = recall
    final_metrics[f'ndcg@{k}'] = ndcg
    print(f"Recall@{k} final: {recall:.4f}")
    print(f"NDCG@{k} final: {ndcg:.4f}")

# 14. Generar gráficos finales de emisiones vs. rendimiento
emissions_tracker.end_training(rmse, final_metrics[f'recall@{config["top_k_values"][0]}'], final_metrics[f'ndcg@{config["top_k_values"][0]}'])

# 15. Guardar resultados finales en un CSV
timestamp = time.strftime("%Y%m%d-%H%M%S")
final_results = {
    'final_rmse': [rmse],
    'final_mae': [mae],
    'final_r2': [r2],
}

# Añadir todas las métricas de ranking
for k in config['top_k_values']:
    final_results[f'final_recall@{k}'] = [final_metrics[f'recall@{k}']]
    final_results[f'final_ndcg@{k}'] = [final_metrics[f'ndcg@{k}']]

final_results_df = pd.DataFrame(final_results)
final_results_file = f"{result_path}/final_results_{timestamp}.csv"
final_results_df.to_csv(final_results_file, index=False)
print(f"Resultados finales guardados en: {final_results_file}")