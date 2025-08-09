import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from codecarbon import EmissionsTracker
import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt

# Importaciones alternativas
Model = tf.keras.models.Model
Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Concatenate = tf.keras.layers.Concatenate

from sklearn.preprocessing import StandardScaler

# Clase para seguimiento de emisiones por época
class EmissionsPerEpochTracker:
    def __init__(self, result_path, model_name="GHRS"):
        self.result_path = result_path
        self.model_name = model_name
        self.epoch_emissions = []
        self.cumulative_emissions = []
        self.epoch_rmse = []
        self.epoch_loss = []
        self.epoch_val_loss = []
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
            save_to_api=False,
            allow_multiple_runs=True
        )
        
        try:
            self.trackers[epoch].start()
        except Exception as e:
            print(f"Advertencia: No se pudo iniciar el tracker para la época {epoch}: {e}")
            self.trackers[epoch] = None
    
    def end_epoch(self, epoch, loss, val_loss=None, val_rmse=None):
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
            
            print(f"Epoch {epoch+1} - Emisiones: {epoch_co2:.8f} kg, Acumulado: {self.total_emissions:.8f} kg")
            print(f"Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}")
            
        except Exception as e:
            print(f"Error al medir emisiones en época {epoch}: {e}")
    
    def end_training(self, final_rmse):
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
                'rmse': self.epoch_rmse if self.epoch_rmse else [None] * len(self.epoch_emissions)
            })
            
            # Crear carpetas si no existen
            os.makedirs(f"{self.result_path}/emissions_reports", exist_ok=True)
            os.makedirs(f"{self.result_path}/emissions_plots", exist_ok=True)
            
            emissions_file = f'{self.result_path}/emissions_reports/emissions_metrics_{self.model_name}_{timestamp}.csv'
            df.to_csv(emissions_file, index=False)
            print(f"Métricas de emisiones guardadas en: {emissions_file}")
            
            # Graficar las relaciones
            self.plot_emissions_vs_metrics(epochs_range, timestamp, final_rmse)
            
        except Exception as e:
            print(f"Error al generar gráficos de emisiones: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_emissions_vs_metrics(self, epochs_range, timestamp, final_rmse=None):
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
X_users = pd.read_pickle("data1m/x_train_Kiva_alpha(0.001).pkl")

# Verificar columnas en X_users
print('Columnas en X_users:', X_users.columns)

# 2. Cargar los ratings (etiquetas) desde la nueva base de datos
ratings = pd.read_csv(r'C:\Users\xpati\Documents\TFG\kiva_ml17.csv')
ratings.rename(columns={'user_id': 'UID', 'item_id': 'MID', 'rating': 'rate'}, inplace=True)
y = ratings['rate']  # Etiquetas (ratings)

# 3. Combinar características de usuarios y películas
data = ratings  # En este caso, no hay características adicionales de películas

# Verificar columnas en data
print('Columnas en data:', data.columns)

# Verificar columnas en X_users y data
if 'UID' not in X_users.columns:
    raise KeyError(f"La columna 'UID' no está presente en X_users. Columnas disponibles: {X_users.columns}")
if 'UID' not in data.columns:
    raise KeyError(f"La columna 'UID' no está presente en data. Columnas disponibles: {data.columns}")

# Combinar con características de usuarios
data = pd.merge(data, X_users, on='UID')

# 4. Dividir los datos en entrenamiento y prueba
X = data.drop('rate', axis=1)  # Características
y = data['rate']               # Etiquetas (ratings)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Normalizar características numéricas
scaler = StandardScaler() 
# Seleccionar solo las columnas numéricas para el escalado
numerical_columns = ['PR', 'CD', 'CC', 'CB', 'LC', 'AND']
X_train_scaled = scaler.fit_transform(X_train[numerical_columns])
X_test_scaled = scaler.transform(X_test[numerical_columns])

# 6. Construir modelo híbrido con TensorFlow/Keras
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
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# 7. Preparar el tracking de emisiones por época
emissions_tracker = EmissionsPerEpochTracker(result_path)

# 8. Define una clase personalizada para TrackingCallback
class TrackingCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, emissions_tracker):
        super().__init__()
        self.val_data = val_data
        self.emissions_tracker = emissions_tracker
        self.val_X, self.val_y = val_data
        
    def on_epoch_begin(self, epoch, logs=None):
        self.emissions_tracker.start_epoch(epoch)
        
    def on_epoch_end(self, epoch, logs=None):
        # Calcular RMSE en el conjunto de validación
        y_pred = self.model.predict(self.val_X, verbose=0)
        val_rmse = mean_squared_error(self.val_y, y_pred.flatten(), squared=False)
        
        # Registrar métricas y emisiones
        self.emissions_tracker.end_epoch(
            epoch, 
            logs.get('loss'), 
            logs.get('val_loss'),
            val_rmse
        )

# 9. Dividir los datos de entrenamiento para validación
X_train_part, X_val, y_train_part, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.1, random_state=42
)

# 10. Crear el callback de tracking
tracking_callback = TrackingCallback((X_val, y_val), emissions_tracker)

# 11. Entrenar con epochs y tracking
print("Iniciando entrenamiento del modelo...")
history = model.fit(
    X_train_part, y_train_part,
    epochs=50,
    batch_size=1024,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[tracking_callback]
)

# 12. Evaluar el modelo final
print("Realizando predicciones finales...")
y_pred = model.predict(X_test_scaled)

# 13. Evaluar el modelo con métricas finales
rmse = mean_squared_error(y_test, y_pred.flatten(), squared=False)
mae = mean_absolute_error(y_test, y_pred.flatten())
r2 = r2_score(y_test, y_pred.flatten())

print(f"RMSE final: {rmse}")
print(f"MAE final: {mae}")
print(f"R² final: {r2}")

# 14. Generar gráficos finales de emisiones vs. rendimiento
emissions_tracker.end_training(rmse)