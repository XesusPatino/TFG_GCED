import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from codecarbon import EmissionsTracker
import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
import psutil

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
        self.epoch_mae = []
        self.epoch_loss = []
        self.epoch_val_loss = []
        self.epoch_times = []
        self.epoch_memory = []
        self.epoch_cpu = []
        self.total_emissions = 0.0
        self.trackers = {}
        self.train_start_time = None
        self.main_tracker = None
        self.best_rmse = float('inf')
        self.best_rmse_epoch = None
        self.best_rmse_emissions = None
        self.best_rmse_cumulative_emissions = None
        self.best_rmse_metrics = None
        
        # Crear directorio para emisiones
        os.makedirs(f"{result_path}/emissions_reports", exist_ok=True)
        os.makedirs(f"{result_path}/emissions_plots", exist_ok=True)
        
        # Inicializar tracker principal
        self.main_tracker = EmissionsTracker(
            project_name=f"{self.model_name}_training",
            output_dir=f"{self.result_path}/emissions_reports",
            save_to_file=True,
            log_level="error",
            save_to_api=False
        )
    
    def start_epoch(self, epoch):
        # Inicializar métricas del sistema para esta época
        self.epoch_start_time = time.time()
        
        # Crear tracker individual para esta época con allow_multiple_runs
        self.trackers[epoch] = EmissionsTracker(
            project_name=f"{self.model_name}_epoch_{epoch}",
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
    
    def end_epoch(self, epoch, loss, val_loss=None, val_rmse=None, val_mae=None):
        try:
            epoch_co2 = 0.0
            if epoch in self.trackers and self.trackers[epoch]:
                try:
                    epoch_co2 = self.trackers[epoch].stop() or 0.0
                except Exception as e:
                    print(f"Advertencia: Error al detener el tracker para la época {epoch}: {e}")
                    epoch_co2 = 0.0
            
            # Calcular tiempo de época
            epoch_time = time.time() - self.epoch_start_time
            
            # Obtener métricas del sistema
            memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
            cpu_usage = psutil.cpu_percent()
            
            # Acumular emisiones totales
            self.total_emissions += epoch_co2
            
            # Guardar datos de esta época
            self.epoch_emissions.append(epoch_co2)
            self.cumulative_emissions.append(self.total_emissions)
            self.epoch_loss.append(loss)
            self.epoch_times.append(epoch_time)
            self.epoch_memory.append(memory_usage)
            self.epoch_cpu.append(cpu_usage)
            
            if val_loss is not None:
                self.epoch_val_loss.append(val_loss)
                
            if val_rmse is not None:
                self.epoch_rmse.append(val_rmse)
                
            if val_mae is not None:
                self.epoch_mae.append(val_mae)
            
            # Rastrear el mejor RMSE y sus emisiones
            if val_rmse is not None and val_rmse < self.best_rmse:
                self.best_rmse = val_rmse
                self.best_rmse_epoch = epoch
                self.best_rmse_emissions = epoch_co2
                self.best_rmse_cumulative_emissions = self.total_emissions
                self.best_rmse_metrics = {
                    'time': epoch_time,
                    'memory': memory_usage,
                    'cpu': cpu_usage,
                    'mae': val_mae
                }
            
            print(f"Epoch {epoch+1}: Time={epoch_time:.2f}s, Memory={memory_usage:.2f}MB, CPU={cpu_usage:.1f}%, RMSE={val_rmse:.4f}, MAE={val_mae:.4f}")
            print(f"Emisiones: {epoch_co2:.8f} kg, Acumulado: {self.total_emissions:.8f} kg")
            
        except Exception as e:
            print(f"Error al medir emisiones en época {epoch}: {e}")
    
    def end_training(self, final_rmse, final_mae, total_training_time, test_time):
        try:
            # Detener el tracker principal y obtener emisiones totales
            total_emissions_main = 0.0
            if self.main_tracker:
                try:
                    total_emissions_main = self.main_tracker.stop() or 0.0
                    print(f"Emisiones totales del tracker principal: {total_emissions_main:.8f} kg")
                except Exception as e:
                    print(f"Error al detener el tracker principal: {e}")
            
            # Asegurarse de que todos los trackers estén detenidos
            for epoch, tracker in self.trackers.items():
                if tracker is not None:
                    try:
                        tracker.stop()
                    except:
                        pass
            
            # Usar las emisiones del tracker principal solo para el total final
            # NO sobrescribir las emisiones de la última época
            if total_emissions_main > 0:
                self.total_emissions = total_emissions_main
                # NO modificar las emisiones acumuladas por época, mantener la progresión natural
            
            # Obtener métricas finales del sistema
            final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
            final_cpu = psutil.cpu_percent()
            
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
                'mae': self.epoch_mae if self.epoch_mae else [None] * len(self.epoch_emissions),
                'time_sec': self.epoch_times,
                'memory_mb': self.epoch_memory,
                'cpu_percent': self.epoch_cpu
            })
            
            # Crear carpetas si no existen
            os.makedirs(f"{self.result_path}/emissions_reports", exist_ok=True)
            os.makedirs(f"{self.result_path}/emissions_plots", exist_ok=True)
            
            emissions_file = f'{self.result_path}/emissions_reports/emissions_metrics_{self.model_name}_{timestamp}.csv'
            df.to_csv(emissions_file, index=False)
            print(f"Métricas de emisiones guardadas en: {emissions_file}")
            
            # Imprimir resumen final como el solicitado
            print("\n=== Final Training Metrics ===")
            for i, row in df.iterrows():
                print(f"Epoch {int(row['epoch'])-1}: Time={row['time_sec']:.2f}s, "
                      f"Memory={row['memory_mb']:.2f}MB, CPU={row['cpu_percent']:.1f}%, "
                      f"RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}")
            
            print("\n=== Final Test Metrics ===")
            print(f"Total Time: {total_training_time:.2f}s (Test: {test_time:.2f}s)")
            print(f"Final Memory: {final_memory:.2f}MB")
            print(f"Final CPU: {final_cpu:.1f}%")
            print(f"Total CO2 Emissions: {self.total_emissions:.8f} kg")
            print(f"RMSE: {final_rmse:.4f}")
            print(f"MAE: {final_mae:.4f}")
            
            # Mostrar información del mejor RMSE durante el entrenamiento
            if self.best_rmse_epoch is not None:
                print(f"\n=== Best Training RMSE ===")
                print(f"Best RMSE: {self.best_rmse:.4f} (Epoch {self.best_rmse_epoch + 1})")
                if self.best_rmse_metrics:
                    print(f"Time: {self.best_rmse_metrics['time']:.2f}s")
                    print(f"Memory: {self.best_rmse_metrics['memory']:.2f}MB")
                    print(f"CPU: {self.best_rmse_metrics['cpu']:.1f}%")
                    if self.best_rmse_metrics['mae'] is not None:
                        print(f"MAE: {self.best_rmse_metrics['mae']:.4f}")
                
                print(f"\n=== Best RMSE and Associated Emissions ===")
                print(f"Best RMSE: {self.best_rmse:.4f} (Epoch {self.best_rmse_epoch + 1})")
                print(f"Emissions at best RMSE: {self.best_rmse_emissions:.8f} kg")
                print(f"Cumulative emissions at best RMSE: {self.best_rmse_cumulative_emissions:.8f} kg")
            
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
            plt.figure(figsize=(15, 12))
            
            plt.subplot(3, 3, 1)
            plt.plot(epochs_range, self.epoch_emissions, 'r-', marker='x')
            plt.title('Emisiones por Época')
            plt.xlabel('Época')
            plt.ylabel('CO₂ Emissions (kg)')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(3, 3, 2)
            plt.plot(epochs_range, self.cumulative_emissions, 'r-', marker='o')
            plt.title('Emisiones Acumuladas por Época')
            plt.xlabel('Época')
            plt.ylabel('CO₂ Emissions (kg)')
            plt.grid(True, alpha=0.3)
            
            if self.epoch_loss:
                plt.subplot(3, 3, 3)
                plt.plot(epochs_range, self.epoch_loss, 'g-', marker='o', label='Train Loss')
                if self.epoch_val_loss:
                    plt.plot(epochs_range, self.epoch_val_loss, 'b-', marker='x', label='Val Loss')
                plt.title('Loss por Época')
                plt.xlabel('Época')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            if self.epoch_rmse:
                plt.subplot(3, 3, 4)
                plt.plot(epochs_range, self.epoch_rmse, 'b-', marker='o')
                plt.title('RMSE por Época')
                plt.xlabel('Época')
                plt.ylabel('RMSE')
                plt.grid(True, alpha=0.3)
            
            if self.epoch_mae:
                plt.subplot(3, 3, 5)
                plt.plot(epochs_range, self.epoch_mae, 'm-', marker='s')
                plt.title('MAE por Época')
                plt.xlabel('Época')
                plt.ylabel('MAE')
                plt.grid(True, alpha=0.3)
            
            if self.epoch_times:
                plt.subplot(3, 3, 6)
                plt.plot(epochs_range, self.epoch_times, 'orange', marker='d')
                plt.title('Tiempo por Época')
                plt.xlabel('Época')
                plt.ylabel('Tiempo (s)')
                plt.grid(True, alpha=0.3)
            
            if self.epoch_memory:
                plt.subplot(3, 3, 7)
                plt.plot(epochs_range, self.epoch_memory, 'purple', marker='^')
                plt.title('Memoria por Época')
                plt.xlabel('Época')
                plt.ylabel('Memoria (MB)')
                plt.grid(True, alpha=0.3)
            
            if self.epoch_cpu:
                plt.subplot(3, 3, 8)
                plt.plot(epochs_range, self.epoch_cpu, 'brown', marker='v')
                plt.title('CPU por Época')
                plt.xlabel('Época')
                plt.ylabel('CPU (%)')
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
X_users = pd.read_pickle("data1m/x_train_ctrpv2_alpha(0.001).pkl")

# Verificar columnas en X_users
print('Columnas en X_users:', X_users.columns)

# 2. Cargar los ratings (etiquetas) desde la nueva base de datos
ratings = pd.read_csv(r'C:\Users\xpati\Documents\TFG\ctrpv2_processed.csv')
ratings.rename(columns={'user_id': 'UID', 'item_id': 'MID', 'rating': 'rate'}, inplace=True)
y = ratings['rate']  # Etiquetas (ratings)

# 3. Combinar características de usuarios y películas
data = ratings  # En este caso, no hay características adicionales de películas

# Verificar columnas en data
print('Columnas en data:', data.columns)

# 3. Combinar con características de usuarios
# Asegúrate de que 'UID' esté en ambos DataFrames
if 'UID' in X_users.columns and 'UID' in data.columns:
    data = pd.merge(data, X_users, on='UID')
else:
    raise KeyError("La columna 'UID' no está presente en ambos DataFrames.")

# 4. Dividir los datos en entrenamiento y prueba
X = data.drop('rate', axis=1)  # Características
y = data['rate']               # Etiquetas (ratings)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Normalizar características numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
emissions_tracker = EmissionsPerEpochTracker(result_path, model_name="GHRS_CTRPv2")

# 8. Define una clase personalizada para TrackingCallback
class TrackingCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, emissions_tracker):
        super().__init__()
        self.val_data = val_data
        self.emissions_tracker = emissions_tracker
        self.val_X, self.val_y = val_data
        
    def on_train_begin(self, logs=None):
        self.emissions_tracker.train_start_time = time.time()
        # Iniciar el tracker principal
        try:
            self.emissions_tracker.main_tracker.start()
            print("Tracker principal de emisiones iniciado")
        except Exception as e:
            print(f"Advertencia: No se pudo iniciar el tracker principal: {e}")
            self.emissions_tracker.main_tracker = None
        
    def on_epoch_begin(self, epoch, logs=None):
        self.emissions_tracker.start_epoch(epoch)
        
    def on_epoch_end(self, epoch, logs=None):
        # Calcular RMSE y MAE en el conjunto de validación
        y_pred = self.model.predict(self.val_X, verbose=0)
        val_rmse = mean_squared_error(self.val_y, y_pred.flatten(), squared=False)
        val_mae = mean_absolute_error(self.val_y, y_pred.flatten())
        
        # Registrar métricas y emisiones
        self.emissions_tracker.end_epoch(
            epoch, 
            logs.get('loss'), 
            logs.get('val_loss'),
            val_rmse,
            val_mae
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
test_start_time = time.time()
y_pred = model.predict(X_test_scaled)
test_time = time.time() - test_start_time

# Calcular tiempo total de entrenamiento
total_training_time = time.time() - emissions_tracker.train_start_time

# 13. Evaluar el modelo con métricas finales
rmse = mean_squared_error(y_test, y_pred.flatten(), squared=False)
mae = mean_absolute_error(y_test, y_pred.flatten())

print(f"RMSE final: {rmse}")
print(f"MAE final: {mae}")

# 14. Generar gráficos finales de emisiones vs. rendimiento
emissions_tracker.end_training(rmse, mae, total_training_time, test_time)
