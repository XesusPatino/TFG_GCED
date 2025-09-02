import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import pandas as pd
from sklearn.model_selection import train_test_split
from model import LRML
import random
from codecarbon import EmissionsTracker
import numpy as np
import time
import psutil
import math
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import traceback


# Clase para seguimiento de emisiones por época
class EmissionsPerEpochTracker:
    def __init__(self, result_path, model_name="LRML"):
        self.result_path = result_path
        self.model_name = model_name
        self.epoch_emissions = []
        self.cumulative_emissions = []
        self.epoch_rmse = []
        self.epoch_mae = []
        self.epoch_loss = []
        self.total_emissions = 0.0
        self.trackers = {}
        
        # Variables para rastrear el mejor RMSE y sus emisiones
        self.best_rmse = float('inf')
        self.best_rmse_epoch = None
        self.best_rmse_emissions = None
        self.best_rmse_cumulative_emissions = None
        self.best_rmse_time = None
        self.best_rmse_memory = None
        self.best_rmse_cpu = None
        self.best_rmse_mae = None
        
        # Crear directorios para emisiones
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
            allow_multiple_runs=True  # Permitir múltiples instancias
        )
        
        try:
            self.trackers[epoch].start()
        except Exception as e:
            print(f"Advertencia: No se pudo iniciar el tracker para la época {epoch}: {e}")
            self.trackers[epoch] = None
    
    def end_epoch(self, epoch, loss, rmse=None, mae=None, epoch_time=None, memory=None, cpu=None):
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
                
            if rmse is not None:
                self.epoch_rmse.append(rmse)
                # Rastrear el mejor RMSE y sus métricas asociadas
                if rmse < self.best_rmse:
                    self.best_rmse = rmse
                    self.best_rmse_epoch = epoch
                    self.best_rmse_emissions = epoch_co2
                    self.best_rmse_cumulative_emissions = self.total_emissions
                    self.best_rmse_time = epoch_time
                    self.best_rmse_memory = memory
                    self.best_rmse_cpu = cpu
                    self.best_rmse_mae = mae
            
            if mae is not None:
                self.epoch_mae.append(mae)
            
            print(f"Epoch {epoch+1} - Emisiones: {epoch_co2:.8f} kg, Acumulado: {self.total_emissions:.8f} kg")
            
        except Exception as e:
            print(f"Error al medir emisiones en época {epoch}: {e}")
    
    def end_training(self, final_rmse=None, final_mae=None):
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
                'rmse': self.epoch_rmse if self.epoch_rmse else [None] * len(self.epoch_emissions),
                'mae': self.epoch_mae if self.epoch_mae else [None] * len(self.epoch_emissions)
            })
            
            emissions_file = f'{self.result_path}/emissions_reports/emissions_metrics_{self.model_name}_{timestamp}.csv'
            df.to_csv(emissions_file, index=False)
            print(f"Métricas de emisiones guardadas en: {emissions_file}")
            
            # Graficar las relaciones
            self.plot_emissions_vs_metrics(epochs_range, timestamp, final_rmse, final_mae)
            
            # Mostrar información del mejor RMSE y sus emisiones
            if self.best_rmse_epoch is not None:
                print(f"\n=== Best RMSE and Associated Emissions ===")
                print(f"Best RMSE: {self.best_rmse:.4f} (Epoch {self.best_rmse_epoch})")
                print(f"Emissions at best RMSE: {self.best_rmse_emissions:.8f} kg")
                print(f"Cumulative emissions at best RMSE: {self.best_rmse_cumulative_emissions:.8f} kg")
            
        except Exception as e:
            print(f"Error al generar gráficos de emisiones: {e}")
            traceback.print_exc()
    
    def plot_emissions_vs_metrics(self, epochs_range, timestamp, final_rmse=None, final_mae=None):
        """Genera gráficos para emisiones vs métricas"""
        
        try:
            from matplotlib.ticker import ScalarFormatter
            
            # 1. Gráfico combinado: Emisiones por época y acumulativas
            plt.figure(figsize=(15, 10))
            
            # Emisiones por época
            plt.subplot(2, 3, 1)
            plt.plot(epochs_range, self.epoch_emissions, 'r-', marker='x')
            plt.title('Emisiones por Época')
            plt.xlabel('Época')
            plt.ylabel('CO₂ Emissions (kg)')
            plt.grid(True, alpha=0.3)
            
            # Emisiones acumuladas
            plt.subplot(2, 3, 2)
            plt.plot(epochs_range, self.cumulative_emissions, 'r-', marker='o')
            plt.title('Emisiones Acumuladas por Época')
            plt.xlabel('Época')
            plt.ylabel('CO₂ Emissions (kg)')
            plt.grid(True, alpha=0.3)
            
            # Loss
            plt.subplot(2, 3, 3)
            plt.plot(epochs_range, self.epoch_loss, 'g-', marker='o', label='Loss')
            plt.title('Loss por Época')
            plt.xlabel('Época')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            
            # RMSE
            if self.epoch_rmse:
                plt.subplot(2, 3, 4)
                plt.plot(epochs_range, self.epoch_rmse, 'b-', marker='o')
                plt.title('RMSE por Época')
                plt.xlabel('Época')
                plt.ylabel('RMSE')
                plt.grid(True, alpha=0.3)
            
            # MAE
            if self.epoch_mae:
                plt.subplot(2, 3, 5)
                plt.plot(epochs_range, self.epoch_mae, 'm-', marker='o')
                plt.title('MAE por Época')
                plt.xlabel('Época')
                plt.ylabel('MAE')
                plt.grid(True, alpha=0.3)
            
            # RMSE vs Emisiones acumuladas
            if self.epoch_rmse:
                plt.subplot(2, 3, 6)
                plt.plot(self.cumulative_emissions, self.epoch_rmse, 'c-', marker='o')
                plt.title('RMSE vs Emisiones Acumuladas')
                plt.xlabel('Emisiones Acumuladas (kg)')
                plt.ylabel('RMSE')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            file_path = f'{self.result_path}/emissions_plots/metrics_by_epoch_{self.model_name}_{timestamp}.png'
            plt.savefig(file_path)
            plt.close()
            print(f"Gráfico de métricas por época guardado en: {file_path}")
            
            # 2. RMSE vs Emisiones acumuladas (en gramos para mejor visualización)
            if self.epoch_rmse:
                plt.figure(figsize=(10, 6))
                # Convertir kg a g (×1000) para mejor visualización
                emissions_in_g = [e * 1000 for e in self.cumulative_emissions]
                plt.plot(emissions_in_g, self.epoch_rmse, 'b-', marker='o')
                
                # Configurar límites del eje x para mostrar desde 0 hasta un poco más del máximo
                plt.xlim(0, max(emissions_in_g) * 1.1)
                
                # Añadir etiquetas con el número de época
                for i, (emissions, rmse) in enumerate(zip(emissions_in_g, self.epoch_rmse)):
                    plt.annotate(f"{i+1}", (emissions, rmse), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9)
                    
                plt.xlabel('Emisiones de CO₂ acumuladas (g)')
                plt.ylabel('RMSE')
                plt.title('Relación entre Emisiones Acumuladas y RMSE')
                plt.grid(True, alpha=0.3)
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_rmse_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path)
                plt.close()
                print(f"Gráfico RMSE vs emisiones guardado en: {file_path}")
            
            # 3. MAE vs Emisiones acumuladas (también en gramos)
            if self.epoch_mae:
                plt.figure(figsize=(10, 6))
                emissions_in_g = [e * 1000 for e in self.cumulative_emissions]
                plt.plot(emissions_in_g, self.epoch_mae, 'm-', marker='o')
                
                # Configurar límites del eje x
                plt.xlim(0, max(emissions_in_g) * 1.1)
                
                # Añadir etiquetas con el número de época
                for i, (emissions, mae) in enumerate(zip(emissions_in_g, self.epoch_mae)):
                    plt.annotate(f"{i+1}", (emissions, mae), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9)
                    
                plt.xlabel('Emisiones de CO₂ acumuladas (g)')
                plt.ylabel('MAE')
                plt.title('Relación entre Emisiones Acumuladas y MAE')
                plt.grid(True, alpha=0.3)
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_mae_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path)
                plt.close()
                print(f"Gráfico MAE vs emisiones guardado en: {file_path}")
                
        except Exception as e:
            print(f"Error al generar los gráficos: {e}")
            traceback.print_exc()


# Función para calcular MAE
def calculate_mae(model, sess, test_data):
    user_ids = [x[0] for x in test_data]
    item_ids = [x[1] for x in test_data]
    true_ratings = [x[2] for x in test_data]
    
    feed_dict = {
        model.user_input: user_ids,
        model.item_input: item_ids,
        model.dropout: 1.0
    }
    
    predicted_ratings = sess.run(model.predict_op, feed_dict=feed_dict)
    predicted_ratings = predicted_ratings.flatten()
    
    # Calcular MAE
    mae = np.mean(np.abs(np.array(predicted_ratings) - np.array(true_ratings)))
    return mae

# Función para calcular RMSE
def calculate_rmse(model, sess, test_data):
    user_ids = [x[0] for x in test_data]
    item_ids = [x[1] for x in test_data]
    true_ratings = [x[2] for x in test_data]
    
    feed_dict = {
        model.user_input: user_ids,  # Cambiado de user_indices a user_input
        model.item_input: item_ids,  # Cambiado de item_indices a item_input
        model.dropout: 1.0  # Durante la evaluación, no aplicamos dropout
    }
    
    # Asumiendo que tu modelo tiene un método de predicción
    predicted_ratings = sess.run(model.predict_op, feed_dict=feed_dict)  # Cambiado de predictions a predict_op
    # Flatten the predictions array if it's not 1D
    predicted_ratings = predicted_ratings.flatten()
    
    # Calcular RMSE
    rmse = np.sqrt(np.mean(np.square(np.array(predicted_ratings) - np.array(true_ratings))))
    return rmse


# Cargar los datos de Kiva
ratings = pd.read_csv('C:/Users/xpati/Documents/TFG/kiva_ml17.csv')

# Preprocesar los datos
user_ids = ratings['user_id'].unique()
item_ids = ratings['item_id'].unique()

user_id_map = {user_id: i for i, user_id in enumerate(user_ids)}
item_id_map = {item_id: i for i, item_id in enumerate(item_ids)}

ratings['user_id'] = ratings['user_id'].map(user_id_map)
ratings['item_id'] = ratings['item_id'].map(item_id_map)

# Dividir los datos en entrenamiento y prueba
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Convertir a listas de tuplas (user_id, item_id, rating)
train_data = list(zip(train_data['user_id'], train_data['item_id'], train_data['rating']))
test_data = list(zip(test_data['user_id'], test_data['item_id'], test_data['rating']))

# Normalizar los ratings (escalarlos a [0,1])
min_rating = min(r for _, _, r in train_data + test_data)
max_rating = max(r for _, _, r in train_data + test_data)
print(f"Rango de ratings original: {min_rating} a {max_rating}")
train_data = [(u, i, (r - min_rating) / (max_rating - min_rating)) 
              for u, i, r in train_data]
test_data = [(u, i, (r - min_rating) / (max_rating - min_rating)) 
             for u, i, r in test_data]

# Definir los argumentos del modelo con valores mejorados
class Args:
    def __init__(self):
        self.std = 0.01
        self.embedding_size = 128      # Aumentado a 150
        self.num_mem = 30             # Aumentado a 30 memorias
        self.dropout = 0.1            # Reducido aún más para mejorar convergencia
        self.margin = 0.3             # Reducido más para facilitar el aprendizaje inicial
        self.l2_reg = 1e-2            # Ajustado para mejor generalización
        self.opt = 'Adam'
        self.learn_rate = 1e-3       # Aumentado para convergencia más rápida
        self.clip_norm = 1.0
        self.constraint = True
        self.rnn_type = 'PAIR'

args = Args()

# Crear el modelo
num_users = len(user_ids)
num_items = len(item_ids)
model = LRML(num_users, num_items, args)

# Pre-calcular items positivos por usuario para muestreo negativo
user_positive_items = defaultdict(set)
for u, i, r in train_data:
    if r >= 0.7:  # Considerar ratings altos como positivos (escala normalizada)
        user_positive_items[u].add(i)

# Calcular los ítems más populares
item_popularity = defaultdict(int)
for u, i, r in train_data:
    if r >= 0.5:  # Considerar ratings positivos
        item_popularity[i] += 1

popular_items = [item for item, count in sorted(
    item_popularity.items(), key=lambda x: x[1], reverse=True
)][:1000]  # Top 1000 ítems

# Crear carpeta para resultados
result_path = './results'
os.makedirs(result_path, exist_ok=True)

# Iniciar el timer para medir el tiempo total
start_time = time.time()

# Eliminar el archivo de lock si existe para evitar el error de múltiples instancias
lock_file = "C:\\Users\\xpati\\AppData\\Local\\Temp\\.codecarbon.lock"
if os.path.exists(lock_file):
    try:
        os.remove(lock_file)
        print(f"Archivo de bloqueo eliminado: {lock_file}")
    except Exception as e:
        print(f"No se pudo eliminar el archivo de bloqueo: {e}")

# Iniciar el tracker de emisiones global (para el total)
global_tracker = EmissionsTracker(
    output_dir=f"{result_path}/emissions_reports",
    allow_multiple_runs=True  # Permitir múltiples ejecuciones
)
global_tracker.start()

# Inicializar el tracker de emisiones por época
epoch_tracker = EmissionsPerEpochTracker(result_path, model_name="LRML")

# IMPORTANTE: La sesión debe contener todo el código de entrenamiento y evaluación
with tf.compat.v1.Session(graph=model.graph) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    '''
    # Implementar entrenamiento con batch adaptativo
    initial_batch_size = 1024
    max_batch_size = 4096
    batch_size = initial_batch_size
    '''
    
    best_rmse = float('inf')
    patience = 10
    patience_counter = 0
    
    # Para medir CPU correctamente
    cpu_measurements = []
    process = psutil.Process()
    process.cpu_percent()  # Primera llamada para inicializar
    
    # Crear un saver para guardar el mejor modelo
    saver = tf.compat.v1.train.Saver()
    
    all_rmse = []
    all_mae = []
    
    # Variables para el resumen final de entrenamiento
    train_metrics = []
    
    # Variables para rastrear el mejor RMSE
    best_rmse = float('inf')
    best_rmse_epoch = None
    best_rmse_metrics = None
    
    for epoch in range(50):
        # Iniciar tracker para esta época
        epoch_tracker.start_epoch(epoch)
        epoch_start_time = time.time()
        
        # Medir CPU al inicio de la época
        epoch_start_cpu = psutil.cpu_percent(interval=None)  # CPU del sistema
        
        '''
        # Aumentar tamaño de batch gradualmente para acelerar convergencia final
        if epoch > 50 and epoch % 10 == 0:
            batch_size = min(batch_size * 2, max_batch_size)
        '''

        batch_size = 1024  # Mantener tamaño de batch constante para evitar problemas de memoria
        
        # Shuffle the training data for each epoch
        random.shuffle(train_data)
        
        total_batches = len(train_data) // batch_size
        total_epoch_loss = 0
        
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(train_data))
            batch = train_data[start_idx:end_idx]
            
            # Mejor estrategia de muestreo negativo - crucial para LRML
            if 'PAIR' in model.args.rnn_type:
                neg_batch = []
                for x in batch:
                    user_id, pos_item, rating = x
                    # Usar muestreo negativo informado con sesgo hacia ítems populares
                    if random.random() < 0.7:
                        # 70% del tiempo, seleccionar aleatoriamente entre todos los ítems
                        attempts = 0
                        while attempts < 10:  # Intentar hasta 10 veces
                            neg_item = random.randint(0, num_items - 1)
                            if neg_item != pos_item and neg_item not in user_positive_items[user_id]:
                                break
                            attempts += 1
                    else:
                        # 30% del tiempo, elegir entre los ítems más populares
                        attempts = 0
                        while attempts < 5:
                            neg_item = popular_items[random.randint(0, min(500, len(popular_items)-1))]
                            if neg_item != pos_item and neg_item not in user_positive_items[user_id]:
                                break
                            attempts += 1
                        
                    neg_batch.append([user_id, neg_item, rating])
            else:
                neg_batch = None
                
            feed_dict, _ = model.get_feed_dict(batch, neg_batch, mode='training')
            _, batch_loss = sess.run([model.train_op, model.cost], feed_dict=feed_dict)
            total_epoch_loss += batch_loss
        
        avg_epoch_loss = total_epoch_loss/total_batches
        print(f'Epoch {epoch + 1}, Loss: {avg_epoch_loss:.4f}')
        
        # Medir CPU al final de la época (promedio durante un intervalo corto)
        epoch_cpu = psutil.cpu_percent(interval=1.0)  # Medir durante 1 segundo
        cpu_measurements.append(epoch_cpu)
        print(f"CPU: {epoch_cpu:.1f}%")
        
        # Calcular métricas de evaluación en cada época
        print("\n--- Métricas de evaluación ---")
        
        # Evaluar en una muestra más pequeña para ahorrar tiempo durante el entrenamiento
        eval_sample = random.sample(test_data, min(len(test_data), 5000))
        
        # Calcular RMSE
        rmse = calculate_rmse(model, sess, eval_sample)
        print(f"RMSE: {rmse:.4f}")
        all_rmse.append(rmse)
        
        # Calcular MAE
        mae = calculate_mae(model, sess, eval_sample)
        print(f"MAE: {mae:.4f}")
        all_mae.append(mae)
        
        # Medir memoria actual
        current_memory = process.memory_info().rss / (1024 * 1024)  # En MB
        
        # Calcular el tiempo de época
        epoch_time = time.time() - epoch_start_time
        
        # Finalizar tracking para esta época y guardar métricas
        epoch_tracker.end_epoch(
            epoch=epoch, 
            loss=avg_epoch_loss, 
            rmse=rmse,
            mae=mae,
            epoch_time=epoch_time,
            memory=current_memory,
            cpu=epoch_cpu
        )
        
        print(f"Tiempo de época: {epoch_time:.2f} segundos")
        print(f"Memoria: {current_memory:.2f} MB")
        print(f"CPU: {epoch_cpu:.1f}%")
        
        # Guardar métricas de esta época para el resumen final
        current_epoch_metrics = {
            'epoch': epoch,
            'time': epoch_time,
            'memory': current_memory,
            'cpu': epoch_cpu,
            'rmse': rmse,
            'mae': mae
        }
        train_metrics.append(current_epoch_metrics)
        
        # Rastrear el mejor RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            best_rmse_epoch = epoch
            best_rmse_metrics = current_epoch_metrics.copy()
        
        # Early stopping más sofisticado (ahora basado en RMSE en lugar de recall)
        if epoch == 0 or rmse < min(all_rmse[:-1]):  # Si es la primera época o si RMSE mejoró
            print(f"¡Nuevo mejor RMSE: {rmse:.4f}!")
            # Guardar el mejor modelo si es significativamente bueno
            if rmse < 1.0:  # Guardar solo si supera un umbral mínimo
                saver.save(sess, './best_model')
                print(f"¡Modelo guardado con RMSE: {rmse:.4f}!")
    
    # Al finalizar el entrenamiento, evaluar sobre todo el conjunto de pruebas
    # Detener el tracker de CodeCarbon global
    emissions = global_tracker.stop()
    total_time = time.time() - start_time
    
    # Tomar una última medición de CPU para asegurar un valor válido
    final_cpu_percent = psutil.cpu_percent(interval=1.0)  # Medir durante 1 segundo
    
    # Calcular métricas finales
    print("\n=========== MÉTRICAS FINALES ===========")
    
    # Calcular RMSE final
    test_start_time = time.time()
    final_rmse = calculate_rmse(model, sess, test_data)
    print(f"RMSE: {final_rmse:.4f}")
    
    # Calcular MAE final
    final_mae = calculate_mae(model, sess, test_data)
    print(f"MAE: {final_mae:.4f}")
    
    test_time = time.time() - test_start_time
    
    # Métricas del sistema
    memory_usage = process.memory_info().rss / (1024 * 1024)  # En MB
    
    # Usar el promedio de las mediciones de CPU durante el entrenamiento
    avg_cpu = np.mean(cpu_measurements) if cpu_measurements else 0.0
    
    # Mostrar métricas del sistema y tiempo
    print(f"Memoria utilizada: {memory_usage:.2f} MB")
    print(f"CPU utilizada (promedio): {avg_cpu:.1f}%")
    print(f"CPU utilizada (último valor): {final_cpu_percent:.1f}%")
    print(f"Emisiones totales: {emissions:.6f} kg CO2")
    print(f"Tiempo total de ejecución: {total_time:.2f} segundos")
    print("=========================================")
    
    # Mostrar resumen final como en el modelo de referencia
    print("\n=== Final Training Metrics ===")
    for m in train_metrics:
        print(f"Epoch {m['epoch']}: Time={m['time']:.2f}s, "
              f"Memory={m['memory']:.2f}MB, CPU={m['cpu']:.1f}%, "
              f"RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}")
    
    # Mostrar información del mejor RMSE durante el entrenamiento
    if best_rmse_epoch is not None:
        print(f"\n=== Best Training RMSE ===")
        print(f"Best RMSE: {best_rmse:.4f} (Epoch {best_rmse_epoch})")
        if best_rmse_metrics:
            print(f"Time: {best_rmse_metrics['time']:.2f}s")
            print(f"Memory: {best_rmse_metrics['memory']:.2f}MB")
            print(f"CPU: {best_rmse_metrics['cpu']:.1f}%")
            print(f"MAE: {best_rmse_metrics['mae']:.4f}")
    
    print("\n=== Final Test Metrics ===")
    print(f"Total Time: {total_time:.2f}s (Test: {test_time:.2f}s)")
    print(f"Final Memory: {memory_usage:.2f}MB")
    print(f"Final CPU: {final_cpu_percent:.1f}%")
    print(f"RMSE: {final_rmse:.4f}")
    print(f"MAE: {final_mae:.4f}")
    
    # Finalizar el seguimiento de emisiones por época
    epoch_tracker.end_training(final_rmse=final_rmse, final_mae=final_mae)

    # Guardar las métricas finales en un archivo CSV
    metrics_df = pd.DataFrame({
        'model': ['LRML'],
        'final_rmse': [final_rmse],
        'final_mae': [final_mae],
        'total_emissions_kg': [emissions],
        'total_time_seconds': [total_time],
        'test_time_seconds': [test_time],
        'average_cpu_percent': [avg_cpu],
        'final_cpu_percent': [final_cpu_percent],
        'memory_usage_mb': [memory_usage]
    })
    metrics_df.to_csv(f'{result_path}/final_metrics_LRML_{time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)
    print(f"Métricas finales guardadas en: {result_path}/final_metrics_LRML_{time.strftime('%Y%m%d-%H%M%S')}.csv")