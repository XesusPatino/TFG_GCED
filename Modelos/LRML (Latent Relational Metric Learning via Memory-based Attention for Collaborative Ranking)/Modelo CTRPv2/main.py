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
        self.epoch_recall = []
        self.epoch_ndcg = []
        self.epoch_loss = []
        self.total_emissions = 0.0
        self.trackers = {}
        
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
    
    def end_epoch(self, epoch, loss, rmse=None, recall=None, ndcg=None):
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
            
            if recall is not None:
                self.epoch_recall.append(recall)
                
            if ndcg is not None:
                self.epoch_ndcg.append(ndcg)
            
            print(f"Epoch {epoch+1} - Emisiones: {epoch_co2:.8f} kg, Acumulado: {self.total_emissions:.8f} kg")
            
        except Exception as e:
            print(f"Error al medir emisiones en época {epoch}: {e}")
    
    def end_training(self, final_rmse=None, final_recall=None, final_ndcg=None):
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
                'recall': self.epoch_recall if self.epoch_recall else [None] * len(self.epoch_emissions),
                'ndcg': self.epoch_ndcg if self.epoch_ndcg else [None] * len(self.epoch_emissions)
            })
            
            emissions_file = f'{self.result_path}/emissions_reports/emissions_metrics_{self.model_name}_{timestamp}.csv'
            df.to_csv(emissions_file, index=False)
            print(f"Métricas de emisiones guardadas en: {emissions_file}")
            
            # Graficar las relaciones
            self.plot_emissions_vs_metrics(epochs_range, timestamp, final_rmse, final_recall, final_ndcg)
            
        except Exception as e:
            print(f"Error al generar gráficos de emisiones: {e}")
            traceback.print_exc()
    
    def plot_emissions_vs_metrics(self, epochs_range, timestamp, final_rmse=None, final_recall=None, final_ndcg=None):
        """Genera gráficos para emisiones vs métricas"""
        
        try:
            from matplotlib.ticker import ScalarFormatter
            
            # 1. Gráfico combinado: Emisiones por época y acumulativas
            plt.figure(figsize=(15, 12))
            
            # Emisiones por época
            plt.subplot(3, 2, 1)
            plt.plot(epochs_range, self.epoch_emissions, 'r-', marker='x')
            plt.title('Emisiones por Época')
            plt.xlabel('Época')
            plt.ylabel('CO₂ Emissions (kg)')
            plt.grid(True, alpha=0.3)
            
            # Emisiones acumuladas
            plt.subplot(3, 2, 2)
            plt.plot(epochs_range, self.cumulative_emissions, 'r-', marker='o')
            plt.title('Emisiones Acumuladas por Época')
            plt.xlabel('Época')
            plt.ylabel('CO₂ Emissions (kg)')
            plt.grid(True, alpha=0.3)
            
            # Loss
            plt.subplot(3, 2, 3)
            plt.plot(epochs_range, self.epoch_loss, 'g-', marker='o', label='Loss')
            plt.title('Loss por Época')
            plt.xlabel('Época')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            
            # RMSE
            if self.epoch_rmse:
                plt.subplot(3, 2, 4)
                plt.plot(epochs_range, self.epoch_rmse, 'b-', marker='o')
                plt.title('RMSE por Época')
                plt.xlabel('Época')
                plt.ylabel('RMSE')
                plt.grid(True, alpha=0.3)
            
            # Recall y NDCG
            if self.epoch_recall:
                plt.subplot(3, 2, 5)
                plt.plot(epochs_range, self.epoch_recall, 'm-', marker='o')
                plt.title('Recall@10 por Época')
                plt.xlabel('Época')
                plt.ylabel('Recall@10')
                plt.grid(True, alpha=0.3)
            
            if self.epoch_ndcg:
                plt.subplot(3, 2, 6)
                plt.plot(epochs_range, self.epoch_ndcg, 'c-', marker='o')
                plt.title('NDCG@10 por Época')
                plt.xlabel('Época')
                plt.ylabel('NDCG@10')
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
            
            # 3. Recall vs Emisiones acumuladas (también en gramos)
            if self.epoch_recall:
                plt.figure(figsize=(10, 6))
                emissions_in_g = [e * 1000 for e in self.cumulative_emissions]
                plt.plot(emissions_in_g, self.epoch_recall, 'm-', marker='o')
                
                # Configurar límites del eje x
                plt.xlim(0, max(emissions_in_g) * 1.1)
                
                # Añadir etiquetas con el número de época
                for i, (emissions, recall) in enumerate(zip(emissions_in_g, self.epoch_recall)):
                    plt.annotate(f"{i+1}", (emissions, recall), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9)
                    
                plt.xlabel('Emisiones de CO₂ acumuladas (g)')
                plt.ylabel('Recall@10')
                plt.title('Relación entre Emisiones Acumuladas y Recall@10')
                plt.grid(True, alpha=0.3)
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_recall_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path)
                plt.close()
                print(f"Gráfico Recall vs emisiones guardado en: {file_path}")
                
        except Exception as e:
            print(f"Error al generar los gráficos: {e}")
            traceback.print_exc()


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

# Función para calcular Recall@K y NDCG@K
def calculate_ranking_metrics(model, sess, test_data, k=10):
    # Agrupar por usuario
    user_test_items = defaultdict(list)
    for user_id, item_id, rating in test_data:
        # Cambiar el umbral para considerar un ítem como relevante en datos normalizados
        if rating >= 0.6:  # 0.6 en lugar de 4 ya que normalizamos a [0,1]
            user_test_items[user_id].append(item_id)
    
    recalls = []
    ndcgs = []
    
    # Verificar que hay usuarios con ítems relevantes
    if len(user_test_items) == 0:
        print("¡Advertencia! No se encontraron ítems relevantes en los datos de prueba.")
        return 0.0, 0.0
    
    num_users_processed = 0
    for user_id, relevant_items in user_test_items.items():
        if not relevant_items:
            continue
            
        # Predecir scores para todos los ítems para este usuario
        items_to_rank = list(range(num_items))
        users_to_rank = [user_id] * len(items_to_rank)
        
        # Para evitar errores de memoria en datasets grandes, limitar el número de ítems si es necesario
        if len(items_to_rank) > 5000:  # Procesar en lotes si hay demasiados ítems
            items_to_rank = random.sample(items_to_rank, 5000)
            users_to_rank = [user_id] * 5000
        
        feed_dict = {
            model.user_input: users_to_rank,
            model.item_input: items_to_rank,
            model.dropout: 1.0
        }
        
        scores = sess.run(model.predict_op, feed_dict=feed_dict)
        scores = scores.flatten()
        
        # Crear pares de (ítem, score) y ordenar por score en orden descendente
        item_scores = list(zip(items_to_rank, scores))
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Obtener los top-K ítems
        top_k_items = [x[0] for x in item_scores[:k]]
        
        # Calcular Recall@K
        hits = len(set(top_k_items) & set(relevant_items))
        recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
        recalls.append(recall)
        
        # Calcular NDCG@K
        dcg = 0
        idcg = 0
        for i, item in enumerate(top_k_items):
            if item in relevant_items:
                dcg += 1 / math.log2(i + 2)  # i+2 porque i empieza en 0
                
        # Calcular IDCG (normalización)
        for i in range(min(len(relevant_items), k)):
            idcg += 1 / math.log2(i + 2)
            
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)
        
        # Limitar el número de usuarios a procesar para evitar tiempos de ejecución excesivos
        num_users_processed += 1
        if num_users_processed >= 500:  # Procesar máximo 500 usuarios
            break
    
    # Verificar que hay resultados válidos antes de calcular el promedio
    if len(recalls) == 0 or len(ndcgs) == 0:
        print("¡Advertencia! No se pudieron calcular métricas de ranking válidas.")
        return 0.0, 0.0
        
    return np.mean(recalls), np.mean(ndcgs)


# Cargar los datos de MovieLens
ratings = pd.read_csv('C:/Users/xpati/Documents/TFG/ctrpv2_processed.csv')

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
    
    best_recall = 0
    patience = 10
    patience_counter = 0
    
    # Para medir CPU correctamente
    cpu_measurements = []
    process = psutil.Process()
    process.cpu_percent()  # Primera llamada para inicializar
    
    # Crear un saver para guardar el mejor modelo
    saver = tf.compat.v1.train.Saver()
    
    all_rmse = []
    all_recall = []
    all_ndcg = []
    
    for epoch in range(50):
        # Iniciar tracker para esta época
        epoch_tracker.start_epoch(epoch)
        epoch_start_time = time.time()
        
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
        
        # Al final de cada época, medir CPU
        cpu_measurements.append(process.cpu_percent())
        print(f"CPU: {cpu_measurements[-1]:.2f}%")
        
        # Calcular métricas de evaluación en cada época
        print("\n--- Métricas de evaluación ---")
        
        # Evaluar en una muestra más pequeña para ahorrar tiempo durante el entrenamiento
        eval_sample = random.sample(test_data, min(len(test_data), 5000))
        
        # Calcular RMSE
        rmse = calculate_rmse(model, sess, eval_sample)
        print(f"RMSE: {rmse:.4f}")
        all_rmse.append(rmse)
        
        # Calcular Recall@10 y NDCG@10
        recall, ndcg = calculate_ranking_metrics(model, sess, eval_sample, k=10)
        print(f"Recall@10: {recall:.4f}")
        print(f"NDCG@10: {ndcg:.4f}")
        all_recall.append(recall)
        all_ndcg.append(ndcg)
        
        # Finalizar tracking para esta época y guardar métricas
        epoch_tracker.end_epoch(
            epoch=epoch, 
            loss=avg_epoch_loss, 
            rmse=rmse,
            recall=recall, 
            ndcg=ndcg
        )
        
        epoch_time = time.time() - epoch_start_time
        print(f"Tiempo de época: {epoch_time:.2f} segundos")
        
        # Early stopping más sofisticado
        if recall > best_recall:
            best_recall = recall
            patience_counter = 0
            print(f"¡Nuevo mejor Recall: {recall:.4f}!")
            # Guardar el mejor modelo si es significativamente mejor
            if recall > 0.03:  # Guardar solo si supera un umbral mínimo
                saver.save(sess, './best_model')
                print(f"¡Modelo guardado con Recall: {recall:.4f}!")
    
    # Al finalizar el entrenamiento, evaluar sobre todo el conjunto de pruebas
    # Detener el tracker de CodeCarbon global
    emissions = global_tracker.stop()
    total_time = time.time() - start_time
    
    # Tomar una última medición de CPU para asegurar un valor válido
    process = psutil.Process()
    process.cpu_percent() # Primera llamada para inicializar
    time.sleep(0.5)  # Esperar un momento para obtener una medición válida
    cpu_percent = process.cpu_percent() # Segunda llamada para obtener el valor
    
    # Calcular métricas finales
    print("\n=========== MÉTRICAS FINALES ===========")
    
    # Calcular RMSE final
    final_rmse = calculate_rmse(model, sess, test_data)
    print(f"RMSE: {final_rmse:.4f}")
    
    # Calcular Recall@10 y NDCG@10 finales
    final_recall, final_ndcg = calculate_ranking_metrics(model, sess, test_data, k=10)
    print(f"Recall@10: {final_recall:.4f}")
    print(f"NDCG@10: {final_ndcg:.4f}")
    
    # Métricas del sistema
    memory_usage = process.memory_info().rss / (1024 * 1024)  # En MB
    
    # Usar el promedio de las mediciones de CPU durante el entrenamiento
    avg_cpu = np.mean(cpu_measurements) if cpu_measurements else 0.0
    
    # Mostrar métricas del sistema y tiempo
    print(f"Memoria utilizada: {memory_usage:.2f} MB")
    print(f"CPU utilizada (promedio): {avg_cpu:.2f}%")
    print(f"CPU utilizada (último valor): {cpu_percent:.2f}%")
    print(f"Emisiones totales: {emissions:.6f} kg CO2")
    print(f"Tiempo total de ejecución: {total_time:.2f} segundos")
    print("=========================================")
    
    # Finalizar el seguimiento de emisiones por época
    epoch_tracker.end_training(final_rmse=final_rmse, final_recall=final_recall, final_ndcg=final_ndcg)

    # Guardar las métricas finales en un archivo CSV
    metrics_df = pd.DataFrame({
        'model': ['LRML'],
        'final_rmse': [final_rmse],
        'final_recall': [final_recall],
        'final_ndcg': [final_ndcg],
        'total_emissions_kg': [emissions],
        'total_time_seconds': [total_time],
        'average_cpu_percent': [avg_cpu],
        'memory_usage_mb': [memory_usage]
    })
    metrics_df.to_csv(f'{result_path}/final_metrics_LRML_{time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)
    print(f"Métricas finales guardadas en: {result_path}/final_metrics_LRML_{time.strftime('%Y%m%d-%H%M%S')}.csv")