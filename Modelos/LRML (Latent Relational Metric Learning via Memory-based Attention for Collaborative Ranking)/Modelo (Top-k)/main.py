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
import argparse


# Configuración de argumentos
parser = argparse.ArgumentParser(description='LRML con enfoque Top-K')
parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--num_mem', type=int, default=30)
parser.add_argument('--train_epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--learn_rate', type=float, default=1e-3)
parser.add_argument('--l2_reg', type=float, default=1e-2)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--margin', type=float, default=0.3,
                   help="margen para la función de pérdida BPR")
parser.add_argument('--top_k_values', type=str, default="5,10,20",
                   help="valores de K para las métricas top-K separados por comas")

args = parser.parse_args()
tf.random.set_seed(42)
np.random.seed(42)

# Parsear valores de K para las métricas
top_k_values = [int(k) for k in args.top_k_values.split(',')]


# Clase para seguimiento de métricas del sistema
class SystemMetricsTracker:
    def __init__(self):
        self.train_metrics = []
        self.test_metrics = {}
        self.start_time = time.time()
        
    def start_epoch(self, epoch):
        self.epoch_start_time = time.time()
        self.current_epoch_metrics = {
            'epoch': epoch,
            'memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'cpu_usage_percent': psutil.cpu_percent(),
        }
        
    def end_epoch(self, epoch, loss, rmse=None, recall=None, ndcg=None):
        epoch_time = time.time() - self.epoch_start_time
        self.current_epoch_metrics['epoch_time_sec'] = epoch_time
        self.current_epoch_metrics['loss'] = loss
        if rmse is not None:
            self.current_epoch_metrics['rmse'] = rmse
        if recall is not None:
            for k, value in recall.items():
                self.current_epoch_metrics[f'recall@{k}'] = value
        if ndcg is not None:
            for k, value in ndcg.items():
                self.current_epoch_metrics[f'ndcg@{k}'] = value
            
        self.train_metrics.append(self.current_epoch_metrics)
        
        # Imprimir resumen de época
        print(f"\nEpoch {epoch+1} Metrics:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Memory: {self.current_epoch_metrics['memory_usage_mb']:.2f}MB")
        print(f"  CPU: {self.current_epoch_metrics['cpu_usage_percent']:.1f}%")
        print(f"  Loss: {loss:.4f}")
        if rmse is not None:
            print(f"  RMSE: {rmse:.4f}")
        if recall is not None:
            for k, value in recall.items():
                print(f"  Recall@{k}: {value:.4f}")
        if ndcg is not None:
            for k, value in ndcg.items():
                print(f"  NDCG@{k}: {value:.4f}")
        
    def end_test(self, rmse, recall=None, ndcg=None):
        self.test_metrics = {
            'test_time_sec': time.time() - self.epoch_start_time,
            'total_time_sec': time.time() - self.start_time,
            'final_memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'final_cpu_usage_percent': psutil.cpu_percent(),
            'test_rmse': rmse,
        }
        
        if recall is not None:
            for k, value in recall.items():
                self.test_metrics[f'test_recall@{k}'] = value
            
        if ndcg is not None:
            for k, value in ndcg.items():
                self.test_metrics[f'test_ndcg@{k}'] = value
        
        # Imprimir métricas finales
        print("\n=== Final Training Metrics ===")
        for m in self.train_metrics:
            metrics_str = f"Epoch {m['epoch']+1}: Time={m['epoch_time_sec']:.2f}s, Memory={m['memory_usage_mb']:.2f}MB, CPU={m['cpu_usage_percent']:.1f}%, Loss={m['loss']:.4f}"
            if 'rmse' in m:
                metrics_str += f", RMSE={m['rmse']:.4f}"
            for k in top_k_values:
                if f'recall@{k}' in m:
                    metrics_str += f", Recall@{k}={m[f'recall@{k}']:.4f}"
                if f'ndcg@{k}' in m:
                    metrics_str += f", NDCG@{k}={m[f'ndcg@{k}']:.4f}"
            print(metrics_str)
        
        print("\n=== Final Test Metrics ===")
        print(f"Total Time: {self.test_metrics['total_time_sec']:.2f}s (Test: {self.test_metrics['test_time_sec']:.2f}s)")
        print(f"Final Memory: {self.test_metrics['final_memory_usage_mb']:.2f}MB")
        print(f"Final CPU: {self.test_metrics['final_cpu_usage_percent']:.1f}%")
        print(f"Test RMSE: {rmse:.4f}")
        if recall is not None:
            for k, value in recall.items():
                print(f"Test Recall@{k}: {value:.4f}")
        if ndcg is not None:
            for k, value in ndcg.items():
                print(f"Test NDCG@{k}: {value:.4f}")
        
        # Guardar métricas en CSV
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        metrics_df = pd.DataFrame(self.train_metrics)
        metrics_df.to_csv(f"{result_path}/system_metrics_{timestamp}.csv", index=False)


# Clase para seguimiento de emisiones por época
class EmissionsPerEpochTracker:
    def __init__(self, result_path, model_name="LRML"):
        self.result_path = result_path
        self.model_name = model_name
        self.epoch_emissions = []
        self.cumulative_emissions = []
        self.epoch_rmse = []
        self.epoch_recall = {}
        self.epoch_ndcg = {}
        self.epoch_loss = []
        self.total_emissions = 0.0
        self.trackers = {}
        
        # Inicializar estructuras para métricas top-k
        for k in top_k_values:
            self.epoch_recall[k] = []
            self.epoch_ndcg[k] = []
        
        # Crear directorios para emisiones
        os.makedirs(f"{result_path}/emissions_reports", exist_ok=True)
        os.makedirs(f"{result_path}/emissions_plots", exist_ok=True)
        
        # Inicializar tracker principal
        self.main_tracker = EmissionsTracker(
            project_name=f"{model_name}_total",
            output_dir=f"{result_path}/emissions_reports",
            save_to_file=True,
            log_level="error",
            save_to_api=False,
            tracking_mode="process"
        )
        try:
            self.main_tracker.start()
            print("Tracker principal iniciado correctamente")
        except Exception as e:
            print(f"Advertencia: No se pudo iniciar el tracker principal: {e}")
            self.main_tracker = None
    
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
            tracking_mode="process",
            measure_power_secs=1,
            allow_multiple_runs=True
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
                for k, value in recall.items():
                    self.epoch_recall[k].append(value)
                
            if ndcg is not None:
                for k, value in ndcg.items():
                    self.epoch_ndcg[k].append(value)
            
            print(f"Epoch {epoch+1} - Emisiones: {epoch_co2:.8f} kg, Acumulado: {self.total_emissions:.8f} kg")
            
        except Exception as e:
            print(f"Error al medir emisiones en época {epoch}: {e}")
    
    def end_training(self, final_rmse=None, final_recall=None, final_ndcg=None):
        try:
            # Detener el tracker principal
            final_emissions = 0.0
            if hasattr(self, 'main_tracker') and self.main_tracker:
                try:
                    final_emissions = self.main_tracker.stop() or 0.0
                    print(f"\nTotal CO2 Emissions: {final_emissions:.6f} kg")
                except Exception as e:
                    print(f"Error al detener el tracker principal: {e}")
                    final_emissions = self.total_emissions
            else:
                final_emissions = self.total_emissions
            
            # Asegurarse de que todos los trackers estén detenidos
            for epoch, tracker in self.trackers.items():
                if tracker is not None:
                    try:
                        tracker.stop()
                    except:
                        pass
            
            # Si no hay datos de emisiones por época pero tenemos emisiones totales,
            # crear al menos una entrada para gráficos
            if not self.epoch_emissions and final_emissions > 0:
                self.epoch_emissions = [final_emissions]
                self.cumulative_emissions = [final_emissions]
                if final_rmse is not None:
                    self.epoch_rmse = [final_rmse]
                if final_recall is not None:
                    for k, value in final_recall.items():
                        self.epoch_recall[k] = [value]
                if final_ndcg is not None:
                    for k, value in final_ndcg.items():
                        self.epoch_ndcg[k] = [value]
            
            # Si no hay datos, salir
            if not self.epoch_emissions:
                print("No hay datos de emisiones para graficar")
                return
            
            # Crear dataframe con todos los datos
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            epochs_range = range(1, len(self.epoch_emissions) + 1)
            
            # Crear diccionario base para el dataframe
            df_data = {
                'epoch': epochs_range,
                'epoch_emissions_kg': self.epoch_emissions,
                'cumulative_emissions_kg': self.cumulative_emissions,
                'loss': self.epoch_loss if self.epoch_loss else [0.0] * len(self.epoch_emissions),
                'rmse': self.epoch_rmse if self.epoch_rmse else [None] * len(self.epoch_emissions)
            }
            
            # Añadir métricas top-k
            for k in top_k_values:
                if k in self.epoch_recall and self.epoch_recall[k]:
                    df_data[f'recall@{k}'] = self.epoch_recall[k]
                if k in self.epoch_ndcg and self.epoch_ndcg[k]:
                    df_data[f'ndcg@{k}'] = self.epoch_ndcg[k]
            
            df = pd.DataFrame(df_data)
            
            emissions_file = f'{self.result_path}/emissions_reports/emissions_metrics_{self.model_name}_{timestamp}.csv'
            df.to_csv(emissions_file, index=False)
            print(f"Métricas de emisiones guardadas en: {emissions_file}")
            
            # Graficar las relaciones
            self.plot_emissions_vs_metrics(timestamp, final_rmse, final_recall, final_ndcg)
            
        except Exception as e:
            print(f"Error al generar informes de emisiones: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_emissions_vs_metrics(self, timestamp, final_rmse=None, final_recall=None, final_ndcg=None):
        """Genera gráficos para emisiones vs métricas"""
        try:
            from matplotlib.ticker import ScalarFormatter
            
            epochs_range = range(1, len(self.epoch_emissions) + 1)
            
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
            
            # Recall para k máximo
            max_k = max(top_k_values)
            if max_k in self.epoch_recall and self.epoch_recall[max_k]:
                plt.subplot(3, 2, 5)
                plt.plot(epochs_range, self.epoch_recall[max_k], 'm-', marker='o')
                plt.title(f'Recall@{max_k} por Época')
                plt.xlabel('Época')
                plt.ylabel(f'Recall@{max_k}')
                plt.grid(True, alpha=0.3)
            
            # NDCG para k máximo
            if max_k in self.epoch_ndcg and self.epoch_ndcg[max_k]:
                plt.subplot(3, 2, 6)
                plt.plot(epochs_range, self.epoch_ndcg[max_k], 'c-', marker='o')
                plt.title(f'NDCG@{max_k} por Época')
                plt.xlabel('Época')
                plt.ylabel(f'NDCG@{max_k}')
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
            
            # 3. Recall vs Emisiones acumuladas para cada K
            for k in top_k_values:
                if k in self.epoch_recall and self.epoch_recall[k]:
                    plt.figure(figsize=(10, 6))
                    emissions_in_g = [e * 1000 for e in self.cumulative_emissions]
                    plt.plot(emissions_in_g, self.epoch_recall[k], 'm-', marker='o')
                    
                    # Configurar límites del eje x
                    plt.xlim(0, max(emissions_in_g) * 1.1)
                    
                    # Añadir etiquetas con el número de época
                    for i, (emissions, recall) in enumerate(zip(emissions_in_g, self.epoch_recall[k])):
                        plt.annotate(f"{i+1}", (emissions, recall), textcoords="offset points", 
                                    xytext=(0,10), ha='center', fontsize=9)
                        
                    plt.xlabel('Emisiones de CO₂ acumuladas (g)')
                    plt.ylabel(f'Recall@{k}')
                    plt.title(f'Relación entre Emisiones Acumuladas y Recall@{k}')
                    plt.grid(True, alpha=0.3)
                    
                    file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_recall{k}_{self.model_name}_{timestamp}.png'
                    plt.savefig(file_path)
                    plt.close()
                    print(f"Gráfico Recall@{k} vs emisiones guardado en: {file_path}")
                
            # 4. NDCG vs Emisiones acumuladas para cada K
            for k in top_k_values:
                if k in self.epoch_ndcg and self.epoch_ndcg[k]:
                    plt.figure(figsize=(10, 6))
                    emissions_in_g = [e * 1000 for e in self.cumulative_emissions]
                    plt.plot(emissions_in_g, self.epoch_ndcg[k], 'c-', marker='o')
                    
                    # Configurar límites del eje x
                    plt.xlim(0, max(emissions_in_g) * 1.1)
                    
                    # Añadir etiquetas con el número de época
                    for i, (emissions, ndcg) in enumerate(zip(emissions_in_g, self.epoch_ndcg[k])):
                        plt.annotate(f"{i+1}", (emissions, ndcg), textcoords="offset points", 
                                    xytext=(0,10), ha='center', fontsize=9)
                        
                    plt.xlabel('Emisiones de CO₂ acumuladas (g)')
                    plt.ylabel(f'NDCG@{k}')
                    plt.title(f'Relación entre Emisiones Acumuladas y NDCG@{k}')
                    plt.grid(True, alpha=0.3)
                    
                    file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_ndcg{k}_{self.model_name}_{timestamp}.png'
                    plt.savefig(file_path)
                    plt.close()
                    print(f"Gráfico NDCG@{k} vs emisiones guardado en: {file_path}")
                    
            # 5. Comparación de métricas top-k
            plt.figure(figsize=(15, 10))
            
            # Recall para diferentes K
            plt.subplot(2, 1, 1)
            for k in top_k_values:
                if k in self.epoch_recall and self.epoch_recall[k]:
                    plt.plot(epochs_range, self.epoch_recall[k], marker='o', label=f'Recall@{k}')
            plt.title('Comparación de Recall@K por Época')
            plt.xlabel('Época')
            plt.ylabel('Recall')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # NDCG para diferentes K
            plt.subplot(2, 1, 2)
            for k in top_k_values:
                if k in self.epoch_ndcg and self.epoch_ndcg[k]:
                    plt.plot(epochs_range, self.epoch_ndcg[k], marker='o', label=f'NDCG@{k}')
            plt.title('Comparación de NDCG@K por Época')
            plt.xlabel('Época')
            plt.ylabel('NDCG')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            file_path = f'{self.result_path}/emissions_plots/topk_metrics_comparison_{self.model_name}_{timestamp}.png'
            plt.savefig(file_path)
            plt.close()
            print(f"Gráfico comparativo top-k guardado en: {file_path}")
            
        except Exception as e:
            print(f"Error al generar los gráficos: {e}")
            traceback.print_exc()


# Función para calcular RMSE básico
def calculate_rmse(model, sess, test_data):
    user_idxs = [x[0] for x in test_data]
    item_idxs = [x[1] for x in test_data]
    true_ratings = [x[2] for x in test_data]
    
    feed_dict = {
        model.user_input: user_idxs,
        model.item_input: item_idxs,
        model.dropout: 1.0
    }
    
    predictions = sess.run(model.predict_op, feed_dict=feed_dict).flatten()
    rmse = np.sqrt(np.mean(np.square(np.array(predictions) - np.array(true_ratings))))
    return rmse


# Función para calcular métricas de ranking (Recall@K y NDCG@K)
def calculate_topk_metrics(model, sess, test_data, user_positive_items, all_item_ids, k_values=[5, 10, 20], max_users=50):
    """Calcula métricas Top-K para diferentes valores de K"""
    # Extraer usuarios únicos de test data
    test_users = set([x[0] for x in test_data])
    
    # Limitar número de usuarios a evaluar para eficiencia
    if len(test_users) > max_users:
        test_users = random.sample(list(test_users), max_users)
    
    # Diccionarios para almacenar métricas
    recall_at_k = {k: [] for k in k_values}
    ndcg_at_k = {k: [] for k in k_values}
    
    print(f"Evaluando métricas top-k para {len(test_users)} usuarios...")
    
    for user_idx in test_users:
        # Ítems relevantes (rating > 0.6) para este usuario en test
        relevant_items = [item for user, item, rating in test_data 
                         if user == user_idx and rating > 0.6]
        
        if not relevant_items:
            continue
        
        # Todos los ítems excepto los que el usuario ya ha valorado positivamente en training
        items_to_rank = list(set(all_item_ids) - set(user_positive_items.get(user_idx, [])))
        
        # Si hay demasiados ítems, limitar para eficiencia
        max_items = 1000
        if len(items_to_rank) > max_items:
            items_to_rank = random.sample(items_to_rank, max_items - len(relevant_items))
            # Asegurarnos de incluir los ítems relevantes
            items_to_rank.extend(relevant_items)
        
        users = [user_idx] * len(items_to_rank)
        
        # Obtener predicciones para todos los ítems
        feed_dict = {
            model.user_input: users,
            model.item_input: items_to_rank,
            model.dropout: 1.0
        }
        
        predictions = sess.run(model.predict_op, feed_dict=feed_dict).flatten()
        
        # Ordenar ítems por predicción en orden descendente
        item_score_pairs = sorted(list(zip(items_to_rank, predictions)), 
                                 key=lambda x: x[1], reverse=True)
        
        # Calcular métricas para cada K
        for k in k_values:
            top_k_items = [x[0] for x in item_score_pairs[:k]]
            
            # Recall@K
            hits = len(set(top_k_items) & set(relevant_items))
            recall = hits / len(relevant_items)
            recall_at_k[k].append(recall)
            
            # NDCG@K
            dcg = 0
            for i, item in enumerate(top_k_items):
                if item in relevant_items:
                    dcg += 1 / math.log2(i + 2)  # posición i+2 porque i empieza en 0
            
            # IDCG - valor ideal de DCG
            idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant_items))))
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_at_k[k].append(ndcg)
    
    # Promediar las métricas
    avg_recall = {k: np.mean(scores) for k, scores in recall_at_k.items() if scores}
    avg_ndcg = {k: np.mean(scores) for k, scores in ndcg_at_k.items() if scores}
    
    return avg_recall, avg_ndcg


# Función para generar batch de tripletas (usuario, ítem positivo, ítem negativo)
def generate_triplet_batch(train_data, user_positive_items, all_item_ids, batch_size):
    """Genera batch de tripletas para entrenamiento BPR"""
    triplets = []
    users = list(user_positive_items.keys())
    
    while len(triplets) < batch_size:
        # Seleccionar usuario aleatorio
        user = random.choice(users)
        
        # Verificar que el usuario tenga ítems positivos
        if not user_positive_items[user]:
            continue
        
        # Seleccionar ítem positivo aleatorio
        pos_item = random.choice(list(user_positive_items[user]))
        
        # Seleccionar ítem negativo (no valorado por el usuario)
        neg_item = random.choice(all_item_ids)
        while neg_item in user_positive_items[user]:
            neg_item = random.choice(all_item_ids)
        
        triplets.append((user, pos_item, neg_item))
    
    return triplets


def main():
    # Configuración de rutas y directorios
    global result_path
    data_name = 'ml-1m'
    path = f"C:/Users/xpati/Documents/TFG/{data_name}/"
    result_path = "./results"
    
    # Crear directorios para resultados
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(f"{result_path}/emissions_reports", exist_ok=True)
    os.makedirs(f"{result_path}/emissions_plots", exist_ok=True)
    
    # Eliminar archivo de lock si existe
    lock_file = "C:\\Users\\xpati\\AppData\\Local\\Temp\\.codecarbon.lock"
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
            print(f"Archivo de bloqueo eliminado: {lock_file}")
        except Exception as e:
            print(f"No se pudo eliminar el archivo de bloqueo: {e}")
    
    print("Cargando datos...")
    
    # Cargar los datos de MovieLens con la codificación correcta
    ratings = pd.read_csv(f'{path}/ratings.dat', 
                        sep='::', 
                        header=None,
                        names=['userId', 'movieId', 'rating', 'timestamp'],
                        engine='python',
                        encoding='latin-1')
                        
    # Cargar películas para información adicional
    movies = pd.read_csv(f'{path}/movies.dat', 
                       sep='::', 
                       header=None,
                       names=['movieId', 'title', 'genres'],
                       engine='python',
                       encoding='latin-1')
    
    # Crear mapeos bidireccionales entre IDs e índices
    user_ids = ratings['userId'].unique()
    item_ids = ratings['movieId'].unique()
    
    user_id_to_idx = {user_id: i for i, user_id in enumerate(user_ids)}
    item_id_to_idx = {item_id: i for i, item_id in enumerate(item_ids)}
    
    idx_to_user_id = {i: user_id for user_id, i in user_id_to_idx.items()}
    idx_to_item_id = {i: item_id for item_id, i in item_id_to_idx.items()}
    
    # Mapeo de IDs a índices
    ratings_mapped = ratings.copy()
    ratings_mapped['userId'] = ratings_mapped['userId'].map(user_id_to_idx)
    ratings_mapped['movieId'] = ratings_mapped['movieId'].map(item_id_to_idx)
    
    # Dividir los datos en entrenamiento y prueba
    train_data, test_data = train_test_split(ratings_mapped, test_size=0.2, random_state=42)
    
    # Convertir a listas de tuplas (userId, movieId, rating)
    train_data = list(zip(train_data['userId'], train_data['movieId'], train_data['rating']))
    test_data = list(zip(test_data['userId'], test_data['movieId'], test_data['rating']))
    
    # Normalizar los ratings (escalarlos a [0,1])
    min_rating = ratings['rating'].min()
    max_rating = ratings['rating'].max()
    
    train_data = [(u, i, (r - min_rating) / (max_rating - min_rating)) 
                for u, i, r in train_data]
    test_data = [(u, i, (r - min_rating) / (max_rating - min_rating)) 
                for u, i, r in test_data]
    
    # Crear mapeo de usuarios a ítems positivos (para muestreo y eval)
    user_positive_items = defaultdict(set)
    for u, i, r in train_data:
        if r > 0.6:  # Considerar ratings > 0.6 como positivos (equivale a ~3.5 en escala 1-5)
            user_positive_items[u].add(i)
    
    all_item_ids = list(range(len(item_ids)))
    
    # Definir argumentos del modelo usando los parámetros de línea de comandos
    class ModelArgs:
        def __init__(self):
            self.std = 0.01
            self.embedding_size = args.embedding_size
            self.num_mem = args.num_mem
            self.dropout = args.dropout
            self.margin = args.margin
            self.l2_reg = args.l2_reg
            self.opt = 'Adam'
            self.learn_rate = args.learn_rate
            self.clip_norm = 1.0
            self.constraint = True
            self.rnn_type = 'PAIR'
    
    model_args = ModelArgs()
    
    # Inicializar trackers
    print("Inicializando trackers de métricas...")
    system_tracker = SystemMetricsTracker()
    emissions_tracker = EmissionsPerEpochTracker(result_path, model_name="LRML_TopK")
    
    # Crear el modelo
    num_users = len(user_ids)
    num_items = len(item_ids)
    print(f"Creando modelo con {num_users} usuarios y {num_items} items...")
    model = LRML(num_users, num_items, model_args)
    
    # Pre-calcular items positivos por usuario para muestreo negativo
    user_pos_items_dict = defaultdict(set)
    for u, i, r in train_data:
        if r >= 0.7:  # Considerar ratings altos como positivos
            user_pos_items_dict[u].add(i)
    
    # Calcular los ítems más populares
    item_popularity = defaultdict(int)
    for u, i, r in train_data:
        if r >= 0.5:  # Ratings positivos
            item_popularity[i] += 1
    
    popular_items = [item for item, count in sorted(
        item_popularity.items(), key=lambda x: x[1], reverse=True
    )][:1000]  # Top 1000 ítems
    
    # Iniciar timer para medir tiempo total
    start_time = time.time()
    
    # Configurar la sesión TensorFlow
    with tf.compat.v1.Session(graph=model.graph) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Configurar entrenamiento
        batch_size = args.batch_size
        
        # Para early stopping
        best_metric = float('-inf')  # Para Recall y NDCG, queremos maximizar
        patience = 10
        patience_counter = 0
        
        # Para medir CPU
        cpu_measurements = []
        process = psutil.Process()
        process.cpu_percent()  # Primera llamada para inicializar
        
        # Crear saver para guardar el mejor modelo
        saver = tf.compat.v1.train.Saver()
        
        # Listas para seguimiento de métricas
        train_losses = []
        epochs_rmse = []
        epochs_recall = {k: [] for k in top_k_values}
        epochs_ndcg = {k: [] for k in top_k_values}
        
        # Loop de entrenamiento
        for epoch in range(args.train_epoch):
            # Iniciar tracking para esta época
            system_tracker.start_epoch(epoch)
            emissions_tracker.start_epoch(epoch)
            epoch_start_time = time.time()
            
            total_batches = len(train_data) // batch_size
            total_epoch_loss = 0
            
            # Entrenar en batches
            print(f"Epoch {epoch+1}/{args.train_epoch} - Procesando {total_batches} batches...")
            for i in range(total_batches):
                # Generar batch de tripletas (usuario, ítem positivo, ítem negativo)
                batch = generate_triplet_batch(train_data, user_pos_items_dict, all_item_ids, batch_size)
                
                # Extraer usuarios, ítems positivos y negativos
                users = [x[0] for x in batch]
                pos_items = [x[1] for x in batch]
                neg_items = [x[2] for x in batch]
                
                # En LRML, usamos la función get_feed_dict para preparar los datos
                if 'PAIR' in model.args.rnn_type:
                    # Crear batch positivo y negativo para LRML
                    pos_batch = [(u, i, 1.0) for u, i in zip(users, pos_items)]  # 1.0 como rating positivo
                    neg_batch = [(u, i, 0.0) for u, i in zip(users, neg_items)]  # 0.0 como rating negativo
                    
                    # Preparar feed_dict para entrenamiento
                    feed_dict, _ = model.get_feed_dict(pos_batch, neg_batch, mode='training')
                else:
                    # Alternativa para otro tipo de modelos (no usado en este caso)
                    feed_dict = {
                        model.user_input: users,
                        model.pos_item_input: pos_items,
                        model.neg_item_input: neg_items,
                        model.dropout: args.dropout
                    }
                
                # Entrenar modelo
                _, batch_loss = sess.run([model.train_op, model.cost], feed_dict=feed_dict)
                total_epoch_loss += batch_loss
                
                # Mostrar progreso cada 50 batches
                if i % 50 == 0 and i > 0:
                    print(f"  Batch {i}/{total_batches}: Loss = {batch_loss:.4f}")
            
            # Calcular pérdida promedio por batch
            avg_epoch_loss = total_epoch_loss/total_batches
            train_losses.append(avg_epoch_loss)
            
            # Evaluar métricas en cada época
            print("\n--- Métricas de evaluación ---")
            
            # Evaluar en una muestra para ahorrar tiempo
            eval_sample = random.sample(test_data, min(len(test_data), 5000))
            
            # Calcular RMSE
            rmse = calculate_rmse(model, sess, eval_sample)
            print(f"RMSE: {rmse:.4f}")
            epochs_rmse.append(rmse)
            
            # Calcular métricas top-K
            recall_dict, ndcg_dict = calculate_topk_metrics(
                model, sess, eval_sample, user_pos_items_dict, all_item_ids, top_k_values
            )
            
            for k, value in recall_dict.items():
                print(f"Recall@{k}: {value:.4f}")
                epochs_recall[k].append(value)
                
            for k, value in ndcg_dict.items():
                print(f"NDCG@{k}: {value:.4f}")
                epochs_ndcg[k].append(value)
                
            # Finalizar tracking para esta época
            system_tracker.end_epoch(
                epoch=epoch, 
                loss=avg_epoch_loss, 
                rmse=rmse,
                recall=recall_dict,
                ndcg=ndcg_dict
            )
            
            emissions_tracker.end_epoch(
                epoch=epoch, 
                loss=avg_epoch_loss, 
                rmse=rmse,
                recall=recall_dict,
                ndcg=ndcg_dict
            )
            
            epoch_time = time.time() - epoch_start_time
            print(f"Tiempo de época: {epoch_time:.2f} segundos")
            
            # Early stopping basado en la métrica top-K principal (Recall@10)
            main_k = 10  # Usamos Recall@10 como métrica principal
            if main_k in recall_dict and recall_dict[main_k] > best_metric:
                best_metric = recall_dict[main_k]
                patience_counter = 0
                print(f"¡Nuevo mejor Recall@{main_k}: {best_metric:.4f}!")
                
                # Guardar modelo
                model_path = f'{result_path}/best_model'
                saver.save(sess, model_path)
                print(f"¡Modelo guardado en: {model_path}!")
            
            '''
            else:
                patience_counter += 1
                print(f"Sin mejora en Recall@{main_k}. Paciencia: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping después de {patience} épocas sin mejora")
                    break
            '''
        
        # Al finalizar el entrenamiento, evaluar métricas finales
        print("\nCalculando métricas finales...")
        system_tracker.start_epoch("final")
        
        # Evaluar en el conjunto completo de prueba o una muestra grande
        print("Evaluando en una muestra grande del conjunto de prueba...")
        final_test_sample = random.sample(test_data, min(len(test_data), 10000))
        
        # Calcular RMSE final
        final_rmse = calculate_rmse(model, sess, final_test_sample)
        
        # Calcular métricas top-K finales
        final_recall, final_ndcg = calculate_topk_metrics(
            model, sess, final_test_sample, user_pos_items_dict, all_item_ids, top_k_values, max_users=100
        )
        
        # Imprimir métricas finales
        print("\n=========== MÉTRICAS FINALES ===========")
        print(f"RMSE: {final_rmse:.4f}")
        for k, value in final_recall.items():
            print(f"Recall@{k}: {value:.4f}")
        for k, value in final_ndcg.items():
            print(f"NDCG@{k}: {value:.4f}")
        
        # Métricas del sistema
        memory_usage = process.memory_info().rss / (1024 * 1024)  # En MB
        
        # Finalizar tracking
        system_tracker.end_test(final_rmse, final_recall, final_ndcg)
        emissions_tracker.end_training(
            final_rmse=final_rmse, 
            final_recall=final_recall, 
            final_ndcg=final_ndcg
        )
        
        # Guardar métricas finales en CSV
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        total_time = time.time() - start_time
        
        # Crear dataframe para métricas finales
        metrics_dict = {
            'model': ['LRML_TopK'],
            'final_rmse': [final_rmse],
            'total_time_seconds': [total_time],
            'memory_usage_mb': [memory_usage]
        }
        
        # Añadir métricas top-K
        for k, value in final_recall.items():
            metrics_dict[f'recall@{k}'] = [value]
        for k, value in final_ndcg.items():
            metrics_dict[f'ndcg@{k}'] = [value]
        
        metrics_df = pd.DataFrame(metrics_dict)
        metrics_file = f'{result_path}/final_metrics_LRML_TopK_{timestamp}.csv'
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Métricas finales guardadas en: {metrics_file}")
        
        # Preparar datos para CSV de métricas por época
        epochs_data = {
            'epoch': list(range(1, len(train_losses) + 1)),
            'train_loss': train_losses,
            'rmse': epochs_rmse
        }
        
        # Añadir métricas top-K por época
        for k in top_k_values:
            if k in epochs_recall and epochs_recall[k]:
                epochs_data[f'recall@{k}'] = epochs_recall[k]
            if k in epochs_ndcg and epochs_ndcg[k]:
                epochs_data[f'ndcg@{k}'] = epochs_ndcg[k]
        
        # Guardar métricas de entrenamiento
        training_df = pd.DataFrame(epochs_data)
        training_file = f'{result_path}/training_metrics_LRML_TopK_{timestamp}.csv'
        training_df.to_csv(training_file, index=False)
        print(f"Métricas de entrenamiento guardadas en: {training_file}")
        
        # Generar gráficos adicionales para métricas top-K
        try:
            plt.figure(figsize=(12, 8))
            
            # Recall@K por época
            plt.subplot(2, 1, 1)
            for k in top_k_values:
                if k in epochs_recall and len(epochs_recall[k]) > 0:
                    plt.plot(range(1, len(epochs_recall[k]) + 1), epochs_recall[k], marker='o', label=f'Recall@{k}')
            plt.title('Recall@K por Época')
            plt.xlabel('Época')
            plt.ylabel('Recall')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # NDCG@K por época
            plt.subplot(2, 1, 2)
            for k in top_k_values:
                if k in epochs_ndcg and len(epochs_ndcg[k]) > 0:
                    plt.plot(range(1, len(epochs_ndcg[k]) + 1), epochs_ndcg[k], marker='o', label=f'NDCG@{k}')
            plt.title('NDCG@K por Época')
            plt.xlabel('Época')
            plt.ylabel('NDCG')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            metrics_plot_file = f'{result_path}/topk_metrics_by_epoch_{timestamp}.png'
            plt.savefig(metrics_plot_file)
            plt.close()
            print(f"Gráfico de métricas top-K guardado en: {metrics_plot_file}")
            
        except Exception as e:
            print(f"Error al generar gráficos adicionales: {e}")


if __name__ == "__main__":
    main()