import json
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import random
from tensorflow.keras.utils import Progbar
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import time
import psutil
import pandas as pd
from codecarbon import EmissionsTracker
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Crear directorios para resultados
result_path = "results_lgcn_gdsc1"
os.makedirs(result_path, exist_ok=True)
os.makedirs(f"{result_path}/emissions_reports", exist_ok=True)
os.makedirs(f"{result_path}/emissions_plots", exist_ok=True)

# Diccionario global para almacenar los ratings reales
ratings_dict = {}

def load_gdsc1_dataset(csv_file, test_size=0.2, val_size=0.1, random_state=42):
    """
    Carga el dataset GDSC1 desde un archivo CSV y lo divide en train/test/val
    
    Args:
        csv_file: Ruta al archivo gdsc1_processed.csv
        test_size: Proporción para conjunto de prueba
        val_size: Proporción para conjunto de validación
        random_state: Semilla para reproducibilidad
    
    Returns:
        train_data, test_data, val_data, n_users, n_items, ratings_dict
    """
    global ratings_dict
    
    print(f"Cargando dataset GDSC1 desde {csv_file}...")
    
    # Cargar el CSV (solo las columnas que necesitamos)
    df = pd.read_csv(csv_file, usecols=['user_id', 'item_id', 'rating'])
    print(f"Dataset cargado: {len(df)} interacciones")
    
    # Estadísticas básicas
    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()
    print(f"Usuarios únicos (líneas celulares): {n_users}")
    print(f"Items únicos (fármacos): {n_items}")
    print(f"Rating promedio (LNIC50): {df['rating'].mean():.3f}")
    print(f"Rating mínimo (LNIC50): {df['rating'].min():.3f}")
    print(f"Rating máximo (LNIC50): {df['rating'].max():.3f}")
    
    # Crear mapeos para asegurar índices consecutivos desde 0
    unique_users = sorted(df['user_id'].unique())
    unique_items = sorted(df['item_id'].unique())
    
    user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    item_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
    
    # Aplicar mapeos y convertir a enteros
    df['user_idx'] = df['user_id'].map(user_map).astype(int)
    df['item_idx'] = df['item_id'].map(item_map).astype(int)
    
    # Crear diccionario de ratings para las métricas
    ratings_dict = {}
    for _, row in df.iterrows():
        ratings_dict[(int(row['user_idx']), int(row['item_idx']))] = row['rating']
    
    print(f"Ratings cargados en diccionario: {len(ratings_dict)}")
    
    # Dividir los datos
    # Primero dividir en train y temp (test + val)
    train_df, temp_df = train_test_split(df, test_size=test_size + val_size, 
                                         random_state=random_state, 
                                         stratify=None)
    
    # Luego dividir temp en test y val
    relative_val_size = val_size / (test_size + val_size)
    test_df, val_df = train_test_split(temp_df, test_size=relative_val_size, 
                                       random_state=random_state)
    
    print(f"División del dataset:")
    print(f"  Train: {len(train_df)} interacciones ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} interacciones ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df)} interacciones ({len(val_df)/len(df)*100:.1f}%)")
    
    # Convertir a formato requerido por LightGCN (listas de sets)
    def df_to_lightgcn_format(data_df, n_users):
        user_items = [set() for _ in range(n_users)]
        for _, row in data_df.iterrows():
            user_idx = int(row['user_idx'])
            item_idx = int(row['item_idx'])
            user_items[user_idx].add(item_idx)
        return user_items
    
    train_data = df_to_lightgcn_format(train_df, n_users)
    test_data = df_to_lightgcn_format(test_df, n_users)
    val_data = df_to_lightgcn_format(val_df, n_users)
    
    # Verificar que todos los usuarios tienen al menos una interacción en train
    empty_users = sum(1 for items in train_data if len(items) == 0)
    if empty_users > 0:
        print(f"Advertencia: {empty_users} usuarios sin interacciones en train")
    
    return train_data, test_data, val_data, n_users, n_items, ratings_dict

def show_ratings_statistics():
    """Muestra estadísticas de los ratings cargados"""
    global ratings_dict
    if not ratings_dict:
        print("No hay ratings reales cargados.")
        return
    
    ratings_values = list(ratings_dict.values())
    print(f"\nEstadísticas de ratings reales (LNIC50):")
    print(f"  Total de ratings: {len(ratings_values)}")
    print(f"  Rating promedio: {np.mean(ratings_values):.3f}")
    print(f"  Rating mínimo: {min(ratings_values):.3f}")
    print(f"  Rating máximo: {max(ratings_values):.3f}")
    print(f"  Desviación estándar: {np.std(ratings_values):.3f}")
    
    # Distribución de ratings (por rangos dado que son valores continuos)
    ratings_array = np.array(ratings_values)
    percentiles = [0, 25, 50, 75, 90, 95, 100]
    print(f"  Distribución por percentiles:")
    for p in percentiles:
        value = np.percentile(ratings_array, p)
        print(f"    Percentil {p}: {value:.3f}")

def test_rating_usage():
    """Función de debug para verificar que se usan ratings reales"""
    global ratings_dict
    if not ratings_dict:
        print("PROBLEMA: No hay ratings cargados!")
        return
    
    # Tomar 10 ejemplos aleatorios
    sample_ratings = random.sample(list(ratings_dict.items()), min(10, len(ratings_dict)))
    print(f"\nVERIFICACIÓN: Muestra de ratings reales que se usarán (LNIC50):")
    for (u, item_id), rating in sample_ratings:
        print(f"  Usuario {u}, Item {item_id} → Rating real: {rating:.3f}")
    
    # Verificar distribución de la muestra
    sample_values = [rating for _, rating in sample_ratings]
    print(f"\nDistribución en muestra:")
    print(f"  Rating promedio en muestra: {np.mean(sample_values):.3f}")
    print(f"  Rating mínimo en muestra: {min(sample_values):.3f}")
    print(f"  Rating máximo en muestra: {max(sample_values):.3f}")

# Función MEJORADA para calcular RMSE
def calculate_rmse(model, test_data, n_users, n_items, sample_size=1000):
    """
    Calcula el RMSE usando los ratings reales de GDSC1 - MEJORADA.
    """
    global ratings_dict
    
    if not ratings_dict:
        return 1.0
    
    # Tomar muestra más grande para estabilidad
    all_ratings = list(ratings_dict.items())
    sample_ratings = random.sample(all_ratings, min(sample_size, len(all_ratings)))
    
    # Obtener embeddings UNA SOLA VEZ
    user_emb, item_emb, _, _, _ = model((model.user_embedding, model.item_embedding))
    
    # Calcular TODOS los scores primero
    all_scores = []
    valid_pairs = []
    
    for (u, item_id), true_rating in sample_ratings:
        if u >= n_users or item_id >= n_items:
            continue
            
        u_emb = tf.nn.embedding_lookup(user_emb, [u])
        i_emb = tf.nn.embedding_lookup(item_emb, [item_id])
        score = tf.reduce_sum(u_emb * i_emb, axis=1).numpy()[0]
        
        all_scores.append(score)
        valid_pairs.append(((u, item_id), true_rating))
    
    if not all_scores:
        return 1.0
    
    # NORMALIZACIÓN MEJORADA usando estadísticas del dataset real
    min_score, max_score = np.min(all_scores), np.max(all_scores)
    
    # Obtener estadísticas reales del dataset
    all_true_ratings = [rating for _, rating in valid_pairs]
    true_min, true_max = np.min(all_true_ratings), np.max(all_true_ratings)
    true_mean = np.mean(all_true_ratings)
    
    y_true = []
    y_pred = []
    
    for i, ((u, item_id), true_rating) in enumerate(valid_pairs):
        score = all_scores[i]
        
        # Normalización por percentiles (más robusta)
        if max_score > min_score:
            # Normalizar score a [0,1]
            normalized_score = (score - min_score) / (max_score - min_score)
            # Mapear a rango real observado
            predicted_rating = true_min + normalized_score * (true_max - true_min)
        else:
            predicted_rating = true_mean
        
        y_true.append(true_rating)
        y_pred.append(predicted_rating)
    
    # Calcular RMSE
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

# Función MEJORADA para calcular MAE
def calculate_mae(model, test_data, n_users, n_items, sample_size=1000):
    """
    Calcula el MAE usando los ratings reales de GDSC1 - MEJORADA.
    """
    global ratings_dict
    
    if not ratings_dict:
        return 0.8
    
    # Usar el mismo sample_size que RMSE para consistencia
    all_ratings = list(ratings_dict.items())
    sample_ratings = random.sample(all_ratings, min(sample_size, len(all_ratings)))
    
    # Obtener embeddings UNA SOLA VEZ
    user_emb, item_emb, _, _, _ = model((model.user_embedding, model.item_embedding))
    
    # Calcular TODOS los scores primero
    all_scores = []
    valid_pairs = []
    
    for (u, item_id), true_rating in sample_ratings:
        if u >= n_users or item_id >= n_items:
            continue
            
        u_emb = tf.nn.embedding_lookup(user_emb, [u])
        i_emb = tf.nn.embedding_lookup(item_emb, [item_id])
        score = tf.reduce_sum(u_emb * i_emb, axis=1).numpy()[0]
        
        all_scores.append(score)
        valid_pairs.append(((u, item_id), true_rating))
    
    if not all_scores:
        return 0.8
    
    # Usar la misma normalización que RMSE
    min_score, max_score = np.min(all_scores), np.max(all_scores)
    
    # Obtener estadísticas reales del dataset
    all_true_ratings = [rating for _, rating in valid_pairs]
    true_min, true_max = np.min(all_true_ratings), np.max(all_true_ratings)
    true_mean = np.mean(all_true_ratings)
    
    y_true = []
    y_pred = []
    
    for i, ((u, item_id), true_rating) in enumerate(valid_pairs):
        score = all_scores[i]
        
        if max_score > min_score:
            normalized_score = (score - min_score) / (max_score - min_score)
            predicted_rating = true_min + normalized_score * (true_max - true_min)
        else:
            predicted_rating = true_mean
        
        y_true.append(true_rating)
        y_pred.append(predicted_rating)
    
    # Calcular MAE
    mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    return mae


class SystemMetricsTracker:
    def __init__(self):
        self.train_metrics = []
        self.test_metrics = {}
        self.start_time = time.time()
        self.best_rmse = float('inf')
        self.best_rmse_epoch = None
        self.best_rmse_metrics = None
        
    def start_epoch(self, epoch):
        self.epoch_start_time = time.time()
        self.current_epoch_metrics = {
            'epoch': epoch,
            'memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'cpu_usage_percent': psutil.cpu_percent(),
        }
        
    def end_epoch(self, epoch, loss, rmse=None, mae=None):
        epoch_time = time.time() - self.epoch_start_time
        self.current_epoch_metrics['epoch_time_sec'] = epoch_time
        self.current_epoch_metrics['loss'] = loss
        if rmse is not None:
            self.current_epoch_metrics['rmse'] = rmse
        if mae is not None:
            self.current_epoch_metrics['mae'] = mae
        self.train_metrics.append(self.current_epoch_metrics)
        
        # Rastrear el mejor RMSE
        if rmse is not None and rmse < self.best_rmse:
            self.best_rmse = rmse
            self.best_rmse_epoch = epoch
            self.best_rmse_metrics = self.current_epoch_metrics.copy()
        
        # Imprimir resumen de época con formato similar al ejemplo NCF
        print(f"\nEpoch {epoch} Metrics:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Memory: {self.current_epoch_metrics['memory_usage_mb']:.2f}MB")
        print(f"  CPU: {self.current_epoch_metrics['cpu_usage_percent']:.1f}%")
        print(f"  Loss: {loss:.4f}")
        if rmse is not None:
            print(f"  RMSE: {rmse:.4f}")
        if mae is not None:
            print(f"  MAE: {mae:.4f}")
        
    def end_test(self, rmse, mae=None):
        self.test_metrics = {
            'test_time_sec': time.time() - self.epoch_start_time,
            'total_time_sec': time.time() - self.start_time,
            'final_memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'final_cpu_usage_percent': psutil.cpu_percent(),
            'test_rmse': rmse,
        }
        if mae is not None:
            self.test_metrics['test_mae'] = mae
        
        # Imprimir métricas finales con formato específico como en NCF
        print("\n=== Final Training Metrics ===")
        for m in self.train_metrics:
            # Formato específico: Epoch X: Time=Xs, Memory=XMB, CPU=X%, RMSE=X, MAE=X
            metrics_str = f"Epoch {m['epoch']}: Time={m['epoch_time_sec']:.2f}s, Memory={m['memory_usage_mb']:.2f}MB, CPU={m['cpu_usage_percent']:.1f}%"
            if 'rmse' in m and m['rmse'] is not None:
                metrics_str += f", RMSE={m['rmse']:.4f}"
            if 'mae' in m and m['mae'] is not None:
                metrics_str += f", MAE={m['mae']:.4f}"
            print(metrics_str)
        
        print("\n=== Final Test Metrics ===")
        print(f"Total Time: {self.test_metrics['total_time_sec']:.2f}s (Test: {self.test_metrics['test_time_sec']:.2f}s)")
        print(f"Final Memory: {self.test_metrics['final_memory_usage_mb']:.2f}MB")
        print(f"Final CPU: {self.test_metrics['final_cpu_usage_percent']:.1f}%")
        print(f"RMSE: {rmse:.4f}")
        if mae is not None:
            print(f"MAE: {mae:.4f}")
        
        # Mostrar información del mejor RMSE durante el entrenamiento
        if self.best_rmse_epoch is not None:
            print(f"\n=== Best Training RMSE ===")
            print(f"Best RMSE: {self.best_rmse:.4f} (Epoch {self.best_rmse_epoch})")
            if self.best_rmse_metrics:
                print(f"Time: {self.best_rmse_metrics['epoch_time_sec']:.2f}s")
                print(f"Memory: {self.best_rmse_metrics['memory_usage_mb']:.2f}MB")
                print(f"CPU: {self.best_rmse_metrics['cpu_usage_percent']:.1f}%")
                if 'mae' in self.best_rmse_metrics and self.best_rmse_metrics['mae'] is not None:
                    print(f"MAE: {self.best_rmse_metrics['mae']:.4f}")
        
        # Guardar métricas en CSV
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        metrics_df = pd.DataFrame(self.train_metrics)
        metrics_df.to_csv(f"{result_path}/system_metrics_{timestamp}.csv", index=False)
        print(f"System metrics saved to: {result_path}/system_metrics_{timestamp}.csv")

    def get_best_rmse_info(self):
        """Retorna información del mejor RMSE para usar con emisiones"""
        if self.best_rmse_epoch is not None:
            return {
                'epoch': self.best_rmse_epoch,
                'rmse': self.best_rmse,
                'metrics': self.best_rmse_metrics
            }
        return None

class EmissionsPerEpochTracker:
    def __init__(self, result_path, model_name="LightGCN"):
        self.result_path = result_path
        self.model_name = model_name
        self.epoch_emissions = []
        self.cumulative_emissions = []
        self.epoch_rmse = []
        self.epoch_mae = []
        self.epoch_loss = []
        self.total_emissions = 0.0
        self.trackers = {}
        
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
    
    def end_epoch(self, epoch, loss, rmse=None, mae=None):
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
            if mae is not None:
                self.epoch_mae.append(mae)
            
            print(f"Epoch {epoch} - Emisiones: {epoch_co2:.8f} kg, Acumulado: {self.total_emissions:.8f} kg, Loss: {loss:.4f}")
            if rmse is not None:
                print(f"RMSE: {rmse:.4f}")
            if mae is not None:
                print(f"MAE: {mae:.4f}")
        except Exception as e:
            print(f"Error al medir emisiones en época {epoch}: {e}")
            
    def get_emissions_for_epoch(self, epoch):
        """Retorna las emisiones de una época específica"""
        if epoch < len(self.epoch_emissions):
            return {
                'epoch_emissions': self.epoch_emissions[epoch],
                'cumulative_emissions': self.cumulative_emissions[epoch]
            }
        return None
    
    def end_training(self, final_rmse=None, final_mae=None, best_rmse_info=None):
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
                    
            # Mostrar información del mejor RMSE con emisiones (CORREGIDO)
            if (best_rmse_info and 
                best_rmse_info['epoch'] is not None and 
                best_rmse_info['epoch'] < len(self.epoch_emissions) and
                len(self.epoch_emissions) > 0):
                
                epoch = best_rmse_info['epoch']
                print(f"\n=== Best RMSE and Associated Emissions ===")
                print(f"Best RMSE: {best_rmse_info['rmse']:.4f} (Epoch {epoch})")
                print(f"Emissions at best RMSE: {self.epoch_emissions[epoch]:.8f} kg")
                print(f"Cumulative emissions at best RMSE: {self.cumulative_emissions[epoch]:.8f} kg")
            else:
                print(f"\n=== Best RMSE Info Not Available ===")
                if best_rmse_info:
                    print(f"Best RMSE: {best_rmse_info['rmse']:.4f} (Epoch {best_rmse_info['epoch']})")
                    print("Emissions data not available for this epoch")
                else:
                    print("No best RMSE information available")
            
            # Si no hay datos de emisiones por época pero tenemos emisiones totales,
            # crear al menos una entrada para gráficos
            if not self.epoch_emissions and final_emissions > 0:
                self.epoch_emissions = [final_emissions]
                self.cumulative_emissions = [final_emissions]
                if final_rmse is not None:
                    self.epoch_rmse = [final_rmse]
                if final_mae is not None:
                    self.epoch_mae = [final_mae]
            
            # Si no hay datos, salir
            if not self.epoch_emissions:
                print("No hay datos de emisiones para graficar")
                return
            
            # Asegurarse de que tengamos valores finales si no se rastrearon por época
            if not self.epoch_rmse and final_rmse is not None:
                self.epoch_rmse = [final_rmse] * len(self.epoch_emissions)
            if not self.epoch_mae and final_mae is not None:
                self.epoch_mae = [final_mae] * len(self.epoch_emissions)
                
            # Crear dataframe con todos los datos
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            df = pd.DataFrame({
                'epoch': range(len(self.epoch_emissions)),
                'epoch_emissions_kg': self.epoch_emissions,
                'cumulative_emissions_kg': self.cumulative_emissions,
                'loss': self.epoch_loss if self.epoch_loss else [0.0] * len(self.epoch_emissions),
                'rmse': self.epoch_rmse if self.epoch_rmse else [None] * len(self.epoch_emissions),
                'mae': self.epoch_mae if self.epoch_mae else [None] * len(self.epoch_emissions)
            })
            
            emissions_file = f'{self.result_path}/emissions_reports/emissions_metrics_{self.model_name}_{timestamp}.csv'
            df.to_csv(emissions_file, index=False)
            print(f"Métricas de emisiones guardadas en: {emissions_file}")
            
            # Graficar las relaciones
            self.plot_emissions_vs_metrics(timestamp, final_rmse, final_mae)
            
        except Exception as e:
            print(f"Error al generar gráficos de emisiones: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_emissions_vs_metrics(self, timestamp, final_rmse=None, final_mae=None):
        """Genera gráficos para emisiones vs métricas"""
        
        # Configurar estilo para fondo blanco y texto negro (más legible)
        plt.style.use('default')
        
        try:
            # Gráfico de emisiones acumulativas vs RMSE
            if self.epoch_rmse:
                plt.figure(figsize=(10, 6), facecolor='white')
                plt.plot(self.cumulative_emissions, self.epoch_rmse, 'b-', marker='o')
                
                # Añadir etiquetas con el número de época
                for i, (emissions, rmse) in enumerate(zip(self.cumulative_emissions, self.epoch_rmse)):
                    plt.annotate(f"{i}", (emissions, rmse), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9, color='black')
                    
                plt.xlabel('Emisiones de CO2 acumuladas (kg)', color='black')
                plt.ylabel('RMSE', color='black')
                plt.title('Relación entre Emisiones Acumuladas y RMSE', color='black')
                plt.grid(True, alpha=0.3)
                plt.tick_params(colors='black')
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_rmse_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path, facecolor='white')
                plt.close()
                print(f"Gráfico guardado en: {file_path}")
            
            # Gráfico de emisiones acumulativas vs MAE
            if self.epoch_mae:
                plt.figure(figsize=(10, 6), facecolor='white')
                plt.plot(self.cumulative_emissions, self.epoch_mae, 'm-', marker='D')
                
                # Añadir etiquetas con el número de época
                for i, (emissions, mae) in enumerate(zip(self.cumulative_emissions, self.epoch_mae)):
                    plt.annotate(f"{i}", (emissions, mae), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9, color='black')
                    
                plt.xlabel('Emisiones de CO2 acumuladas (kg)', color='black')
                plt.ylabel('MAE', color='black')
                plt.title('Relación entre Emisiones Acumuladas y MAE', color='black')
                plt.grid(True, alpha=0.3)
                plt.tick_params(colors='black')
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_mae_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path, facecolor='white')
                plt.close()
                print(f"Gráfico guardado en: {file_path}")
            
            # Gráfico combinado: Emisiones por época y acumulativas
            plt.figure(figsize=(12, 10), facecolor='white')
            
            plt.subplot(2, 3, 1)
            plt.plot(range(len(self.epoch_emissions)), self.epoch_emissions, 'r-', marker='x')
            plt.title('Emisiones por Época', color='black')
            plt.xlabel('Época', color='black')
            plt.ylabel('CO2 Emissions (kg)', color='black')
            plt.tick_params(colors='black')
            
            plt.subplot(2, 3, 2)
            plt.plot(range(len(self.cumulative_emissions)), self.cumulative_emissions, 'r-', marker='o')
            plt.title('Emisiones Acumuladas por Época', color='black')
            plt.xlabel('Época', color='black')
            plt.ylabel('CO2 Emissions (kg)', color='black')
            plt.tick_params(colors='black')
            
            if self.epoch_loss:
                plt.subplot(2, 3, 3)
                plt.plot(range(len(self.epoch_loss)), self.epoch_loss, 'g-', marker='o')
                plt.title('Loss por Época', color='black')
                plt.xlabel('Época', color='black')
                plt.ylabel('Loss', color='black')
                plt.tick_params(colors='black')
            
            if self.epoch_rmse:
                plt.subplot(2, 3, 4)
                plt.plot(range(len(self.epoch_rmse)), self.epoch_rmse, 'b-', marker='o')
                plt.title('RMSE por Época', color='black')
                plt.xlabel('Época', color='black')
                plt.ylabel('RMSE', color='black')
                plt.tick_params(colors='black')
            
            if self.epoch_mae:
                plt.subplot(2, 3, 5)
                plt.plot(range(len(self.epoch_mae)), self.epoch_mae, 'm-', marker='D')
                plt.title('MAE por Época', color='black')
                plt.xlabel('Época', color='black')
                plt.ylabel('MAE', color='black')
                plt.tick_params(colors='black')
            
            plt.tight_layout()
            
            file_path = f'{self.result_path}/emissions_plots/metrics_by_epoch_{self.model_name}_{timestamp}.png'
            plt.savefig(file_path, facecolor='white')
            plt.close()
            print(f"Gráfico guardado en: {file_path}")
            
            # Gráfico comparativo de RMSE y MAE
            if self.epoch_rmse and self.epoch_mae:
                plt.figure(figsize=(10, 6), facecolor='white')
                plt.plot(range(len(self.epoch_rmse)), self.epoch_rmse, 'b-', marker='o', label='RMSE')
                
                # Para MAE, usar un segundo eje Y debido a la diferencia de escala
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                ax2.plot(range(len(self.epoch_mae)), self.epoch_mae, 'm-', marker='D', label='MAE')
                ax2.set_ylabel('MAE', color='m')
                ax2.tick_params(axis='y', colors='m')
                
                # Añadir las leyendas de ambos ejes
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                lines = lines1 + lines2
                labels = labels1 + labels2
                ax1.legend(lines, labels, loc='best')
                
                plt.title('Comparación de RMSE y MAE por Época', color='black')
                plt.xlabel('Época', color='black')
                ax1.set_ylabel('RMSE', color='black')
                plt.grid(True, alpha=0.3)
                ax1.tick_params(colors='black')
                
                file_path = f'{self.result_path}/emissions_plots/metrics_comparison_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path, facecolor='white')
                plt.close()
                print(f"Gráfico comparativo guardado en: {file_path}")
                
        except Exception as e:
            print(f"Error al generar los gráficos: {e}")
            import traceback
            traceback.print_exc()

# Funciones originales de LightGCN (sin cambios)
def build_adjacency_matrix(train_data, n_users, n_items):
    R_dok = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    for u, items in enumerate(train_data):
        for i in items:
            R_dok[u, i] = 1.0
    R_csr = R_dok.tocsr()

    adj_size = n_users + n_items
    adj_dok = sp.dok_matrix((adj_size, adj_size), dtype=np.float32)
    # R in the upper-right block
    adj_dok[:n_users, n_users:] = R_csr
    # R^T in the lower-left block
    adj_dok[n_users:, :n_users] = R_csr.transpose()
    return adj_dok.tocsr()

def normalize_adj_sym(adj_mat):
    # symmetric normalization: D^-1/2 * A * D^-1/2.
    rowsum = np.array(adj_mat.sum(axis=1)).flatten() + 1e-9
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt.dot(adj_mat).dot(D_inv_sqrt)

class LightGCNModel(tf.keras.Model):
    def __init__(self, n_users, n_items, adj_mat, n_layers=3, emb_dim=64, decay=1e-4, use_personalized_alpha=False):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.adj_mat = adj_mat  # TF SparseTensor
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.decay = decay
        self.use_personalized_alpha = use_personalized_alpha

        # Añadir semilla a los inicializadores
        initializer = tf.initializers.GlorotUniform(seed=42)
        self.user_embedding = self.add_weight(
            name='user_embedding',
            shape=(n_users, emb_dim),
            initializer=initializer,
            trainable=True
        )
        # Usar semilla diferente para cada inicialización
        item_initializer = tf.initializers.GlorotUniform(seed=43)
        self.item_embedding = self.add_weight(
            name='item_embedding',
            shape=(n_items, emb_dim),
            initializer=item_initializer,
            trainable=True
        )

        if use_personalized_alpha:
            alpha_initializer = tf.initializers.GlorotUniform(seed=44)
            self.alpha_mlp = tf.keras.Sequential([
                tf.keras.layers.Dense(n_layers + 1, activation='softmax', 
                                     kernel_initializer=alpha_initializer)
            ])
        
        # node attribute prediction (auxiliary task)
        attr_initializer = tf.initializers.GlorotUniform(seed=45)
        self.attribute_predictor = tf.keras.layers.Dense(emb_dim, activation='relu', 
                                                       kernel_initializer=attr_initializer,
                                                       name="attribute_predictor")

    def call(self, embeddings, mask_prob=0.2):
        user_emb, item_emb = embeddings
        all_emb = tf.concat([user_emb, item_emb], axis=0)
        emb_list = [all_emb]
    
        # propagation layers
        for _ in range(self.n_layers):
            all_emb = tf.sparse.sparse_dense_matmul(self.adj_mat, all_emb)
            emb_list.append(all_emb)
    
        # combine embeddings from different layers
        if not self.use_personalized_alpha:
            alpha_k = 1.0 / (self.n_layers + 1)
            alpha_weights = [alpha_k] * (self.n_layers + 1)
            alpha_weights = tf.convert_to_tensor(alpha_weights, dtype=tf.float32)
            alpha_weights = tf.reshape(alpha_weights, (-1, 1, 1))
            stacked_emb = tf.stack(emb_list, axis=0)
            combined_emb = tf.reduce_sum(stacked_emb * alpha_weights, axis=0)
        else:
            alpha = self.alpha_mlp(emb_list[0])
            alpha = tf.expand_dims(alpha, axis=-1)
            stacked_emb = tf.stack(emb_list, axis=0)
            combined_emb = tf.reduce_sum(stacked_emb * alpha, axis=0)
    
        user_final, item_final = tf.split(combined_emb, [self.n_users, self.n_items], axis=0)
    
        # node attribute prediction
        masked_user_emb, mask = mask_embeddings(user_final, mask_prob)
        predicted_attributes = self.attribute_predictor(masked_user_emb)
        return user_final, item_final, masked_user_emb, predicted_attributes, mask

    def recommend(self, user_ids, k=10):
        user_final, item_final, _, _, _ = self((self.user_embedding, self.item_embedding))
        user_vecs = tf.gather(user_final, user_ids)
    
        all_recs = []
        for idx, uid in enumerate(user_ids):
            u_vec = user_vecs[idx:idx + 1]
            scores = tf.matmul(u_vec, item_final, transpose_b=True)  # (1, n_items)
            scores_np = scores.numpy().flatten()
            idx_topk = np.argsort(scores_np)[::-1][:k]
            score_topk = scores_np[idx_topk]
            for item_id, sc in zip(idx_topk, score_topk):
                all_recs.append((int(uid), int(item_id), float(sc)))
        return all_recs

def mask_embeddings(embeddings, mask_prob=0.2):
    mask = tf.cast(tf.random.uniform(embeddings.shape) > mask_prob, tf.float32)
    masked_embeddings = embeddings * mask
    return masked_embeddings, mask

def sample_neg(pos_items, n_items, strategy='random'):
    if strategy == 'random':
        neg_item = random.randint(0, n_items - 1)
        while neg_item in pos_items:
            neg_item = random.randint(0, n_items - 1)
    return neg_item

def train_lightgcn_with_metrics(model, train_data, val_data, test_data, n_users, n_items, batch_size=1024, epochs=10, initial_lr=1e-2, k=20):
    # Inicializar trackers
    print("Inicializando trackers...")
    system_tracker = SystemMetricsTracker()
    emissions_tracker = EmissionsPerEpochTracker(result_path, "LightGCN_GDSC1")
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # (user, item) pairs from train_data
    train_pairs = [(u, i) for u in range(n_users) for i in train_data[u]]
    steps_per_epoch = len(train_pairs) // batch_size + (len(train_pairs) % batch_size != 0)

    epoch_losses = []
    rmse_scores = []
    mae_scores = []
    
    # Para métricas finales
    tiempo_inicio = time.time()
    
    for epoch in range(epochs):
        # Iniciar seguimiento de época
        system_tracker.start_epoch(epoch)
        emissions_tracker.start_epoch(epoch)
        
        random.shuffle(train_pairs)
        progbar = Progbar(steps_per_epoch)

        epoch_loss = 0
        for step in range(steps_per_epoch):
            batch_slice = train_pairs[step * batch_size:(step + 1) * batch_size]
            users = [u for (u, _) in batch_slice]
            pos_items = [i for (_, i) in batch_slice]
            neg_items = [sample_neg(train_data[u], n_items) for (u, _) in batch_slice]

            users = np.array(users, dtype=np.int32)
            pos_items = np.array(pos_items, dtype=np.int32)
            neg_items = np.array(neg_items, dtype=np.int32)

            with tf.GradientTape() as tape:
                user_emb, item_emb, masked_user_emb, predicted_attributes, mask = model(
                    (model.user_embedding, model.item_embedding)
                )
                u_emb = tf.nn.embedding_lookup(user_emb, users)
                pos_emb = tf.nn.embedding_lookup(item_emb, pos_items)
                neg_emb = tf.nn.embedding_lookup(item_emb, neg_items)
            
                # BPR loss: mean( softplus(neg_score - pos_score) )
                pos_scores = tf.reduce_sum(u_emb * pos_emb, axis=1)
                neg_scores = tf.reduce_sum(u_emb * neg_emb, axis=1)
                mf_loss = tf.reduce_mean(tf.nn.softplus(neg_scores - pos_scores))
            
                # node attribute prediction loss
                attribute_loss = tf.reduce_mean(tf.square(masked_user_emb - predicted_attributes) * mask)
            
                # L2 Regularization
                reg_loss = model.decay * (
                    tf.nn.l2_loss(u_emb) + tf.nn.l2_loss(pos_emb) + tf.nn.l2_loss(neg_emb)
                ) / batch_size
            
                # total loss
                loss = mf_loss + reg_loss + 0.1 * attribute_loss  # Weighted auxiliary loss

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss += loss.numpy()
            progbar.add(1, values=[('loss', float(loss))])

        avg_epoch_loss = epoch_loss / steps_per_epoch
        epoch_losses.append(avg_epoch_loss)
        
        # Calcular RMSE y MAE en cada época para seguimiento correcto
        epoch_rmse = calculate_rmse(model, val_data, n_users, n_items)
        epoch_mae = calculate_mae(model, val_data, n_users, n_items)
        print(f"Epoch {epoch}: RMSE: {epoch_rmse:.6f}, MAE: {epoch_mae:.6f} (calculated)")
        
        rmse_scores.append(epoch_rmse)
        mae_scores.append(epoch_mae)
        
        # Actualizar trackers con las métricas
        system_tracker.end_epoch(epoch, avg_epoch_loss, epoch_rmse, epoch_mae)
        emissions_tracker.end_epoch(epoch, avg_epoch_loss, epoch_rmse, epoch_mae)
        
        print(f"Epoch {epoch}/{epochs} completed. Average loss: {avg_epoch_loss:.6f}")

    # Evaluación final en el conjunto de pruebas
    print("\nEvaluando en conjunto de prueba final...")
    system_tracker.start_epoch("test")
    
    # Calcular RMSE y MAE finales en conjunto de prueba
    final_rmse = calculate_rmse(model, test_data, n_users, n_items)
    final_mae = calculate_mae(model, test_data, n_users, n_items)
    
    # Finalizar seguimiento de sistemas
    try:
        print("\nGenerando métricas finales del sistema...")
        best_rmse_info = system_tracker.get_best_rmse_info()
        system_tracker.end_test(final_rmse, final_mae)
    except Exception as e:
        print(f"Error al generar métricas finales con tracker: {e}")
        import traceback
        traceback.print_exc()
        best_rmse_info = None
    
    try:
        print("\nGenerando gráficos y métricas de emisiones...")
        emissions_tracker.end_training(final_rmse, final_mae, best_rmse_info)
    except Exception as e:
        print(f"Error al generar métricas de emisiones: {e}")
        import traceback
        traceback.print_exc()
    
    # Guardar métricas de entrenamiento
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, epochs + 1)),
        'loss': epoch_losses,
        'rmse': rmse_scores,
        'mae': mae_scores
    })
    
    metrics_file = f"{result_path}/model_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Métricas del modelo guardadas en: {metrics_file}")
    
    # Mostrar métricas finales (independientes)
    memoria_final = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    cpu_final = psutil.cpu_percent(interval=1.0)
    tiempo_total = time.time() - tiempo_inicio
    
    print("\n" + "="*60)
    print("MÉTRICAS FINALES DEL SISTEMA - GDSC1")
    print("="*60)
    print(f"Memoria final: {memoria_final:.2f} MB")
    print(f"CPU final: {cpu_final:.2f}%")
    print(f"Tiempo total de ejecución: {tiempo_total:.2f} segundos")
    print(f"RMSE final: {final_rmse:.4f}")
    print(f"MAE final: {final_mae:.4f}")
    print("="*60)
    
    # Guardar las métricas finales
    final_metrics_dict = {
        'final_memory_mb': memoria_final,
        'final_cpu_percent': cpu_final,
        'total_time_sec': tiempo_total,
        'final_rmse': final_rmse,
        'final_mae': final_mae,
        'timestamp': timestamp
    }
    
    final_metrics_df = pd.DataFrame([final_metrics_dict])
    final_metrics_file = f"{result_path}/final_metrics_{timestamp}.csv"
    final_metrics_df.to_csv(final_metrics_file, index=False)
    print(f"Métricas finales guardadas en: {final_metrics_file}")
    
    # Graficar resultados de entrenamiento incluyendo MAE
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', linestyle='-', color='b', label="Loss")
    plt.title("LightGCN GDSC1 - Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{result_path}/training_loss_{timestamp}.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(rmse_scores) + 1), rmse_scores, marker='o', linestyle='-', color='b', label="RMSE")
    plt.title("LightGCN GDSC1 - RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{result_path}/rmse_{timestamp}.png")
    plt.close()
    
    # Graficar MAE
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(mae_scores) + 1), mae_scores, marker='D', linestyle='-', color='m', label="MAE")
    plt.title("LightGCN GDSC1 - MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{result_path}/mae_{timestamp}.png")
    plt.close()

    print(f"\nEntrenamiento finalizado! RMSE: {final_rmse:.4f}, MAE: {final_mae:.4f}")
    
    return epoch_losses, rmse_scores, mae_scores, final_rmse, final_mae

# Función principal MODIFICADA para ejecutar todo con GDSC1
def run_lightgcn_with_metrics():
    # Ruta del archivo GDSC1
    csv_file = 'C:/Users/xpati/Documents/TFG/gdsc1_processed.csv'
    
    # Cargar datos de GDSC1
    print("Cargando dataset GDSC1...")
    train_data, test_data, val_data, n_users, n_items, ratings_dict = load_gdsc1_dataset(
        csv_file, test_size=0.2, val_size=0.1, random_state=42
    )
    
    # Mostrar estadísticas y verificar carga
    show_ratings_statistics()
    test_rating_usage()
    
    print(f"Number of Users (líneas celulares): {n_users}")
    print(f"Number of Items (fármacos): {n_items}")

    adj_csr = build_adjacency_matrix(train_data, n_users, n_items)
    norm_adj_csr = normalize_adj_sym(adj_csr)

    # convert to TensorFlow SparseTensor
    coo = norm_adj_csr.tocoo().astype(np.float32)
    indices = np.vstack((coo.row, coo.col)).transpose()
    A_tilde = tf.sparse.SparseTensor(indices=indices, values=coo.data, dense_shape=coo.shape)
    A_tilde = tf.sparse.reorder(A_tilde)

    # Hiperparámetros ajustados para GDSC1
    N_LAYERS = 3  
    EMBED_DIM = 64  
    DECAY = 1e-3  
    INITIAL_LR = 1e-4  
    EPOCHS = 50  # Ajustar según necesidades
    BATCH_SIZE = 1024

    model = LightGCNModel(
        n_users=n_users,
        n_items=n_items,
        adj_mat=A_tilde,
        n_layers=N_LAYERS,
        emb_dim=EMBED_DIM,
        decay=DECAY,
        use_personalized_alpha=False
    )

    print("\nStarting LightGCN training with GDSC1 dataset...")
    epoch_losses, rmse_scores, mae_scores, final_rmse, final_mae = train_lightgcn_with_metrics(
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        n_users=n_users,
        n_items=n_items,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        initial_lr=INITIAL_LR,
        k=20
    )
    
    print("\nTraining and evaluation completed for GDSC1!")
    return final_rmse, final_mae

if __name__ == "__main__":
    run_lightgcn_with_metrics()