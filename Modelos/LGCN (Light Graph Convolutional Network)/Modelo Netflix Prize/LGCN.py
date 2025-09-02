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
result_path = "results_lgcn_netflix"
os.makedirs(result_path, exist_ok=True)
os.makedirs(f"{result_path}/emissions_reports", exist_ok=True)
os.makedirs(f"{result_path}/emissions_plots", exist_ok=True)

# Diccionario global para almacenar los ratings reales
ratings_dict = {}

def load_netflix_dataset(csv_file, test_size=0.2, val_size=0.1, random_state=42, sample_fraction=None):
    """
    Carga el dataset Netflix desde un archivo CSV y lo divide en train/test/val
    
    Args:
        csv_file: Ruta al archivo netflix.csv
        test_size: Proporción para conjunto de prueba
        val_size: Proporción para conjunto de validación
        random_state: Semilla para reproducibilidad
        sample_fraction: Fracción del dataset a usar (None para usar todo)
    
    Returns:
        train_data, test_data, val_data, n_users, n_items, ratings_dict
    """
    global ratings_dict
    
    print(f"Cargando dataset Netflix desde {csv_file}...")
    
    # Cargar el CSV
    df = pd.read_csv(csv_file)
    print(f"Dataset original cargado: {len(df)} interacciones")
    
    # Muestreo opcional para datasets muy grandes
    if sample_fraction and sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=random_state)
        print(f"Usando muestra del {sample_fraction*100:.1f}%: {len(df)} interacciones")
    
    # Verificar columnas esperadas (ajustar según el formato real del CSV)
    expected_columns = ['user_id', 'item_id', 'rating']
    if not all(col in df.columns for col in expected_columns):
        print(f"Columnas disponibles: {list(df.columns)}")
        # Intentar mapear columnas comunes del dataset Netflix
        if 'userId' in df.columns:
            df.rename(columns={'userId': 'user_id'}, inplace=True)
        if 'movieId' in df.columns:
            df.rename(columns={'movieId': 'item_id'}, inplace=True)
        if 'movie_id' in df.columns:
            df.rename(columns={'movie_id': 'item_id'}, inplace=True)
        print(f"Columnas después del mapeo: {list(df.columns)}")
    
    # Estadísticas básicas
    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()
    print(f"Usuarios únicos: {n_users}")
    print(f"Items únicos (películas): {n_items}")
    print(f"Rating promedio: {df['rating'].mean():.3f}")
    print(f"Rating mínimo: {df['rating'].min():.3f}")
    print(f"Rating máximo: {df['rating'].max():.3f}")
    
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
    print(f"\nEstadísticas de ratings reales:")
    print(f"  Total de ratings: {len(ratings_values)}")
    print(f"  Rating promedio: {np.mean(ratings_values):.3f}")
    print(f"  Rating mínimo: {min(ratings_values):.3f}")
    print(f"  Rating máximo: {max(ratings_values):.3f}")
    print(f"  Desviación estándar: {np.std(ratings_values):.3f}")
    
    # Distribución de ratings específica para Netflix (1-5)
    ratings_array = np.array(ratings_values)
    unique_ratings, counts = np.unique(ratings_array, return_counts=True)
    print(f"  Distribución de ratings:")
    for rating, count in zip(unique_ratings, counts):
        percentage = count / len(ratings_values) * 100
        print(f"    Rating {rating}: {count} ({percentage:.1f}%)")

def test_rating_usage():
    """Función de debug para verificar que se usan ratings reales"""
    global ratings_dict
    if not ratings_dict:
        print("PROBLEMA: No hay ratings cargados!")
        return
    
    # Tomar 10 ejemplos aleatorios
    sample_ratings = random.sample(list(ratings_dict.items()), min(10, len(ratings_dict)))
    print(f"\nVERIFICACIÓN: Muestra de ratings reales que se usarán:")
    for (u, item_id), rating in sample_ratings:
        print(f"  Usuario {u}, Item {item_id} → Rating real: {rating:.1f}")
    
    # Verificar distribución de la muestra
    sample_values = [rating for _, rating in sample_ratings]
    print(f"\nDistribución en muestra:")
    print(f"  Rating promedio en muestra: {np.mean(sample_values):.3f}")
    print(f"  Rating mínimo en muestra: {min(sample_values):.1f}")
    print(f"  Rating máximo en muestra: {max(sample_values):.1f}")

# Función OPTIMIZADA para calcular RMSE - ADAPTADA PARA NETFLIX
def calculate_rmse(model, test_data, n_users, n_items, sample_size=1000):
    """
    Calcula el RMSE usando los ratings reales de Netflix (escala 1-5) - OPTIMIZADA.
    """
    global ratings_dict
    
    if not ratings_dict:
        return 1.0
    
    # Tomar muestra para estabilidad
    all_ratings = list(ratings_dict.items())
    sample_ratings = random.sample(all_ratings, min(sample_size, len(all_ratings)))
    
    # Obtener embeddings UNA SOLA VEZ
    user_emb, item_emb, _, _, _ = model((model.user_embedding, model.item_embedding))
    
    # Calcular TODOS los scores primero para normalización eficiente
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
    
    # Normalización global EFICIENTE para el rango de Netflix (1-5)
    min_score, max_score = np.min(all_scores), np.max(all_scores)
    
    y_true = []
    y_pred = []
    
    for i, ((u, item_id), true_rating) in enumerate(valid_pairs):
        score = all_scores[i]
        
        # Mapeo mejorado a escala del dataset Netflix [1, 5]
        if max_score > min_score:
            # Mapear a rango Netflix 1-5
            predicted_rating = 1.0 + 4.0 * ((score - min_score) / (max_score - min_score))
        else:
            predicted_rating = 3.0  # valor neutro en la escala Netflix
        
        y_true.append(true_rating)
        y_pred.append(predicted_rating)
    
    # Calcular RMSE
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

# Función OPTIMIZADA para calcular MAE - ADAPTADA PARA NETFLIX
def calculate_mae(model, test_data, n_users, n_items, sample_size=1000):
    """
    Calcula el MAE usando los ratings reales de Netflix (escala 1-5) - OPTIMIZADA.
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
    
    # Normalización global
    min_score, max_score = np.min(all_scores), np.max(all_scores)
    
    y_true = []
    y_pred = []
    
    for i, ((u, item_id), true_rating) in enumerate(valid_pairs):
        score = all_scores[i]
        
        if max_score > min_score:
            # Mapear a rango del dataset Netflix 1-5
            predicted_rating = 1.0 + 4.0 * ((score - min_score) / (max_score - min_score))
        else:
            predicted_rating = 3.0
        
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
        
        # Imprimir resumen de época
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
        
        # Imprimir métricas finales
        print("\n=== Final Training Metrics ===")
        for m in self.train_metrics:
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
                    
            # Mostrar información del mejor RMSE con emisiones
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


def build_adjacency_matrix(train_data, n_users, n_items):
    """
    Construye la matriz de adyacencia de forma optimizada para datasets grandes
    """
    print(f"Construyendo matriz de adyacencia para {n_users} usuarios y {n_items} items...")
    
    # Usar lil_matrix para construcción eficiente con datasets grandes
    R_lil = sp.lil_matrix((n_users, n_items), dtype=np.float32)
    
    # Llenar la matriz por lotes para evitar problemas de memoria
    batch_size = 10000
    total_interactions = 0
    
    for u in range(0, n_users, batch_size):
        end_u = min(u + batch_size, n_users)
        for user_idx in range(u, end_u):
            if user_idx < len(train_data):
                items = train_data[user_idx]
                for item in items:
                    if item < n_items:  # Verificar límites
                        R_lil[user_idx, item] = 1.0
                        total_interactions += 1
        
        if (u // batch_size) % 100 == 0:
            print(f"Procesado batch {u//batch_size}, usuarios {u}-{end_u-1}")
    
    print(f"Total de interacciones procesadas: {total_interactions}")
    
    # Convertir a CSR para operaciones eficientes
    print("Convirtiendo a formato CSR...")
    R_csr = R_lil.tocsr()
    del R_lil  # Liberar memoria
    
    print("Construyendo matriz de adyacencia bipartita...")
    adj_size = n_users + n_items
    
    # Usar lil_matrix para la matriz de adyacencia también
    adj_lil = sp.lil_matrix((adj_size, adj_size), dtype=np.float32)
    
    # Copiar R en el bloque superior derecho
    print("Copiando interacciones usuario-item...")
    adj_lil[:n_users, n_users:] = R_csr
    
    # Copiar R^T en el bloque inferior izquierdo
    print("Copiando interacciones item-usuario...")
    adj_lil[n_users:, :n_users] = R_csr.transpose()
    
    print("Convirtiendo matriz de adyacencia final a CSR...")
    adj_csr = adj_lil.tocsr()
    del adj_lil  # Liberar memoria
    
    print(f"Matriz de adyacencia construida: {adj_csr.shape}")
    return adj_csr

def load_netflix_dataset(csv_file, test_size=0.2, val_size=0.1, random_state=42, sample_fraction=None, max_users=None, min_interactions=5):
    """
    Carga el dataset Netflix con optimizaciones para memoria
    
    Args:
        csv_file: Ruta al archivo netflix.csv
        test_size: Proporción para conjunto de prueba
        val_size: Proporción para conjunto de validación
        random_state: Semilla para reproducibilidad
        sample_fraction: Fracción del dataset a usar (None para usar todo)
        max_users: Número máximo de usuarios a incluir (None para todos)
        min_interactions: Mínimo de interacciones por usuario para incluirlo
    """
    global ratings_dict
    
    print(f"Cargando dataset Netflix desde {csv_file}...")
    
    # Cargar el CSV con chunks para manejar archivos grandes
    chunk_size = 1000000
    df_chunks = []
    
    print("Leyendo archivo CSV por chunks...")
    for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
        df_chunks.append(chunk)
        if len(df_chunks) % 10 == 0:
            print(f"Procesados {len(df_chunks)} chunks...")
    
    df = pd.concat(df_chunks, ignore_index=True)
    del df_chunks  # Liberar memoria
    
    print(f"Dataset original cargado: {len(df)} interacciones")
    
    # Mapear columnas
    if 'movie_id' in df.columns:
        df.rename(columns={'movie_id': 'item_id'}, inplace=True)
    
    # Filtrar usuarios con pocas interacciones
    print(f"Filtrando usuarios con menos de {min_interactions} interacciones...")
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    df = df[df['user_id'].isin(valid_users)]
    print(f"Después del filtrado: {len(df)} interacciones, {df['user_id'].nunique()} usuarios")
    
    # Limitar número de usuarios si se especifica
    if max_users is not None:
        top_users = df['user_id'].value_counts().head(max_users).index
        df = df[df['user_id'].isin(top_users)]
        print(f"Limitado a top {max_users} usuarios: {len(df)} interacciones")
    
    # Muestreo opcional
    if sample_fraction and sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=random_state)
        print(f"Usando muestra del {sample_fraction*100:.1f}%: {len(df)} interacciones")
    
    # Estadísticas
    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()
    print(f"Usuarios únicos: {n_users}")
    print(f"Items únicos (películas): {n_items}")
    print(f"Rating promedio: {df['rating'].mean():.3f}")
    
    # Crear mapeos para índices consecutivos
    print("Creando mapeos de usuarios e items...")
    unique_users = sorted(df['user_id'].unique())
    unique_items = sorted(df['item_id'].unique())
    
    user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    item_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
    
    # Aplicar mapeos
    df['user_idx'] = df['user_id'].map(user_map).astype(int)
    df['item_idx'] = df['item_id'].map(item_map).astype(int)
    
    # Crear diccionario de ratings de forma eficiente
    print("Creando diccionario de ratings...")
    ratings_dict = {}
    for _, row in df.iterrows():
        ratings_dict[(int(row['user_idx']), int(row['item_idx']))] = row['rating']
    
    print(f"Ratings cargados en diccionario: {len(ratings_dict)}")
    
    # División de datos
    print("Dividiendo dataset...")
    train_df, temp_df = train_test_split(df, test_size=test_size + val_size, 
                                         random_state=random_state)
    
    relative_val_size = val_size / (test_size + val_size)
    test_df, val_df = train_test_split(temp_df, test_size=relative_val_size, 
                                       random_state=random_state)
    
    print(f"División del dataset:")
    print(f"  Train: {len(train_df)} interacciones ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} interacciones ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df)} interacciones ({len(val_df)/len(df)*100:.1f}%)")
    
    # Convertir a formato LightGCN
    def df_to_lightgcn_format(data_df, n_users):
        user_items = [set() for _ in range(n_users)]
        for _, row in data_df.iterrows():
            user_idx = int(row['user_idx'])
            item_idx = int(row['item_idx'])
            if user_idx < n_users:  # Verificar límites
                user_items[user_idx].add(item_idx)
        return user_items
    
    train_data = df_to_lightgcn_format(train_df, n_users)
    test_data = df_to_lightgcn_format(test_df, n_users)
    val_data = df_to_lightgcn_format(val_df, n_users)
    
    # Verificaciones finales
    empty_users = sum(1 for items in train_data if len(items) == 0)
    if empty_users > 0:
        print(f"Advertencia: {empty_users} usuarios sin interacciones en train")
    
    return train_data, test_data, val_data, n_users, n_items, ratings_dict


def normalize_adj_sym(adj_mat):
    """Normalización simétrica de la matriz de adyacencia"""
    rowsum = np.array(adj_mat.sum(axis=1)).flatten() + 1e-9
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt.dot(adj_mat).dot(D_inv_sqrt)

def mask_embeddings(embeddings, mask_prob=0.2):
    """Aplica masking a los embeddings para regularización"""
    mask = tf.cast(tf.random.uniform(embeddings.shape) > mask_prob, tf.float32)
    masked_embeddings = embeddings * mask
    return masked_embeddings

def sample_neg(pos_items, n_items, strategy='random'):
    """Muestreo de items negativos"""
    neg_item = random.randint(0, n_items - 1)
    while neg_item in pos_items:
        neg_item = random.randint(0, n_items - 1)
    return neg_item

class LightGCNModel(tf.keras.Model):
    def __init__(self, n_users, n_items, adj_mat, n_layers=3, emb_dim=64, decay=1e-4, use_personalized_alpha=False):
        super(LightGCNModel, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.adj_mat = adj_mat
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.decay = decay
        self.use_personalized_alpha = use_personalized_alpha
        
        # Inicializar embeddings
        self.user_embedding = tf.Variable(
            tf.random.normal([n_users, emb_dim], stddev=0.1),
            name='user_embedding'
        )
        self.item_embedding = tf.Variable(
            tf.random.normal([n_items, emb_dim], stddev=0.1),
            name='item_embedding'
        )
        
        # Parámetros de alpha personalizados si se usan
        if use_personalized_alpha:
            self.alphas = tf.Variable(
                tf.ones([n_layers + 1]) / (n_layers + 1),
                name='alphas'
            )

    def call(self, embeddings, mask_prob=0.2):
        user_emb, item_emb = embeddings
        
        # Aplicar masking si está en entrenamiento
        if mask_prob > 0:
            user_emb = mask_embeddings(user_emb, mask_prob)
            item_emb = mask_embeddings(item_emb, mask_prob)
        
        # Concatenar embeddings de usuarios e items
        all_emb = tf.concat([user_emb, item_emb], axis=0)
        embs = [all_emb]
        
        # Propagación por capas
        for layer in range(self.n_layers):
            all_emb = tf.sparse.sparse_dense_matmul(self.adj_mat, all_emb)
            embs.append(all_emb)
        
        # Combinar embeddings de todas las capas
        if self.use_personalized_alpha:
            final_emb = tf.zeros_like(embs[0])
            for i, emb in enumerate(embs):
                final_emb += self.alphas[i] * emb
        else:
            final_emb = tf.reduce_mean(tf.stack(embs, axis=1), axis=1)
        
        # Separar embeddings de usuarios e items
        user_final = final_emb[:self.n_users]
        item_final = final_emb[self.n_users:]
        
        return user_final, item_final, user_emb, item_emb, all_emb

    def recommend(self, user_ids, k=10):
        """Genera recomendaciones para usuarios dados"""
        user_emb, item_emb, _, _, _ = self((self.user_embedding, self.item_embedding), mask_prob=0.0)
        
        user_embeddings = tf.nn.embedding_lookup(user_emb, user_ids)
        scores = tf.matmul(user_embeddings, item_emb, transpose_b=True)
        
        # Obtener top-k items
        _, top_items = tf.nn.top_k(scores, k=k)
        
        return top_items.numpy()

def train_lightgcn_with_metrics(model, train_data, val_data, test_data, n_users, n_items, 
                               batch_size=1024, epochs=10, initial_lr=1e-2, k=20):
    """Función de entrenamiento con métricas"""
    
    system_tracker = SystemMetricsTracker()
    emissions_tracker = EmissionsPerEpochTracker(result_path, "LightGCN_Netflix")
    
    # Configurar optimizador
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr, 
        decay_steps=1000, 
        decay_rate=0.96, 
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Preparar datos de entrenamiento
    train_pairs = []
    for u in range(n_users):
        if u < len(train_data):
            for item in train_data[u]:
                train_pairs.append((u, item))
    
    steps_per_epoch = len(train_pairs) // batch_size + (len(train_pairs) % batch_size != 0)
    
    # Usuarios de validación y test
    val_users = [u for u in range(min(n_users, len(val_data))) if val_data[u]]
    test_users = [u for u in range(min(n_users, len(test_data))) if test_data[u]]

    epoch_losses = []
    rmse_scores = []
    mae_scores = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        system_tracker.start_epoch(epoch)
        emissions_tracker.start_epoch(epoch)
        
        # Shuffle datos de entrenamiento
        random.shuffle(train_pairs)
        
        epoch_loss = 0.0
        progress_bar = Progbar(steps_per_epoch)
        
        for step in range(steps_per_epoch):
            # Crear batch
            start_idx = step * batch_size
            end_idx = min(start_idx + batch_size, len(train_pairs))
            batch_pairs = train_pairs[start_idx:end_idx]
            
            if not batch_pairs:
                continue
                
            batch_users = []
            batch_pos_items = []
            batch_neg_items = []
            
            for u, pos_item in batch_pairs:
                # Muestrear item negativo
                if u < len(train_data):
                    neg_item = sample_neg(train_data[u], n_items)
                    batch_users.append(u)
                    batch_pos_items.append(pos_item)
                    batch_neg_items.append(neg_item)
            
            if not batch_users:
                continue
                
            batch_users = tf.constant(batch_users, dtype=tf.int32)
            batch_pos_items = tf.constant(batch_pos_items, dtype=tf.int32)
            batch_neg_items = tf.constant(batch_neg_items, dtype=tf.int32)
            
            # Entrenamiento step
            with tf.GradientTape() as tape:
                user_emb, item_emb, _, _, _ = model((model.user_embedding, model.item_embedding))
                
                # Obtener embeddings del batch
                u_emb = tf.nn.embedding_lookup(user_emb, batch_users)
                pos_i_emb = tf.nn.embedding_lookup(item_emb, batch_pos_items)
                neg_i_emb = tf.nn.embedding_lookup(item_emb, batch_neg_items)
                
                # Calcular scores
                pos_scores = tf.reduce_sum(u_emb * pos_i_emb, axis=1)
                neg_scores = tf.reduce_sum(u_emb * neg_i_emb, axis=1)
                
                # BPR Loss
                bpr_loss = -tf.reduce_mean(tf.nn.log_sigmoid(pos_scores - neg_scores))
                
                # Regularización L2
                reg_loss = model.decay * (
                    tf.reduce_sum(tf.square(u_emb)) +
                    tf.reduce_sum(tf.square(pos_i_emb)) +
                    tf.reduce_sum(tf.square(neg_i_emb))
                )
                
                total_loss = bpr_loss + reg_loss
            
            # Aplicar gradientes
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_loss += total_loss.numpy()
            progress_bar.update(step + 1, values=[('loss', total_loss.numpy())])
        
        avg_epoch_loss = epoch_loss / steps_per_epoch
        epoch_losses.append(avg_epoch_loss)
        
        # Evaluar en validación
        print("\nEvaluando en validación...")
        val_rmse = calculate_rmse(model, val_data, n_users, n_items, sample_size=1000)
        val_mae = calculate_mae(model, val_data, n_users, n_items, sample_size=1000)
        
        rmse_scores.append(val_rmse)
        mae_scores.append(val_mae)
        
        system_tracker.end_epoch(epoch, avg_epoch_loss, val_rmse, val_mae)
        emissions_tracker.end_epoch(epoch, avg_epoch_loss, val_rmse, val_mae)

    # Evaluación final en test
    print("\nEvaluando en conjunto de prueba...")
    system_tracker.start_epoch("test")
    
    final_rmse = calculate_rmse(model, test_data, n_users, n_items, sample_size=2000)
    final_mae = calculate_mae(model, test_data, n_users, n_items, sample_size=2000)
    
    system_tracker.end_test(final_rmse, final_mae)
    best_rmse_info = system_tracker.get_best_rmse_info()
    emissions_tracker.end_training(final_rmse, final_mae, best_rmse_info)
    
    print("\nEntrenamiento y evaluación completados!")
    return epoch_losses, rmse_scores, mae_scores, final_rmse, final_mae

# Actualizar la función run_lightgcn_with_metrics para usar los parámetros correctos
def run_lightgcn_with_metrics():
    # Ruta del archivo Netflix
    csv_file = 'C:/Users/xpati/Documents/TFG/netflix.csv'
    
    # Parámetros más conservadores para evitar problemas de memoria
    sample_fraction = 0.1  # 10% del dataset
    max_users = 20000     # Limitar a 20K usuarios máximo
    min_interactions = 5   # Usuarios con al menos 5 interacciones
    
    print(f"Configuración para dataset grande:")
    print(f"  - Muestra: {sample_fraction*100:.1f}% del dataset")
    print(f"  - Máximo usuarios: {max_users}")
    print(f"  - Mínimo interacciones por usuario: {min_interactions}")
    
    # Cargar datos de Netflix usando la función original (más simple)
    print("\nCargando dataset Netflix...")
    train_data, test_data, val_data, n_users, n_items = load_netflix_dataset(
        csv_file, 
        test_size=0.2, 
        val_size=0.1, 
        random_state=42, 
        sample_fraction=sample_fraction
    )
    
    # Verificar que el tamaño sea manejable
    estimated_memory_gb = (n_users + n_items) ** 2 * 4 / (1024**3)
    if estimated_memory_gb > 8:
        print(f"ADVERTENCIA: La matriz de adyacencia podría usar ~{estimated_memory_gb:.1f} GB de RAM")
        print("Considerando reducir sample_fraction")
        return None, None
    
    # Mostrar estadísticas
    show_ratings_statistics()
    test_rating_usage()
    
    print(f"\nDimensiones finales:")
    print(f"  Usuarios: {n_users}")
    print(f"  Items: {n_items}")
    print(f"  Total nodos: {n_users + n_items}")
    
    # Construir matriz de adyacencia
    print(f"Number of Users: {n_users}")
    print(f"Number of Items (movies): {n_items}")
    
    adj_csr = build_adjacency_matrix(train_data, n_users, n_items)
    norm_adj_csr = normalize_adj_sym(adj_csr)
    
    # Convertir a TensorFlow SparseTensor
    coo = norm_adj_csr.tocoo().astype(np.float32)
    indices = np.vstack((coo.row, coo.col)).transpose()
    A_tilde = tf.sparse.SparseTensor(indices=indices, values=coo.data, dense_shape=coo.shape)
    A_tilde = tf.sparse.reorder(A_tilde)

    # Hiperparámetros
    N_LAYERS = 2
    EMBED_DIM = 64
    DECAY = 1e-4
    INITIAL_LR = 1e-3
    EPOCHS = 5
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

    print(f"\nIniciando entrenamiento LightGCN con dataset Netflix...")
    print(f"Configuración: {N_LAYERS} capas, {EMBED_DIM}D embeddings, {EPOCHS} épocas")
    
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
    
    print("\nEntrenamiento completado exitosamente!")
    return final_rmse, final_mae

if __name__ == "__main__":
    run_lightgcn_with_metrics()