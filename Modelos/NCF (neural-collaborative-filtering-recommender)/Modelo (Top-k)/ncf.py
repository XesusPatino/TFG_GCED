import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import time
import psutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from codecarbon import EmissionsTracker
from collections import defaultdict
import random

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
    'top_k_values': [5, 10, 20, 50]  # Valores de K para las métricas Top-K
}

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
        
    def end_epoch(self, epoch, loss, rmse=None, recall_metrics=None, ndcg_metrics=None):
        epoch_time = time.time() - self.epoch_start_time
        self.current_epoch_metrics['epoch_time_sec'] = epoch_time
        self.current_epoch_metrics['loss'] = loss
        if rmse is not None:
            self.current_epoch_metrics['rmse'] = rmse
        if recall_metrics is not None:
            self.current_epoch_metrics.update(recall_metrics)
        if ndcg_metrics is not None:
            self.current_epoch_metrics.update(ndcg_metrics)
        self.train_metrics.append(self.current_epoch_metrics)
        
        # Rastrear el mejor RMSE
        if rmse is not None and rmse < self.best_rmse:
            self.best_rmse = rmse
            self.best_rmse_epoch = epoch
            self.best_rmse_metrics = self.current_epoch_metrics.copy()
        
        # Imprimir resumen de época en formato compacto
        memory_mb = self.current_epoch_metrics.get('memory_usage_mb', 0)
        cpu_percent = self.current_epoch_metrics.get('cpu_usage_percent', 0)
        
        output_parts = [f"Epoch {epoch}: Time={epoch_time:.2f}s, Memory={memory_mb:.2f}MB, CPU={cpu_percent:.1f}%"]
        if rmse is not None:
            output_parts.append(f"RMSE={rmse:.4f}")
        if recall_metrics:
            for k, v in recall_metrics.items():
                if k.startswith('recall_'):
                    k_val = k.split('_')[1]
                    output_parts.append(f"Recall@{k_val}={v:.4f}")
        if ndcg_metrics:
            for k, v in ndcg_metrics.items():
                if k.startswith('ndcg_'):
                    k_val = k.split('_')[1]
                    output_parts.append(f"NDCG@{k_val}={v:.4f}")
        
        print(", ".join(output_parts))
        
    def end_test(self, rmse, recall_metrics=None, ndcg_metrics=None):
        self.test_metrics = {
            'test_time_sec': time.time() - self.epoch_start_time,
            'total_time_sec': time.time() - self.start_time,
            'final_memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'final_cpu_usage_percent': psutil.cpu_percent(),
            'test_rmse': rmse,
        }
        if recall_metrics is not None:
            self.test_metrics.update(recall_metrics)
        if ndcg_metrics is not None:
            self.test_metrics.update(ndcg_metrics)
        
        # Imprimir métricas finales
        print("\n=== Final Training Metrics ===")
        for m in self.train_metrics:
            # Formato específico requerido
            output_parts = [f"Epoch {m['epoch']}: Time={m['epoch_time_sec']:.2f}s, Memory={m['memory_usage_mb']:.2f}MB, CPU={m['cpu_usage_percent']:.1f}%"]
            if 'rmse' in m and m['rmse'] is not None:
                output_parts.append(f"RMSE={m['rmse']:.4f}")
            
            # Añadir métricas de recall y ndcg en orden específico
            for k_val in [5, 10, 20, 50]:
                if f'recall_{k_val}' in m:
                    output_parts.append(f"Recall@{k_val}={m[f'recall_{k_val}']:.4f}")
                if f'ndcg_{k_val}' in m:
                    output_parts.append(f"NDCG@{k_val}={m[f'ndcg_{k_val}']:.4f}")
            
            print(", ".join(output_parts))
        
        print("\n=== Final Test Metrics ===")
        print(f"Total Time: {self.test_metrics['total_time_sec']:.2f}s (Test: {self.test_metrics['test_time_sec']:.2f}s)")
        print(f"Final Memory: {self.test_metrics['final_memory_usage_mb']:.2f}MB")
        print(f"Final CPU: {self.test_metrics['final_cpu_usage_percent']:.1f}%")
        print(f"RMSE: {rmse:.4f}")
        
        # Mostrar métricas Top-k finales
        for k_val in [5, 10, 20, 50]:
            if recall_metrics and f'recall_{k_val}' in recall_metrics:
                print(f"Recall@{k_val}: {recall_metrics[f'recall_{k_val}']:.4f}")
            if ndcg_metrics and f'ndcg_{k_val}' in ndcg_metrics:
                print(f"NDCG@{k_val}: {ndcg_metrics[f'ndcg_{k_val}']:.4f}")
        
        # Mostrar información del mejor RMSE durante el entrenamiento
        if self.best_rmse_epoch is not None:
            print(f"\n=== Best Training RMSE ===")
            print(f"Best RMSE: {self.best_rmse:.4f} (Epoch {self.best_rmse_epoch})")
            if self.best_rmse_metrics:
                print(f"Time: {self.best_rmse_metrics['epoch_time_sec']:.2f}s")
                print(f"Memory: {self.best_rmse_metrics['memory_usage_mb']:.2f}MB")
                print(f"CPU: {self.best_rmse_metrics['cpu_usage_percent']:.1f}%")
        
        # Guardar métricas en CSV
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        metrics_df = pd.DataFrame(self.train_metrics)
        metrics_df.to_csv(f"{result_path}/system_metrics_{timestamp}.csv", index=False)
        
        
class EmissionsPerEpochTracker:
    def __init__(self, result_path, model_name="NCF"):
        self.result_path = result_path
        self.model_name = model_name
        self.epoch_emissions = []
        self.cumulative_emissions = []
        self.epoch_rmse = []
        self.epoch_loss = []
        self.epoch_recall_5 = []
        self.epoch_recall_10 = []
        self.epoch_recall_20 = []
        self.epoch_recall_50 = []
        self.epoch_ndcg_5 = []
        self.epoch_ndcg_10 = []
        self.epoch_ndcg_20 = []
        self.epoch_ndcg_50 = []
        self.total_emissions = 0.0
        self.trackers = {}
        self.best_rmse = float('inf')
        self.best_rmse_epoch = None
        self.best_rmse_emissions = None
        self.best_rmse_cumulative_emissions = None
        
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
    
    def end_epoch(self, epoch, loss, rmse=None, recall_metrics=None, ndcg_metrics=None):
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
                # Rastrear el mejor RMSE y sus emisiones
                if rmse < self.best_rmse:
                    self.best_rmse = rmse
                    self.best_rmse_epoch = epoch
                    self.best_rmse_emissions = epoch_co2
                    self.best_rmse_cumulative_emissions = self.total_emissions
            
            # Guardar métricas Top-k de cada época
            if recall_metrics:
                self.epoch_recall_5.append(recall_metrics.get('recall_5', 0.0))
                self.epoch_recall_10.append(recall_metrics.get('recall_10', 0.0))
                self.epoch_recall_20.append(recall_metrics.get('recall_20', 0.0))
                self.epoch_recall_50.append(recall_metrics.get('recall_50', 0.0))
            else:
                self.epoch_recall_5.append(0.0)
                self.epoch_recall_10.append(0.0)
                self.epoch_recall_20.append(0.0)
                self.epoch_recall_50.append(0.0)
            
            if ndcg_metrics:
                self.epoch_ndcg_5.append(ndcg_metrics.get('ndcg_5', 0.0))
                self.epoch_ndcg_10.append(ndcg_metrics.get('ndcg_10', 0.0))
                self.epoch_ndcg_20.append(ndcg_metrics.get('ndcg_20', 0.0))
                self.epoch_ndcg_50.append(ndcg_metrics.get('ndcg_50', 0.0))
            else:
                self.epoch_ndcg_5.append(0.0)
                self.epoch_ndcg_10.append(0.0)
                self.epoch_ndcg_20.append(0.0)
                self.epoch_ndcg_50.append(0.0)
            
        except Exception as e:
            print(f"Error al medir emisiones en época {epoch}: {e}")
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
                # Rastrear el mejor RMSE y sus emisiones
                if rmse < self.best_rmse:
                    self.best_rmse = rmse
                    self.best_rmse_epoch = epoch
                    self.best_rmse_emissions = epoch_co2
                    self.best_rmse_cumulative_emissions = self.total_emissions
            
        except Exception as e:
            print(f"Error al medir emisiones en época {epoch}: {e}")
    
    def end_training(self, final_rmse, final_recall_metrics=None, final_ndcg_metrics=None):
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
            
            # Si no hay datos, salir
            if not self.epoch_emissions:
                print("No hay datos de emisiones para graficar")
                return
            
            # Asegurarse de que tengamos un RMSE final si no se rastreó por época
            if not self.epoch_rmse and final_rmse is not None:
                self.epoch_rmse = [final_rmse] * len(self.epoch_emissions)
            
            # Crear dataframe con todos los datos incluyendo métricas Top-k
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            df = pd.DataFrame({
                'epoch': range(len(self.epoch_emissions)),
                'epoch_emissions_kg': self.epoch_emissions,
                'cumulative_emissions_kg': self.cumulative_emissions,
                'loss': self.epoch_loss if self.epoch_loss else [0.0] * len(self.epoch_emissions),
                'rmse': self.epoch_rmse if self.epoch_rmse else [None] * len(self.epoch_emissions),
                'recall_5': self.epoch_recall_5,
                'recall_10': self.epoch_recall_10,
                'recall_20': self.epoch_recall_20,
                'recall_50': self.epoch_recall_50,
                'ndcg_5': self.epoch_ndcg_5,
                'ndcg_10': self.epoch_ndcg_10,
                'ndcg_20': self.epoch_ndcg_20,
                'ndcg_50': self.epoch_ndcg_50
            })
            
            emissions_file = f'{self.result_path}/emissions_reports/emissions_metrics_{self.model_name}_{timestamp}.csv'
            df.to_csv(emissions_file, index=False)
            print(f"Métricas de emisiones guardadas en: {emissions_file}")
            
            # Mostrar información del mejor RMSE y sus emisiones
            if self.best_rmse_epoch is not None:
                print(f"\n=== Best RMSE and Associated Emissions ===")
                print(f"Best RMSE: {self.best_rmse:.4f} (Epoch {self.best_rmse_epoch})")
                print(f"Emissions at best RMSE: {self.best_rmse_emissions:.8f} kg")
                print(f"Cumulative emissions at best RMSE: {self.best_rmse_cumulative_emissions:.8f} kg")
            
            # Mostrar métricas Top-k finales si están disponibles
            if final_recall_metrics or final_ndcg_metrics:
                print(f"\n=== Final Top-K Metrics ===")
                for k_val in [5, 10, 20, 50]:
                    if final_recall_metrics and f'recall_{k_val}' in final_recall_metrics:
                        print(f"Recall@{k_val}: {final_recall_metrics[f'recall_{k_val}']:.4f}")
                    if final_ndcg_metrics and f'ndcg_{k_val}' in final_ndcg_metrics:
                        print(f"NDCG@{k_val}: {final_ndcg_metrics[f'ndcg_{k_val}']:.4f}")
            
            # Graficar las relaciones (sin MAE)
            self.plot_emissions_vs_metrics(timestamp, final_rmse)
            
        except Exception as e:
            print(f"Error al generar gráficos de emisiones: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_emissions_vs_metrics(self, timestamp, final_rmse=None):
        """Genera gráficos para emisiones vs métricas"""
        
        # Configurar estilo para fondo blanco y texto negro (más legible)
        plt.style.use('default')
        
        # Usar RMSE por época si está disponible, sino crear lista con el RMSE final
        if not self.epoch_rmse and final_rmse is not None:
            self.epoch_rmse = [final_rmse] * len(self.epoch_emissions)
        
        try:
            if self.epoch_rmse:
                # 1. Emisiones acumulativas vs RMSE
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
            
            # 2. Gráfico combinado: Emisiones por época y acumulativas
            plt.figure(figsize=(12, 10), facecolor='white')
            
            plt.subplot(2, 2, 1)
            plt.plot(range(len(self.epoch_emissions)), self.epoch_emissions, 'r-', marker='x')
            plt.title('Emisiones por Época', color='black')
            plt.xlabel('Época', color='black')
            plt.ylabel('CO2 Emissions (kg)', color='black')
            plt.tick_params(colors='black')
            
            plt.subplot(2, 2, 2)
            plt.plot(range(len(self.cumulative_emissions)), self.cumulative_emissions, 'r-', marker='o')
            plt.title('Emisiones Acumuladas por Época', color='black')
            plt.xlabel('Época', color='black')
            plt.ylabel('CO2 Emissions (kg)', color='black')
            plt.tick_params(colors='black')
            
            if self.epoch_loss:
                plt.subplot(2, 2, 3)
                plt.plot(range(len(self.epoch_loss)), self.epoch_loss, 'g-', marker='o')
                plt.title('Loss por Época', color='black')
                plt.xlabel('Época', color='black')
                plt.ylabel('Loss', color='black')
                plt.tick_params(colors='black')
            
            if self.epoch_rmse:
                plt.subplot(2, 2, 4)
                plt.plot(range(len(self.epoch_rmse)), self.epoch_rmse, 'b-', marker='o')
                plt.title('RMSE por Época', color='black')
                plt.xlabel('Época', color='black')
                plt.ylabel('RMSE', color='black')
                plt.tick_params(colors='black')
            
            plt.tight_layout()
            
            file_path = f'{self.result_path}/emissions_plots/metrics_by_epoch_{self.model_name}_{timestamp}.png'
            plt.savefig(file_path, facecolor='white')
            plt.close()
            print(f"Gráfico guardado en: {file_path}")
            
            if self.epoch_rmse:
                # 3. Scatter plot de rendimiento frente a emisiones acumulativas
                plt.figure(figsize=(10, 6), facecolor='white')
                
                # Ajustar tamaño de los puntos según la época
                sizes = [(i+1)*20 for i in range(len(self.cumulative_emissions))]
                
                scatter = plt.scatter(self.epoch_rmse, self.cumulative_emissions, 
                            color='blue', marker='o', s=sizes, alpha=0.7)
                
                # Añadir etiquetas de época
                for i, (rmse, em) in enumerate(zip(self.epoch_rmse, self.cumulative_emissions)):
                    plt.annotate(f"{i}", (rmse, em), textcoords="offset points", 
                                xytext=(0,5), ha='center', fontsize=9, color='black')
                
                plt.ylabel('Emisiones de CO2 acumuladas (kg)', color='black')
                plt.xlabel('RMSE', color='black')
                plt.title('Relación entre RMSE y Emisiones Acumuladas', color='black')
                plt.grid(True, alpha=0.3)
                plt.tick_params(colors='black')
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_performance_scatter_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path, facecolor='white')
                plt.close()
                print(f"Gráfico guardado en: {file_path}")
                
            # 4. Si tenemos RMSE y MAE, crear un gráfico comparativo
            if self.epoch_mae and self.epoch_rmse:
                plt.figure(figsize=(10, 6), facecolor='white')
                plt.plot(range(len(self.epoch_rmse)), self.epoch_rmse, 'b-', marker='o', label='RMSE')
                plt.plot(range(len(self.epoch_mae)), self.epoch_mae, 'g-', marker='s', label='MAE')
                plt.title('Comparación de RMSE y MAE por Época', color='black')
                plt.xlabel('Época', color='black')
                plt.ylabel('Error', color='black')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tick_params(colors='black')
                
                file_path = f'{self.result_path}/emissions_plots/rmse_vs_mae_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path, facecolor='white')
                plt.close()
                print(f"Gráfico comparativo guardado en: {file_path}")
                
        except Exception as e:
            print(f"Error al generar los gráficos: {e}")
            import traceback
            traceback.print_exc()
            
            
# Load MovieLens dataset
movies_df = pd.read_csv('C://Users//xpati//Documents//TFG//ml-1m//movies.dat', 
                       sep='::', 
                       header=None, 
                       names=['movieId', 'title', 'genres'],
                       encoding='latin-1',
                       engine='python')

ratings_df = pd.read_csv('C://Users//xpati//Documents//TFG//ml-1m//ratings.dat', 
                        sep='::', 
                        header=None, 
                        names=['userId', 'movieId', 'rating', 'timestamp'],
                        encoding='latin-1',
                        engine='python')

# Display first few rows
print(movies_df.head())
print(ratings_df.head())


# Label encoding of user and movie IDs
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings_df['userId'] = user_encoder.fit_transform(ratings_df['userId'])
ratings_df['movieId'] = movie_encoder.fit_transform(ratings_df['movieId'])

# Obtener número de usuarios y películas únicas
n_users = len(user_encoder.classes_)
n_movies = len(movie_encoder.classes_)

print(f'Number of users: {n_users}')
print(f'Number of movies: {n_movies}')


class BPRDataset(Dataset):
    def __init__(self, ratings_df):
        self.ratings_df = ratings_df
        self.user_item_map = defaultdict(list)
        
        # Mapeo de usuario a ítems positivos (valorados)
        for _, row in ratings_df.iterrows():
            self.user_item_map[row['userId']].append(row['movieId'])
        
        # Lista de todos los ítems para muestreo negativo
        self.all_items = ratings_df['movieId'].unique()
        self.users = list(self.user_item_map.keys())
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        user = self.users[idx]
        pos_items = self.user_item_map[user]
        
        # Seleccionar un ítem positivo aleatorio
        pos_item = random.choice(pos_items)
        
        # Seleccionar un ítem negativo aleatorio (no valorado por el usuario)
        neg_item = random.choice(self.all_items)
        while neg_item in pos_items:
            neg_item = random.choice(self.all_items)
        
        return torch.tensor(user, dtype=torch.long), \
               torch.tensor(pos_item, dtype=torch.long), \
               torch.tensor(neg_item, dtype=torch.long)
               
               
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim=50, hidden_dim=128):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, user, item):
        user_embedded = self.user_embedding(user)
        item_embedded = self.movie_embedding(item)
        x = torch.cat([user_embedded, item_embedded], dim=1)
        return self.mlp(x).squeeze()
    
    def predict(self, user, item):
        return self.forward(user, item)
    
    
class MovieLensDataset(Dataset):
    """Dataset para cálculo de RMSE (usa tripletas usuario-ítem-rating)"""
    def __init__(self, ratings_df):
        self.users = ratings_df['userId'].values
        self.movies = ratings_df['movieId'].values
        self.ratings = ratings_df['rating'].values

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return torch.tensor(self.users[idx], dtype=torch.long), \
               torch.tensor(self.movies[idx], dtype=torch.long), \
               torch.tensor(self.ratings[idx], dtype=torch.float32)
               
               
def get_user_positive_items(ratings_df):
    """Crea un diccionario con los ítems positivos para cada usuario"""
    user_pos_items = {}
    for user in ratings_df['userId'].unique():
        user_pos_items[user] = set(ratings_df[ratings_df['userId'] == user]['movieId'].values)
    return user_pos_items

def generate_recommendations(model, user_ids, all_item_ids, k=10):
    """Genera recomendaciones Top-K para una lista de usuarios"""
    model.eval()
    recommendations = {}
    
    with torch.no_grad():
        for user_id in user_ids:
            # Crear tensores para el usuario y todos los ítems
            user_tensor = torch.tensor([user_id] * len(all_item_ids), dtype=torch.long)
            item_tensor = torch.tensor(all_item_ids, dtype=torch.long)
            
            # Obtener predicciones
            scores = model.predict(user_tensor, item_tensor)
            
            # Ordenar y obtener los Top-K ítems
            _, top_indices = torch.topk(scores, k=k)
            recommended_items = item_tensor[top_indices].numpy()
            recommendations[user_id] = recommended_items
    
    return recommendations

def calculate_topk_metrics(model, test_df, train_df, user_pos_items, all_item_ids, k_values=[5, 10, 20, 50]):
    """Calcula métricas Top-K para diferentes valores de K"""
    test_users = test_df['userId'].unique()
    metrics = {f'recall_{k}': [] for k in k_values}
    metrics.update({f'ndcg_{k}': [] for k in k_values})
    
    # Generar recomendaciones para todos los usuarios de test
    recommendations = generate_recommendations(model, test_users, all_item_ids, k=max(k_values))
    
    for user_id in test_users:
        # Ítems relevantes en test (consideramos relevantes rating >= 4)
        relevant_items = set(test_df[(test_df['userId'] == user_id) & (test_df['rating'] >= 4)]['movieId'].values)
        if not relevant_items:
            continue
            
        # Ítems ya vistos en train (para excluirlos)
        seen_items = user_pos_items.get(user_id, set())
        
        # Recomendaciones para este usuario (ya excluyen ítems vistos)
        recommended_items = recommendations[user_id]
        
        for k in k_values:
            # Tomar los primeros k ítems recomendados
            top_k = recommended_items[:k]
            
            # Calcular Recall@K
            hits = len(set(top_k) & relevant_items)
            recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0.0
            metrics[f'recall_{k}'].append(recall)
            
            # Calcular NDCG@K
            dcg = 0.0
            for i, item in enumerate(top_k):
                if item in relevant_items:
                    dcg += 1.0 / np.log2(i + 2)  # +2 porque el ranking empieza en 1
            
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            metrics[f'ndcg_{k}'].append(ndcg)
    
    # Promediar las métricas
    avg_metrics = {k: np.mean(v) for k, v in metrics.items() if v}
    return avg_metrics


def calculate_rmse(model, ratings_df):
    """Calcula RMSE para todas las valoraciones en el dataframe"""
    model.eval()
    total_squared_error = 0
    total_samples = 0
    
    # Usar un DataLoader para procesamiento por lotes
    dataset = MovieLensDataset(ratings_df)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    
    with torch.no_grad():
        for user, movie, rating in loader:
            output = model(user, movie)
            squared_error = (output - rating) ** 2
            total_squared_error += squared_error.sum().item()
            total_samples += len(rating)
    
    return np.sqrt(total_squared_error / total_samples)


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
    
    def forward(self, pos_preds, neg_preds):
        """Bayesian Personalized Ranking Loss"""
        return -torch.log(torch.sigmoid(pos_preds - neg_preds)).mean()
    

def train_model():
    # Inicializar trackers
    print("Inicializando trackers...")
    system_tracker = SystemMetricsTracker()
    emissions_tracker = EmissionsPerEpochTracker(result_path, "NCF")
    
    # Inicializar listas para métricas
    train_losses = []
    val_rmses = []  # Para almacenar RMSE por época
    top_k_metrics = {f'recall@{k}': [] for k in config['top_k_values']}
    top_k_metrics.update({f'ndcg@{k}': [] for k in config['top_k_values']})
    
    # Preparar datos para evaluación
    user_pos_items = get_user_positive_items(train_df)
    all_item_ids = ratings_df['movieId'].unique()
    
    # Crear datasets y dataloaders
    train_dataset = BPRDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Para cálculo de RMSE
    val_dataset_rmse = MovieLensDataset(val_df)
    val_loader_rmse = DataLoader(val_dataset_rmse, batch_size=config['batch_size'], shuffle=False)
    
    # Función de pérdida y optimizador
    criterion = BPRLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    print(f"\nComenzando entrenamiento ({config['epochs']} épocas)...")
    for epoch in range(config['epochs']):
        # Iniciar seguimiento
        system_tracker.start_epoch(epoch)
        emissions_tracker.start_epoch(epoch)
        
        # Modo entrenamiento
        model.train()
        total_loss = 0
        total_samples = 0
        
        for user, pos_item, neg_item in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            pos_pred = model(user, pos_item)
            neg_pred = model(user, neg_item)
            
            # Calcular pérdida
            loss = criterion(pos_pred, neg_pred)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(user)
            total_samples += len(user)
        
        # Calcular pérdida promedio
        avg_train_loss = total_loss / total_samples
        train_losses.append(avg_train_loss)
        
        # Calcular RMSE en conjunto de validación
        val_rmse = calculate_rmse(model, val_df)
        val_rmses.append(val_rmse)
        
        # Evaluar métricas Top-K
        epoch_metrics = calculate_topk_metrics(
            model, val_df, train_df, user_pos_items, all_item_ids, config['top_k_values']
        )
        
        # Guardar métricas
        for k in config['top_k_values']:
            if f'recall_{k}' in epoch_metrics:
                top_k_metrics[f'recall@{k}'].append(epoch_metrics[f'recall_{k}'])
            if f'ndcg_{k}' in epoch_metrics:
                top_k_metrics[f'ndcg@{k}'].append(epoch_metrics[f'ndcg_{k}'])
        
        # Preparar métricas para los trackers
        recall_metrics = {f'recall_{k}': epoch_metrics.get(f'recall_{k}', 0.0) for k in config['top_k_values']}
        ndcg_metrics = {f'ndcg_{k}': epoch_metrics.get(f'ndcg_{k}', 0.0) for k in config['top_k_values']}
        
        # Actualizar trackers con todas las métricas
        system_tracker.end_epoch(epoch, avg_train_loss, val_rmse, recall_metrics, ndcg_metrics)
        emissions_tracker.end_epoch(epoch, avg_train_loss, val_rmse, recall_metrics, ndcg_metrics)
    
    # Evaluación final
    print("\nEvaluando modelo en conjunto de prueba final...")
    system_tracker.start_epoch("test")
    
    # Calcular métricas finales
    final_rmse = calculate_rmse(model, val_df)
    final_metrics = calculate_topk_metrics(
        model, val_df, train_df, user_pos_items, all_item_ids, config['top_k_values']
    )
    
    # Preparar métricas finales para los trackers
    final_recall_metrics = {f'recall_{k}': final_metrics.get(f'recall_{k}', 0.0) for k in config['top_k_values']}
    final_ndcg_metrics = {f'ndcg_{k}': final_metrics.get(f'ndcg_{k}', 0.0) for k in config['top_k_values']}
    
    # Finalizar trackers
    try:
        # Pasar la información del mejor RMSE del system_tracker al emissions_tracker
        if system_tracker.best_rmse_epoch is not None:
            emissions_tracker.best_rmse = system_tracker.best_rmse
            emissions_tracker.best_rmse_epoch = system_tracker.best_rmse_epoch
            # Buscar las emisiones correspondientes al mejor epoch
            if system_tracker.best_rmse_epoch < len(emissions_tracker.epoch_emissions):
                emissions_tracker.best_rmse_emissions = emissions_tracker.epoch_emissions[system_tracker.best_rmse_epoch]
                emissions_tracker.best_rmse_cumulative_emissions = emissions_tracker.cumulative_emissions[system_tracker.best_rmse_epoch]
        
        system_tracker.end_test(final_rmse, final_recall_metrics, final_ndcg_metrics)
    except Exception as e:
        print(f"Error al generar métricas finales con tracker: {e}")
    
    try:
        emissions_tracker.end_training(final_rmse, final_recall_metrics, final_ndcg_metrics)
    except Exception as e:
        print(f"Error al generar métricas de emisiones: {e}")
    
    # Guardar todas las métricas en un CSV
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    metrics_data = {
        'epoch': list(range(config['epochs'])),
        'train_loss': train_losses,
        'val_rmse': val_rmses,
        **top_k_metrics
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_file = f"{result_path}/model_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Métricas del modelo guardadas en: {metrics_file}")
    
    # Guardar resultados finales en un CSV separado
    final_results = {
        'final_rmse': [final_rmse],
        **{f'final_{k}': [final_metrics[k]] for k in final_metrics}
    }
    
    final_results_df = pd.DataFrame(final_results)
    final_results_file = f"{result_path}/final_results_{timestamp}.csv"
    final_results_df.to_csv(final_results_file, index=False)
    print(f"Resultados finales guardados en: {final_results_file}")
    
    return final_rmse, final_metrics

# Preparar los datos
print("\nPreparando los datasets...")

train_df, val_df = train_test_split(ratings_df, test_size=config['test_size'], random_state=config['random_state'])

# Crear datasets y dataloaders
train_dataset = BPRDataset(train_df)
val_dataset = BPRDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

print(f"Datos preparados: {len(train_df)} muestras de entrenamiento, {len(val_df) } muestras de validación")

# Inicializar modelo
model = NeuralCollaborativeFiltering(n_users, n_movies, 
                                    embedding_dim=config['embedding_dim'], 
                                    hidden_dim=config['hidden_dim'])

# Entrenar modelo
final_rmse, final_metrics = train_model() 

print("Entrenamiento completado exitosamente!")