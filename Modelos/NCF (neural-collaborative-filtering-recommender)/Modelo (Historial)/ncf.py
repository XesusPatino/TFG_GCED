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
    'random_state': 42
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
        
        # Imprimir resumen de época en formato más compacto
        memory_mb = self.current_epoch_metrics.get('memory_usage_mb', 0)
        cpu_percent = self.current_epoch_metrics.get('cpu_usage_percent', 0)
        
        output_parts = [f"Epoch {epoch}: Time={epoch_time:.0f}s, Memory={memory_mb:.0f}MB, CPU={cpu_percent:.1f}%"]
        if rmse is not None:
            output_parts.append(f"RMSE={rmse:.4f}")
        if mae is not None:
            output_parts.append(f"MAE={mae:.4f}")
        
        print(", ".join(output_parts))
        
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
            # Formato específico solicitado: Epoch X: Time=Xs, Memory=XMB, CPU=X%, RMSE=X, MAE=X
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
        
        
class EmissionsPerEpochTracker:
    def __init__(self, result_path, model_name="NCF"):
        self.result_path = result_path
        self.model_name = model_name
        self.epoch_emissions = []
        self.cumulative_emissions = []
        self.epoch_rmse = []
        self.epoch_mae = []
        self.epoch_loss = []
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
                # Rastrear el mejor RMSE y sus emisiones
                if rmse < self.best_rmse:
                    self.best_rmse = rmse
                    self.best_rmse_epoch = epoch
                    self.best_rmse_emissions = epoch_co2
                    self.best_rmse_cumulative_emissions = self.total_emissions
            if mae is not None:
                self.epoch_mae.append(mae)
            
            print(f"Epoch {epoch} - Emisiones: {epoch_co2:.8f} kg, Acumulado: {self.total_emissions:.8f} kg, Loss: {loss:.4f}")
            if rmse is not None:
                print(f"  RMSE: {rmse:.4f}")
            if mae is not None:
                print(f"  MAE: {mae:.4f}")
        except Exception as e:
            print(f"Error al medir emisiones en época {epoch}: {e}")
    
    def end_training(self, final_rmse, final_mae=None):
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
                if final_mae is not None:
                    self.epoch_mae = [final_mae]
            
            # Si no hay datos, salir
            if not self.epoch_emissions:
                print("No hay datos de emisiones para graficar")
                return
            
            # Asegurarse de que tengamos un RMSE final si no se rastreó por época
            if not self.epoch_rmse and final_rmse is not None:
                self.epoch_rmse = [final_rmse] * len(self.epoch_emissions)
            
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
            
            # Mostrar información del mejor RMSE y sus emisiones
            if self.best_rmse_epoch is not None:
                print(f"\n=== Best RMSE and Associated Emissions ===")
                print(f"Best RMSE: {self.best_rmse:.4f} (Epoch {self.best_rmse_epoch})")
                print(f"Emissions at best RMSE: {self.best_rmse_emissions:.8f} kg")
                print(f"Cumulative emissions at best RMSE: {self.best_rmse_cumulative_emissions:.8f} kg")
            
            # Graficar las relaciones
            self.plot_emissions_vs_metrics(timestamp, final_rmse, final_mae)
            
        except Exception as e:
            print(f"Error al generar gráficos de emisiones: {e}")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"Error al medir emisiones en época {epoch}: {e}")
    
    def end_training(self, final_rmse, final_mae=None):
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
                if final_mae is not None:
                    self.epoch_mae = [final_mae]
            
            # Si no hay datos, salir
            if not self.epoch_emissions:
                print("No hay datos de emisiones para graficar")
                return
            
            # Asegurarse de que tengamos un RMSE final si no se rastreó por época
            if not self.epoch_rmse and final_rmse is not None:
                self.epoch_rmse = [final_rmse] * len(self.epoch_emissions)
            
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



class MovieLensDataset(Dataset):
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
               
               
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim=50, hidden_dim=128):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        # Embedding layers for users and movies
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        
        # MLP layers for neural network
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user, movie):
        # Get embeddings
        user_embedded = self.user_embedding(user)
        movie_embedded = self.movie_embedding(movie)
        
        # Concatenate embeddings
        x = torch.cat([user_embedded, movie_embedded], dim=1)
        
        # Pass through MLP
        return self.mlp(x).squeeze()
    
    
    
class NCFWithHistory(nn.Module):
    """Neural Collaborative Filtering con recomendaciones basadas en historial"""
    def __init__(self, n_users, n_movies, embedding_dim=50, hidden_dim=128, history_weight=0.3):
        super(NCFWithHistory, self).__init__()
        
        # Parámetros básicos
        self.n_users = n_users
        self.n_movies = n_movies
        self.history_weight = history_weight
        
        # Capas de embedding para usuarios y películas
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        
        # Capas MLP para la red neuronal
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Historial de usuario y matriz de similitud
        self.user_histories = {}
        self.item_similarities = None
    
    def forward(self, user, movie):
        """Forward pass a través del modelo"""
        # Obtener embeddings
        user_embedded = self.user_embedding(user)
        movie_embedded = self.movie_embedding(movie)
        
        # Concatenar embeddings
        x = torch.cat([user_embedded, movie_embedded], dim=1)
        
        # Pasar por la red MLP
        return self.mlp(x).squeeze()
    
    def set_user_histories(self, ratings_df):
        """Configurar historiales de usuario a partir de datos de entrenamiento"""
        print("Construyendo historiales de usuario...")
        self.user_histories = {}
        
        # Agrupar valoraciones por usuario
        for user_id in range(self.n_users):
            # Filtrar valoraciones para este usuario
            user_data = ratings_df[ratings_df['userId'] == user_id]
            
            if len(user_data) > 0:
                # Crear historial como un diccionario {item_id: rating}
                user_history = dict(zip(user_data['movieId'].values, user_data['rating'].values))
                self.user_histories[user_id] = user_history
        
        print(f"Construido historial para {len(self.user_histories)} usuarios")
    
    def compute_item_similarities(self, ratings_df):
        """Calcular matriz de similitud entre ítems basada en valoraciones de usuario"""
        print("Calculando similitudes entre películas...")
        
        # Inicializar matriz de similitud
        self.item_similarities = np.zeros((self.n_movies, self.n_movies))
        
        # Crear matriz de valoraciones usuarios-películas
        rating_matrix = np.zeros((self.n_users, self.n_movies))
        for index, row in ratings_df.iterrows():
            rating_matrix[row['userId'], row['movieId']] = row['rating']
        
        # Obtener perfiles de película (transponer matriz para obtener perfiles por película)
        item_profiles = rating_matrix.T
        
        # Calcular normas para todas las películas
        item_norms = np.array([np.linalg.norm(item_profiles[i]) for i in range(self.n_movies)])
        
        # Solo procesar películas con normas no cero
        valid_items = np.where(item_norms > 0)[0]
        print(f"Calculando similitudes para {len(valid_items)} películas válidas...")
        
        # Procesar por lotes para mostrar progreso
        batch_size = 100
        total_batches = (len(valid_items) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(valid_items))
            batch_items = valid_items[batch_start:batch_end]
            
            if batch_idx % 10 == 0:
                print(f"  Procesando lote {batch_idx+1}/{total_batches}...")
            
            for i_idx in range(len(batch_items)):
                i = batch_items[i_idx]
                i_profile = item_profiles[i]
                i_norm = item_norms[i]
                
                # Calcular similitudes con otras películas
                for j_idx in range(i_idx, len(valid_items)):
                    j = valid_items[j_idx]
                    if j < i:
                        continue  # Omitir si ya calculamos este par
                    
                    j_profile = item_profiles[j]
                    j_norm = item_norms[j]
                    
                    # Calcular similitud de coseno
                    if i_norm * j_norm > 0:
                        sim = np.dot(i_profile, j_profile) / (i_norm * j_norm)
                    else:
                        sim = 0.0
                    
                    # Actualizar matriz de similitud (simétrica)
                    self.item_similarities[i, j] = sim
                    if i != j:  # Evitar establecer valores diagonales dos veces
                        self.item_similarities[j, i] = sim
        
        print("Similitudes calculadas")
    
    def predict_with_history(self, user_id, item_id, ncf_prediction):
        """
        Ajustar predicción de NCF usando historial de usuario
        Retorna el promedio ponderado de la predicción NCF y la predicción basada en historial
        """
        # Si no hay historial o datos de similitud, retornar predicción NCF
        if user_id not in self.user_histories or self.item_similarities is None:
            return ncf_prediction
        
        user_history = self.user_histories[user_id]
        
        # Si el usuario no tiene historial, retornar predicción NCF
        if not user_history:
            return ncf_prediction
        
        # Obtener similitudes entre película objetivo y películas calificadas
        similar_items = []
        
        for hist_item_id, hist_rating in user_history.items():
            similarity = self.item_similarities[item_id, hist_item_id]
            
            # Solo considerar películas algo similares (umbral arbitrario y ajustable)
            if similarity > 0.1:
                similar_items.append((hist_item_id, hist_rating, similarity))
        
        # Si no se encontraron películas similares, retornar predicción NCF
        if not similar_items:
            return ncf_prediction
        
        # Calcular promedio ponderado de valoraciones de películas similares
        total_sim = sum(sim for _, _, sim in similar_items)
        history_prediction = sum(rating * sim for _, rating, sim in similar_items) / total_sim
        
        # Combinar predicciones
        final_prediction = (1 - self.history_weight) * ncf_prediction + self.history_weight * history_prediction
        
        return final_prediction
    
    def predict_for_user(self, user_id, movie_id):
        """Predecir valoración para un usuario y película usando NCF y historial"""
        # Convertir a tensores
        user_tensor = torch.tensor([user_id], dtype=torch.long)
        movie_tensor = torch.tensor([movie_id], dtype=torch.long)
        
        # Obtener predicción NCF
        with torch.no_grad():
            ncf_prediction = self.forward(user_tensor, movie_tensor).item()
        
        # Ajustar con historial
        adjusted_prediction = self.predict_with_history(user_id, movie_id, ncf_prediction)
        
        return ncf_prediction, adjusted_prediction
    
    def demonstrate_prediction(self, user_id, item_id):
        """Demostrar proceso de predicción para una película"""
        # Obtener predicción NCF
        user_tensor = torch.tensor([user_id], dtype=torch.long)
        movie_tensor = torch.tensor([item_id], dtype=torch.long)
        
        with torch.no_grad():
            ncf_prediction = self.forward(user_tensor, movie_tensor).item()
        
        # Obtener predicción ajustada
        adjusted_prediction = self.predict_with_history(user_id, item_id, ncf_prediction)
        
        # Imprimir información sobre la predicción
        print(f"\nDemostrando predicción para Usuario {user_id}, Película {item_id}:")
        print(f"Predicción NCF: {ncf_prediction:.4f}")
        
        if user_id in self.user_histories:
            history = self.user_histories[user_id]
            print(f"Usuario ha calificado {len(history)} películas")
            
            # Encontrar películas similares
            similar_items = []
            for hist_item_id, hist_rating in history.items():
                if self.item_similarities is not None:
                    similarity = self.item_similarities[item_id, hist_item_id]
                    if similarity > 0.1:
                        similar_items.append((hist_item_id, hist_rating, similarity))
            
            if similar_items:
                print("\nPelículas similares encontradas en historial:")
                print(f"{'ID Película':<10}{'Valoración':<10}{'Similitud':<15}")
                
                for hist_item, hist_rating, sim in sorted(similar_items, key=lambda x: x[2], reverse=True)[:5]:
                    print(f"{hist_item:<10}{hist_rating:<10.2f}{sim:<15.4f}")
                
                total_sim = sum(sim for _, _, sim in similar_items)
                history_prediction = sum(rating * sim for _, rating, sim in similar_items) / total_sim
                print(f"\nPredicción basada en historial: {history_prediction:.4f}")
                print(f"Predicción final ponderada ({self.history_weight} peso historial): {adjusted_prediction:.4f}")
            else:
                print("No se encontraron películas similares en historial de usuario")
        else:
            print("Usuario no tiene historial de valoraciones")
        
        return adjusted_prediction
    

def calculate_metrics(model, loader):
    model.eval()
    total_loss = 0
    total_squared_error = 0
    total_absolute_error = 0
    total_samples = 0
    
    with torch.no_grad():
        for user, movie, rating in loader:
            output = model(user, movie)
            loss = criterion(output, rating)
            total_loss += loss.item() * len(rating)
            
            # Calculate squared error and absolute error
            squared_error = (output - rating) ** 2
            absolute_error = torch.abs(output - rating)
            
            # Accumulate total errors
            total_squared_error += squared_error.sum().item()
            total_absolute_error += absolute_error.sum().item()
            total_samples += len(rating)
    
    # Compute metrics
    avg_loss = total_loss / total_samples
    rmse = np.sqrt(total_squared_error / total_samples)
    mae = total_absolute_error / total_samples
    
    return avg_loss, rmse, mae

def train_model():
    # Inicializar trackers
    print("Inicializando trackers...")
    system_tracker = SystemMetricsTracker()
    emissions_tracker = EmissionsPerEpochTracker(result_path, "NCF")
    
    # Inicializar listas para seguimiento de métricas
    train_losses = []
    val_losses = []
    val_rmses = []
    val_maes = []
    
    # Guardar tiempo de inicio para medir tiempo total
    tiempo_inicio = time.time()
    
    # Training loop
    print(f"\nComenzando entrenamiento ({config['epochs']} épocas)...")
    for epoch in range(config['epochs']):
        # Iniciar seguimiento de época
        system_tracker.start_epoch(epoch)
        emissions_tracker.start_epoch(epoch)
        
        # Cambiar a modo de entrenamiento
        model.train()
        total_loss = 0
        total_samples = 0
        
        for user, movie, rating in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(user, movie)
            
            # Compute loss
            loss = criterion(output, rating)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(rating)
            total_samples += len(rating)
        
        # Calcular loss promedio de esta época
        avg_train_loss = total_loss / total_samples
        train_losses.append(avg_train_loss)
        
        # Evaluar en conjunto de validación
        val_loss, val_rmse, val_mae = calculate_metrics(model, val_loader)
        val_losses.append(val_loss)
        val_rmses.append(val_rmse)
        val_maes.append(val_mae)
        
        # Actualizar trackers
        system_tracker.end_epoch(epoch, avg_train_loss, val_rmse, val_mae)
        emissions_tracker.end_epoch(epoch, avg_train_loss, val_rmse, val_mae)
    
    print("\nEvaluando modelo en conjunto de prueba final...")
    system_tracker.start_epoch("test")
    
    # Preparar historial después del entrenamiento
    if isinstance(model, NCFWithHistory):
        print("\nConfiguración de historiales de usuario y similitudes entre películas")
        model.set_user_histories(train_df)
        model.compute_item_similarities(train_df)
        
        # Evaluación final con ajuste de historial
        print("\nCalculando RMSE con ajuste de historial")
        total_squared_error = 0
        total_absolute_error = 0
        total_samples = 0
        examples = []
        
        # Muestrear hasta 5 ejemplos para mostrar
        sample_users = np.random.choice(n_users, min(10, n_users), replace=False)
        
        for user_id in range(n_users):
            user_test_data = val_df[val_df['userId'] == user_id]
            if len(user_test_data) == 0:
                continue
                
            for _, row in user_test_data.iterrows():
                movie_id = row['movieId']
                actual_rating = row['rating']
                
                # Obtener predicciones
                ncf_pred, adjusted_pred = model.predict_for_user(user_id, movie_id)
                
                # Calcular errores
                squared_error = (adjusted_pred - actual_rating) ** 2
                absolute_error = abs(adjusted_pred - actual_rating)
                
                total_squared_error += squared_error
                total_absolute_error += absolute_error
                total_samples += 1
                
                # Guardar algunos ejemplos para mostrar
                if user_id in sample_users and len(examples) < 5:
                    examples.append((user_id, movie_id, actual_rating, ncf_pred, adjusted_pred))
        
        # Calcular métricas finales
        final_rmse = np.sqrt(total_squared_error / total_samples)
        final_mae = total_absolute_error / total_samples
        
        # Mostrar ejemplos
        print("\nEjemplos de predicciones:")
        for user_id, movie_id, actual, ncf_pred, adj_pred in examples:
            print(f"Usuario {user_id}, Película {movie_id}: Real={actual:.2f}, NCF={ncf_pred:.2f}, Ajustada={adj_pred:.2f}")
            # Mostrar detalles de uno de los ejemplos
            if examples.index((user_id, movie_id, actual, ncf_pred, adj_pred)) == 0:
                model.demonstrate_prediction(user_id, movie_id)
    else:
        # Evaluación estándar para modelo NCF normal
        final_loss, final_rmse, final_mae = calculate_metrics(model, val_loader)
    
    # Finalizar seguimiento - Asegurarnos que esto se ejecute sin excepciones
    metrics_displayed = False
    
    try:
        print("\nGenerando métricas finales del sistema con SystemMetricsTracker...")
        system_tracker.end_test(final_rmse, final_mae)
        metrics_displayed = True
    except Exception as e:
        print(f"Error al generar métricas finales con tracker: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print("\nGenerando gráficos y métricas de emisiones...")
        # Pasar la información del mejor RMSE del system_tracker al emissions_tracker
        if system_tracker.best_rmse_epoch is not None:
            emissions_tracker.best_rmse = system_tracker.best_rmse
            emissions_tracker.best_rmse_epoch = system_tracker.best_rmse_epoch
            # Buscar las emisiones correspondientes al mejor epoch
            if system_tracker.best_rmse_epoch < len(emissions_tracker.epoch_emissions):
                emissions_tracker.best_rmse_emissions = emissions_tracker.epoch_emissions[system_tracker.best_rmse_epoch]
                emissions_tracker.best_rmse_cumulative_emissions = emissions_tracker.cumulative_emissions[system_tracker.best_rmse_epoch]
        
        emissions_tracker.end_training(final_rmse, final_mae)
    except Exception as e:
        print(f"Error al generar métricas de emisiones: {e}")
        import traceback
        traceback.print_exc()
    
    # Guardar métricas de entrenamiento
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    metrics_df = pd.DataFrame({
        'epoch': list(range(config['epochs'])),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_rmse': val_rmses,
        'val_mae': val_maes
    })
    
    metrics_file = f"{result_path}/model_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Métricas del modelo guardadas en: {metrics_file}")
    
    # Siempre mostrar las métricas finales, incluso si los otros métodos fallaron
    print("\n" + "="*60)
    print("MÉTRICAS FINALES DEL SISTEMA (MEDICIÓN INDEPENDIENTE GARANTIZADA)")
    print("="*60)
    
    # Mediciones directas
    memoria_final = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    cpu_final = psutil.cpu_percent(interval=1.0)  # Medición de 1 segundo
    tiempo_total = time.time() - tiempo_inicio
    
    print(f"Memoria final: {memoria_final:.2f} MB")
    print(f"CPU final: {cpu_final:.2f}%")
    print(f"Tiempo total de ejecución: {tiempo_total:.2f} segundos")
    print(f"RMSE final: {final_rmse:.4f}")
    print(f"MAE final: {final_mae:.4f}")
    
    # Mostrar información del mejor RMSE
    if system_tracker.best_rmse_epoch is not None:
        print(f"\nMejor RMSE durante entrenamiento: {system_tracker.best_rmse:.4f} (Época {system_tracker.best_rmse_epoch})")
        if (system_tracker.best_rmse_epoch < len(emissions_tracker.epoch_emissions) and 
            system_tracker.best_rmse_epoch < len(emissions_tracker.cumulative_emissions)):
            best_epoch_emissions = emissions_tracker.epoch_emissions[system_tracker.best_rmse_epoch]
            best_cumulative_emissions = emissions_tracker.cumulative_emissions[system_tracker.best_rmse_epoch]
            print(f"Emisiones en mejor época: {best_epoch_emissions:.8f} kg")
            print(f"Emisiones acumuladas en mejor época: {best_cumulative_emissions:.8f} kg")
    
    print("="*60)
    
    # Guardar las métricas finales en un archivo separado para asegurar que se capturen
    final_metrics = {
        'final_memory_mb': memoria_final,
        'final_cpu_percent': cpu_final,
        'total_time_sec': tiempo_total,
        'final_rmse': final_rmse,
        'final_mae': final_mae,
        'timestamp': timestamp
    }
    
    final_metrics_df = pd.DataFrame([final_metrics])
    final_metrics_file = f"{result_path}/final_metrics_{timestamp}.csv"
    final_metrics_df.to_csv(final_metrics_file, index=False)
    print(f"Métricas finales guardadas en: {final_metrics_file}")
    
    print(f"\nEntrenamiento finalizado!")
    print(f"Métricas finales - RMSE: {final_rmse:.4f}, MAE: {final_mae:.4f}")
    
    return final_rmse, final_mae

# Preparar los datos
print("\nPreparando los datasets...")
train_df, val_df = train_test_split(ratings_df, test_size=config['test_size'], random_state=config['random_state'])

# Crear datasets y dataloaders
train_dataset = MovieLensDataset(train_df)
val_dataset = MovieLensDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

print(f"Datos preparados: {len(train_df)} muestras de entrenamiento, {len(val_df)} muestras de validación")


# Inicializar modelo con soporte de historial
model = NCFWithHistory(n_users, n_movies, 
                     embedding_dim=config['embedding_dim'], 
                     hidden_dim=config['hidden_dim'],
                     history_weight=0.3)  # Ajustar este valor según sea necesario

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# Ejecutar entrenamiento
final_rmse, final_mae = train_model()

print("Programa finalizado correctamente.")



def predict_rating_for_user_and_movie(user_id, movie_id):
    """
    Predecir la valoración que un usuario daría a una película específica,
    utilizando el modelo NCF ajustado con historial.
    """
    if not isinstance(model, NCFWithHistory):
        raise TypeError("Esta función requiere un modelo NCFWithHistory")
    
    # Verificar que el usuario y la película existan en nuestros datos
    if user_id >= n_users or user_id < 0:
        raise ValueError(f"ID de usuario inválido. Debe estar entre 0 y {n_users-1}")
    
    if movie_id >= n_movies or movie_id < 0:
        raise ValueError(f"ID de película inválido. Debe estar entre 0 y {n_movies-1}")
    
    # Obtener predicciones
    ncf_pred, adjusted_pred = model.predict_for_user(user_id, movie_id)
    
    # Mostrar detalles del proceso de predicción
    model.demonstrate_prediction(user_id, movie_id)
    
    return ncf_pred, adjusted_pred

# Ejemplo de uso
print("\nProbando predicción para usuario y película específicos")
try:
    # Elegir un usuario y película aleatorios para probar
    sample_user = np.random.randint(0, n_users)
    sample_movie = np.random.randint(0, n_movies)
    
    print(f"\nPrediciendo valoración para usuario {sample_user}, película {sample_movie}")
    ncf_pred, history_pred = predict_rating_for_user_and_movie(sample_user, sample_movie)
    print(f"\nResumen de predicción:")
    print(f"Predicción NCF: {ncf_pred:.4f}")
    print(f"Predicción ajustada con historial: {history_pred:.4f}")
    
except Exception as e:
    print(f"Error al predecir: {e}")