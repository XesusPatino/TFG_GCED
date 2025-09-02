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
            
            
# Load IMF DOTS 2023 dataset
print("Cargando dataset IMF DOTS 2023...")
ratings_df = pd.read_csv('C:/Users/xpati/Documents/TFG/dot_2023_processed.csv')

# Renombrar columnas para mantener compatibilidad con el resto del código
ratings_df = ratings_df.rename(columns={'user_id': 'userId', 'item_id': 'movieId'})

# Display first few rows
print("Primeras filas del dataset IMF DOTS 2023:")
print(ratings_df.head())
print(f"\nForma del dataset: {ratings_df.shape}")
print(f"Columnas disponibles: {list(ratings_df.columns)}")
print(f"Rango de ratings: {ratings_df['rating'].min():.4f} - {ratings_df['rating'].max():.4f}")

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



class IMFDOTS2023Dataset(Dataset):
    def __init__(self, ratings_df):
        self.users = ratings_df['userId'].values
        self.items = ratings_df['movieId'].values  # Mantenemos 'movieId' para compatibilidad
        self.ratings = ratings_df['rating'].values

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return torch.tensor(self.users[idx], dtype=torch.long), \
               torch.tensor(self.items[idx], dtype=torch.long), \
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
        
        # Mostrar progreso
        if (epoch + 1) % config['display_step'] == 0:
            print(f"Epoch {epoch + 1}/{config['epochs']}, Training Loss: {avg_train_loss:.4f}, Validation RMSE: {val_rmse:.4f}, Validation MAE: {val_mae:.4f}")
    
    print("\nEvaluando modelo en conjunto de prueba final...")
    system_tracker.start_epoch("test")
    
    # Evaluación final
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
train_dataset = IMFDOTS2023Dataset(train_df)
val_dataset = IMFDOTS2023Dataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

print(f"Datos preparados: {len(train_df)} muestras de entrenamiento, {len(val_df)} muestras de validación")


# Inicializar modelo, función de pérdida y optimizador
model = NeuralCollaborativeFiltering(n_users, n_movies, 
                                   embedding_dim=config['embedding_dim'], 
                                   hidden_dim=config['hidden_dim'])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# Ejecutar entrenamiento
final_rmse, final_mae = train_model()

print("Programa finalizado correctamente.")



