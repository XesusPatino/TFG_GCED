from data_preprocessor import *
from AutoRec import AutoRec
from codecarbon import EmissionsTracker
import tensorflow as tf
import time
import argparse
import numpy as np
import os
import psutil
import matplotlib.pyplot as plt
import pandas as pd

# Configuración de argumentos
parser = argparse.ArgumentParser(description='I-AutoRec ')
parser.add_argument('--hidden_neuron', type=int, default=500)
parser.add_argument('--lambda_value', type=float, default=1)
parser.add_argument('--train_epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--optimizer_method', choices=['Adam', 'RMSProp'], default='Adam')
parser.add_argument('--grad_clip', type=bool, default=False)
parser.add_argument('--base_lr', type=float, default=5e-4)  # Reducir learning rate para más estabilidad
parser.add_argument('--decay_epoch_step', type=int, default=10, help="decay the learning rate for each n epochs")  # Decay más frecuente
parser.add_argument('--random_seed', type=int, default=1000)  
parser.add_argument('--display_step', type=int, default=1)

args = parser.parse_args()
tf.random.set_seed(42)
np.random.seed(42)

# Configuración de rutas y directorios
data_name = 'ml-1m'
path = f"C:/Users/xpati/Documents/TFG/{data_name}/"
result_path = "C:/Users/xpati/Documents/TFG/Pruebas(Metricas)/AutoRec (Autoencoders Meet Collaborative Filtering)/results/"

# Crear directorios para resultados si no existen
os.makedirs(result_path, exist_ok=True)
os.makedirs(f"{result_path}/emissions_reports", exist_ok=True)
os.makedirs(f"{result_path}/emissions_plots", exist_ok=True)

# Preparación de datos
num_users, num_items, num_total_ratings = 6040, 3952, 1000209
train_ratio = 0.9

print("Cargando y procesando datos...")
# Verificar que el archivo de datos existe
ratings_file = os.path.join(path, "ratings.dat")
if not os.path.exists(ratings_file):
    raise FileNotFoundError(f"El archivo de datos no se encontró en: {ratings_file}")
else:
    print(f"Archivo de datos encontrado: {ratings_file}")

R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R, num_train_ratings, num_test_ratings, user_train_set, item_train_set, user_test_set, item_test_set = read_rating(
    path, num_users, num_items, num_total_ratings, 1, 0, train_ratio
)

# Verificar que los datos se cargaron correctamente
if np.sum(train_mask_R) == 0:
    raise ValueError("No hay valores en los datos de entrenamiento. Verifica la carga de datos.")

if np.sum(test_mask_R) == 0:
    raise ValueError("No hay valores en los datos de prueba. Verifica la carga de datos.")

print(f"Datos cargados: {num_train_ratings} valoraciones de entrenamiento, {num_test_ratings} valoraciones de prueba")
print(f"Suma de máscara de entrenamiento: {np.sum(train_mask_R)}")
print(f"Suma de máscara de prueba: {np.sum(test_mask_R)}")

# Clase para seguimiento de métricas del sistema
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
        
    def end_epoch(self, epoch, loss, rmse=None, topk_metrics=None):
        epoch_time = time.time() - self.epoch_start_time
        self.current_epoch_metrics['epoch_time_sec'] = epoch_time
        self.current_epoch_metrics['loss'] = loss
        if rmse is not None:
            self.current_epoch_metrics['rmse'] = rmse
        if topk_metrics is not None:
            self.current_epoch_metrics.update(topk_metrics)
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
        if topk_metrics is not None:
            for metric, value in topk_metrics.items():
                print(f"  {metric}: {value:.4f}")
        
    def end_test(self, rmse, topk_metrics=None):
        self.test_metrics = {
            'test_time_sec': time.time() - self.epoch_start_time,
            'total_time_sec': time.time() - self.start_time,
            'final_memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'final_cpu_usage_percent': psutil.cpu_percent(),
            'test_rmse': rmse,
        }
        if topk_metrics is not None:
            self.test_metrics.update(topk_metrics)
        
        # Imprimir métricas finales
        print("\n=== Final Training Metrics ===")
        for m in self.train_metrics:
            metrics_str = f"Epoch {m['epoch']}: Time={m['epoch_time_sec']:.2f}s, Memory={m['memory_usage_mb']:.2f}MB, CPU={m['cpu_usage_percent']:.1f}%"
            if 'rmse' in m:
                metrics_str += f", RMSE={m['rmse']:.4f}"
            if 'recall@5' in m:
                metrics_str += f", Recall@5={m['recall@5']:.4f}, Recall@10={m['recall@10']:.4f}"
            if 'ndcg@5' in m:
                metrics_str += f", NDCG@5={m['ndcg@5']:.4f}, NDCG@10={m['ndcg@10']:.4f}"
            print(metrics_str)
        
        print("\n=== Final Test Metrics ===")
        print(f"Total Time: {self.test_metrics['total_time_sec']:.2f}s (Test: {self.test_metrics['test_time_sec']:.2f}s)")
        print(f"Final Memory: {self.test_metrics['final_memory_usage_mb']:.2f}MB")
        print(f"Final CPU: {self.test_metrics['final_cpu_usage_percent']:.1f}%")
        print(f"RMSE: {rmse:.4f}")
        if topk_metrics is not None:
            for k in [5, 10, 20, 50]:
                if f'recall@{k}' in topk_metrics:
                    print(f"Recall@{k}: {topk_metrics[f'recall@{k}']:.4f}")
            for k in [5, 10, 20, 50]:
                if f'ndcg@{k}' in topk_metrics:
                    print(f"NDCG@{k}: {topk_metrics[f'ndcg@{k}']:.4f}")
        
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

# Clase para seguimiento de emisiones por época
class EmissionsPerEpochTracker:
    def __init__(self, result_path, model_name="AutoRec"):
        self.result_path = result_path
        self.model_name = model_name
        self.epoch_emissions = []
        self.cumulative_emissions = []
        self.epoch_rmse = []
        self.epoch_loss = []
        self.epoch_topk_metrics = []
        self.total_emissions = 0.0
        self.trackers = {}
        self.best_rmse = float('inf')
        self.best_rmse_epoch = None
        self.best_rmse_emissions = None
        self.best_rmse_cumulative_emissions = None
        
        # Crear directorio para emisiones
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
    
    def end_epoch(self, epoch, loss, rmse=None, topk_metrics=None):
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
            if topk_metrics is not None:
                self.epoch_topk_metrics.append(topk_metrics)
            
            print(f"Epoch {epoch} - Emisiones: {epoch_co2:.8f} kg, Acumulado: {self.total_emissions:.8f} kg, Loss: {loss:.4f}")
            if rmse is not None:
                print(f"RMSE: {rmse:.4f}")
            if topk_metrics is not None:
                for metric, value in topk_metrics.items():
                    print(f"{metric}: {value:.4f}")
        except Exception as e:
            print(f"Error al medir emisiones en época {epoch}: {e}")
    
    def end_training(self, final_rmse, final_topk_metrics=None):
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
                if final_topk_metrics is not None:
                    self.epoch_topk_metrics = [final_topk_metrics]
            
            # Si no hay datos, salir
            if not self.epoch_emissions:
                print("No hay datos de emisiones para graficar")
                return
            
            # Asegurarse de que tengamos un RMSE final si no se rastreó por época
            if not self.epoch_rmse and final_rmse is not None:
                self.epoch_rmse = [final_rmse] * len(self.epoch_emissions)
            
            # Crear dataframe con todos los datos
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            df_data = {
                'epoch': range(len(self.epoch_emissions)),
                'epoch_emissions_kg': self.epoch_emissions,
                'cumulative_emissions_kg': self.cumulative_emissions,
                'loss': self.epoch_loss if self.epoch_loss else [0.0] * len(self.epoch_emissions),
                'rmse': self.epoch_rmse if self.epoch_rmse else [None] * len(self.epoch_emissions)
            }
            
            # Añadir métricas Top-K si hay datos
            if self.epoch_topk_metrics:
                for k in [5, 10, 20, 50]:
                    recall_key = f'recall@{k}'
                    ndcg_key = f'ndcg@{k}'
                    if recall_key in self.epoch_topk_metrics[0]:
                        df_data[recall_key] = [metrics[recall_key] for metrics in self.epoch_topk_metrics]
                    if ndcg_key in self.epoch_topk_metrics[0]:
                        df_data[ndcg_key] = [metrics[ndcg_key] for metrics in self.epoch_topk_metrics]
            
            df = pd.DataFrame(df_data)
            
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
            self.plot_emissions_vs_metrics(timestamp, final_rmse, final_topk_metrics)
            
        except Exception as e:
            print(f"Error al generar gráficos de emisiones: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_emissions_vs_metrics(self, timestamp, final_rmse=None, final_topk_metrics=None):
        """Genera gráficos para emisiones vs métricas incluyendo Top-K"""
        
        # Usar RMSE por época si está disponible, sino crear lista con el RMSE final
        if not self.epoch_rmse and final_rmse is not None:
            self.epoch_rmse = [final_rmse] * len(self.epoch_emissions)
        
        try:
            if self.epoch_rmse:
                # 1. Emisiones acumulativas vs RMSE
                plt.figure(figsize=(10, 6))
                plt.plot(self.cumulative_emissions, self.epoch_rmse, 'b-', marker='o')
                
                # Añadir etiquetas con el número de época
                for i, (emissions, rmse) in enumerate(zip(self.cumulative_emissions, self.epoch_rmse)):
                    plt.annotate(f"{i}", (emissions, rmse), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9)
                    
                plt.xlabel('Emisiones de CO2 acumuladas (kg)')
                plt.ylabel('RMSE')
                plt.title('Relación entre Emisiones Acumuladas y RMSE')
                plt.grid(True, alpha=0.3)
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_rmse_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path)
                plt.close()
                print(f"Gráfico guardado en: {file_path}")
            
            # 2. Gráfico combinado: Emisiones por época y métricas
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Emisiones por época
            axes[0, 0].plot(range(len(self.epoch_emissions)), self.epoch_emissions, 'r-', marker='x')
            axes[0, 0].set_title('Emisiones por Época')
            axes[0, 0].set_xlabel('Época')
            axes[0, 0].set_ylabel('CO2 Emissions (kg)')
            
            # Emisiones acumuladas
            axes[0, 1].plot(range(len(self.cumulative_emissions)), self.cumulative_emissions, 'r-', marker='o')
            axes[0, 1].set_title('Emisiones Acumuladas por Época')
            axes[0, 1].set_xlabel('Época')
            axes[0, 1].set_ylabel('CO2 Emissions (kg)')
            
            # RMSE por época
            if self.epoch_rmse:
                axes[0, 2].plot(range(len(self.epoch_rmse)), self.epoch_rmse, 'b-', marker='o')
                axes[0, 2].set_title('RMSE por Época')
                axes[0, 2].set_xlabel('Época')
                axes[0, 2].set_ylabel('RMSE')
            
            # Loss por época
            if self.epoch_loss:
                axes[1, 0].plot(range(len(self.epoch_loss)), self.epoch_loss, 'g-', marker='o')
                axes[1, 0].set_title('Loss por Época')
                axes[1, 0].set_xlabel('Época')
                axes[1, 0].set_ylabel('Loss')
            
            # Métricas Top-K si están disponibles
            if self.epoch_topk_metrics:
                # Recall@10
                recall_10 = [metrics.get('recall@10', 0) for metrics in self.epoch_topk_metrics]
                axes[1, 1].plot(range(len(recall_10)), recall_10, 'm-', marker='s')
                axes[1, 1].set_title('Recall@10 por Época')
                axes[1, 1].set_xlabel('Época')
                axes[1, 1].set_ylabel('Recall@10')
                
                # NDCG@10
                ndcg_10 = [metrics.get('ndcg@10', 0) for metrics in self.epoch_topk_metrics]
                axes[1, 2].plot(range(len(ndcg_10)), ndcg_10, 'c-', marker='d')
                axes[1, 2].set_title('NDCG@10 por Época')
                axes[1, 2].set_xlabel('Época')
                axes[1, 2].set_ylabel('NDCG@10')
            
            plt.tight_layout()
            
            file_path = f'{self.result_path}/emissions_plots/metrics_by_epoch_{self.model_name}_{timestamp}.png'
            plt.savefig(file_path)
            plt.close()
            print(f"Gráfico guardado en: {file_path}")
            
            # 3. Gráfico específico de métricas Top-K vs Emisiones
            if self.epoch_topk_metrics and self.cumulative_emissions:
                plt.figure(figsize=(15, 10))
                
                # Extraer métricas
                recall_5 = [metrics.get('recall@5', 0) for metrics in self.epoch_topk_metrics]
                recall_10 = [metrics.get('recall@10', 0) for metrics in self.epoch_topk_metrics]
                ndcg_5 = [metrics.get('ndcg@5', 0) for metrics in self.epoch_topk_metrics]
                ndcg_10 = [metrics.get('ndcg@10', 0) for metrics in self.epoch_topk_metrics]
                
                # Subplot 1: Recall vs Emisiones
                plt.subplot(2, 2, 1)
                plt.plot(self.cumulative_emissions, recall_5, 'b-', marker='o', label='Recall@5')
                plt.plot(self.cumulative_emissions, recall_10, 'g-', marker='s', label='Recall@10')
                plt.xlabel('Emisiones Acumuladas (kg)')
                plt.ylabel('Recall')
                plt.title('Recall vs Emisiones Acumuladas')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Subplot 2: NDCG vs Emisiones
                plt.subplot(2, 2, 2)
                plt.plot(self.cumulative_emissions, ndcg_5, 'r-', marker='o', label='NDCG@5')
                plt.plot(self.cumulative_emissions, ndcg_10, 'm-', marker='s', label='NDCG@10')
                plt.xlabel('Emisiones Acumuladas (kg)')
                plt.ylabel('NDCG')
                plt.title('NDCG vs Emisiones Acumuladas')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Subplot 3: Recall por época
                plt.subplot(2, 2, 3)
                epochs = range(len(recall_5))
                plt.plot(epochs, recall_5, 'b-', marker='o', label='Recall@5')
                plt.plot(epochs, recall_10, 'g-', marker='s', label='Recall@10')
                plt.xlabel('Época')
                plt.ylabel('Recall')
                plt.title('Recall por Época')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Subplot 4: NDCG por época
                plt.subplot(2, 2, 4)
                plt.plot(epochs, ndcg_5, 'r-', marker='o', label='NDCG@5')
                plt.plot(epochs, ndcg_10, 'm-', marker='s', label='NDCG@10')
                plt.xlabel('Época')
                plt.ylabel('NDCG')
                plt.title('NDCG por Época')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                file_path = f'{self.result_path}/emissions_plots/topk_vs_emissions_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path)
                plt.close()
                print(f"Gráfico Top-K vs Emisiones guardado en: {file_path}")
            
            # 4. Scatter plot de rendimiento vs emisiones
            if self.epoch_rmse:
                plt.figure(figsize=(10, 6))
                
                # Ajustar tamaño de los puntos según la época
                sizes = [(i+1)*20 for i in range(len(self.cumulative_emissions))]
                
                scatter = plt.scatter(self.epoch_rmse, self.cumulative_emissions, 
                            color='blue', marker='o', s=sizes, alpha=0.7)
                
                # Añadir etiquetas de época
                for i, (rmse, em) in enumerate(zip(self.epoch_rmse, self.cumulative_emissions)):
                    plt.annotate(f"{i}", (rmse, em), textcoords="offset points", 
                                xytext=(0,5), ha='center', fontsize=9)
                
                plt.ylabel('Emisiones de CO2 acumuladas (kg)')
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

# Inicializar trackers
print("Inicializando trackers...")
system_tracker = SystemMetricsTracker()
emissions_tracker = EmissionsPerEpochTracker(result_path)

# Inicializar modelo AutoRec
print("Construyendo modelo AutoRec...")
model = AutoRec(args, num_users, num_items, R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R, 
                num_train_ratings, num_test_ratings, user_train_set, item_train_set, user_test_set, 
                item_test_set, result_path)

# Función para calcular RMSE
def calculate_rmse(model, input_R, mask_R, num_test_ratings):
    """Función para calcular RMSE independientemente"""
    if np.sum(mask_R) == 0:
        print("ADVERTENCIA: Máscara de valoraciones vacía al calcular RMSE!")
        return 0.0, model.model(input_R)
        
    output = model.model(input_R)
    squared_error_sum = np.sum(np.square(input_R - output) * mask_R)
    rmse = np.sqrt(squared_error_sum / max(1, np.sum(mask_R)))
    
    # Verificación adicional
    print(f"Cálculo de RMSE - Suma de errores cuadrados: {squared_error_sum}, Valoraciones consideradas: {np.sum(mask_R)}")
    
    return rmse, output

# Función mejorada para calcular métricas top-K
def calculate_topk_metrics(model, test_users, k_values=[5, 10, 20, 50], threshold=2.5):
    """
    Calcula métricas de top-K (Recall y NDCG) de manera más eficiente y precisa
    
    Args:
        model: Modelo AutoRec entrenado
        test_users: Lista de índices de usuarios para evaluar
        k_values: Lista de valores de k para evaluar
        threshold: Umbral para considerar un ítem como relevante (reducido a 2.5)
    
    Returns:
        dict: Métricas calculadas para cada valor de k
    """
    metrics = {}
    
    for k in k_values:
        recall_scores = []
        ndcg_scores = []
        total_relevant_items = 0
        total_recommended_items = 0
        
        for user_idx in test_users:
            # Obtener las valoraciones reales del usuario en el conjunto de test
            user_test_ratings = test_R[user_idx]
            user_test_mask = test_mask_R[user_idx]
            
            # Solo considerar ítems que el usuario efectivamente valoró en el test
            rated_items_in_test = np.where(user_test_mask > 0)[0]
            if len(rated_items_in_test) == 0:
                continue
                
            # Generar predicciones para todos los ítems
            user_input = np.copy(train_R[user_idx:user_idx+1])
            predictions = model.model(user_input).numpy()[0]
            
            # CAMBIO IMPORTANTE: Permitir recomendar tanto ítems conocidos como desconocidos
            # Esto es más realista ya que evaluamos si el modelo predice bien las preferencias
            # Solo eliminar ítems con predicciones muy bajas o inválidas
            valid_items = ~np.isnan(predictions) & ~np.isinf(predictions)
            masked_predictions = predictions.copy()
            masked_predictions[~valid_items] = -np.inf
            
            # Obtener top-k ítems con mayor predicción
            if np.all(masked_predictions == -np.inf):
                continue
                
            top_k_items = np.argsort(masked_predictions)[-k:][::-1]
            
            # Identificar ítems relevantes en el test (valoraciones >= threshold)
            # CAMBIO: Usar umbral más bajo y más realista
            relevant_items_in_test = rated_items_in_test[user_test_ratings[rated_items_in_test] >= threshold]
            
            if len(relevant_items_in_test) == 0:
                # Si no hay ítems relevantes con el umbral, usar los mejor valorados
                if len(rated_items_in_test) > 0:
                    # Tomar el top 50% de las valoraciones del usuario como relevantes
                    user_ratings_test = user_test_ratings[rated_items_in_test]
                    median_rating = np.median(user_ratings_test)
                    relevant_items_in_test = rated_items_in_test[user_test_ratings[rated_items_in_test] >= median_rating]
                else:
                    continue
                
            total_relevant_items += len(relevant_items_in_test)
            total_recommended_items += k
                
            # Calcular Recall: ¿Cuántos ítems relevantes están en el top-k?
            hits = len(np.intersect1d(top_k_items, relevant_items_in_test))
            recall = hits / len(relevant_items_in_test) if len(relevant_items_in_test) > 0 else 0.0
            recall_scores.append(recall)
            
            # Calcular NDCG con peso por valoración real
            dcg = 0.0
            for rank, item_idx in enumerate(top_k_items):
                if item_idx in relevant_items_in_test:
                    # Usar la valoración real como peso de relevancia
                    if item_idx < len(user_test_ratings) and user_test_mask[item_idx] > 0:
                        relevance = user_test_ratings[item_idx] - 1  # Normalizar desde 1-5 a 0-4
                    else:
                        relevance = 1.0  # Peso base si no tenemos la valoración
                    dcg += (2**relevance - 1) / np.log2(rank + 2)
            
            # IDCG (DCG ideal) - ordenar ítems relevantes por valoración
            relevant_ratings = []
            for item_idx in relevant_items_in_test:
                if item_idx < len(user_test_ratings) and user_test_mask[item_idx] > 0:
                    relevant_ratings.append(user_test_ratings[item_idx] - 1)
                else:
                    relevant_ratings.append(1.0)
            
            # Ordenar de mayor a menor valoración
            relevant_ratings = sorted(relevant_ratings, reverse=True)
            
            idcg = 0.0
            for rank, rating in enumerate(relevant_ratings[:k]):
                idcg += (2**rating - 1) / np.log2(rank + 2)
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        # Promediar métricas
        metrics[f'recall@{k}'] = np.mean(recall_scores) if recall_scores else 0.0
        metrics[f'ndcg@{k}'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
        
        print(f"K={k}: {len(recall_scores)} usuarios evaluados, {total_relevant_items} ítems relevantes totales")
    
    return metrics

def modified_run():
    print("\nIniciando entrenamiento...")
    # Verificar matrices antes de entrenar
    print(f"Verificación previa: Valoraciones en train_R: {np.sum(train_mask_R)}, Valoraciones en test_R: {np.sum(test_mask_R)}")
    
    # Lista para almacenar métricas por época
    metrics_by_epoch = []
    
    # Seleccionar usuarios para evaluación Top-K (muestra representativa)
    num_eval_users = min(1000, model.num_users)  # Usar más usuarios para mayor estabilidad
    eval_user_indices = np.random.choice(model.num_users, num_eval_users, replace=False)
    print(f"Evaluando Top-K con {num_eval_users} usuarios por época")
    
    # Entrenar el modelo con seguimiento de métricas
    for epoch in range(model.train_epoch):
        # Iniciar seguimiento de época
        system_tracker.start_epoch(epoch)
        emissions_tracker.start_epoch(epoch)
        
        # Entrenar una época
        start_time = time.time()
        total_loss = 0
        batch_count = 0
        
        print(f"Epoch {epoch+1}/{model.train_epoch} - Procesando {model.num_batch} batches...")
        for i in range(model.num_batch):
            batch_idx = np.random.choice(model.num_users, model.batch_size, replace=False)
            batch_train = model.train_R[batch_idx]
            batch_mask = model.train_mask_R[batch_idx]
            
            # Verificar datos antes del entrenamiento
            ratings_in_batch = np.sum(batch_mask)
            if ratings_in_batch == 0:
                print(f"¡Advertencia! Batch {i} no contiene valoraciones (máscara vacía)")
                continue
                
            loss = model.train_step(batch_train, batch_mask)
            total_loss += loss.numpy()
            batch_count += 1
            
            if i % 50 == 0:  # Reducir frecuencia de prints
                print(f"  Batch {i}: {ratings_in_batch} valoraciones, Loss: {loss.numpy():.4f}")
        
        # Calcular pérdida promedio por batch
        avg_loss = total_loss / max(1, batch_count)
        model.train_cost_list.append(avg_loss)
        
        # Calcular RMSE tradicional en el conjunto de validación
        rmse, _ = calculate_rmse(model, test_R, test_mask_R, num_test_ratings)
        test_loss = model.loss_function(test_R, test_mask_R, model.model(test_R)).numpy()
        
        model.test_cost_list.append(test_loss)
        model.test_rmse_list.append(rmse)
        
        # Calcular métricas Top-K usando la función mejorada
        print(f"Calculando métricas Top-K para época {epoch+1}...")
        topk_metrics = calculate_topk_metrics(model, eval_user_indices, k_values=[5, 10, 20, 50], threshold=2.5)
        
        # Guardar métricas de la época
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'test_rmse': rmse,
            **topk_metrics  # Incluir todas las métricas Top-K
        }
        metrics_by_epoch.append(epoch_metrics)
        
        # Finalizar seguimiento de época
        system_tracker.end_epoch(epoch, avg_loss, rmse, topk_metrics)
        emissions_tracker.end_epoch(epoch, avg_loss, rmse, topk_metrics)
        
        # Mostrar progreso
        if (epoch + 1) % model.display_step == 0:
            progress_str = f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Test RMSE: {rmse:.4f}"
            progress_str += f" | Recall@5: {topk_metrics['recall@5']:.4f} | Recall@10: {topk_metrics['recall@10']:.4f}"
            progress_str += f" | NDCG@5: {topk_metrics['ndcg@5']:.4f} | NDCG@10: {topk_metrics['ndcg@10']:.4f}"
            progress_str += f" | Time: {int(time.time() - start_time)}s"
            print(progress_str)
    
    print("\nEvaluando modelo en conjunto de prueba...")
    # Evaluar modelo en conjunto de prueba
    system_tracker.start_epoch("test")
    
    # Evaluación RMSE tradicional
    final_rmse, final_output = calculate_rmse(model, test_R, test_mask_R, num_test_ratings)
    
    # Evaluación Top-K en conjunto de prueba completo con más usuarios
    num_test_users = min(2000, model.num_users)  # Usar más usuarios para el test final
    test_user_indices = np.random.choice(model.num_users, num_test_users, replace=False)
    
    print(f"Evaluando Top-K final con {num_test_users} usuarios...")
    final_topk_metrics = calculate_topk_metrics(model, test_user_indices, k_values=[5, 10, 20, 50], threshold=2.5)
    
    # Guardar algunos ejemplos de predicciones
    print("\nEjemplos de predicciones:")
    sample_size = min(5, np.sum(test_mask_R > 0))
    sample_indices = np.where(test_mask_R > 0)
    sampled_idx = np.random.choice(len(sample_indices[0]), sample_size, replace=False)
    
    for i in range(sample_size):
        u_idx = sample_indices[0][sampled_idx[i]]
        i_idx = sample_indices[1][sampled_idx[i]]
        print(f"Usuario {u_idx}, Ítem {i_idx}: Real = {test_R[u_idx, i_idx]:.2f}, Predicción = {final_output[u_idx, i_idx]:.2f}")
    
    # Ejemplo de recomendaciones Top-5 para un usuario aleatorio
    random_user_idx = np.random.randint(len(test_user_indices))
    random_user = test_user_indices[random_user_idx]
    
    # Generar predicciones para este usuario
    user_input = np.copy(train_R[random_user:random_user+1])
    user_predictions = model.model(user_input).numpy()[0]
    
    # Enmascarar ítems ya valorados en entrenamiento
    not_rated_in_train = train_mask_R[random_user] == 0
    masked_preds = user_predictions.copy()
    masked_preds[~not_rated_in_train] = -np.inf
    
    # Obtener top-5 ítems
    top5_items = np.argsort(masked_preds)[-5:][::-1]
    
    print(f"\nTop-5 recomendaciones para usuario {random_user}:")
    for rank, item_idx in enumerate(top5_items):
        rating_status = "No valorado en entrenamiento"
        if train_mask_R[random_user, item_idx] > 0:
            rating_status = f"Ya valorado: {train_R[random_user, item_idx]:.1f}"
        print(f"  {rank+1}. Ítem {item_idx}: Score = {user_predictions[item_idx]:.2f} ({rating_status})")
    
    # Finalizar seguimiento de prueba
    system_tracker.end_test(final_rmse, final_topk_metrics)
    emissions_tracker.end_training(final_rmse, final_topk_metrics)
    
    # Imprimir resumen final de métricas
    print(f"\n=== Métricas Finales ===")
    print(f"RMSE: {final_rmse:.4f}")
    for k in [5, 10, 20, 50]:
        print(f"Recall@{k}: {final_topk_metrics[f'recall@{k}']:.4f}")
    for k in [5, 10, 20, 50]:
        print(f"NDCG@{k}: {final_topk_metrics[f'ndcg@{k}']:.4f}")
    
    # Guardar resultados
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    metrics_df = pd.DataFrame(metrics_by_epoch)
    
    metrics_file = f"{result_path}/model_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Métricas del modelo guardadas en: {metrics_file}")
    
    # Guardar métricas finales
    final_metrics = {'rmse': final_rmse, **final_topk_metrics}
    final_metrics_df = pd.DataFrame([final_metrics])
    final_metrics_file = f"{result_path}/final_metrics_{timestamp}.csv"
    final_metrics_df.to_csv(final_metrics_file, index=False)
    print(f"Métricas finales guardadas en: {final_metrics_file}")
    
    return final_metrics

# Reemplazar el método run() original
model.run = modified_run

# Ejecutar el modelo
print("Comenzando ejecución del modelo...")
try:
    final_metrics = model.run()
    print(f"\n=== Ejecución Completada ===")
    print(f"RMSE final: {final_metrics['rmse']:.4f}")
    print("Métricas Top-K finales:")
    for k in [5, 10, 20, 50]:
        print(f"  Recall@{k}: {final_metrics[f'recall@{k}']:.4f}")
        print(f"  NDCG@{k}: {final_metrics[f'ndcg@{k}']:.4f}")
except Exception as e:
    print(f"Error durante la ejecución: {e}")
    import traceback
    traceback.print_exc()