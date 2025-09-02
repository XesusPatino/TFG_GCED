from data_preprocessor import *
from AutoRec import AutoRec # Asegúrate que AutoRec.py está adaptado para sparse
from codecarbon import EmissionsTracker
import tensorflow as tf
import time
import argparse
import numpy as np
import os
import psutil
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csr_matrix # Importar csr_matrix

# Configuración de argumentos
parser = argparse.ArgumentParser(description='I-AutoRec ')
parser.add_argument('--hidden_neuron', type=int, default=500)
parser.add_argument('--lambda_value', type=float, default=1)
# Reducir épocas y batch size para pruebas iniciales con Netflix Prize
parser.add_argument('--train_epoch', type=int, default=50)  
parser.add_argument('--batch_size', type=int, default=1024) 
parser.add_argument('--optimizer_method', choices=['Adam', 'RMSProp'], default='Adam')
parser.add_argument('--grad_clip', type=bool, default=False)
parser.add_argument('--base_lr', type=float, default=1e-3)
parser.add_argument('--decay_epoch_step', type=int, default=50, help="decay the learning rate for each n epochs")
parser.add_argument('--random_seed', type=int, default=1000)
parser.add_argument('--display_step', type=int, default=1)

args = parser.parse_args()
# Usar semillas consistentes
tf.random.set_seed(args.random_seed)
np.random.seed(args.random_seed)
random.seed(args.random_seed) # Añadir semilla para 'random' usado en data_preprocessor

# Configuración de rutas y directorios
path = "C:/Users/xpati/Documents/TFG/" # Directorio que contiene netflix.csv
result_path = "C:/Users/xpati/Documents/TFG/Pruebas(Metricas)/AutoRec (Autoencoders Meet Collaborative Filtering)/Modelo Netflix Prize/results"

# Crear directorios para resultados si no existen
os.makedirs(result_path, exist_ok=True)
os.makedirs(f"{result_path}/emissions_reports", exist_ok=True)
os.makedirs(f"{result_path}/emissions_plots", exist_ok=True)

# Preparación de datos
train_ratio = 0.9

print("Cargando y procesando datos...")
# Verificar que el archivo de datos existe
ratings_file = os.path.join(path, "netflix.csv")
if not os.path.exists(ratings_file):
    raise FileNotFoundError(f"El archivo de datos no se encontró en: {ratings_file}")
else:
    print(f"Archivo de datos encontrado: {ratings_file}")

# Llamar a read_rating SIN los tamaños, y recibirlos de vuelta
# Asegúrate que read_rating devuelve matrices dispersas (csr_matrix)
(R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,
 num_train_ratings, num_test_ratings,
 user_train_set, item_train_set, user_test_set, item_test_set,
 num_users, num_items, num_total_ratings) = read_rating(path, train_ratio)

# Verificar que los datos se cargaron correctamente usando .nnz para matrices dispersas
if not isinstance(train_R, csr_matrix) or not isinstance(test_R, csr_matrix):
     raise TypeError("read_rating no devolvió matrices CSR dispersas como se esperaba.")

if train_mask_R.nnz == 0: # Usar .nnz para contar elementos no cero
    raise ValueError("No hay valores en los datos de entrenamiento (máscara dispersa vacía). Verifica la carga de datos.")

if test_mask_R.nnz == 0: # Usar .nnz
    raise ValueError("No hay valores en los datos de prueba (máscara dispersa vacía). Verifica la carga de datos.")

print(f"Datos cargados: {num_train_ratings} valoraciones de entrenamiento, {num_test_ratings} valoraciones de prueba")
# Modificar esta línea para usar num_users y num_items directamente
print(f"Dimensiones de la matriz: ({num_users}, {num_items}) (Usuarios: {num_users}, Ítems: {num_items})")
print(f"Elementos no cero en máscara de entrenamiento: {train_mask_R.nnz}") # Usar .nnz
print(f"Elementos no cero en máscara de prueba: {test_mask_R.nnz}") # Usar .nnz

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
            if 'rmse' in m:
                metrics_str += f", RMSE={m['rmse']:.4f}"
            if 'mae' in m:
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
        metrics_file = f"{result_path}/system_metrics_{timestamp}.csv"
        try:
            metrics_df.to_csv(metrics_file, index=False)
            print(f"Métricas del sistema guardadas en: {metrics_file}")
        except Exception as e:
            print(f"Error al guardar métricas del sistema: {e}")

# Clase para seguimiento de emisiones por época
class EmissionsPerEpochTracker:
    def __init__(self, result_path, model_name="AutoRec"):
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
            tracking_mode="process",
            allow_multiple_runs=True # AÑADIDO
        )
        try:
            self.main_tracker.start()
            print("Tracker principal de emisiones iniciado.")
        except Exception as e:
            print(f"Advertencia: No se pudo iniciar el tracker principal de emisiones: {e}")
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
            allow_multiple_runs=True # AÑADIDO
        )
        try:
            self.trackers[epoch].start()
        except Exception as e:
            print(f"Advertencia: No se pudo iniciar el tracker de emisiones para la época {epoch}: {e}")
            self.trackers[epoch] = None
    
    def end_epoch(self, epoch, loss, rmse=None, mae=None):
        try:
            epoch_co2 = 0.0
            if epoch in self.trackers and self.trackers[epoch]:
                try:
                    emissions_data = self.trackers[epoch].stop()
                    epoch_co2 = emissions_data if emissions_data is not None else 0.0
                except Exception as e:
                    print(f"Advertencia: Error al detener el tracker de emisiones para la época {epoch}: {e}")
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
            
            print(f"Epoch {epoch} - Emisiones: {epoch_co2:.8f} kg, Acumulado: {self.total_emissions:.8f} kg, Loss: {loss:.4f}", end="")
            if rmse is not None:
                print(f", RMSE: {rmse:.4f}")
            if mae is not None:
                print(f", MAE: {mae:.4f}")
            if rmse is None and mae is None:
                print() # Nueva línea si no hay métricas
        except Exception as e:
            print(f"Error al medir emisiones en época {epoch}: {e}")
    
    def end_training(self, final_rmse, final_mae=None):
        try:
            # Detener el tracker principal
            final_emissions = 0.0
            if hasattr(self, 'main_tracker') and self.main_tracker:
                try:
                    emissions_data = self.main_tracker.stop()
                    final_emissions = emissions_data if emissions_data is not None else 0.0
                    print(f"\nTotal CO2 Emissions (tracker principal): {final_emissions:.6f} kg")
                except Exception as e:
                    print(f"Error al detener el tracker principal de emisiones: {e}")
                    final_emissions = self.total_emissions
                    print(f"\nTotal CO2 Emissions (acumulado): {final_emissions:.6f} kg")
            else:
                final_emissions = self.total_emissions
                print(f"\nTotal CO2 Emissions (acumulado): {final_emissions:.6f} kg")
            
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
                print("Advertencia: No hay emisiones por época, usando total para gráficos.")
                self.epoch_emissions = [final_emissions]
                self.cumulative_emissions = [final_emissions]
                if final_rmse is not None:
                    self.epoch_rmse = [final_rmse]
                if final_mae is not None:
                    self.epoch_mae = [final_mae]
                if self.epoch_loss:
                     self.epoch_loss = [self.epoch_loss[-1]]
                else:
                     self.epoch_loss = [0.0]
            
            # Si no hay datos, salir
            if not self.epoch_emissions:
                print("No hay datos de emisiones para guardar o graficar.")
                return
            
            # Asegurar longitudes consistentes
            num_epochs_recorded = len(self.epoch_emissions)
            if len(self.epoch_rmse) < num_epochs_recorded and final_rmse is not None:
                 last_rmse = self.epoch_rmse[-1] if self.epoch_rmse else final_rmse
                 self.epoch_rmse.extend([last_rmse] * (num_epochs_recorded - len(self.epoch_rmse)))
            elif len(self.epoch_rmse) > num_epochs_recorded:
                 self.epoch_rmse = self.epoch_rmse[:num_epochs_recorded]

            if len(self.epoch_mae) < num_epochs_recorded and final_mae is not None:
                 last_mae = self.epoch_mae[-1] if self.epoch_mae else final_mae
                 self.epoch_mae.extend([last_mae] * (num_epochs_recorded - len(self.epoch_mae)))
            elif len(self.epoch_mae) > num_epochs_recorded:
                 self.epoch_mae = self.epoch_mae[:num_epochs_recorded]

            if len(self.epoch_loss) < num_epochs_recorded:
                 last_loss = self.epoch_loss[-1] if self.epoch_loss else 0.0
                 self.epoch_loss.extend([last_loss] * (num_epochs_recorded - len(self.epoch_loss)))
            elif len(self.epoch_loss) > num_epochs_recorded:
                 self.epoch_loss = self.epoch_loss[:num_epochs_recorded]
            
            # Crear dataframe con todos los datos
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            df_data = {
                'epoch': list(range(num_epochs_recorded)),
                'epoch_emissions_kg': self.epoch_emissions,
                'cumulative_emissions_kg': self.cumulative_emissions,
                'loss': self.epoch_loss,
            }
            
            # Añadir RMSE y MAE solo si hay datos
            if self.epoch_rmse:
                df_data['rmse'] = self.epoch_rmse
            else:
                df_data['rmse'] = [None] * num_epochs_recorded
                
            if self.epoch_mae:
                df_data['mae'] = self.epoch_mae
            
            df = pd.DataFrame(df_data)
            
            emissions_file = f'{self.result_path}/emissions_reports/emissions_metrics_{self.model_name}_{timestamp}.csv'
            try:
                df.to_csv(emissions_file, index=False)
                print(f"Métricas de emisiones guardadas en: {emissions_file}")
            except Exception as e:
                print(f"Error al guardar métricas de emisiones: {e}")
            
            # Mostrar información del mejor RMSE y sus emisiones
            if self.best_rmse_epoch is not None:
                print(f"\n=== Best RMSE and Associated Emissions ===")
                print(f"Best RMSE: {self.best_rmse:.4f} (Epoch {self.best_rmse_epoch})")
                print(f"Emissions at best RMSE: {self.best_rmse_emissions:.8f} kg")
                print(f"Cumulative emissions at best RMSE: {self.best_rmse_cumulative_emissions:.8f} kg")
            
            # Graficar las relaciones
            self.plot_emissions_vs_metrics(timestamp, final_rmse, final_mae)
            
        except Exception as e:
            print(f"Error general en end_training de EmissionsTracker: {e}")
            import traceback
            traceback.print_exc()
            
    def plot_emissions_vs_metrics(self, timestamp, final_rmse=None, final_mae=None):
        """Genera gráficos para emisiones vs métricas"""
        if not self.epoch_emissions:
             print("No hay datos de emisiones para graficar.")
             return

        # Usar RMSE por época si está disponible
        plot_rmse = self.epoch_rmse and len(self.epoch_rmse) == len(self.epoch_emissions)
        plot_mae = self.epoch_mae and len(self.epoch_mae) == len(self.epoch_emissions)
        
        try:
            if plot_rmse:
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
            
            # 2. Gráfico combinado: Emisiones por época y acumulativas con métricas
            num_plots = 2 + (1 if self.epoch_loss else 0) + (1 if plot_rmse else 0) + (1 if plot_mae else 0)
            rows = (num_plots + 2) // 3  # Organizar en filas de hasta 3 columnas
            cols = min(3, num_plots)
            
            plt.figure(figsize=(15, 5 * rows))
            plot_index = 1
            
            plt.subplot(rows, cols, plot_index)
            plt.plot(range(len(self.epoch_emissions)), self.epoch_emissions, 'r-', marker='x')
            plt.title('Emisiones por Época')
            plt.xlabel('Época')
            plt.ylabel('CO2 Emissions (kg)')
            plot_index += 1
            
            plt.subplot(rows, cols, plot_index)
            plt.plot(range(len(self.cumulative_emissions)), self.cumulative_emissions, 'r-', marker='o')
            plt.title('Emisiones Acumuladas por Época')
            plt.xlabel('Época')
            plt.ylabel('CO2 Emissions (kg)')
            plot_index += 1
            
            if self.epoch_loss:
                plt.subplot(rows, cols, plot_index)
                plt.plot(range(len(self.epoch_loss)), self.epoch_loss, 'g-', marker='o')
                plt.title('Loss por Época')
                plt.xlabel('Época')
                plt.ylabel('Loss')
                plot_index += 1
            
            if plot_rmse:
                plt.subplot(rows, cols, plot_index)
                plt.plot(range(len(self.epoch_rmse)), self.epoch_rmse, 'b-', marker='o')
                plt.title('RMSE por Época')
                plt.xlabel('Época')
                plt.ylabel('RMSE')
                plot_index += 1
            
            if plot_mae:
                plt.subplot(rows, cols, plot_index)
                plt.plot(range(len(self.epoch_mae)), self.epoch_mae, 'm-', marker='s')
                plt.title('MAE por Época')
                plt.xlabel('Época')
                plt.ylabel('MAE')
                plot_index += 1
                
            plt.tight_layout()
            
            file_path = f'{self.result_path}/emissions_plots/metrics_by_epoch_{self.model_name}_{timestamp}.png'
            plt.savefig(file_path)
            plt.close()
            print(f"Gráfico guardado en: {file_path}")
            
            # 3. Scatter plots de rendimiento vs emisiones
            if plot_rmse:
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
                
            if plot_mae:
                plt.figure(figsize=(10, 6))
                plt.plot(self.cumulative_emissions, self.epoch_mae, 'purple', marker='s')
                
                # Añadir etiquetas con el número de época
                for i, (emissions, mae) in enumerate(zip(self.cumulative_emissions, self.epoch_mae)):
                    plt.annotate(f"{i}", (emissions, mae), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9)
                    
                plt.xlabel('Emisiones de CO2 acumuladas (kg)')
                plt.ylabel('MAE')
                plt.title('Relación entre Emisiones Acumuladas y MAE')
                plt.grid(True, alpha=0.3)
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_mae_{self.model_name}_{timestamp}.png'
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
# ¡¡CRÍTICO!! Asegúrate que la clase AutoRec en AutoRec.py está adaptada
# para recibir y usar estas matrices dispersas CSR.
print("Construyendo modelo AutoRec...")
model = AutoRec(args, num_users, num_items, R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,
                num_train_ratings, num_test_ratings, user_train_set, item_train_set, user_test_set,
                item_test_set, result_path)

# Función para calcular RMSE y MAE usando matrices dispersas (ADAPTADA PARA BATCHING)
def calculate_metrics(model, input_R_sparse, mask_R_sparse):
    """
    Calcula RMSE y MAE usando matrices dispersas, procesando en batches para evitar MemoryError.
    """
    num_ratings = mask_R_sparse.nnz
    if num_ratings == 0:
        print("ADVERTENCIA: Máscara de valoraciones vacía al calcular métricas!")
        return 0.0, 0.0, None

    total_squared_error = 0.0
    total_absolute_error = 0.0
    processed_ratings = 0
    batch_size = model.batch_size
    num_users = input_R_sparse.shape[0]

    print(f"Calculando métricas en batches de tamaño {batch_size}...")
    try:
        for i in range(0, num_users, batch_size):
            batch_idx = slice(i, min(i + batch_size, num_users))

            batch_input_sparse = input_R_sparse[batch_idx]
            batch_mask_sparse = mask_R_sparse[batch_idx]

            batch_num_ratings = batch_mask_sparse.nnz
            if batch_num_ratings == 0:
                continue

            # Obtener predicciones para el batch
            batch_output_dense = model.predict_step(batch_input_sparse)
            if not isinstance(batch_output_dense, tf.Tensor):
                 batch_output_dense = tf.convert_to_tensor(batch_output_dense, dtype=tf.float32)

            # Extraer índices y valores reales del batch
            batch_mask_coo = batch_mask_sparse.tocoo()
            row_indices_rel = batch_mask_coo.row
            col_indices = batch_mask_coo.col
            tf_indices_rel = tf.stack([row_indices_rel, col_indices], axis=1)
            tf_indices_rel = tf.cast(tf_indices_rel, dtype=tf.int64)

            batch_input_coo = batch_input_sparse.tocoo()
            actual_values = tf.convert_to_tensor(batch_input_coo.data, dtype=tf.float32)

            # Extraer valores predichos
            predicted_values = tf.gather_nd(batch_output_dense, tf_indices_rel)

            # Calcular errores del batch
            squared_errors = tf.square(actual_values - predicted_values)
            absolute_errors = tf.abs(actual_values - predicted_values)
            
            total_squared_error += tf.reduce_sum(squared_errors).numpy()
            total_absolute_error += tf.reduce_sum(absolute_errors).numpy()
            processed_ratings += batch_num_ratings

            if (i // batch_size) % 50 == 0:
                 print(f"  Métricas Batch {i // batch_size}: Procesados {processed_ratings}/{num_ratings} ratings...")

        if processed_ratings == 0:
             print("ADVERTENCIA: No se procesaron ratings válidos durante el cálculo de métricas.")
             return np.nan, np.nan, None

        # Calcular métricas finales
        rmse = np.sqrt(total_squared_error / processed_ratings)
        mae = total_absolute_error / processed_ratings

        print(f"Cálculo de métricas completado - RMSE: {rmse:.4f}, MAE: {mae:.4f}, Valoraciones: {processed_ratings}")

        return rmse, mae, None # Devolver None para la salida si no se necesita completa

    except Exception as e:
        print(f"Error durante el cálculo de métricas: {e}")
        import traceback
        traceback.print_exc()
        return np.nan, np.nan, None

def modified_run():
    print("\nIniciando entrenamiento...")
    # Verificar matrices dispersas antes de entrenar
    print(f"Verificación previa: Valoraciones en train_R: {train_R.nnz}, Valoraciones en test_R: {test_R.nnz}")

    # Entrenar el modelo con seguimiento de métricas
    for epoch in range(model.train_epoch):
        epoch_start_time_total = time.time()
        # Iniciar seguimiento de época
        system_tracker.start_epoch(epoch)
        emissions_tracker.start_epoch(epoch)

        # Entrenar una época
        total_loss = 0
        batch_count = 0
        processed_users = 0

        # --- Bucle de entrenamiento con batches de usuarios ---
        user_indices = np.arange(num_users)
        np.random.shuffle(user_indices)

        print(f"\nEpoch {epoch+1}/{model.train_epoch} - Procesando usuarios...")
        epoch_start_time_loop = time.time()

        for i in range(0, num_users, model.batch_size):
            batch_idx = user_indices[i:min(i + model.batch_size, num_users)]
            if len(batch_idx) == 0: continue

            # Obtener batches dispersos
            batch_train_sparse = model.train_R[batch_idx]
            batch_mask_sparse = model.train_mask_R[batch_idx]

            ratings_in_batch = batch_mask_sparse.nnz
            if ratings_in_batch == 0:
                continue

            # Entrenar
            loss = model.train_step(batch_train_sparse, batch_mask_sparse)

            if loss is not None and not np.isnan(loss):
                 current_loss = loss.numpy() if hasattr(loss, 'numpy') else loss
                 total_loss += current_loss
                 batch_count += 1
                 processed_users += len(batch_idx)

                 if batch_count % 100 == 0: # Reducir frecuencia de prints
                     print(f"  Epoch {epoch+1}, Users processed: {processed_users}/{num_users}, Batches: {batch_count}, Avg Loss: {total_loss/batch_count:.4f}")
            else:
                 print(f"  Epoch {epoch+1}, Batch starting at user {i}: Skipped (invalid loss)")

        epoch_loop_time = time.time() - epoch_start_time_loop
        print(f"Epoch {epoch+1} - Training loop time: {epoch_loop_time:.2f}s")

        # Calcular pérdida promedio
        avg_loss = total_loss / max(1, batch_count)
        model.train_cost_list.append(avg_loss)

        # --- Calcular RMSE y MAE en el conjunto de prueba ---
        print(f"Epoch {epoch+1} - Calculando métricas en conjunto de prueba...")
        eval_start_time = time.time()
        rmse, mae, _ = calculate_metrics(model, test_R, test_mask_R)
        eval_time = time.time() - eval_start_time
        print(f"Epoch {epoch+1} - Evaluación métricas time: {eval_time:.2f}s")

        # Convertir a float para evitar problemas con tipos
        current_rmse = float(rmse) if rmse is not None and not np.isnan(rmse) else None
        current_mae = float(mae) if mae is not None and not np.isnan(mae) else None
        
        model.test_rmse_list.append(current_rmse)

        # Finalizar seguimiento de época
        system_tracker.end_epoch(epoch, avg_loss, current_rmse, current_mae)
        emissions_tracker.end_epoch(epoch, avg_loss, current_rmse, current_mae)

        # Mostrar progreso
        epoch_total_time = time.time() - epoch_start_time_total
        if (epoch + 1) % model.display_step == 0:
            rmse_str = f"{current_rmse:.4f}" if current_rmse is not None else "N/A"
            mae_str = f"{current_mae:.4f}" if current_mae is not None else "N/A"
            print(f"Epoch {epoch+1} Summary | Train Loss: {avg_loss:.4f} | Test RMSE: {rmse_str} | Test MAE: {mae_str} | Total Time: {epoch_total_time:.2f}s")

    print("\nEvaluando modelo final en conjunto de prueba...")
    # Evaluar modelo final
    system_tracker.start_epoch("test")

    final_rmse, final_mae, _ = calculate_metrics(model, test_R, test_mask_R)
    final_rmse = float(final_rmse) if final_rmse is not None and not np.isnan(final_rmse) else None
    final_mae = float(final_mae) if final_mae is not None and not np.isnan(final_mae) else None

    # Guardar algunos ejemplos de predicciones
    print("\nEjemplos de predicciones:")
    if test_mask_R.nnz > 0:
        sample_size = min(5, test_mask_R.nnz)
        test_mask_coo = test_mask_R.tocoo()
        sample_indices_flat = np.random.choice(len(test_mask_coo.data), sample_size, replace=False)

        print("  (Re-prediciendo para ejemplos seleccionados...)")
        example_user_indices = sorted(list(set(test_mask_coo.row[sample_indices_flat])))
        example_input_sparse = test_R[example_user_indices]
        example_output_dense = model.predict_step(example_input_sparse)

        user_map_example = {global_idx: relative_idx for relative_idx, global_idx in enumerate(example_user_indices)}

        for i in range(sample_size):
            idx_in_coo = sample_indices_flat[i]
            u_idx = test_mask_coo.row[idx_in_coo]
            i_idx = test_mask_coo.col[idx_in_coo]

            real_rating = test_R[u_idx, i_idx]

            if u_idx in user_map_example:
                u_idx_rel = user_map_example[u_idx]
                predicted_rating = example_output_dense[u_idx_rel, i_idx].numpy()
                print(f"Usuario {u_idx}, Ítem {i_idx}: Real = {real_rating:.2f}, Predicción = {predicted_rating:.2f}")
            else:
                 print(f"Usuario {u_idx}, Ítem {i_idx}: Real = {real_rating:.2f}, Predicción = Error (usuario no encontrado)")
    else:
        print("No se pueden mostrar ejemplos (sin valoraciones de prueba).")

    # Finalizar seguimiento
    system_tracker.end_test(final_rmse if final_rmse is not None else -1.0, final_mae)
    
    # Sincronizar mejor RMSE entre trackers
    if system_tracker.best_rmse_epoch is not None:
        emissions_tracker.best_rmse = system_tracker.best_rmse
        emissions_tracker.best_rmse_epoch = system_tracker.best_rmse_epoch
        if system_tracker.best_rmse_epoch < len(emissions_tracker.epoch_emissions):
            emissions_tracker.best_rmse_emissions = emissions_tracker.epoch_emissions[system_tracker.best_rmse_epoch]
            emissions_tracker.best_rmse_cumulative_emissions = emissions_tracker.cumulative_emissions[system_tracker.best_rmse_epoch]
    
    emissions_tracker.end_training(final_rmse if final_rmse is not None else -1.0, final_mae)

    rmse_str = f"{final_rmse:.4f}" if final_rmse is not None else "N/A"
    mae_str = f"{final_mae:.4f}" if final_mae is not None else "N/A"
    print(f"\nTest RMSE Final: {rmse_str}")
    print(f"Test MAE Final: {mae_str}")

    # Guardar resultados
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    num_epochs_run = len(model.train_cost_list)
    test_rmse_list_padded = model.test_rmse_list + [None] * (num_epochs_run - len(model.test_rmse_list))

    metrics_data = {
        'epoch': list(range(num_epochs_run)),
        'train_loss': model.train_cost_list[:num_epochs_run],
        'test_rmse': test_rmse_list_padded[:num_epochs_run]
    }

    metrics_df = pd.DataFrame(metrics_data)

    metrics_file = f"{result_path}/model_metrics_{timestamp}.csv"
    try:
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Métricas del modelo guardadas en: {metrics_file}")
    except Exception as e:
        print(f"Error al guardar métricas del modelo: {e}")

    return final_rmse

# Reemplazar el método run() original con nuestra versión modificada
model.run = modified_run

# Ejecutar el modelo
print("Comenzando ejecución del modelo...")
try:
    final_rmse = model.run()
    rmse_str = f"{final_rmse:.4f}" if final_rmse is not None else "N/A"
    print(f"\nEjecución completada con RMSE final: {rmse_str}")
except Exception as e:
    print(f"\nError durante la ejecución principal: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Asegurarse de que los trackers se detienen incluso si hay un error
    print("Deteniendo trackers (si están activos)...")
    if 'emissions_tracker' in locals() and hasattr(emissions_tracker, 'main_tracker') and emissions_tracker.main_tracker:
        try:
            emissions_tracker.main_tracker.stop()
        except: pass
    for tracker in emissions_tracker.trackers.values():
         if tracker:
             try: tracker.stop()
             except: pass