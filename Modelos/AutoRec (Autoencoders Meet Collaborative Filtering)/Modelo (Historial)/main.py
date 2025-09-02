from data_preprocessor import *
from AutoRecWithHistory import AutoRecWithHistory
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
parser.add_argument('--base_lr', type=float, default=1e-3)
parser.add_argument('--decay_epoch_step', type=int, default=50, help="decay the learning rate for each n epochs")
parser.add_argument('--random_seed', type=int, default=1000)  
parser.add_argument('--display_step', type=int, default=1)
parser.add_argument('--history_weight', type=float, default=0.3, help="weight for history-based prediction")
parser.add_argument('--use_history', type=bool, default=True, help="whether to use history-based prediction")

args = parser.parse_args()
tf.random.set_seed(42)
np.random.seed(42)

# Configuración de rutas y directorios
data_name = 'ml-1m'
path = f"C:/Users/xpati/Documents/TFG/{data_name}/"
result_path = "C:/Users/xpati/Documents/TFG/Pruebas(Metricas)/AutoRec (Autoencoders Meet Collaborative Filtering)/Modelo (Historial)/results"

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
        metrics_df.to_csv(f"{result_path}/system_metrics_{timestamp}.csv", index=False)

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
                print(f"RMSE: {rmse:.4f}")
            if mae is not None:
                print(f"MAE: {mae:.4f}")
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
            
            # Asegurarse de que tengamos un MAE final si no se rastreó por época
            if not self.epoch_mae and final_mae is not None:
                self.epoch_mae = [final_mae] * len(self.epoch_emissions)
            
            # Crear dataframe con todos los datos
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            df_data = {
                'epoch': range(len(self.epoch_emissions)),
                'epoch_emissions_kg': self.epoch_emissions,
                'cumulative_emissions_kg': self.cumulative_emissions,
                'loss': self.epoch_loss if self.epoch_loss else [0.0] * len(self.epoch_emissions),
                'rmse': self.epoch_rmse if self.epoch_rmse else [None] * len(self.epoch_emissions)
            }
            
            # Añadir MAE solo si hay datos
            if self.epoch_mae:
                df_data['mae'] = self.epoch_mae
            
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
            self.plot_emissions_vs_metrics(timestamp, final_rmse, final_mae)
            
        except Exception as e:
            print(f"Error al generar gráficos de emisiones: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_emissions_vs_metrics(self, timestamp, final_rmse=None, final_mae=None):
        """Genera gráficos para emisiones vs métricas"""
        
        # Usar RMSE por época si está disponible, sino crear lista con el RMSE final
        if not self.epoch_rmse and final_rmse is not None:
            self.epoch_rmse = [final_rmse] * len(self.epoch_emissions)
        
        # Usar MAE por época si está disponible, sino crear lista con el MAE final
        if not self.epoch_mae and final_mae is not None:
            self.epoch_mae = [final_mae] * len(self.epoch_emissions)
        
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
            
            # 2. Gráfico combinado: Emisiones por época y acumulativas con métricas
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            plt.plot(range(len(self.epoch_emissions)), self.epoch_emissions, 'r-', marker='x')
            plt.title('Emisiones por Época')
            plt.xlabel('Época')
            plt.ylabel('CO2 Emissions (kg)')
            
            plt.subplot(2, 3, 2)
            plt.plot(range(len(self.cumulative_emissions)), self.cumulative_emissions, 'r-', marker='o')
            plt.title('Emisiones Acumuladas por Época')
            plt.xlabel('Época')
            plt.ylabel('CO2 Emissions (kg)')
            
            if self.epoch_rmse:
                plt.subplot(2, 3, 3)
                plt.plot(range(len(self.epoch_rmse)), self.epoch_rmse, 'b-', marker='o')
                plt.title('RMSE por Época')
                plt.xlabel('Época')
                plt.ylabel('RMSE')
            
            if self.epoch_mae:
                plt.subplot(2, 3, 4)
                plt.plot(range(len(self.epoch_mae)), self.epoch_mae, 'm-', marker='s')
                plt.title('MAE por Época')
                plt.xlabel('Época')
                plt.ylabel('MAE')
            
            if self.epoch_loss:
                plt.subplot(2, 3, 5)
                plt.plot(range(len(self.epoch_loss)), self.epoch_loss, 'g-', marker='o')
                plt.title('Loss por Época')
                plt.xlabel('Época')
                plt.ylabel('Loss')
                
            # Gráfico combinado de RMSE vs Emisiones Acumuladas
            if self.epoch_rmse:
                plt.subplot(2, 3, 6)
                plt.plot(self.cumulative_emissions, self.epoch_rmse, 'orange', marker='o')
                plt.title('RMSE vs Emisiones Acumuladas')
                plt.xlabel('Emisiones Acumuladas (kg)')
                plt.ylabel('RMSE')
            
            plt.tight_layout()
            
            file_path = f'{self.result_path}/emissions_plots/metrics_by_epoch_{self.model_name}_{timestamp}.png'
            plt.savefig(file_path)
            plt.close()
            print(f"Gráfico guardado en: {file_path}")
            
            # 3. Gráfico específico para MAE vs Emisiones Acumuladas
            if self.epoch_mae:
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
            
            if self.epoch_rmse:
                # 4. Scatter plot de rendimiento frente a emisiones acumulativas
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

# Modificar la parte donde inicializas el modelo:

# Inicializar modelo AutoRec (con o sin historial)
print("Construyendo modelo AutoRec...")
if args.use_history:
    print("Usando AutoRec con ajuste basado en historial de usuario")
    # Importar AutoRec original para poder usarlo
    from AutoRec import AutoRec
    
    # Crear primero el modelo AutoRec estándar
    model = AutoRec(args, num_users, num_items, R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R, 
                    num_train_ratings, num_test_ratings, user_train_set, item_train_set, user_test_set, 
                    item_test_set, result_path)
    
    # Luego crear el modelo con historial
    history_model = AutoRecWithHistory(num_users, num_items, args.hidden_neuron, args.lambda_value, args.history_weight)
    
    # Reemplazar el modelo interno de AutoRec con nuestro modelo con historial
    model.model = history_model
    
    # Configurar historiales después de inicializar el modelo
    print("Configurando historiales de usuario para predicciones...")
    history_model.set_user_histories(train_R, train_mask_R)
    history_model.compute_item_similarities(train_R, train_mask_R)
else:
    # Usar modelo AutoRec original
    from AutoRec import AutoRec
    model = AutoRec(args, num_users, num_items, R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R, 
                num_train_ratings, num_test_ratings, user_train_set, item_train_set, user_test_set, 
                item_test_set, result_path)

# Función para calcular RMSE y MAE
def calculate_metrics(model, input_R, mask_R, num_test_ratings):
    """Función para calcular RMSE y MAE independientemente"""
    if np.sum(mask_R) == 0:
        print("ADVERTENCIA: Máscara de valoraciones vacía al calcular métricas!")
        return 0.0, 0.0, model.model(input_R)
        
    output = model.model(input_R)
    
    # Calcular RMSE
    squared_error_sum = np.sum(np.square(input_R - output) * mask_R)
    rmse = np.sqrt(squared_error_sum / max(1, np.sum(mask_R)))
    
    # Calcular MAE
    absolute_error_sum = np.sum(np.abs(input_R - output) * mask_R)
    mae = absolute_error_sum / max(1, np.sum(mask_R))
    
    # Verificación adicional
    print(f"Cálculo de métricas - Suma de errores cuadrados: {squared_error_sum}, Suma de errores absolutos: {absolute_error_sum}, Valoraciones consideradas: {np.sum(mask_R)}")
    
    return rmse, mae, output

def modified_run():
    print("\nIniciando entrenamiento...")
    # Verificar matrices antes de entrenar
    print(f"Verificación previa: Valoraciones en train_R: {np.sum(train_mask_R)}, Valoraciones en test_R: {np.sum(test_mask_R)}")
    
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
            
            if i % 10 == 0:
                print(f"  Batch {i}: {ratings_in_batch} valoraciones, Loss: {loss.numpy():.4f}")
        
        # Calcular pérdida promedio por batch
        avg_loss = total_loss / max(1, batch_count)
        model.train_cost_list.append(avg_loss)
        
        # Calcular RMSE y MAE en el conjunto de validación
        rmse, mae, _ = calculate_metrics(model, test_R, test_mask_R, num_test_ratings)
        test_loss = model.loss_function(test_R, test_mask_R, model.model(test_R)).numpy()
        
        model.test_cost_list.append(test_loss)
        model.test_rmse_list.append(rmse)
        
        # Finalizar seguimiento de época
        system_tracker.end_epoch(epoch, avg_loss, rmse, mae)
        emissions_tracker.end_epoch(epoch, avg_loss, rmse, mae)
        
        # Mostrar progreso
        if (epoch + 1) % model.display_step == 0:
            print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Test RMSE: {rmse:.4f} | Test MAE: {mae:.4f} | Time: {int(time.time() - start_time)}s")

    # Si usamos el modelo con historial, configurarlo ahora
    if args.use_history and hasattr(model.model, 'set_user_histories'):
        print("\nConfigurando historiales de usuario para predicciones...")
        model.model.set_user_histories(train_R, train_mask_R)
        model.model.compute_item_similarities(train_R, train_mask_R)
    
    print("\nEvaluando modelo en conjunto de prueba...")
    # Evaluar modelo en conjunto de prueba
    system_tracker.start_epoch("test")
    
    # Si usamos historial, calculamos RMSE y MAE con predicciones ajustadas
    if args.use_history and hasattr(model.model, 'predict_for_user'):
        # Calcular RMSE y MAE con predicciones ajustadas por historial
        squared_error_sum = 0
        absolute_error_sum = 0
        test_count = 0
        
        print("\nCalculando RMSE y MAE con predicciones ajustadas por historial...")
        # Mostrar progreso para usuarios con más valoraciones
        progress_step = max(1, num_users // 10)  # Mostrar progreso cada ~10%
        
        try:
            for user_id in range(num_users):
                # Solo procesar usuarios con valoraciones en el conjunto de prueba
                if np.sum(test_mask_R[user_id]) > 0:
                    # Mostrar progreso
                    if user_id % progress_step == 0:
                        print(f"  Procesando usuario {user_id}/{num_users} ({user_id/num_users*100:.1f}%)")
                        
                    # Obtener predicciones ajustadas para este usuario
                    try:
                        predictions = model.model.predict_for_user(user_id, test_R)
                        
                        # Calcular errores para las valoraciones de prueba
                        user_squared_error = np.sum(np.square(test_R[user_id] - predictions) * test_mask_R[user_id])
                        user_absolute_error = np.sum(np.abs(test_R[user_id] - predictions) * test_mask_R[user_id])
                        user_test_count = np.sum(test_mask_R[user_id])
                        
                        squared_error_sum += user_squared_error
                        absolute_error_sum += user_absolute_error
                        test_count += user_test_count
                    except Exception as e:
                        print(f"  Error procesando usuario {user_id}: {e}")
                        continue
            
            final_rmse = np.sqrt(squared_error_sum / max(1, test_count))
            final_mae = absolute_error_sum / max(1, test_count)
            print(f"Test RMSE (con ajuste de historial): {final_rmse:.4f}")
            print(f"Test MAE (con ajuste de historial): {final_mae:.4f}")
        except Exception as e:
            print(f"Error en el cálculo con historial: {e}")
            # Fallback al cálculo estándar
            final_rmse, final_mae, final_output = calculate_metrics(model, test_R, test_mask_R, num_test_ratings)
    else:
        # Usar cálculo de métricas estándar
        final_rmse, final_mae, final_output = calculate_metrics(model, test_R, test_mask_R, num_test_ratings)    
    
    # Guardar algunos ejemplos de predicciones
    print("\nEjemplos de predicciones:")
    sample_size = min(5, np.sum(test_mask_R > 0))
    sample_indices = np.where(test_mask_R > 0)
    sampled_idx = np.random.choice(len(sample_indices[0]), sample_size, replace=False)
    
    for i in range(sample_size):
        u_idx = sample_indices[0][sampled_idx[i]]
        i_idx = sample_indices[1][sampled_idx[i]]
        
        # Obtener predicción (ajustada con historial si corresponde)
        if args.use_history and hasattr(model.model, 'predict_with_history'):
            autorec_pred = model.model(test_R[u_idx:u_idx+1])[0].numpy()[i_idx]
            adjusted_pred = model.model.predict_with_history(u_idx, i_idx, autorec_pred)
            
            print(f"Usuario {u_idx}, Ítem {i_idx}: Real = {test_R[u_idx, i_idx]:.2f}, " +
                 f"AutoRec = {autorec_pred:.2f}, Ajustada = {adjusted_pred:.2f}")

            # Demostrar cómo se calcula la predicción con historial
            if i == 0:  # Solo para el primer ejemplo
                model.model.demonstrate_prediction(u_idx, i_idx, test_R)
        else:
            # Predicción estándar
            if 'final_output' in locals():
                pred = final_output[u_idx, i_idx]
            else:
                pred = model.model(test_R[u_idx:u_idx+1])[0].numpy()[i_idx]
            print(f"Usuario {u_idx}, Ítem {i_idx}: Real = {test_R[u_idx, i_idx]:.2f}, Predicción = {pred:.2f}")
    
    # Finalizar seguimiento de prueba
    system_tracker.end_test(final_rmse, final_mae)
    
    # Pasar la información del mejor RMSE del system_tracker al emissions_tracker
    if system_tracker.best_rmse_epoch is not None:
        emissions_tracker.best_rmse = system_tracker.best_rmse
        emissions_tracker.best_rmse_epoch = system_tracker.best_rmse_epoch
        # Buscar las emisiones correspondientes al mejor epoch
        if system_tracker.best_rmse_epoch < len(emissions_tracker.epoch_emissions):
            emissions_tracker.best_rmse_emissions = emissions_tracker.epoch_emissions[system_tracker.best_rmse_epoch]
            emissions_tracker.best_rmse_cumulative_emissions = emissions_tracker.cumulative_emissions[system_tracker.best_rmse_epoch]
    
    emissions_tracker.end_training(final_rmse, final_mae)
    
    print(f"Test RMSE: {final_rmse:.4f}")
    print(f"Test MAE: {final_mae:.4f}")
    
    # Guardar resultados
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    metrics_df = pd.DataFrame({
        'epoch': list(range(len(model.train_cost_list))),
        'train_loss': model.train_cost_list,
        'test_loss': model.test_cost_list,
        'test_rmse': model.test_rmse_list
    })

    metrics_file = f"{result_path}/model_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Métricas del modelo guardadas en: {metrics_file}")
    
    return final_rmse

# Reemplazar el método run() original
model.run = modified_run

# Ejecutar el modelo
print("Comenzando ejecución del modelo...")
try:
    final_rmse = model.run()
    print(f"Ejecución completada con RMSE final: {final_rmse:.4f}")
except Exception as e:
    print(f"Error durante la ejecución: {e}")
    import traceback
    traceback.print_exc()