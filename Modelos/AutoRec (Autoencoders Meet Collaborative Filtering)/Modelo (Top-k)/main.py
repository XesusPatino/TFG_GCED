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
parser.add_argument('--train_epoch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--optimizer_method', choices=['Adam', 'RMSProp'], default='Adam')
parser.add_argument('--grad_clip', type=bool, default=False)
parser.add_argument('--base_lr', type=float, default=1e-3)
parser.add_argument('--decay_epoch_step', type=int, default=50, help="decay the learning rate for each n epochs")
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
        
    def start_epoch(self, epoch):
        self.epoch_start_time = time.time()
        self.current_epoch_metrics = {
            'epoch': epoch,
            'memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'cpu_usage_percent': psutil.cpu_percent(),
        }
        
    def end_epoch(self, epoch, loss, rmse=None):
        epoch_time = time.time() - self.epoch_start_time
        self.current_epoch_metrics['epoch_time_sec'] = epoch_time
        self.current_epoch_metrics['loss'] = loss
        if rmse is not None:
            self.current_epoch_metrics['rmse'] = rmse
        self.train_metrics.append(self.current_epoch_metrics)
        
        # Imprimir resumen de época
        print(f"\nEpoch {epoch} Metrics:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Memory: {self.current_epoch_metrics['memory_usage_mb']:.2f}MB")
        print(f"  CPU: {self.current_epoch_metrics['cpu_usage_percent']:.1f}%")
        print(f"  Loss: {loss:.4f}")
        if rmse is not None:
            print(f"  RMSE: {rmse:.4f}")
        
    def end_test(self, rmse):
        self.test_metrics = {
            'test_time_sec': time.time() - self.epoch_start_time,
            'total_time_sec': time.time() - self.start_time,
            'final_memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'final_cpu_usage_percent': psutil.cpu_percent(),
            'test_rmse': rmse,
        }
        
        # Imprimir métricas finales
        print("\n=== Final Training Metrics ===")
        for m in self.train_metrics:
            metrics_str = f"Epoch {m['epoch']}: Time={m['epoch_time_sec']:.2f}s, Memory={m['memory_usage_mb']:.2f}MB, CPU={m['cpu_usage_percent']:.1f}%, Loss={m['loss']:.4f}"
            if 'rmse' in m:
                metrics_str += f", RMSE={m['rmse']:.4f}"
            print(metrics_str)
        
        print("\n=== Final Test Metrics ===")
        print(f"Total Time: {self.test_metrics['total_time_sec']:.2f}s (Test: {self.test_metrics['test_time_sec']:.2f}s)")
        print(f"Final Memory: {self.test_metrics['final_memory_usage_mb']:.2f}MB")
        print(f"Final CPU: {self.test_metrics['final_cpu_usage_percent']:.1f}%")
        print(f"Test RMSE: {rmse:.4f}")
        
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
        self.total_emissions = 0.0
        self.trackers = {}
        
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
    
    def end_epoch(self, epoch, loss, rmse=None):
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
            
            print(f"Epoch {epoch} - Emisiones: {epoch_co2:.8f} kg, Acumulado: {self.total_emissions:.8f} kg, Loss: {loss:.4f}")
            if rmse is not None:
                print(f"RMSE: {rmse:.4f}")
        except Exception as e:
            print(f"Error al medir emisiones en época {epoch}: {e}")
    
    def end_training(self, final_rmse):
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
            
            # Crear dataframe con todos los datos
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            df = pd.DataFrame({
                'epoch': range(len(self.epoch_emissions)),
                'epoch_emissions_kg': self.epoch_emissions,
                'cumulative_emissions_kg': self.cumulative_emissions,
                'loss': self.epoch_loss if self.epoch_loss else [0.0] * len(self.epoch_emissions),
                'rmse': self.epoch_rmse if self.epoch_rmse else [None] * len(self.epoch_emissions)
            })
            
            emissions_file = f'{self.result_path}/emissions_reports/emissions_metrics_{self.model_name}_{timestamp}.csv'
            df.to_csv(emissions_file, index=False)
            print(f"Métricas de emisiones guardadas en: {emissions_file}")
            
            # Graficar las relaciones
            self.plot_emissions_vs_metrics(timestamp, final_rmse)
            
        except Exception as e:
            print(f"Error al generar gráficos de emisiones: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_emissions_vs_metrics(self, timestamp, final_rmse=None):
        """Genera gráficos para emisiones vs métricas"""
        
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
            
            # 2. Gráfico combinado: Emisiones por época y acumulativas
            plt.figure(figsize=(12, 10))
            
            plt.subplot(2, 2, 1)
            plt.plot(range(len(self.epoch_emissions)), self.epoch_emissions, 'r-', marker='x')
            plt.title('Emisiones por Época')
            plt.xlabel('Época')
            plt.ylabel('CO2 Emissions (kg)')
            
            plt.subplot(2, 2, 2)
            plt.plot(range(len(self.cumulative_emissions)), self.cumulative_emissions, 'r-', marker='o')
            plt.title('Emisiones Acumuladas por Época')
            plt.xlabel('Época')
            plt.ylabel('CO2 Emissions (kg)')
            
            if self.epoch_loss:
                plt.subplot(2, 2, 3)
                plt.plot(range(len(self.epoch_loss)), self.epoch_loss, 'g-', marker='o')
                plt.title('Loss por Época')
                plt.xlabel('Época')
                plt.ylabel('Loss')
            
            if self.epoch_rmse:
                plt.subplot(2, 2, 4)
                plt.plot(range(len(self.epoch_rmse)), self.epoch_rmse, 'b-', marker='o')
                plt.title('RMSE por Época')
                plt.xlabel('Época')
                plt.ylabel('RMSE')
            
            plt.tight_layout()
            
            file_path = f'{self.result_path}/emissions_plots/metrics_by_epoch_{self.model_name}_{timestamp}.png'
            plt.savefig(file_path)
            plt.close()
            print(f"Gráfico guardado en: {file_path}")
            
            if self.epoch_rmse:
                # 3. Scatter plot de rendimiento frente a emisiones acumulativas
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

# Función para generar predicciones para todos los ítems para un grupo de usuarios
def generate_predictions_for_users(model, user_batch, num_items):
    """
    Genera predicciones para todos los ítems para un lote de usuarios,
    aprovechando las valoraciones conocidas del usuario
    
    Args:
        model: El modelo AutoRec
        user_batch: Array numpy de IDs de usuario
        num_items: Número total de ítems
    
    Returns:
        Matriz de predicciones de shape [batch_size, num_items]
    """
    predictions = np.zeros((len(user_batch), num_items))
    
    # Para cada usuario, predecir valoraciones para todos los ítems
    for i, user_idx in enumerate(user_batch):
        # Crear entrada usando las valoraciones conocidas del usuario
        user_input = train_R[user_idx].reshape(1, -1).copy()
        user_mask = train_mask_R[user_idx].reshape(1, -1)
        
        # Hacer predicción conservando las valoraciones observadas
        output = model.model(user_input)
        predictions[i] = output[0]
    
    return predictions

# Función para calcular métricas top-K
def calculate_topk_metrics(predictions, true_ratings, k=10):
    """
    Calcula métricas de top-K como recall y NDCG
    
    Args:
        predictions: Matriz de predicciones [num_usuarios, num_items]
        true_ratings: Matriz de valoraciones reales [num_usuarios, num_items]
        k: Número de ítems para recomendar
    
    Returns:
        Dict con métricas recall@k y ndcg@k
    """
    num_users = predictions.shape[0]
    
    # Convertir valoraciones a relevancia binaria (1 si rating >= threshold, else 0)
    threshold = 3.0
    binary_ratings = (true_ratings >= threshold).astype(float)
    
    # Resultados para cada usuario
    recall_list = []
    ndcg_list = []
    
    for user_idx in range(num_users):
        user_preds = predictions[user_idx]
        user_ratings = binary_ratings[user_idx]
        
        # Ignorar ítems que el usuario ya ha valorado en el entrenamiento
        user_mask = ~(true_ratings[user_idx] > 0)
        
        # Aplicar máscara
        masked_preds = user_preds.copy()
        masked_preds[~user_mask] = -np.inf
        
        # Obtener top-k ítems
        top_k_items = np.argsort(masked_preds)[-k:][::-1]  # Invertimos para tener orden descendente
        
        # Calcular recall: relevantes_en_topk / total_relevantes
        relevant_items = np.where(user_ratings == 1)[0]
        if len(relevant_items) > 0:
            hits = np.isin(top_k_items, relevant_items).sum()
            recall = hits / len(relevant_items)
            recall_list.append(recall)
        
        # Calcular NDCG
        dcg = 0
        idcg = 0
        
        # DCG = suma(rel_i / log2(i+1))
        for i, item_idx in enumerate(top_k_items):
            if item_idx in relevant_items:
                dcg += 1 / np.log2(i + 2)  # +2 porque i empieza en 0 y log2(1) = 0
        
        # IDCG (DCG ideal cuando los ítems relevantes están al principio)
        for i in range(min(len(relevant_items), k)):
            idcg += 1 / np.log2(i + 2)
        
        if idcg > 0:
            ndcg_list.append(dcg / idcg)
    
    # Calcular promedios
    recall = np.mean(recall_list) if recall_list else 0
    ndcg = np.mean(ndcg_list) if ndcg_list else 0
    
    return {'recall@k': recall, 'ndcg@k': ndcg}

def modified_run():
    print("\nIniciando entrenamiento...")
    # Verificar matrices antes de entrenar
    print(f"Verificación previa: Valoraciones en train_R: {np.sum(train_mask_R)}, Valoraciones en test_R: {np.sum(test_mask_R)}")
    
    # Lista para almacenar métricas por época
    metrics_by_epoch = []
    
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
        
        # Calcular RMSE tradicional en el conjunto de validación para comparación
        rmse, _ = calculate_rmse(model, test_R, test_mask_R, num_test_ratings)
        test_loss = model.loss_function(test_R, test_mask_R, model.model(test_R)).numpy()
        
        model.test_cost_list.append(test_loss)
        model.test_rmse_list.append(rmse)
        
        # ===== TOP-K EVALUATION =====
        # Tomar una muestra de usuarios para la evaluación top-K 
        num_eval_users = min(500, model.num_users)
        eval_user_indices = np.random.choice(model.num_users, num_eval_users, replace=False)
        
        # Generar predicciones para todos los ítems para esta muestra de usuarios
        print(f"Evaluando top-K para {num_eval_users} usuarios...")
        
        # Versión mejorada de generate_predictions_for_users
        eval_predictions = np.zeros((len(eval_user_indices), model.num_items))
        for i, user_idx in enumerate(eval_user_indices):
            # Usar valoraciones conocidas del entrenamiento como entrada
            user_input = np.copy(train_R[user_idx:user_idx+1])
            
            # Obtener predicciones del modelo para todos los ítems
            output = model.model(user_input).numpy()
            eval_predictions[i] = output[0]
        
        # Extraer valoraciones reales para esos usuarios del test
        eval_true_ratings = test_R[eval_user_indices]
        
        # Calcular métricas top-K
        topk_metrics = calculate_topk_metrics(eval_predictions, eval_true_ratings, k=10)
        
        # Finalizar seguimiento de época
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'test_rmse': rmse,
            'recall@10': topk_metrics['recall@k'],
            'ndcg@10': topk_metrics['ndcg@k'],
        }
        metrics_by_epoch.append(epoch_metrics)
        
        system_tracker.end_epoch(epoch, avg_loss, rmse)
        emissions_tracker.end_epoch(epoch, avg_loss, rmse)
        
        # Mostrar progreso
        if (epoch + 1) % model.display_step == 0:
            print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Test RMSE: {rmse:.4f} | Recall@10: {topk_metrics['recall@k']:.4f} | NDCG@10: {topk_metrics['ndcg@k']:.4f} | Time: {int(time.time() - start_time)}s")
    
    print("\nEvaluando modelo en conjunto de prueba...")
    # Evaluar modelo en conjunto de prueba
    system_tracker.start_epoch("test")
    
    # Evaluación RMSE tradicional
    final_rmse, final_output = calculate_rmse(model, test_R, test_mask_R, num_test_ratings)
    
    # Evaluación Top-K en conjunto de prueba completo
    # Para el conjunto de prueba, podemos permitirnos usar más usuarios
    num_test_users = min(1000, model.num_users)
    test_user_indices = np.random.choice(model.num_users, num_test_users, replace=False)
    
    print(f"Evaluando top-K para {num_test_users} usuarios en conjunto de prueba...")
    
    # Generar predicciones para usuarios de prueba
    test_predictions = np.zeros((len(test_user_indices), model.num_items))
    for i, user_idx in enumerate(test_user_indices):
        # Usar valoraciones conocidas del entrenamiento como entrada
        user_input = np.copy(train_R[user_idx:user_idx+1])
        
        # Obtener predicciones del modelo para todos los ítems
        output = model.model(user_input).numpy()
        test_predictions[i] = output[0]
    
    test_true_ratings = test_R[test_user_indices]
    
    # Verificar valores extremos en las predicciones
    print(f"Rango de predicciones: Min={np.min(test_predictions):.2f}, Max={np.max(test_predictions):.2f}")
    print(f"Rango de valoraciones reales: Min={np.min(test_true_ratings):.2f}, Max={np.max(test_true_ratings):.2f}")
    
    # Calculamos para diferentes valores de k
    final_metrics = {}
    
    # Calcular métricas para k=5, k=10, k=50
    for k in [5, 10, 50]:
        topk_result = calculate_topk_metrics(test_predictions, test_true_ratings, k=k)
        final_metrics[f'recall@{k}'] = topk_result['recall@k']
        final_metrics[f'ndcg@{k}'] = topk_result['ndcg@k']
    
    # Añadir RMSE a las métricas finales
    final_metrics['rmse'] = final_rmse
    
    # Guardar algunos ejemplos de predicciones
    print("\nEjemplos de predicciones:")
    sample_size = min(5, np.sum(test_mask_R > 0))
    sample_indices = np.where(test_mask_R > 0)
    sampled_idx = np.random.choice(len(sample_indices[0]), sample_size, replace=False)
    
    for i in range(sample_size):
        u_idx = sample_indices[0][sampled_idx[i]]
        i_idx = sample_indices[1][sampled_idx[i]]
        print(f"Usuario {u_idx}, Ítem {i_idx}: Real = {test_R[u_idx, i_idx]:.2f}, Predicción = {final_output[u_idx, i_idx]:.2f}")
    
    # Top-5 recomendaciones para un usuario aleatorio
    random_user_idx = np.random.randint(len(test_user_indices))
    random_user = test_user_indices[random_user_idx]
    user_preds = test_predictions[random_user_idx]
    
    # No recomendar ítems que el usuario ya haya valorado en entrenamiento
    user_mask = train_mask_R[random_user] == 0  # Invertir la máscara: 1 donde no hay valoración
    masked_preds = user_preds.copy()
    masked_preds[~user_mask] = -np.inf
    
    # Verificar que hay suficientes ítems para recomendar
    non_rated_count = np.sum(user_mask)
    if non_rated_count < 5:
        print(f"\nAdvertencia: El usuario {random_user} solo tiene {non_rated_count} ítems no valorados.")
    
    # Obtener top-5 ítems
    top5_items = np.argsort(masked_preds)[-5:][::-1]
    
    print(f"\nTop-5 recomendaciones para usuario {random_user}:")
    for rank, item_idx in enumerate(top5_items):
        rating_status = "No valorado" if train_mask_R[random_user, item_idx] == 0 else f"Ya valorado: {train_R[random_user, item_idx]:.1f}"
        print(f"  {rank+1}. Ítem {item_idx}: Score = {user_preds[item_idx]:.2f} ({rating_status})")
    
    # Finalizar seguimiento de prueba
    system_tracker.end_test(final_rmse)
    emissions_tracker.end_training(final_rmse)
    
    # Imprimir métricas top-K finales
    print(f"\n=== Top-K Metrics ===")
    print(f"Recall@5:  {final_metrics['recall@5']:.4f}")
    print(f"NDCG@5:    {final_metrics['ndcg@5']:.4f}")
    print(f"Recall@10: {final_metrics['recall@10']:.4f}")
    print(f"NDCG@10:   {final_metrics['ndcg@10']:.4f}")
    print(f"Recall@50: {final_metrics['recall@50']:.4f}")
    print(f"NDCG@50:   {final_metrics['ndcg@50']:.4f}")
    print(f"RMSE:      {final_metrics['rmse']:.4f}")
    
    # Guardar resultados
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    metrics_df = pd.DataFrame(metrics_by_epoch)
    
    metrics_file = f"{result_path}/model_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Métricas del modelo guardadas en: {metrics_file}")
    
    # También guardar métricas finales
    final_metrics_df = pd.DataFrame([final_metrics])
    final_metrics_file = f"{result_path}/final_metrics_{timestamp}.csv"
    final_metrics_df.to_csv(final_metrics_file, index=False)
    print(f"Métricas finales guardadas en: {final_metrics_file}")
    
    # Generar gráficas adicionales para Top-K
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    # Convertir a arrays numpy antes de graficar
    epochs = np.array(metrics_df['epoch'])
    recall_values = np.array(metrics_df['recall@10'])
    plt.plot(epochs, recall_values, 'b-', marker='o')
    plt.title('Recall@10 durante entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Recall@10')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    ndcg_values = np.array(metrics_df['ndcg@10'])
    plt.plot(epochs, ndcg_values, 'g-', marker='o')
    plt.title('NDCG@10 durante entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('NDCG@10')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    topk_plot_file = f"{result_path}/emissions_plots/topk_metrics_{timestamp}.png"
    plt.savefig(topk_plot_file)
    plt.close()
    print(f"Gráfico de métricas Top-K guardado en: {topk_plot_file}")
    
    # Gráfica de métricas vs. emisiones
    plt.figure(figsize=(10, 6))
    emissions = np.array(emissions_tracker.cumulative_emissions)
    plt.plot(emissions, recall_values, 'b-', marker='o', label='Recall@10')
    plt.plot(emissions, ndcg_values, 'g-', marker='s', label='NDCG@10')
    
    # Añadir etiquetas con número de época
    for i, (em, rec) in enumerate(zip(emissions, recall_values)):
        plt.annotate(f"{i}", (em, rec), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.xlabel('Emisiones de CO2 acumuladas (kg)')
    plt.ylabel('Valor de la métrica')
    plt.title('Métricas Top-K vs. Emisiones Acumulativas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    em_topk_file = f"{result_path}/emissions_plots/emissions_vs_topk_{timestamp}.png"
    plt.savefig(em_topk_file)
    plt.close()
    print(f"Gráfico de emisiones vs. Top-K guardado en: {em_topk_file}")
    
    return final_metrics

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