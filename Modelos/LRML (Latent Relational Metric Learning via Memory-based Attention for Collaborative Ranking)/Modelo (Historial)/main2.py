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
from history_recommender import UserHistoryRecommender
import argparse


# Configuración de argumentos (similar al modelo AutoRec)
parser = argparse.ArgumentParser(description='LRML con ajuste por historial')
parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--num_mem', type=int, default=30)
parser.add_argument('--train_epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--learn_rate', type=float, default=1e-3)
parser.add_argument('--l2_reg', type=float, default=1e-2)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--history_weight', type=float, default=0.3,
                   help="peso para predicciones basadas en historial")
parser.add_argument('--use_history', type=bool, default=True,
                   help="si usar ajuste basado en historial")

args = parser.parse_args()
tf.random.set_seed(42)
np.random.seed(42)


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
            self.current_epoch_metrics['recall'] = recall
        if ndcg is not None:
            self.current_epoch_metrics['ndcg'] = ndcg
            
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
            print(f"  Recall@10: {recall:.4f}")
        if ndcg is not None:
            print(f"  NDCG@10: {ndcg:.4f}")
        
    def end_test(self, rmse, recall=None, ndcg=None):
        self.test_metrics = {
            'test_time_sec': time.time() - self.epoch_start_time,
            'total_time_sec': time.time() - self.start_time,
            'final_memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'final_cpu_usage_percent': psutil.cpu_percent(),
            'test_rmse': rmse,
        }
        
        if recall is not None:
            self.test_metrics['test_recall'] = recall
            
        if ndcg is not None:
            self.test_metrics['test_ndcg'] = ndcg
        
        # Imprimir métricas finales
        print("\n=== Final Training Metrics ===")
        for m in self.train_metrics:
            metrics_str = f"Epoch {m['epoch']+1}: Time={m['epoch_time_sec']:.2f}s, Memory={m['memory_usage_mb']:.2f}MB, CPU={m['cpu_usage_percent']:.1f}%, Loss={m['loss']:.4f}"
            if 'rmse' in m:
                metrics_str += f", RMSE={m['rmse']:.4f}"
            if 'recall' in m:
                metrics_str += f", Recall={m['recall']:.4f}"
            if 'ndcg' in m:
                metrics_str += f", NDCG={m['ndcg']:.4f}"
            print(metrics_str)
        
        print("\n=== Final Test Metrics ===")
        print(f"Total Time: {self.test_metrics['total_time_sec']:.2f}s (Test: {self.test_metrics['test_time_sec']:.2f}s)")
        print(f"Final Memory: {self.test_metrics['final_memory_usage_mb']:.2f}MB")
        print(f"Final CPU: {self.test_metrics['final_cpu_usage_percent']:.1f}%")
        print(f"Test RMSE: {rmse:.4f}")
        if recall is not None:
            print(f"Test Recall@10: {recall:.4f}")
        if ndcg is not None:
            print(f"Test NDCG@10: {ndcg:.4f}")
        
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
        self.epoch_recall = []
        self.epoch_ndcg = []
        self.epoch_loss = []
        self.total_emissions = 0.0
        self.trackers = {}
        
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
                self.epoch_recall.append(recall)
                
            if ndcg is not None:
                self.epoch_ndcg.append(ndcg)
            
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
                    self.epoch_recall = [final_recall]
                if final_ndcg is not None:
                    self.epoch_ndcg = [final_ndcg]
            
            # Si no hay datos, salir
            if not self.epoch_emissions:
                print("No hay datos de emisiones para graficar")
                return
            
            # Crear dataframe con todos los datos
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            epochs_range = range(1, len(self.epoch_emissions) + 1)
            df = pd.DataFrame({
                'epoch': epochs_range,
                'epoch_emissions_kg': self.epoch_emissions,
                'cumulative_emissions_kg': self.cumulative_emissions,
                'loss': self.epoch_loss if self.epoch_loss else [0.0] * len(self.epoch_emissions),
                'rmse': self.epoch_rmse if self.epoch_rmse else [None] * len(self.epoch_emissions),
                'recall': self.epoch_recall if self.epoch_recall else [None] * len(self.epoch_emissions),
                'ndcg': self.epoch_ndcg if self.epoch_ndcg else [None] * len(self.epoch_emissions)
            })
            
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
            
            # Recall@10
            if self.epoch_recall:
                plt.subplot(3, 2, 5)
                plt.plot(epochs_range, self.epoch_recall, 'm-', marker='o')
                plt.title('Recall@10 por Época')
                plt.xlabel('Época')
                plt.ylabel('Recall@10')
                plt.grid(True, alpha=0.3)
            
            # NDCG@10
            if self.epoch_ndcg:
                plt.subplot(3, 2, 6)
                plt.plot(epochs_range, self.epoch_ndcg, 'c-', marker='o')
                plt.title('NDCG@10 por Época')
                plt.xlabel('Época')
                plt.ylabel('NDCG@10')
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
            
            # 3. Recall vs Emisiones acumuladas (también en gramos)
            if self.epoch_recall:
                plt.figure(figsize=(10, 6))
                emissions_in_g = [e * 1000 for e in self.cumulative_emissions]
                plt.plot(emissions_in_g, self.epoch_recall, 'm-', marker='o')
                
                # Configurar límites del eje x
                plt.xlim(0, max(emissions_in_g) * 1.1)
                
                # Añadir etiquetas con el número de época
                for i, (emissions, recall) in enumerate(zip(emissions_in_g, self.epoch_recall)):
                    plt.annotate(f"{i+1}", (emissions, recall), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9)
                    
                plt.xlabel('Emisiones de CO₂ acumuladas (g)')
                plt.ylabel('Recall@10')
                plt.title('Relación entre Emisiones Acumuladas y Recall@10')
                plt.grid(True, alpha=0.3)
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_recall_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path)
                plt.close()
                print(f"Gráfico Recall vs emisiones guardado en: {file_path}")
                
            # 4. NDCG vs Emisiones acumuladas (también en gramos)
            if self.epoch_ndcg:
                plt.figure(figsize=(10, 6))
                emissions_in_g = [e * 1000 for e in self.cumulative_emissions]
                plt.plot(emissions_in_g, self.epoch_ndcg, 'c-', marker='o')
                
                # Configurar límites del eje x
                plt.xlim(0, max(emissions_in_g) * 1.1)
                
                # Añadir etiquetas con el número de época
                for i, (emissions, ndcg) in enumerate(zip(emissions_in_g, self.epoch_ndcg)):
                    plt.annotate(f"{i+1}", (emissions, ndcg), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9)
                    
                plt.xlabel('Emisiones de CO₂ acumuladas (g)')
                plt.ylabel('NDCG@10')
                plt.title('Relación entre Emisiones Acumuladas y NDCG@10')
                plt.grid(True, alpha=0.3)
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_ndcg_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path)
                plt.close()
                print(f"Gráfico NDCG vs emisiones guardado en: {file_path}")
                
            # 5. Comparación modelo base vs combinado (si existe el entrenamiento)
            training_metrics_file = None
            for file in os.listdir(self.result_path):
                if file.startswith('training_metrics_LRML_') and file.endswith('.csv'):
                    training_metrics_file = os.path.join(self.result_path, file)
                    break
            
            if training_metrics_file and os.path.exists(training_metrics_file):
                try:
                    # Cargar métricas de entrenamiento guardadas
                    training_df = pd.read_csv(training_metrics_file)
                    
                    if 'model_rmse' in training_df.columns and 'combined_rmse' in training_df.columns:
                        plt.figure(figsize=(15, 5))
                        
                        # RMSE: Modelo Base vs Combinado
                        plt.subplot(1, 3, 1)
                        plt.plot(training_df['epoch'], training_df['model_rmse'], 'b-', marker='o', label='Base Model')
                        plt.plot(training_df['epoch'], training_df['combined_rmse'], 'g-', marker='x', label='With History')
                        plt.title('RMSE: Modelo Base vs Con Historial')
                        plt.xlabel('Época')
                        plt.ylabel('RMSE')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        
                        # Recall: Modelo Base vs Combinado
                        if 'model_recall' in training_df.columns and 'combined_recall' in training_df.columns:
                            plt.subplot(1, 3, 2)
                            plt.plot(training_df['epoch'], training_df['model_recall'], 'b-', marker='o', label='Base Model')
                            plt.plot(training_df['epoch'], training_df['combined_recall'], 'g-', marker='x', label='With History')
                            plt.title('Recall@10: Modelo Base vs Con Historial')
                            plt.xlabel('Época')
                            plt.ylabel('Recall@10')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                        
                        # NDCG: Modelo Base vs Combinado
                        if 'model_ndcg' in training_df.columns and 'combined_ndcg' in training_df.columns:
                            plt.subplot(1, 3, 3)
                            plt.plot(training_df['epoch'], training_df['model_ndcg'], 'b-', marker='o', label='Base Model')
                            plt.plot(training_df['epoch'], training_df['combined_ndcg'], 'g-', marker='x', label='With History')
                            plt.title('NDCG@10: Modelo Base vs Con Historial')
                            plt.xlabel('Época')
                            plt.ylabel('NDCG@10')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        comparison_file = f'{self.result_path}/emissions_plots/model_comparison_{self.model_name}_{timestamp}.png'
                        plt.savefig(comparison_file)
                        plt.close()
                        print(f"Gráfico de comparación guardado en: {comparison_file}")
                    
                    # 6. Gráfico de eficiencia energética: métricas vs emisiones
                    plt.figure(figsize=(15, 5))
                    
                    # Preparar las emisiones acumuladas por época
                    emissions_by_epoch = np.cumsum(self.epoch_emissions)
                    
                    # Relación RMSE/Emisiones (menor es mejor)
                    plt.subplot(1, 3, 1)
                    if 'combined_rmse' in training_df.columns and len(emissions_by_epoch) == len(training_df):
                        rmse_efficiency = training_df['combined_rmse'] / emissions_by_epoch
                        plt.plot(training_df['epoch'], rmse_efficiency, 'm-', marker='o')
                        plt.title('Eficiencia Energética: RMSE')
                        plt.xlabel('Época')
                        plt.ylabel('RMSE / kg CO₂')
                        plt.grid(True, alpha=0.3)
                    
                    # Relación Recall/Emisiones (mayor es mejor)
                    plt.subplot(1, 3, 2)
                    if 'combined_recall' in training_df.columns and len(emissions_by_epoch) == len(training_df):
                        recall_efficiency = training_df['combined_recall'] / emissions_by_epoch
                        plt.plot(training_df['epoch'], recall_efficiency, 'g-', marker='o')
                        plt.title('Eficiencia Energética: Recall')
                        plt.xlabel('Época')
                        plt.ylabel('Recall@10 / kg CO₂')
                        plt.grid(True, alpha=0.3)
                    
                    # Relación NDCG/Emisiones (mayor es mejor)
                    plt.subplot(1, 3, 3)
                    if 'combined_ndcg' in training_df.columns and len(emissions_by_epoch) == len(training_df):
                        ndcg_efficiency = training_df['combined_ndcg'] / emissions_by_epoch
                        plt.plot(training_df['epoch'], ndcg_efficiency, 'c-', marker='o')
                        plt.title('Eficiencia Energética: NDCG')
                        plt.xlabel('Época')
                        plt.ylabel('NDCG@10 / kg CO₂')
                        plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    efficiency_file = f'{self.result_path}/emissions_plots/efficiency_metrics_{self.model_name}_{timestamp}.png'
                    plt.savefig(efficiency_file)
                    plt.close()
                    print(f"Gráfico de eficiencia energética guardado en: {efficiency_file}")
                    
                except Exception as e:
                    print(f"Error al generar gráficos comparativos: {e}")
                    traceback.print_exc()
            
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


# Función para calcular RMSE con ajuste por historial
def calculate_rmse_combined(model, sess, test_data, history_recommender, 
                           user_id_to_idx, item_id_to_idx, idx_to_user_id, idx_to_item_id,
                           min_rating, max_rating, history_weight=0.3):
    user_idxs = [x[0] for x in test_data]
    item_idxs = [x[1] for x in test_data]
    true_ratings = [x[2] for x in test_data]
    
    feed_dict = {
        model.user_input: user_idxs,
        model.item_input: item_idxs,
        model.dropout: 1.0
    }
    
    # Obtener predicciones del modelo LRML
    model_preds = sess.run(model.predict_op, feed_dict=feed_dict).flatten()
    
    # Desnormalizar las predicciones
    model_preds = model_preds * (max_rating - min_rating) + min_rating
    
    # Combinar con predicciones basadas en historial
    combined_preds = []
    
    for i, (user_idx, item_idx, true_rating) in enumerate(zip(user_idxs, item_idxs, true_ratings)):
        # Convertir índices a IDs originales
        user_id = idx_to_user_id[user_idx]
        item_id = idx_to_item_id[item_idx]
        
        # Obtener predicción del modelo LRML
        model_pred = model_preds[i]
        
        # Obtener predicción basada en historial
        history_pred, confidence = history_recommender.predict_rating_from_history(user_id, item_id)
        
        # Combinar predicciones
        if history_pred is not None:
            # Ajustar peso según la confianza
            effective_weight = history_weight * confidence
            combined_pred = (1 - effective_weight) * model_pred + effective_weight * history_pred
        else:
            # Si no hay predicción por historial, usar solo modelo LRML
            combined_pred = model_pred
            
        combined_preds.append(combined_pred)
    
    # Re-normalizar para cálculo de RMSE
    true_ratings_denorm = [r * (max_rating - min_rating) + min_rating for r in true_ratings]
    
    # Calcular RMSE
    rmse = np.sqrt(np.mean(np.square(np.array(combined_preds) - np.array(true_ratings_denorm))))
    return rmse


# Función para calcular métricas de ranking (Recall@K y NDCG@K)
def calculate_ranking_metrics(model, sess, test_data, k=10):
    # Agrupar por usuario
    user_test_items = defaultdict(list)
    for user_idx, item_idx, rating in test_data:
        # Considerar un ítem como relevante si su rating normalizado es >= 0.6
        if rating >= 0.6:  
            user_test_items[user_idx].append(item_idx)
    
    recalls = []
    ndcgs = []
    
    if len(user_test_items) == 0:
        print("¡Advertencia! No se encontraron ítems relevantes en los datos de prueba.")
        return 0.0, 0.0
    
    # Limitar el número de usuarios para evaluación (opcional)
    max_users = 50
    users_to_process = list(user_test_items.keys())
    if len(users_to_process) > max_users:
        users_to_process = random.sample(users_to_process, max_users)
    
    for user_idx in users_to_process:
        relevant_items = user_test_items.get(user_idx, [])
        if not relevant_items:
            continue
            
        # Predecir scores para todos los ítems
        max_items_per_user = 300  # Limitar para mayor eficiencia
        all_items = list(range(model.num_items))
        
        # Si hay demasiados ítems, muestrear aleatoriamente
        if len(all_items) > max_items_per_user:
            # Asegurarse de incluir los ítems relevantes
            sampled_items = set(relevant_items)
            # Añadir ítems aleatorios hasta alcanzar max_items_per_user
            remaining = max_items_per_user - len(sampled_items)
            if remaining > 0:
                other_items = list(set(all_items) - sampled_items)
                if len(other_items) > remaining:
                    other_items = random.sample(other_items, remaining)
                sampled_items.update(other_items)
            items_to_rank = list(sampled_items)
        else:
            items_to_rank = all_items
            
        # Generar inputs para todos los pares usuario-ítem
        users_to_rank = [user_idx] * len(items_to_rank)
        
        # Obtener predicciones
        feed_dict = {
            model.user_input: users_to_rank,
            model.item_input: items_to_rank,
            model.dropout: 1.0
        }
        
        try:
            predictions = sess.run(model.predict_op, feed_dict=feed_dict).flatten()
            
            # Crear pares (ítem, predicción) y ordenarlos
            item_score_pairs = list(zip(items_to_rank, predictions))
            item_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Obtener los top-K ítems
            top_k_items = [x[0] for x in item_score_pairs[:k]]
            
            # Calcular Recall@K
            num_hits = len(set(top_k_items) & set(relevant_items))
            recall = num_hits / len(relevant_items)
            recalls.append(recall)
            
            # Calcular NDCG@K
            dcg = 0
            idcg = 0
            
            for i, item in enumerate(top_k_items):
                if item in relevant_items:
                    # Posición i+1 porque i empieza en 0
                    dcg += 1 / math.log2(i + 2)
                    
            # Calcular IDCG (normalización)
            for i in range(min(len(relevant_items), k)):
                # Posición i+1 porque i empieza en 0
                idcg += 1 / math.log2(i + 2)
                
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)
            
        except Exception as e:
            print(f"Error al calcular métricas de ranking para usuario {user_idx}: {e}")
            continue
    
    # Promediar métricas entre usuarios
    if not recalls or not ndcgs:
        print("¡Advertencia! No se pudieron calcular métricas de ranking.")
        return 0.0, 0.0
        
    return np.mean(recalls), np.mean(ndcgs)


# Función para calcular métricas de ranking con ajuste por historial
def calculate_ranking_metrics_combined(model, sess, test_data, history_recommender, 
                                      user_id_to_idx, item_id_to_idx, idx_to_user_id, idx_to_item_id, 
                                      min_rating, max_rating, k=10, history_weight=0.3,
                                      max_users=50, max_items_per_user=300):
    # Agrupar por usuario
    user_test_items = defaultdict(list)
    for user_idx, item_idx, rating in test_data:
        # Considerar un ítem como relevante si su rating normalizado es >= 0.6
        if rating >= 0.6:  
            user_test_items[user_idx].append(item_idx)
    
    recalls = []
    ndcgs = []
    
    if len(user_test_items) == 0:
        print("¡Advertencia! No se encontraron ítems relevantes en los datos de prueba.")
        return 0.0, 0.0
    
    # Limitar el número de usuarios para evaluación
    users_to_process = list(user_test_items.keys())
    if len(users_to_process) > max_users:
        users_to_process = random.sample(users_to_process, max_users)
    
    num_users_processed = 0
    print(f"Evaluando métricas de ranking para {len(users_to_process)} usuarios...")
    
    for user_idx in users_to_process:
        relevant_items = user_test_items.get(user_idx, [])
        if not relevant_items:
            continue
            
        # Convertir user_idx a user_id original
        user_id = idx_to_user_id[user_idx]
        
        # Predecir scores para un conjunto limitado de ítems para mayor eficiencia
        all_items = set(range(len(idx_to_item_id)))
        non_relevant_items = all_items - set(relevant_items)
        
        # Seleccionar solo algunos ítems aleatorios para evaluar
        if len(non_relevant_items) > (max_items_per_user - len(relevant_items)):
            sampled_non_relevant = random.sample(list(non_relevant_items), 
                                               max_items_per_user - len(relevant_items))
        else:
            sampled_non_relevant = list(non_relevant_items)
        
        # Combinar ítems relevantes y los aleatorios seleccionados
        items_to_rank = list(relevant_items) + sampled_non_relevant
        users_to_rank = [user_idx] * len(items_to_rank)
        
        feed_dict = {
            model.user_input: users_to_rank,
            model.item_input: items_to_rank,
            model.dropout: 1.0
        }
        
        # Obtener predicciones del modelo LRML
        try:
            model_scores = sess.run(model.predict_op, feed_dict=feed_dict).flatten()
        except Exception as e:
            print(f"Error al obtener predicciones del modelo: {e}")
            continue
        
        # Desnormalizar las predicciones
        model_scores = model_scores * (max_rating - min_rating) + min_rating
        
        # Combinar con predicciones basadas en historial
        combined_scores = []
        
        for i, item_idx in enumerate(items_to_rank):
            # Convertir a ID original
            item_id = idx_to_item_id[item_idx]
            
            # Obtener predicción del modelo LRML
            model_score = model_scores[i]
            
            # Obtener predicción basada en historial
            history_score, confidence = history_recommender.predict_rating_from_history(user_id, item_id)
            
            # Combinar predicciones
            if history_score is not None:
                effective_weight = history_weight * confidence
                combined_score = (1 - effective_weight) * model_score + effective_weight * history_score
            else:
                combined_score = model_score
                
            combined_scores.append(combined_score)
        
        # Crear pares de (ítem, score) y ordenar por score en orden descendente
        item_scores = list(zip(items_to_rank, combined_scores))
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Obtener los top-K ítems
        top_k_items = [x[0] for x in item_scores[:k]]
        
        # Calcular Recall@K
        hits = len(set(top_k_items) & set(relevant_items))
        recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
        recalls.append(recall)
        
        # Calcular NDCG@K
        dcg = 0
        idcg = 0
        for i, item in enumerate(top_k_items):
            if item in relevant_items:
                dcg += 1 / math.log2(i + 2)  # i+2 porque i empieza en 0
                
        # Calcular IDCG (normalización)
        for i in range(min(len(relevant_items), k)):
            idcg += 1 / math.log2(i + 2)
            
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)
        
        num_users_processed += 1
        if num_users_processed % 10 == 0:
            print(f"  Procesados {num_users_processed}/{len(users_to_process)} usuarios")
    
    if len(recalls) == 0 or len(ndcgs) == 0:
        print("¡Advertencia! No se pudieron calcular métricas de ranking válidas.")
        return 0.0, 0.0
        
    return np.mean(recalls), np.mean(ndcgs)


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
                        
    # Cargar películas para obtener información de géneros
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
    
    # Inicializar recomendador basado en historial
    print("Inicializando recomendador basado en historial...")
    history_recommender = UserHistoryRecommender(
        ratings_df=ratings,
        movies_df=movies,
        history_weight=args.history_weight  # Usar el valor del argumento
    )
    
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
    
    # Definir argumentos del modelo usando los parámetros de línea de comandos
    class ModelArgs:
        def __init__(self):
            self.std = 0.01
            self.embedding_size = args.embedding_size
            self.num_mem = args.num_mem
            self.dropout = args.dropout
            self.margin = 0.3
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
    emissions_tracker = EmissionsPerEpochTracker(result_path, model_name="LRML_History")
    
    # Crear el modelo
    num_users = len(user_ids)
    num_items = len(item_ids)
    print(f"Creando modelo con {num_users} usuarios y {num_items} items...")
    model = LRML(num_users, num_items, model_args)
    
    # Pre-calcular items positivos por usuario para muestreo negativo
    user_positive_items = defaultdict(set)
    for u, i, r in train_data:
        if r >= 0.7:  # Considerar ratings altos como positivos
            user_positive_items[u].add(i)
    
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
        best_metric = float('inf')  # Para RMSE, queremos minimizar
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
        model_rmses = []
        combined_rmses = []
        model_recalls = []
        combined_recalls = []
        model_ndcgs = []
        combined_ndcgs = []
        
        # Loop de entrenamiento
        for epoch in range(args.train_epoch):
            # Iniciar tracking para esta época
            system_tracker.start_epoch(epoch)
            emissions_tracker.start_epoch(epoch)
            epoch_start_time = time.time()
            
            # Barajar datos de entrenamiento
            random.shuffle(train_data)
            
            total_batches = len(train_data) // batch_size
            total_epoch_loss = 0
            
            # Entrenar en batches
            print(f"Epoch {epoch+1}/{args.train_epoch} - Procesando {total_batches} batches...")
            for i in range(total_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(train_data))
                batch = train_data[start_idx:end_idx]
                
                # Muestreo negativo mejorado
                if 'PAIR' in model.args.rnn_type:
                    neg_batch = []
                    for x in batch:
                        user_id, pos_item, rating = x
                        if random.random() < 0.7:
                            # 70% aleatorio entre todos los ítems
                            attempts = 0
                            while attempts < 10:
                                neg_item = random.randint(0, num_items - 1)
                                if neg_item != pos_item and neg_item not in user_positive_items[user_id]:
                                    break
                                attempts += 1
                        else:
                            # 30% entre los ítems más populares
                            attempts = 0
                            while attempts < 5:
                                neg_item = popular_items[random.randint(0, min(500, len(popular_items)-1))]
                                if neg_item != pos_item and neg_item not in user_positive_items[user_id]:
                                    break
                                attempts += 1
                            
                        neg_batch.append([user_id, neg_item, rating])
                else:
                    neg_batch = None
                    
                # Entrenar modelo
                feed_dict, _ = model.get_feed_dict(batch, neg_batch, mode='training')
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
            
            # Métricas solo con modelo LRML
            rmse = calculate_rmse(model, sess, eval_sample)
            print(f"RMSE (solo modelo): {rmse:.4f}")
            model_rmses.append(rmse)

            # Calcular métricas de ranking del modelo base en cada época
            recall, ndcg = calculate_ranking_metrics(model, sess, eval_sample, k=10)
            print(f"Recall@10 (solo modelo): {recall:.4f}")
            print(f"NDCG@10 (solo modelo): {ndcg:.4f}")
            model_recalls.append(recall)
            model_ndcgs.append(ndcg)

            # Métricas combinadas (modelo + historial)
            combined_rmse = calculate_rmse_combined(
                model, sess, eval_sample, history_recommender,
                user_id_to_idx, item_id_to_idx, idx_to_user_id, idx_to_item_id,
                min_rating, max_rating, args.history_weight
            )
            print(f"RMSE combinado (modelo + historial): {combined_rmse:.4f}")
            combined_rmses.append(combined_rmse)

            # Calcular métricas de ranking combinadas en cada época (ya no solo cada 5)
            print("Calculando métricas de ranking combinadas...")
            combined_recall, combined_ndcg = calculate_ranking_metrics_combined(
                model, sess, eval_sample, history_recommender,
                user_id_to_idx, item_id_to_idx, idx_to_user_id, idx_to_item_id,
                min_rating, max_rating, k=10, history_weight=args.history_weight
            )
            print(f"Recall@10 combinado: {combined_recall:.4f}")
            print(f"NDCG@10 combinado: {combined_ndcg:.4f}")
            combined_recalls.append(combined_recall)
            combined_ndcgs.append(combined_ndcg)
            
            # Finalizar tracking para esta época
            system_tracker.end_epoch(
                epoch=epoch, 
                loss=avg_epoch_loss, 
                rmse=combined_rmse,  # Métricas combinadas para tracking principal
                recall=combined_recall, 
                ndcg=combined_ndcg
            )
            
            emissions_tracker.end_epoch(
                epoch=epoch, 
                loss=avg_epoch_loss, 
                rmse=combined_rmse,
                recall=combined_recall, 
                ndcg=combined_ndcg
            )
            
            epoch_time = time.time() - epoch_start_time
            print(f"Tiempo de época: {epoch_time:.2f} segundos")
            
            # Early stopping basado en RMSE combinado
            if combined_rmse < best_metric:
                best_metric = combined_rmse
                patience_counter = 0
                print(f"¡Nuevo mejor RMSE combinado: {combined_rmse:.4f}!")
                
                # Guardar modelo
                model_path = f'{result_path}/best_model'
                saver.save(sess, model_path)
                print(f"¡Modelo guardado en: {model_path}!")
            
            '''
            else:
                patience_counter += 1
                print(f"Sin mejora en RMSE. Paciencia: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping después de {patience} épocas sin mejora")
                    break
            '''
        
        # Al finalizar el entrenamiento, evaluar métricas finales
        print("\nCalculando métricas finales...")
        system_tracker.start_epoch("final")
        
        # Evaluar en el conjunto completo de prueba
        print("Evaluando en el conjunto completo de prueba...")
        
        # Métricas solo con modelo
        print("Calculando métricas finales del modelo base...")
        final_rmse = calculate_rmse(model, sess, test_data)
        final_recall, final_ndcg = calculate_ranking_metrics(model, sess, 
                                                           random.sample(test_data, min(len(test_data), 10000)),
                                                           k=10)
        
        # Métricas combinadas (modelo + historial)
        print("Calculando métricas finales combinadas con historial...")
        test_sample = random.sample(test_data, min(len(test_data), 10000))
        final_combined_rmse = calculate_rmse_combined(
            model, sess, test_sample, history_recommender,
            user_id_to_idx, item_id_to_idx, idx_to_user_id, idx_to_item_id,
            min_rating, max_rating, args.history_weight
        )
        
        final_combined_recall, final_combined_ndcg = calculate_ranking_metrics_combined(
            model, sess, test_sample, history_recommender,
            user_id_to_idx, item_id_to_idx, idx_to_user_id, idx_to_item_id,
            min_rating, max_rating, k=10, history_weight=args.history_weight
        )
        
        # Imprimir métricas finales
        print("\n=========== MÉTRICAS FINALES ===========")
        print(f"RMSE (solo modelo): {final_rmse:.4f}")
        print(f"Recall@10 (solo modelo): {final_recall:.4f}")
        print(f"NDCG@10 (solo modelo): {final_ndcg:.4f}")
        print(f"RMSE combinado (modelo + historial): {final_combined_rmse:.4f}")
        print(f"Recall@10 combinado: {final_combined_recall:.4f}")
        print(f"NDCG@10 combinado: {final_combined_ndcg:.4f}")
        
        # Métricas del sistema
        memory_usage = process.memory_info().rss / (1024 * 1024)  # En MB
        avg_cpu = np.mean(cpu_measurements) if cpu_measurements else 0.0
        
        # Finalizar tracking
        system_tracker.end_test(final_combined_rmse, final_combined_recall, final_combined_ndcg)
        emissions_tracker.end_training(
            final_rmse=final_combined_rmse, 
            final_recall=final_combined_recall, 
            final_ndcg=final_combined_ndcg
        )
        
        # Guardar métricas finales en CSV
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        total_time = time.time() - start_time
        
        metrics_df = pd.DataFrame({
            'model': ['LRML_Base', 'LRML_History'],
            'final_rmse': [final_rmse, final_combined_rmse],
            'final_recall': [final_recall, final_combined_recall],
            'final_ndcg': [final_ndcg, final_combined_ndcg],
            'history_weight': [0.0, args.history_weight],
            'total_time_seconds': [total_time, total_time],
            'memory_usage_mb': [memory_usage, memory_usage]
        })
        
        metrics_file = f'{result_path}/final_metrics_LRML_{timestamp}.csv'
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Métricas finales guardadas en: {metrics_file}")
        
        # Verificar que todas las listas tienen la misma longitud
        min_length = min(len(train_losses), len(model_rmses), len(combined_rmses),
                        len(model_recalls), len(combined_recalls), len(model_ndcgs), len(combined_ndcgs))

        # Recortar todas las listas a la misma longitud
        train_losses = train_losses[:min_length]
        model_rmses = model_rmses[:min_length]
        combined_rmses = combined_rmses[:min_length]
        model_recalls = model_recalls[:min_length]
        combined_recalls = combined_recalls[:min_length]
        model_ndcgs = model_ndcgs[:min_length]
        combined_ndcgs = combined_ndcgs[:min_length]

        # Guardar datos de entrenamiento para gráficos
        training_df = pd.DataFrame({
            'epoch': list(range(1, min_length + 1)),
            'train_loss': train_losses,
            'model_rmse': model_rmses,
            'combined_rmse': combined_rmses,
            'model_recall': model_recalls,
            'combined_recall': combined_recalls,
            'model_ndcg': model_ndcgs,
            'combined_ndcg': combined_ndcgs
        })
        
        training_file = f'{result_path}/training_metrics_LRML_{timestamp}.csv'
        training_df.to_csv(training_file, index=False)
        print(f"Métricas de entrenamiento guardadas en: {training_file}")


if __name__ == "__main__":
    main()