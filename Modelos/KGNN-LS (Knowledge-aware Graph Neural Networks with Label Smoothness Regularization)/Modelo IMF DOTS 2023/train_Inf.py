import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
from model import KGNN_LS
import pandas as pd
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
import os
import time
import psutil
import traceback
from sklearn.metrics import mean_squared_error


# Añadir función para calcular RMSE
def calculate_rmse(y_true, y_pred, mask=None):
    """
    Calcula el RMSE entre las etiquetas verdaderas y predichas
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        mask: Máscara opcional para filtrar valores
    
    Returns:
        Valor RMSE
    """
    if mask is None:
        return np.sqrt(mean_squared_error(y_true, y_pred))
    else:
        # Solo considerar elementos donde mask=True
        return np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))


# Añadir función para calcular MAE
def calculate_mae(y_true, y_pred, mask=None):
    """
    Calcula el MAE entre las etiquetas verdaderas y predichas
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        mask: Máscara opcional para filtrar valores
    
    Returns:
        Valor MAE
    """
    if mask is None:
        return np.mean(np.abs(y_true - y_pred))
    else:
        # Solo considerar elementos donde mask=True
        return np.mean(np.abs(y_true[mask] - y_pred[mask]))


# Añadir función para calcular RMSE en un batch
def calculate_rmse_for_batch(sess, model, data, start, end):
    """
    Calcula RMSE para un batch específico usando predicciones en escala 1-5
    """
    feed_dict = get_feed_dict(model, data, start, end)
    # Obtener las puntuaciones normalizadas (1-5) para calcular RMSE
    scores_normalized = sess.run(model.scores_normalized, feed_dict)
    true_labels = data[start:end, 2]
    return calculate_rmse(true_labels, scores_normalized)


# Añadir función para calcular MAE en un batch
def calculate_mae_for_batch(sess, model, data, start, end):
    """
    Calcula MAE para un batch específico usando predicciones en escala 1-5
    """
    feed_dict = get_feed_dict(model, data, start, end)
    # Obtener las puntuaciones normalizadas (1-5) para calcular MAE
    scores_normalized = sess.run(model.scores_normalized, feed_dict)
    true_labels = data[start:end, 2]
    return calculate_mae(true_labels, scores_normalized)


class EmissionsPerEpochTracker:
    def __init__(self, result_path, model_name="KGNN_LS"):
        self.result_path = result_path
        self.model_name = model_name
        self.epoch_emissions = []
        self.cumulative_emissions = []
        self.epoch_loss = []
        self.epoch_rmse = []  # Lista para RMSE
        self.epoch_mae = []   # Lista para MAE (reemplaza F1 y Recall)
        self.total_emissions = 0.0
        self.trackers = {}
        self.best_rmse = float('inf')
        self.best_rmse_epoch = None
        self.best_rmse_emissions = None
        self.best_rmse_cumulative_emissions = None
        
        # Create directories for emissions reports and plots
        os.makedirs(f"{result_path}", exist_ok=True)
        os.makedirs(f"{result_path}/emissions_reports", exist_ok=True)
        os.makedirs(f"{result_path}/emissions_plots", exist_ok=True)
        
        # Initialize main tracker
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
            print("Main tracker started successfully")
        except Exception as e:
            print(f"Warning: Could not start main tracker: {e}")
            self.main_tracker = None

    def start_epoch(self, epoch):
        # Create a tracker with a unique name based on timestamp
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
            print(f"Warning: Could not start tracker for epoch {epoch}: {e}")
            self.trackers[epoch] = None

    def end_epoch(self, epoch, loss, rmse=None, mae=None):
        try:
            epoch_co2 = 0.0
            if epoch in self.trackers and self.trackers[epoch]:
                try:
                    epoch_co2 = self.trackers[epoch].stop() or 0.0
                except Exception as e:
                    print(f"Warning: Error stopping tracker for epoch {epoch}: {e}")
                    epoch_co2 = 0.0
            
            # Accumulate total emissions
            self.total_emissions += epoch_co2
            
            # Save data for this epoch
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
            
            print(f"Epoch {epoch} - Emissions: {epoch_co2:.8f} kg, Cumulative: {self.total_emissions:.8f} kg, Loss: {loss:.4f}")
            if rmse is not None:
                print(f"RMSE: {rmse:.4f}")
            if mae is not None:
                print(f"MAE: {mae:.4f}")
        except Exception as e:
            print(f"Error measuring emissions in epoch {epoch}: {e}")

    def end_training(self, final_rmse=None, final_mae=None):
        try:
            # Stop the main tracker
            final_emissions = 0.0
            if hasattr(self, 'main_tracker') and self.main_tracker:
                try:
                    final_emissions = self.main_tracker.stop() or 0.0
                    print(f"\nTotal CO2 Emissions: {final_emissions:.6f} kg")
                except Exception as e:
                    print(f"Error stopping main tracker: {e}")
                    final_emissions = self.total_emissions
            else:
                final_emissions = self.total_emissions
            
            # Make sure all trackers are stopped
            for epoch, tracker in self.trackers.items():
                if tracker is not None:
                    try:
                        tracker.stop()
                    except:
                        pass
            
            # If no epoch emission data but we have total emissions,
            # create at least one entry for graphs
            if not self.epoch_emissions and final_emissions > 0:
                self.epoch_emissions = [final_emissions]
                self.cumulative_emissions = [final_emissions]
                if final_rmse is not None:
                    self.epoch_rmse = [final_rmse]
                if final_mae is not None:
                    self.epoch_mae = [final_mae]
            
            # If no data, exit
            if not self.epoch_emissions:
                print("No emission data to plot")
                return
            
            # Make sure we have final metrics if not tracked by epoch
            if not self.epoch_rmse and final_rmse is not None:
                self.epoch_rmse = [final_rmse] * len(self.epoch_emissions)
            if not self.epoch_mae and final_mae is not None:
                self.epoch_mae = [final_mae] * len(self.epoch_emissions)
            
            # Create dataframe with all data
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
            print(f"Emission metrics saved to: {emissions_file}")
            
            # Mostrar información del mejor RMSE y sus emisiones
            if self.best_rmse_epoch is not None:
                print(f"\n=== Best RMSE and Associated Emissions ===")
                print(f"Best RMSE: {self.best_rmse:.4f} (Epoch {self.best_rmse_epoch})")
                print(f"Emissions at best RMSE: {self.best_rmse_emissions:.8f} kg")
                print(f"Cumulative emissions at best RMSE: {self.best_rmse_cumulative_emissions:.8f} kg")
            
            # Plot relationships
            self.plot_emissions_vs_metrics(timestamp, final_rmse, final_mae)
            
        except Exception as e:
            print(f"Error generating emission plots: {e}")
            import traceback
            traceback.print_exc()

    def plot_emissions_vs_metrics(self, timestamp, final_rmse=None, final_mae=None):
        """Generate plots for emissions vs metrics"""
        
        # Use metrics by epoch if available, else create list with final values
        if not self.epoch_rmse and final_rmse is not None:
            self.epoch_rmse = [final_rmse] * len(self.epoch_emissions)
        if not self.epoch_mae and final_mae is not None:
            self.epoch_mae = [final_mae] * len(self.epoch_emissions)
        
        try:
            # RMSE plot
            if self.epoch_rmse:
                plt.figure(figsize=(10, 6))
                plt.plot(self.cumulative_emissions, self.epoch_rmse, 'm-', marker='o')
                
                # Add labels with epoch number
                for i, (emissions, rmse) in enumerate(zip(self.cumulative_emissions, self.epoch_rmse)):
                    plt.annotate(f"{i}", (emissions, rmse), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9)
                    
                plt.xlabel('Cumulative CO2 Emissions (kg)')
                plt.ylabel('RMSE')
                plt.title('Relationship between Cumulative Emissions and RMSE')
                plt.grid(True, alpha=0.3)
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_rmse_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path)
                plt.close()
                print(f"Plot saved to: {file_path}")
                
            # MAE plot
            if self.epoch_mae:
                plt.figure(figsize=(10, 6))
                plt.plot(self.cumulative_emissions, self.epoch_mae, 'c-', marker='o')
                
                # Add labels with epoch number
                for i, (emissions, mae) in enumerate(zip(self.cumulative_emissions, self.epoch_mae)):
                    plt.annotate(f"{i}", (emissions, mae), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9)
                    
                plt.xlabel('Cumulative CO2 Emissions (kg)')
                plt.ylabel('MAE')
                plt.title('Relationship between Cumulative Emissions and MAE')
                plt.grid(True, alpha=0.3)
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_mae_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path)
                plt.close()
                print(f"Plot saved to: {file_path}")
            
            # Combined plot: Emissions per epoch and cumulative
            plt.figure(figsize=(12, 8))
            
            # Configurar el layout para 6 subplots (2 filas, 3 columnas)
            plt.subplot(2, 3, 1)
            plt.plot(range(len(self.epoch_emissions)), self.epoch_emissions, 'r-', marker='x')
            plt.title('Emissions per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('CO2 Emissions (kg)')
            
            plt.subplot(2, 3, 2)
            plt.plot(range(len(self.cumulative_emissions)), self.cumulative_emissions, 'r-', marker='o')
            plt.title('Cumulative Emissions per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('CO2 Emissions (kg)')
            
            if self.epoch_loss:
                plt.subplot(2, 3, 3)
                plt.plot(range(len(self.epoch_loss)), self.epoch_loss, 'orange', marker='s')
                plt.title('Loss per Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
            
            if self.epoch_rmse:
                plt.subplot(2, 3, 4)
                plt.plot(range(len(self.epoch_rmse)), self.epoch_rmse, 'b-', marker='o')
                plt.title('RMSE per Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('RMSE')
                
            if self.epoch_mae:
                plt.subplot(2, 3, 5)
                plt.plot(range(len(self.epoch_mae)), self.epoch_mae, 'm-', marker='s')
                plt.title('MAE per Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('MAE')
            
            plt.tight_layout()
            
            file_path = f'{self.result_path}/emissions_plots/metrics_by_epoch_{self.model_name}_{timestamp}.png'
            plt.savefig(file_path)
            plt.close()
            print(f"Plot saved to: {file_path}")
            
            # Scatter plot of performance vs cumulative emissions
            if self.epoch_rmse and self.epoch_mae:
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                # Ajustar tamaño de los puntos según la época
                sizes = [(i+1)*20 for i in range(len(self.cumulative_emissions))]
                
                scatter = plt.scatter(self.epoch_rmse, self.cumulative_emissions, 
                            color='blue', marker='o', s=sizes, alpha=0.7)
                
                # Añadir etiquetas de época
                for i, (rmse, em) in enumerate(zip(self.epoch_rmse, self.cumulative_emissions)):
                    plt.annotate(f"{i}", (rmse, em), textcoords="offset points", 
                                xytext=(0,5), ha='center', fontsize=9)
                
                plt.ylabel('Cumulative CO2 Emissions (kg)')
                plt.xlabel('RMSE')
                plt.title('RMSE vs Cumulative Emissions')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                scatter = plt.scatter(self.epoch_mae, self.cumulative_emissions, 
                            color='green', marker='s', s=sizes, alpha=0.7)
                
                # Añadir etiquetas de época
                for i, (mae, em) in enumerate(zip(self.epoch_mae, self.cumulative_emissions)):
                    plt.annotate(f"{i}", (mae, em), textcoords="offset points", 
                                xytext=(0,5), ha='center', fontsize=9)
                
                plt.ylabel('Cumulative CO2 Emissions (kg)')
                plt.xlabel('MAE')
                plt.title('MAE vs Cumulative Emissions')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_performance_scatter_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path)
                plt.close()
                print(f"Plot saved to: {file_path}")
                
        except Exception as e:
            print(f"Error generating plots: {e}")
            import traceback
            traceback.print_exc()


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
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Metrics:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Memory: {self.current_epoch_metrics['memory_usage_mb']:.2f}MB")
        print(f"  CPU: {self.current_epoch_metrics['cpu_usage_percent']:.1f}%")
        print(f"  Loss: {loss:.4f}")
        if rmse is not None:
            print(f"  RMSE: {rmse:.4f}")
        if mae is not None:
            print(f"  MAE: {mae:.4f}")

    def end_test(self, rmse=None, mae=None):
        self.test_metrics = {
            'test_time_sec': time.time() - self.epoch_start_time,
            'total_time_sec': time.time() - self.start_time,
            'final_memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'final_cpu_usage_percent': psutil.cpu_percent()
        }
        
        if rmse is not None:
            self.test_metrics['test_rmse'] = rmse
            
        if mae is not None:
            self.test_metrics['test_mae'] = mae
        
        # Print final metrics - formato como el ejemplo que proporcionaste
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
        if rmse is not None:
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
        
        # Save metrics to CSV
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        metrics_df = pd.DataFrame(self.train_metrics)
        metrics_df.to_csv(f"./results/system_metrics_{timestamp}.csv", index=False)


def train(args, data, show_loss, show_topk):
    # Create results directory if it doesn't exist
    result_path = "./results"
    os.makedirs(result_path, exist_ok=True)
    
    # Initialize the trackers
    print("Initializing performance and emissions trackers...")
    system_tracker = SystemMetricsTracker()
    emissions_tracker = EmissionsPerEpochTracker(result_path)
    
    # Lists to store metrics
    train_loss_list = []
    train_rmse_list = []  
    train_mae_list = []   # Nueva lista para MAE
    test_rmse_list = []   
    test_mae_list = []    # Nueva lista para MAE
    
    # Variables para almacenar métricas finales
    final_test_rmse = None
    final_test_mae = None

    try:
        n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
        train_data, eval_data, test_data = data[4], data[5], data[6]
        adj_entity, adj_relation = data[7], data[8]

        interaction_table, offset = get_interaction_table(train_data, n_entity)
        model = KGNN_LS(args, n_user, n_entity, n_relation, adj_entity, adj_relation, interaction_table, offset)

        # top-K evaluation settings
        user_list, train_record, test_record, item_set, k_list = topk_settings(show_topk, train_data, test_data, n_item)

        with tf.compat.v1.Session() as sess:
            # Initialize all variables, including the hash table
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.tables_initializer())  # Initialize the hash table

            for step in range(args.n_epochs):
                # Start epoch tracking
                system_tracker.start_epoch(step)
                emissions_tracker.start_epoch(step)
                
                # Training
                np.random.shuffle(train_data)
                start = 0
                epoch_loss = 0.0
                batch_count = 0
                
                # Arrays para acumular predicciones y valores reales para RMSE
                rmse_values = []
                
                # Skip the last incomplete minibatch if its size < batch size
                print(f"Epoch {step+1}/{args.n_epochs} - Processing batches...")
                while start + args.batch_size <= train_data.shape[0]:
                    # Entrenar con el batch actual
                    _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                    
                    # Calcular RMSE para este batch (solo de vez en cuando para ahorrar tiempo)
                    if batch_count % 5 == 0:
                        try:
                            batch_rmse = calculate_rmse_for_batch(sess, model, train_data, start, start + args.batch_size)
                            rmse_values.append(batch_rmse)
                        except Exception as e:
                            print(f"  Error calculating batch RMSE: {e}")
                    
                    start += args.batch_size
                    epoch_loss += loss
                    batch_count += 1
                    
                    if show_loss and batch_count % 5 == 0:
                        print(f"  Batch {batch_count}: Loss: {loss:.4f}")
                
                # Calculate average loss and RMSE
                avg_loss = epoch_loss / (batch_count if batch_count > 0 else 1)
                train_loss_list.append(avg_loss)
                
                # Calcular RMSE y MAE promedio si tenemos valores
                train_rmse = float(np.mean(rmse_values)) if rmse_values else None
                if train_rmse is not None:
                    train_rmse_list.append(train_rmse)
                    print(f"  Train RMSE: {train_rmse:.4f}")

                # Calcular MAE para entrenamiento
                try:
                    mae_values = []
                    mae_start = 0
                    while mae_start + args.batch_size <= train_data.shape[0]:
                        batch_mae = calculate_mae_for_batch(sess, model, train_data, mae_start, mae_start + args.batch_size)
                        mae_values.append(batch_mae)
                        mae_start += args.batch_size
                    train_mae = float(np.mean(mae_values)) if mae_values else None
                    if train_mae is not None:
                        train_mae_list.append(train_mae)
                        print(f"  Train MAE: {train_mae:.4f}")
                except Exception as e:
                    print(f"  Error calculating train MAE: {e}")
                    train_mae = None
                
                # Calcular RMSE y MAE en datos de test
                try:
                    test_rmse_values = []
                    test_mae_values = []
                    test_start = 0
                    # Solo usar una muestra de los datos de prueba para ahorrar tiempo
                    sample_size = min(10000, test_data.shape[0])
                    test_sample = test_data[:sample_size]
                    
                    while test_start + args.batch_size <= test_sample.shape[0]:
                        batch_test_rmse = calculate_rmse_for_batch(sess, model, test_sample, test_start, test_start + args.batch_size)
                        batch_test_mae = calculate_mae_for_batch(sess, model, test_sample, test_start, test_start + args.batch_size)
                        test_rmse_values.append(batch_test_rmse)
                        test_mae_values.append(batch_test_mae)
                        test_start += args.batch_size
                    
                    test_rmse = float(np.mean(test_rmse_values)) if test_rmse_values else None
                    test_mae = float(np.mean(test_mae_values)) if test_mae_values else None
                    
                    if test_rmse is not None:
                        test_rmse_list.append(test_rmse)
                        print(f"  Test RMSE: {test_rmse:.4f}")
                    if test_mae is not None:
                        test_mae_list.append(test_mae)
                        print(f"  Test MAE: {test_mae:.4f}")
                        
                except Exception as e:
                    print(f"  Error calculating test RMSE/MAE: {e}")
                    test_rmse = test_mae = None

                # End epoch tracking
                system_tracker.end_epoch(step, avg_loss, train_rmse, train_mae)
                emissions_tracker.end_epoch(step, avg_loss, train_rmse, train_mae)
                
                # Imprimir resultados incluyendo RMSE y MAE
                print('epoch %d    train rmse: %s mae: %s    test rmse: %s mae: %s'
                    % (step, 
                        f"{train_rmse:.4f}" if train_rmse is not None else "N/A",
                        f"{train_mae:.4f}" if train_mae is not None else "N/A",
                        f"{test_rmse:.4f}" if test_rmse is not None else "N/A",
                        f"{test_mae:.4f}" if test_mae is not None else "N/A"))

                # top-K evaluation
                if show_topk:
                    precision, recall = topk_eval(
                        sess, model, user_list, train_record, test_record, item_set, k_list, args.batch_size)
                    print('precision: ', end='')
                    for i in precision:
                        print('%.4f\t' % i, end='')
                    print()
                    print('recall: ', end='')
                    for i in recall:
                        print('%.4f\t' % i, end='')
                    print('\n')

            # Final test metrics
            system_tracker.start_epoch("final")
            print("\nFinal evaluation on test set...")
            
            # Calcular RMSE y MAE finales
            final_test_rmse = None
            final_test_mae = None
            try:
                if test_rmse_list:
                    final_test_rmse = test_rmse_list[-1]
                if test_mae_list:
                    final_test_mae = test_mae_list[-1]
                    
                # Si no tenemos valores, calcular en una muestra de datos de prueba
                if final_test_rmse is None or final_test_mae is None:
                    test_rmse_values = []
                    test_mae_values = []
                    test_start = 0
                    test_sample = test_data[:min(10000, test_data.shape[0])]
                    while test_start + args.batch_size <= test_sample.shape[0]:
                        if final_test_rmse is None:
                            test_batch_rmse = calculate_rmse_for_batch(sess, model, test_sample, test_start, test_start + args.batch_size)
                            test_rmse_values.append(test_batch_rmse)
                        if final_test_mae is None:
                            test_batch_mae = calculate_mae_for_batch(sess, model, test_sample, test_start, test_start + args.batch_size)
                            test_mae_values.append(test_batch_mae)
                        test_start += args.batch_size
                    
                    if final_test_rmse is None and test_rmse_values:
                        final_test_rmse = float(np.mean(test_rmse_values))
                    if final_test_mae is None and test_mae_values:
                        final_test_mae = float(np.mean(test_mae_values))
                        print(f"Final Test RMSE: {final_test_rmse:.4f}")
                        print(f"Final Test MAE: {final_test_mae:.4f}")
            except Exception as e:
                print(f"Error calculating final test metrics: {e}")
            
            system_tracker.end_test(final_test_rmse, final_test_mae)            
            print(f"\nFinal metrics - RMSE: {final_test_rmse if final_test_rmse is not None else 'N/A'}, MAE: {final_test_mae if final_test_mae is not None else 'N/A'}")
            
            # Save training metrics to CSV
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            metrics_df = pd.DataFrame({
                'epoch': range(len(train_loss_list)),
                'train_loss': train_loss_list
            })
            
            # Añadir RMSE y MAE si están disponibles
            if train_rmse_list:
                metrics_df['train_rmse'] = pd.Series(train_rmse_list).reindex(metrics_df.index)
            
            if train_mae_list:
                metrics_df['train_mae'] = pd.Series(train_mae_list).reindex(metrics_df.index)
            
            if test_rmse_list:
                metrics_df['test_rmse'] = pd.Series(test_rmse_list).reindex(metrics_df.index)
                
            if test_mae_list:
                metrics_df['test_mae'] = pd.Series(test_mae_list).reindex(metrics_df.index)
            
            metrics_file = f"{result_path}/model_metrics_{timestamp}.csv"
            metrics_df.to_csv(metrics_file, index=False)
            print(f"Model metrics saved to: {metrics_file}")
            
            # Construir y devolver diccionario de resultados
            return {
                'train_rmse': train_rmse_list[-1] if train_rmse_list else None,
                'train_mae': train_mae_list[-1] if train_mae_list else None,
                'test_rmse': final_test_rmse,
                'test_mae': final_test_mae,
                'all_metrics': {
                    'train_loss': train_loss_list,
                    'train_rmse': train_rmse_list,
                    'train_mae': train_mae_list,
                    'test_rmse': test_rmse_list,
                    'test_mae': test_mae_list
                }
            }
            

    finally:
        # Sincronizar información del mejor RMSE entre trackers
        if system_tracker.best_rmse_epoch is not None:
            emissions_tracker.best_rmse = system_tracker.best_rmse
            emissions_tracker.best_rmse_epoch = system_tracker.best_rmse_epoch
            # Buscar las emisiones correspondientes al mejor epoch
            if system_tracker.best_rmse_epoch < len(emissions_tracker.epoch_emissions):
                emissions_tracker.best_rmse_emissions = emissions_tracker.epoch_emissions[system_tracker.best_rmse_epoch]
                emissions_tracker.best_rmse_cumulative_emissions = emissions_tracker.cumulative_emissions[system_tracker.best_rmse_epoch]
        
        # End emissions tracking and generate final reports
        emissions_tracker.end_training(final_test_rmse, final_test_mae)
        
        # Visualize emissions results
        visualize_emissions_results()


def visualize_emissions_results():
    # Load the CSV file generated by CodeCarbon
    try:
        results = pd.read_csv("./codecarbon_results/emissions.csv")
        print("\n--- Emission and Energy Consumption Results ---")
        print(results[["timestamp", "project_name", "duration", "emissions", "energy_consumed"]])
    except FileNotFoundError:
        print("Emissions file not found. Make sure CodeCarbon executed correctly.")


def get_interaction_table(train_data, n_entity):
    offset = len(str(n_entity))
    offset = 10 ** offset
    keys = train_data[:, 0] * offset + train_data[:, 1]
    keys = keys.astype(np.int64)
    values = train_data[:, 2].astype(np.float32)

    # Use tf.lookup.StaticHashTable instead of tf.contrib.lookup.HashTable
    # Default value 3.0 for rating range 1-5 (middle value)
    interaction_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values), default_value=3.0)
    return interaction_table, offset


def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 100
        k_list = [1, 2, 5, 10, 20, 50, 100]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def get_feed_dict(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict


def ctr_eval(sess, model, data, batch_size):
    start = 0
    auc_list = []
    f1_list = []
    while start + batch_size <= data.shape[0]:
        auc, f1 = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))
        auc_list.append(auc)
        f1_list.append(f1)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list))


def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]

    return precision, recall


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict