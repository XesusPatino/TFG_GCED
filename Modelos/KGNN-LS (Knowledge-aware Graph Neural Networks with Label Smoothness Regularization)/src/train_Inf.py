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


# Añadir función para calcular RMSE en un batch
def calculate_rmse_for_batch(sess, model, data, start, end):
    """
    Calcula RMSE para un batch específico
    """
    feed_dict = get_feed_dict(model, data, start, end)
    # Obtener las puntuaciones (raw scores) para calcular RMSE
    _, scores = model.get_scores(sess, feed_dict)
    true_labels = data[start:end, 2]
    return calculate_rmse(true_labels, scores)
    
    # Add inside calculate_rmse_for_batch
    print(f"Debug - Score range: min={scores.min()}, max={scores.max()}, mean={scores.mean()}")     


class EmissionsPerEpochTracker:
    def __init__(self, result_path, model_name="KGNN_LS"):
        self.result_path = result_path
        self.model_name = model_name
        self.epoch_emissions = []
        self.cumulative_emissions = []
        self.epoch_auc = []
        self.epoch_f1 = []
        self.epoch_loss = []
        self.epoch_rmse = []  # Añadir lista para RMSE
        self.epoch_recall = []  # Añadir lista para Recall
        self.total_emissions = 0.0
        self.trackers = {}
        
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

    def end_epoch(self, epoch, loss, auc=None, f1=None, rmse=None, recall=None):  # Añadir parámetro recall
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
            if auc is not None:
                self.epoch_auc.append(auc)
            if f1 is not None:
                self.epoch_f1.append(f1)
            if rmse is not None:
                self.epoch_rmse.append(rmse)
            if recall is not None:
                self.epoch_recall.append(recall)
            
            print(f"Epoch {epoch} - Emissions: {epoch_co2:.8f} kg, Cumulative: {self.total_emissions:.8f} kg, Loss: {loss:.4f}")
            if auc is not None:
                print(f"AUC: {auc:.4f}")
            if f1 is not None:
                print(f"F1: {f1:.4f}")
            if rmse is not None:
                print(f"RMSE: {rmse:.4f}")
            if recall is not None:
                print(f"Recall: {recall:.4f}")
        except Exception as e:
            print(f"Error measuring emissions in epoch {epoch}: {e}")

    def end_training(self, final_auc=None, final_f1=None, final_rmse=None, final_recall=None):  # Añadir parámetro final_recall
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
                if final_auc is not None:
                    self.epoch_auc = [final_auc]
                if final_f1 is not None:
                    self.epoch_f1 = [final_f1]
                if final_rmse is not None:
                    self.epoch_rmse = [final_rmse]
                if final_recall is not None:
                    self.epoch_recall = [final_recall]
            
            # If no data, exit
            if not self.epoch_emissions:
                print("No emission data to plot")
                return
            
            # Make sure we have a final AUC if not tracked by epoch
            if not self.epoch_auc and final_auc is not None:
                self.epoch_auc = [final_auc] * len(self.epoch_emissions)
            if not self.epoch_f1 and final_f1 is not None:
                self.epoch_f1 = [final_f1] * len(self.epoch_emissions)
            if not self.epoch_rmse and final_rmse is not None:
                self.epoch_rmse = [final_rmse] * len(self.epoch_emissions)
            if not self.epoch_recall and final_recall is not None:
                self.epoch_recall = [final_recall] * len(self.epoch_emissions)
            
            # Create dataframe with all data
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            df = pd.DataFrame({
                'epoch': range(len(self.epoch_emissions)),
                'epoch_emissions_kg': self.epoch_emissions,
                'cumulative_emissions_kg': self.cumulative_emissions,
                'loss': self.epoch_loss if self.epoch_loss else [0.0] * len(self.epoch_emissions),
                'auc': self.epoch_auc if self.epoch_auc else [None] * len(self.epoch_emissions),
                'f1': self.epoch_f1 if self.epoch_f1 else [None] * len(self.epoch_emissions),
                'rmse': self.epoch_rmse if self.epoch_rmse else [None] * len(self.epoch_emissions),  # Añadir RMSE
                'recall': self.epoch_recall if self.epoch_recall else [None] * len(self.epoch_emissions)  # Añadir Recall
            })
            
            emissions_file = f'{self.result_path}/emissions_reports/emissions_metrics_{self.model_name}_{timestamp}.csv'
            df.to_csv(emissions_file, index=False)
            print(f"Emission metrics saved to: {emissions_file}")
            
            # Plot relationships
            self.plot_emissions_vs_metrics(timestamp, final_auc, final_f1, final_rmse, final_recall)
            
        except Exception as e:
            print(f"Error generating emission plots: {e}")
            import traceback
            traceback.print_exc()

    def plot_emissions_vs_metrics(self, timestamp, final_auc=None, final_f1=None, final_rmse=None, final_recall=None):
        """Generate plots for emissions vs metrics"""
        
        # Use AUC by epoch if available, else create list with final AUC
        if not self.epoch_auc and final_auc is not None:
            self.epoch_auc = [final_auc] * len(self.epoch_emissions)
        if not self.epoch_f1 and final_f1 is not None:
            self.epoch_f1 = [final_f1] * len(self.epoch_emissions)
        if not self.epoch_rmse and final_rmse is not None:
            self.epoch_rmse = [final_rmse] * len(self.epoch_emissions)
        if not self.epoch_recall and final_recall is not None:
            self.epoch_recall = [final_recall] * len(self.epoch_emissions)
        
        try:
            if self.epoch_auc:
                # 1. Cumulative emissions vs AUC
                plt.figure(figsize=(10, 6))
                plt.plot(self.cumulative_emissions, self.epoch_auc, 'b-', marker='o')
                
                # Add labels with epoch number
                for i, (emissions, auc) in enumerate(zip(self.cumulative_emissions, self.epoch_auc)):
                    plt.annotate(f"{i}", (emissions, auc), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9)
                    
                plt.xlabel('Cumulative CO2 Emissions (kg)')
                plt.ylabel('AUC')
                plt.title('Relationship between Cumulative Emissions and AUC')
                plt.grid(True, alpha=0.3)
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_auc_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path)
                plt.close()
                print(f"Plot saved to: {file_path}")
            
            if self.epoch_f1:
                # 2. Cumulative emissions vs F1
                plt.figure(figsize=(10, 6))
                plt.plot(self.cumulative_emissions, self.epoch_f1, 'g-', marker='o')
                
                # Add labels with epoch number
                for i, (emissions, f1) in enumerate(zip(self.cumulative_emissions, self.epoch_f1)):
                    plt.annotate(f"{i}", (emissions, f1), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9)
                    
                plt.xlabel('Cumulative CO2 Emissions (kg)')
                plt.ylabel('F1 Score')
                plt.title('Relationship between Cumulative Emissions and F1 Score')
                plt.grid(True, alpha=0.3)
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_f1_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path)
                plt.close()
                print(f"Plot saved to: {file_path}")
            
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
                
            # Recall plot
            if self.epoch_recall:
                plt.figure(figsize=(10, 6))
                plt.plot(self.cumulative_emissions, self.epoch_recall, 'c-', marker='o')
                
                # Add labels with epoch number
                for i, (emissions, recall) in enumerate(zip(self.cumulative_emissions, self.epoch_recall)):
                    plt.annotate(f"{i}", (emissions, recall), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9)
                    
                plt.xlabel('Cumulative CO2 Emissions (kg)')
                plt.ylabel('Recall')
                plt.title('Relationship between Cumulative Emissions and Recall')
                plt.grid(True, alpha=0.3)
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_recall_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path)
                plt.close()
                print(f"Plot saved to: {file_path}")
            
            # 3. Combined plot: Emissions per epoch and cumulative
            plt.figure(figsize=(15, 10))  # Hacer más grande para incluir recall
            
            # Configurar el layout para 8 subplots (2 filas, 4 columnas)
            plt.subplot(2, 4, 1)
            plt.plot(range(len(self.epoch_emissions)), self.epoch_emissions, 'r-', marker='x')
            plt.title('Emissions per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('CO2 Emissions (kg)')
            
            plt.subplot(2, 4, 2)
            plt.plot(range(len(self.cumulative_emissions)), self.cumulative_emissions, 'r-', marker='o')
            plt.title('Cumulative Emissions per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('CO2 Emissions (kg)')
            
            if self.epoch_loss:
                plt.subplot(2, 4, 3)
                plt.plot(range(len(self.epoch_loss)), self.epoch_loss, 'g-', marker='o')
                plt.title('Loss per Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
            
            if self.epoch_auc:
                plt.subplot(2, 4, 4)
                plt.plot(range(len(self.epoch_auc)), self.epoch_auc, 'b-', marker='o')
                plt.title('AUC per Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('AUC')
            
            if self.epoch_f1:
                plt.subplot(2, 4, 5)
                plt.plot(range(len(self.epoch_f1)), self.epoch_f1, 'g-', marker='o')
                plt.title('F1 Score per Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('F1')
                
            if self.epoch_rmse:
                plt.subplot(2, 4, 6)
                plt.plot(range(len(self.epoch_rmse)), self.epoch_rmse, 'm-', marker='o')
                plt.title('RMSE per Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('RMSE')
                
            if self.epoch_recall:
                plt.subplot(2, 4, 7)
                plt.plot(range(len(self.epoch_recall)), self.epoch_recall, 'c-', marker='o')
                plt.title('Recall per Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('Recall')
            
            plt.tight_layout()
            
            file_path = f'{self.result_path}/emissions_plots/metrics_by_epoch_{self.model_name}_{timestamp}.png'
            plt.savefig(file_path)
            plt.close()
            print(f"Plot saved to: {file_path}")
            
            # 4. Scatter plot of performance vs cumulative emissions
            if self.epoch_auc or self.epoch_f1 or self.epoch_rmse or self.epoch_recall:
                plt.figure(figsize=(10, 6))
                
                # Adjust point size by epoch
                sizes = [(i+1)*20 for i in range(len(self.cumulative_emissions))]
                
                if self.epoch_auc:
                    plt.scatter(self.epoch_auc, self.cumulative_emissions, 
                                color='blue', marker='o', s=sizes, alpha=0.7, label='AUC')
                
                if self.epoch_f1:
                    plt.scatter(self.epoch_f1, self.cumulative_emissions, 
                                color='green', marker='x', s=sizes, alpha=0.7, label='F1')
                
                if self.epoch_rmse:
                    plt.scatter(self.epoch_rmse, self.cumulative_emissions, 
                                color='magenta', marker='s', s=sizes, alpha=0.7, label='RMSE')
                                
                if self.epoch_recall:
                    plt.scatter(self.epoch_recall, self.cumulative_emissions, 
                                color='cyan', marker='^', s=sizes, alpha=0.7, label='Recall')
                
                # Add epoch labels for one of the series
                if self.epoch_auc:
                    for i, (auc, em) in enumerate(zip(self.epoch_auc, self.cumulative_emissions)):
                        plt.annotate(f"{i}", (auc, em), textcoords="offset points", 
                                     xytext=(0,5), ha='center', fontsize=9)
                
                plt.ylabel('Cumulative CO2 Emissions (kg)')
                plt.xlabel('Performance Metrics')
                plt.title('Relationship between Performance Metrics and Cumulative Emissions')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
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
        
    def start_epoch(self, epoch):
        self.epoch_start_time = time.time()
        self.current_epoch_metrics = {
            'epoch': epoch,
            'memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'cpu_usage_percent': psutil.cpu_percent(),
        }
        
    def end_epoch(self, epoch, loss, auc=None, f1=None, rmse=None, recall=None):  # Añadir recall
        epoch_time = time.time() - self.epoch_start_time
        self.current_epoch_metrics['epoch_time_sec'] = epoch_time
        self.current_epoch_metrics['loss'] = loss
        if auc is not None:
            self.current_epoch_metrics['auc'] = auc
        if f1 is not None:
            self.current_epoch_metrics['f1'] = f1
        if rmse is not None:
            self.current_epoch_metrics['rmse'] = rmse
        if recall is not None:
            self.current_epoch_metrics['recall'] = recall
        self.train_metrics.append(self.current_epoch_metrics)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Metrics:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Memory: {self.current_epoch_metrics['memory_usage_mb']:.2f}MB")
        print(f"  CPU: {self.current_epoch_metrics['cpu_usage_percent']:.1f}%")
        print(f"  Loss: {loss:.4f}")
        if auc is not None:
            print(f"  AUC: {auc:.4f}")
        if f1 is not None:
            print(f"  F1: {f1:.4f}")
        if rmse is not None:
            print(f"  RMSE: {rmse:.4f}")
        if recall is not None:
            print(f"  Recall: {recall:.4f}")

    def end_test(self, auc, f1, rmse=None, recall=None):  # Añadir recall
        self.test_metrics = {
            'test_time_sec': time.time() - self.epoch_start_time,
            'total_time_sec': time.time() - self.start_time,
            'final_memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'final_cpu_usage_percent': psutil.cpu_percent(),
            'test_auc': auc,
            'test_f1': f1
        }
        
        if rmse is not None:
            self.test_metrics['test_rmse'] = rmse
            
        if recall is not None:
            self.test_metrics['test_recall'] = recall  # Añadir recall
        
        # Print final metrics
        print("\n=== Final Training Metrics ===")
        for m in self.train_metrics:
            metrics_str = f"Epoch {m['epoch']}: Time={m['epoch_time_sec']:.2f}s, Memory={m['memory_usage_mb']:.2f}MB, CPU={m['cpu_usage_percent']:.1f}%, Loss={m['loss']:.4f}"
            if 'auc' in m:
                metrics_str += f", AUC={m['auc']:.4f}"
            if 'f1' in m:
                metrics_str += f", F1={m['f1']:.4f}"
            if 'rmse' in m:
                metrics_str += f", RMSE={m['rmse']:.4f}"
            if 'recall' in m:
                metrics_str += f", Recall={m['recall']:.4f}"
            print(metrics_str)
        
        print("\n=== Final Test Metrics ===")
        print(f"Total Time: {self.test_metrics['total_time_sec']:.2f}s (Test: {self.test_metrics['test_time_sec']:.2f}s)")
        print(f"Final Memory: {self.test_metrics['final_memory_usage_mb']:.2f}MB")
        print(f"Final CPU: {self.test_metrics['final_cpu_usage_percent']:.1f}%")
        print_str = f"Test AUC: {auc:.4f}, F1: {f1:.4f}"
        if recall is not None:
            print_str += f", Recall: {recall:.4f}"
        if rmse is not None:
            print_str += f", RMSE: {rmse:.4f}"
        print(print_str)
        
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
    train_auc_list = []
    train_f1_list = []
    train_recall_list = []  # Nueva lista para recall
    test_auc_list = []
    test_f1_list = []
    test_recall_list = []  # Nueva lista para recall
    train_rmse_list = []  
    test_rmse_list = []   
    
    # Variables para almacenar métricas finales
    final_test_auc = None
    final_test_f1 = None
    final_test_recall = None  # Nueva variable para recall final
    final_test_rmse = None

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
                
                # Calcular RMSE promedio si tenemos valores
                train_rmse = float(np.mean(rmse_values)) if rmse_values else None
                if train_rmse is not None:
                    train_rmse_list.append(train_rmse)
                    print(f"  Train RMSE: {train_rmse:.4f}")

                # Evaluation
                train_auc, train_f1, train_recall, train_precision = ctr_eval_with_recall(sess, model, train_data, args.batch_size)
                train_auc_list.append(train_auc)
                train_f1_list.append(train_f1)
                train_recall_list.append(train_recall)
                
                eval_auc, eval_f1, eval_recall, eval_precision = ctr_eval_with_recall(sess, model, eval_data, args.batch_size)
                test_auc, test_f1, test_recall, test_precision = ctr_eval_with_recall(sess, model, test_data, args.batch_size)
                test_auc_list.append(test_auc)
                test_f1_list.append(test_f1)
                test_recall_list.append(test_recall)
                
                # Calcular RMSE en datos de test
                try:
                    test_rmse_values = []
                    test_start = 0
                    # Solo usar una muestra de los datos de prueba para ahorrar tiempo
                    sample_size = min(10000, test_data.shape[0])
                    test_sample = test_data[:sample_size]
                    
                    while test_start + args.batch_size <= test_sample.shape[0]:
                        test_batch_rmse = calculate_rmse_for_batch(sess, model, test_sample, test_start, test_start + args.batch_size)
                        test_rmse_values.append(test_batch_rmse)
                        test_start += args.batch_size
                    
                    if test_rmse_values:
                        test_rmse = float(np.mean(test_rmse_values))
                        test_rmse_list.append(test_rmse)
                        print(f"  Test RMSE: {test_rmse:.4f}")
                    else:
                        test_rmse = None
                except Exception as e:
                    print(f"  Error calculating test RMSE: {e}")
                    test_rmse = None

                # End epoch tracking
                system_tracker.end_epoch(step, avg_loss, eval_auc, eval_f1, train_rmse, train_recall)  # Añadir train_recall
                emissions_tracker.end_epoch(step, avg_loss, eval_auc, eval_f1, train_rmse, train_recall)  # Añadir train_recall
                
                # Imprimir resultados incluyendo RMSE
                print('epoch %d    train auc: %.4f  f1: %.4f  recall: %.4f  rmse: %s    eval auc: %.4f  f1: %.4f  recall: %.4f    test auc: %.4f  f1: %.4f  recall: %.4f  rmse: %s'
                    % (step, train_auc, train_f1, train_recall,
                        f"{train_rmse:.4f}" if train_rmse is not None else "N/A", 
                        eval_auc, eval_f1, eval_recall,
                        test_auc, test_f1, test_recall,
                        f"{test_rmse:.4f}" if test_rmse is not None else "N/A"))

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
            final_test_auc, final_test_f1, final_test_recall, final_test_precision = ctr_eval_with_recall(sess, model, test_data, args.batch_size)
            
            # Calcular RMSE final
            final_test_rmse = None
            try:
                if test_rmse_list:
                    final_test_rmse = test_rmse_list[-1]
                else:
                    # Intentar calcular en los datos de prueba
                    test_rmse_values = []
                    test_start = 0
                    test_sample = test_data[:min(10000, test_data.shape[0])]
                    while test_start + args.batch_size <= test_sample.shape[0]:
                        test_batch_rmse = calculate_rmse_for_batch(sess, model, test_sample, test_start, test_start + args.batch_size)
                        test_rmse_values.append(test_batch_rmse)
                        test_start += args.batch_size
                    
                    if test_rmse_values:
                        final_test_rmse = float(np.mean(test_rmse_values))
                        print(f"Final Test RMSE: {final_test_rmse:.4f}")
            except Exception as e:
                print(f"Error calculating final test RMSE: {e}")
            
            system_tracker.end_test(final_test_auc, final_test_f1, final_test_rmse, final_test_recall)            
            print(f"\nFinal metrics - AUC: {final_test_auc:.4f}, F1: {final_test_f1:.4f}, Recall: {final_test_recall:.4f}, RMSE: {final_test_rmse if final_test_rmse is not None else 'N/A'}")
            
            # Debugging - añade esto antes del return en la función train
            print("\n======== DEBUG RECALL VALUES =========")
            print(f"Train recall list: {train_recall_list}")
            print(f"Test recall list: {test_recall_list}")
            print(f"Final test recall: {final_test_recall}")
            print("======================================")
    
            
            # Save training metrics to CSV
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            metrics_df = pd.DataFrame({
                'epoch': range(len(train_loss_list)),
                'train_loss': train_loss_list,
                'train_auc': train_auc_list,
                'train_f1': train_f1_list,
                'train_recall': train_recall_list,  # Asegúrate de que esté aquí
                'test_auc': test_auc_list,
                'test_f1': test_f1_list,
                'test_recall': test_recall_list     # Y también aquí
            })
            
            
            # Añadir RMSE si está disponible
            if train_rmse_list:
                metrics_df['train_rmse'] = pd.Series(train_rmse_list).reindex(metrics_df.index)
            
            if test_rmse_list:
                metrics_df['test_rmse'] = pd.Series(test_rmse_list).reindex(metrics_df.index)
            
            metrics_file = f"{result_path}/model_metrics_{timestamp}.csv"
            metrics_df.to_csv(metrics_file, index=False)
            print(f"Model metrics saved to: {metrics_file}")
            
            # Construir y devolver diccionario de resultados
            return {
                'train_auc': train_auc_list[-1] if train_auc_list else None,
                'train_f1': train_f1_list[-1] if train_f1_list else None,
                'train_recall': train_recall_list[-1] if train_recall_list else None,  # Añade recall
                'train_rmse': train_rmse_list[-1] if train_rmse_list else None,
                'test_auc': final_test_auc,
                'test_f1': final_test_f1,
                'test_recall': final_test_recall,  # Añade recall
                'test_rmse': final_test_rmse,
                'all_metrics': {
                    'train_loss': train_loss_list,
                    'train_auc': train_auc_list,
                    'train_f1': train_f1_list,
                    'train_recall': train_recall_list,  # Añade recall
                    'train_rmse': train_rmse_list,
                    'test_auc': test_auc_list,
                    'test_f1': test_f1_list,
                    'test_recall': test_recall_list,  # Añade recall
                    'test_rmse': test_rmse_list
                }
            }
            

    finally:
        # End emissions tracking and generate final reports
        emissions_tracker.end_training(final_test_auc, final_test_f1, final_test_rmse, final_test_recall)
        
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
    interaction_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values), default_value=0.5)
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


def ctr_eval_with_recall(sess, model, data, batch_size):
    """Evaluación extendida que incluye recall junto a AUC y F1"""
    start = 0
    auc_list = []
    f1_list = []
    recall_list = []
    precision_list = []
    
    while start + batch_size <= data.shape[0]:
        feed_dict = get_feed_dict(model, data, start, start + batch_size)
        auc, f1 = model.eval(sess, feed_dict)
        
        # Calcular recall manualmente
        # Primero obtenemos las predicciones
        items, scores = model.get_scores(sess, feed_dict)
        labels = data[start:start + batch_size, 2]
        
        # Convertir scores en predicciones binarias
        predictions = (scores > 0.5).astype(int)
        
        # Calcular recall: tp / (tp + fn)
        true_positives = np.sum((predictions == 1) & (labels == 1))
        false_negatives = np.sum((predictions == 0) & (labels == 1))
        
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0.0
            
        # Calcular precisión: tp / (tp + fp)
        false_positives = np.sum((predictions == 1) & (labels == 0))
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0.0
        
        auc_list.append(auc)
        f1_list.append(f1)
        recall_list.append(recall)
        precision_list.append(precision)
        
        start += batch_size
        
    if len(recall_list) > 0:
        print(f"DEBUG - Recall en batch: {recall_list[-1]:.4f}")    
        
    return float(np.mean(auc_list)), float(np.mean(f1_list)), float(np.mean(recall_list)), float(np.mean(precision_list))


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