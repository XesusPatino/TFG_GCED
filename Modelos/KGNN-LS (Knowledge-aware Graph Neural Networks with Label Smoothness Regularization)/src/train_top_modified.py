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


# Función para calcular RMSE
def calculate_rmse(y_true, y_pred, mask=None):
    """
    Calcula el RMSE entre las etiquetas verdaderas y predichas
    """
    if mask is None:
        return np.sqrt(mean_squared_error(y_true, y_pred))
    else:
        return np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))


# Función para calcular RMSE en un batch
def calculate_rmse_for_batch(sess, model, data, start, end):
    """
    Calcula RMSE para un batch específico usando predicciones en escala 1-5
    """
    feed_dict = get_feed_dict(model, data, start, end)
    scores_normalized = sess.run(model.scores_normalized, feed_dict)
    true_labels = data[start:end, 2]
    return calculate_rmse(true_labels, scores_normalized)


# Función para calcular métricas NDCG y Recall@K correctamente
def calculate_ndcg_k(y_true, y_pred, k):
    """
    Calcula NDCG@K para un usuario específico
    """
    # Ordenar por predicción descendente
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order]
    
    # Calcular DCG@K
    dcg = 0.0
    for i in range(min(k, len(y_true_sorted))):
        if y_true_sorted[i] >= 4.0:  # Relevante si rating >= 4
            dcg += 1.0 / np.log2(i + 2)
    
    # Calcular IDCG@K (ideal DCG)
    relevance_sorted = np.sort(y_true)[::-1]
    idcg = 0.0
    for i in range(min(k, len(relevance_sorted))):
        if relevance_sorted[i] >= 4.0:
            idcg += 1.0 / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_recall_k(y_true, y_pred, k):
    """
    Calcula Recall@K para un usuario específico
    """
    # Encontrar ítems relevantes (rating >= 4)
    relevant_items = np.where(y_true >= 4.0)[0]
    if len(relevant_items) == 0:
        return 0.0
    
    # Obtener top-k ítems predichos
    top_k_items = np.argsort(y_pred)[::-1][:k]
    
    # Calcular intersección
    hits = len(set(top_k_items) & set(relevant_items))
    
    return hits / len(relevant_items)


class EmissionsPerEpochTracker:
    def __init__(self, result_path, model_name="KGNN_LS"):
        self.result_path = result_path
        self.model_name = model_name
        self.epoch_emissions = []
        self.cumulative_emissions = []
        self.epoch_loss = []
        self.epoch_rmse = []
        self.epoch_times = []
        self.epoch_memory = []
        self.epoch_cpu = []
        
        # Diccionarios para métricas top-k
        self.epoch_recall = {5: [], 10: [], 20: [], 50: []}
        self.epoch_ndcg = {5: [], 10: [], 20: [], 50: []}
        
        self.total_emissions = 0.0
        self.trackers = {}
        self.best_rmse = float('inf')
        self.best_rmse_epoch = None
        self.best_rmse_emissions = None
        self.best_rmse_cumulative_emissions = None
        self.best_rmse_metrics = None
        
        # Create directories
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
        self.epoch_start_time = time.time()
        
        # Create tracker for this epoch
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

    def end_epoch(self, epoch, loss, rmse=None, recall_dict=None, ndcg_dict=None):
        try:
            epoch_co2 = 0.0
            if epoch in self.trackers and self.trackers[epoch]:
                try:
                    epoch_co2 = self.trackers[epoch].stop() or 0.0
                except Exception as e:
                    print(f"Warning: Error stopping tracker for epoch {epoch}: {e}")
                    epoch_co2 = 0.0
            
            # Calculate epoch time and system metrics
            epoch_time = time.time() - self.epoch_start_time
            memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
            cpu_usage = psutil.cpu_percent()
            
            # Accumulate total emissions
            self.total_emissions += epoch_co2
            
            # Save data for this epoch
            self.epoch_emissions.append(epoch_co2)
            self.cumulative_emissions.append(self.total_emissions)
            self.epoch_loss.append(loss)
            self.epoch_times.append(epoch_time)
            self.epoch_memory.append(memory_usage)
            self.epoch_cpu.append(cpu_usage)
            
            if rmse is not None:
                self.epoch_rmse.append(rmse)
                # Track best RMSE
                if rmse < self.best_rmse:
                    self.best_rmse = rmse
                    self.best_rmse_epoch = epoch
                    self.best_rmse_emissions = epoch_co2
                    self.best_rmse_cumulative_emissions = self.total_emissions
                    self.best_rmse_metrics = {
                        'time': epoch_time,
                        'memory': memory_usage,
                        'cpu': cpu_usage
                    }
            
            # Store recall and ndcg metrics
            if recall_dict:
                for k in [5, 10, 20, 50]:
                    if k in recall_dict:
                        self.epoch_recall[k].append(recall_dict[k])
                    else:
                        self.epoch_recall[k].append(0.0)
            else:
                for k in [5, 10, 20, 50]:
                    self.epoch_recall[k].append(0.0)
                    
            if ndcg_dict:
                for k in [5, 10, 20, 50]:
                    if k in ndcg_dict:
                        self.epoch_ndcg[k].append(ndcg_dict[k])
                    else:
                        self.epoch_ndcg[k].append(0.0)
            else:
                for k in [5, 10, 20, 50]:
                    self.epoch_ndcg[k].append(0.0)
            
            # Print epoch summary in the requested format
            recall_5 = recall_dict.get(5, 0.0) if recall_dict else 0.0
            recall_10 = recall_dict.get(10, 0.0) if recall_dict else 0.0
            ndcg_5 = ndcg_dict.get(5, 0.0) if ndcg_dict else 0.0
            ndcg_10 = ndcg_dict.get(10, 0.0) if ndcg_dict else 0.0
            
            print(f"Epoch {epoch}: Time={epoch_time:.2f}s, Memory={memory_usage:.2f}MB, CPU={cpu_usage:.1f}%, "
                  f"RMSE={rmse:.4f}, Recall@5={recall_5:.4f}, Recall@10={recall_10:.4f}, "
                  f"NDCG@5={ndcg_5:.4f}, NDCG@10={ndcg_10:.4f}")
            
        except Exception as e:
            print(f"Error measuring emissions in epoch {epoch}: {e}")

    def end_training(self, final_rmse=None, final_recall_dict=None, final_ndcg_dict=None, 
                    total_training_time=None, test_time=None):
        try:
            # Stop the main tracker
            final_emissions = 0.0
            if hasattr(self, 'main_tracker') and self.main_tracker:
                try:
                    final_emissions = self.main_tracker.stop() or 0.0
                except Exception as e:
                    print(f"Error stopping main tracker: {e}")
                    final_emissions = self.total_emissions
            else:
                final_emissions = self.total_emissions
            
            # Stop all epoch trackers
            for epoch, tracker in self.trackers.items():
                if tracker is not None:
                    try:
                        tracker.stop()
                    except:
                        pass
            
            # Get final system metrics
            final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
            final_cpu = psutil.cpu_percent()
            
            # Create dataframe with all metrics
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            epochs_range = range(len(self.epoch_emissions))
            
            # Base dataframe
            df_data = {
                'epoch': epochs_range,
                'epoch_emissions_kg': self.epoch_emissions,
                'cumulative_emissions_kg': self.cumulative_emissions,
                'loss': self.epoch_loss,
                'rmse': self.epoch_rmse if self.epoch_rmse else [None] * len(self.epoch_emissions),
                'time_sec': self.epoch_times,
                'memory_mb': self.epoch_memory,
                'cpu_percent': self.epoch_cpu
            }
            
            # Add recall and ndcg metrics
            for k in [5, 10, 20, 50]:
                df_data[f'recall@{k}'] = self.epoch_recall[k]
                df_data[f'ndcg@{k}'] = self.epoch_ndcg[k]
            
            df = pd.DataFrame(df_data)
            
            # Save emissions metrics
            emissions_file = f'{self.result_path}/emissions_reports/emissions_metrics_{self.model_name}_{timestamp}.csv'
            df.to_csv(emissions_file, index=False)
            print(f"Emissions metrics saved to: {emissions_file}")
            
            # Print final training metrics in the requested format
            print("\n=== Final Training Metrics ===")
            for i, row in df.iterrows():
                epoch_num = int(row['epoch'])
                time_val = row['time_sec']
                memory_val = row['memory_mb']
                cpu_val = row['cpu_percent']
                rmse_val = row['rmse']
                recall5_val = row['recall@5']
                recall10_val = row['recall@10']
                ndcg5_val = row['ndcg@5']
                ndcg10_val = row['ndcg@10']
                
                print(f"Epoch {epoch_num}: Time={time_val:.2f}s, Memory={memory_val:.2f}MB, CPU={cpu_val:.1f}%, "
                      f"RMSE={rmse_val:.4f}, Recall@5={recall5_val:.4f}, Recall@10={recall10_val:.4f}, "
                      f"NDCG@5={ndcg5_val:.4f}, NDCG@10={ndcg10_val:.4f}")
            
            # Print final test metrics
            print("\n=== Final Test Metrics ===")
            if total_training_time and test_time:
                print(f"Total Time: {total_training_time:.2f}s (Test: {test_time:.2f}s)")
            print(f"Final Memory: {final_memory:.2f}MB")
            print(f"Final CPU: {final_cpu:.1f}%")
            print(f"RMSE: {final_rmse:.4f}")
            
            # Print final recall and ndcg metrics
            if final_recall_dict:
                for k in [5, 10, 20, 50]:
                    if k in final_recall_dict:
                        print(f"Recall@{k}: {final_recall_dict[k]:.4f}")
            
            if final_ndcg_dict:
                for k in [5, 10, 20, 50]:
                    if k in final_ndcg_dict:
                        print(f"NDCG@{k}: {final_ndcg_dict[k]:.4f}")
            
            # Print best training RMSE information
            if self.best_rmse_epoch is not None:
                print(f"\n=== Best Training RMSE ===")
                print(f"Best RMSE: {self.best_rmse:.4f} (Epoch {self.best_rmse_epoch})")
                if self.best_rmse_metrics:
                    print(f"Time: {self.best_rmse_metrics['time']:.2f}s")
                    print(f"Memory: {self.best_rmse_metrics['memory']:.2f}MB")
                    print(f"CPU: {self.best_rmse_metrics['cpu']:.1f}%")
                
                print(f"\n=== Best RMSE and Associated Emissions ===")
                print(f"Best RMSE: {self.best_rmse:.4f} (Epoch {self.best_rmse_epoch})")
                print(f"Emissions at best RMSE: {self.best_rmse_emissions:.8f} kg")
                print(f"Cumulative emissions at best RMSE: {self.best_rmse_cumulative_emissions:.8f} kg")
            
            # Generate plots
            self.plot_emissions_vs_metrics(epochs_range, timestamp)
            
        except Exception as e:
            print(f"Error generating final reports: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_emissions_vs_metrics(self, epochs_range, timestamp):
        """Generate plots for emissions vs metrics"""
        try:
            # Combined plot: Emissions and metrics
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            plt.plot(epochs_range, self.epoch_emissions, 'r-', marker='x')
            plt.title('Emissions per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('CO₂ Emissions (kg)')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 3, 2)
            plt.plot(epochs_range, self.cumulative_emissions, 'r-', marker='o')
            plt.title('Cumulative Emissions')
            plt.xlabel('Epoch')
            plt.ylabel('CO₂ Emissions (kg)')
            plt.grid(True, alpha=0.3)
            
            if self.epoch_rmse:
                plt.subplot(2, 3, 3)
                plt.plot(epochs_range, self.epoch_rmse, 'b-', marker='o')
                plt.title('RMSE per Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('RMSE')
                plt.grid(True, alpha=0.3)
            
            if self.epoch_recall[5]:
                plt.subplot(2, 3, 4)
                plt.plot(epochs_range, self.epoch_recall[5], 'g-', marker='s', label='Recall@5')
                plt.plot(epochs_range, self.epoch_recall[10], 'orange', marker='^', label='Recall@10')
                plt.title('Recall per Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('Recall')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            if self.epoch_ndcg[5]:
                plt.subplot(2, 3, 5)
                plt.plot(epochs_range, self.epoch_ndcg[5], 'm-', marker='d', label='NDCG@5')
                plt.plot(epochs_range, self.epoch_ndcg[10], 'c-', marker='v', label='NDCG@10')
                plt.title('NDCG per Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('NDCG')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            file_path = f'{self.result_path}/emissions_plots/metrics_by_epoch_{self.model_name}_{timestamp}.png'
            plt.savefig(file_path)
            plt.close()
            print(f"Plot saved to: {file_path}")
            
        except Exception as e:
            print(f"Error generating plots: {e}")


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
            'start_time': self.epoch_start_time
        }
        
    def end_epoch(self, epoch, loss, rmse=None, recall_dict=None, ndcg_dict=None):
        if hasattr(self, 'current_epoch_metrics'):
            epoch_time = time.time() - self.current_epoch_metrics['start_time']
            memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
            cpu_usage = psutil.cpu_percent()
            
            metrics = {
                'epoch': epoch,
                'time': epoch_time,
                'memory': memory_usage,
                'cpu': cpu_usage,
                'loss': loss,
                'rmse': rmse
            }
            
            if recall_dict:
                for k, v in recall_dict.items():
                    metrics[f'recall@{k}'] = v
                    
            if ndcg_dict:
                for k, v in ndcg_dict.items():
                    metrics[f'ndcg@{k}'] = v
            
            self.train_metrics.append(metrics)
            
            if rmse is not None and rmse < self.best_rmse:
                self.best_rmse = rmse
                self.best_rmse_epoch = epoch
                self.best_rmse_metrics = {
                    'time': epoch_time,
                    'memory': memory_usage,
                    'cpu': cpu_usage
                }
    
    def end_test(self, test_rmse=None, test_recall_dict=None, test_ndcg_dict=None):
        total_time = time.time() - self.start_time
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        cpu_usage = psutil.cpu_percent()
        
        self.test_metrics = {
            'total_time': total_time,
            'memory': memory_usage,
            'cpu': cpu_usage,
            'rmse': test_rmse
        }
        
        if test_recall_dict:
            for k, v in test_recall_dict.items():
                self.test_metrics[f'recall@{k}'] = v
                
        if test_ndcg_dict:
            for k, v in test_ndcg_dict.items():
                self.test_metrics[f'ndcg@{k}'] = v


def train(args, data, show_loss, show_topk):
    # Create results directory
    result_path = "./results"
    os.makedirs(result_path, exist_ok=True)
    
    # Initialize trackers
    print("Initializing performance and emissions trackers...")
    system_tracker = SystemMetricsTracker()
    emissions_tracker = EmissionsPerEpochTracker(result_path)
    
    # Lists to store metrics
    train_loss_list = []
    train_rmse_list = []
    
    # Final metrics variables
    final_test_rmse = None
    final_test_recall_dict = None
    final_test_ndcg_dict = None

    try:
        n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
        train_data, eval_data, test_data = data[4], data[5], data[6]
        adj_entity, adj_relation = data[7], data[8]

        interaction_table, offset = get_interaction_table(train_data, n_entity)
        model = KGNN_LS(args, n_user, n_entity, n_relation, adj_entity, adj_relation, interaction_table, offset)

        # Top-K evaluation settings
        user_list, train_record, test_record, item_set, k_list = topk_settings(True, train_data, test_data, n_item)

        with tf.compat.v1.Session() as sess:
            # Initialize all variables
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.tables_initializer())

            for step in range(args.n_epochs):
                # Start epoch tracking
                system_tracker.start_epoch(step)
                emissions_tracker.start_epoch(step)
                
                # Training
                np.random.shuffle(train_data)
                start = 0
                epoch_loss = 0.0
                batch_count = 0
                rmse_values = []
                
                print(f"Epoch {step+1}/{args.n_epochs} - Processing batches...")
                while start + args.batch_size <= train_data.shape[0]:
                    # Train with current batch
                    _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                    
                    # Calculate RMSE for this batch occasionally
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
                
                train_rmse = float(np.mean(rmse_values)) if rmse_values else None
                if train_rmse is not None:
                    train_rmse_list.append(train_rmse)

                # Top-K evaluation with NDCG and Recall
                recall_dict, ndcg_dict = topk_eval_with_ndcg(
                    sess, model, user_list, train_record, test_record, item_set, [5, 10, 20, 50], args.batch_size)

                # End epoch tracking
                system_tracker.end_epoch(step, avg_loss, train_rmse, recall_dict, ndcg_dict)
                emissions_tracker.end_epoch(step, avg_loss, train_rmse, recall_dict, ndcg_dict)

            # Final test evaluation
            print("\nFinal evaluation on test set...")
            test_start_time = time.time()
            
            # Calculate final test RMSE
            test_rmse_values = []
            test_start = 0
            test_sample = test_data[:min(10000, test_data.shape[0])]
            
            while test_start + args.batch_size <= test_sample.shape[0]:
                test_batch_rmse = calculate_rmse_for_batch(sess, model, test_sample, test_start, test_start + args.batch_size)
                test_rmse_values.append(test_batch_rmse)
                test_start += args.batch_size
            
            final_test_rmse = float(np.mean(test_rmse_values)) if test_rmse_values else None
            
            # Calculate final top-k metrics
            final_test_recall_dict, final_test_ndcg_dict = topk_eval_with_ndcg(
                sess, model, user_list, train_record, test_record, item_set, [5, 10, 20, 50], args.batch_size)
            
            test_time = time.time() - test_start_time
            total_training_time = time.time() - system_tracker.start_time
            
            # End tracking
            system_tracker.end_test(final_test_rmse, final_test_recall_dict, final_test_ndcg_dict)
            
            print(f"\nFinal Test RMSE: {final_test_rmse:.4f}")
            for k in [5, 10, 20, 50]:
                if k in final_test_recall_dict:
                    print(f"Final Test Recall@{k}: {final_test_recall_dict[k]:.4f}")
                if k in final_test_ndcg_dict:
                    print(f"Final Test NDCG@{k}: {final_test_ndcg_dict[k]:.4f}")
            
            # Save training metrics to CSV
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            metrics_df = pd.DataFrame({
                'epoch': range(len(train_loss_list)),
                'train_loss': train_loss_list,
                'train_rmse': pd.Series(train_rmse_list).reindex(range(len(train_loss_list)))
            })
            
            metrics_file = f"{result_path}/model_metrics_{timestamp}.csv"
            metrics_df.to_csv(metrics_file, index=False)
            print(f"Model metrics saved to: {metrics_file}")
            
            return {
                'train_rmse': train_rmse_list[-1] if train_rmse_list else None,
                'test_rmse': final_test_rmse,
                'test_recall': final_test_recall_dict,
                'test_ndcg': final_test_ndcg_dict
            }

    finally:
        # Sync best RMSE information between trackers
        if system_tracker.best_rmse_epoch is not None:
            emissions_tracker.best_rmse = system_tracker.best_rmse
            emissions_tracker.best_rmse_epoch = system_tracker.best_rmse_epoch
            if system_tracker.best_rmse_epoch < len(emissions_tracker.epoch_emissions):
                emissions_tracker.best_rmse_emissions = emissions_tracker.epoch_emissions[system_tracker.best_rmse_epoch]
                emissions_tracker.best_rmse_cumulative_emissions = emissions_tracker.cumulative_emissions[system_tracker.best_rmse_epoch]
        
        # End emissions tracking and generate final reports
        emissions_tracker.end_training(final_test_rmse, final_test_recall_dict, final_test_ndcg_dict, 
                                     system_tracker.test_metrics.get('total_time'), test_time if 'test_time' in locals() else None)


def get_interaction_table(train_data, n_entity):
    offset = len(str(n_entity))
    offset = 10 ** offset
    keys = train_data[:, 0] * offset + train_data[:, 1]
    keys = keys.astype(np.int64)
    values = train_data[:, 2].astype(np.float32)

    # Use tf.lookup.StaticHashTable with default value 3.0 for rating range 1-5
    interaction_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values), default_value=3.0)
    return interaction_table, offset


def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 100
        k_list = [5, 10, 20, 50]
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


def topk_eval_with_ndcg(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    """
    Evalúa métricas top-k incluyendo NDCG y Recall correctamente calculados
    """
    recall_dict = {k: [] for k in k_list}
    ndcg_dict = {k: [] for k in k_list}

    for user in user_list:
        # Get test items (items not in training set)
        test_item_list = list(item_set - train_record[user])
        if len(test_item_list) == 0:
            continue
            
        # Get scores for all test items
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {
                model.user_indices: [user] * batch_size,
                model.item_indices: test_item_list[start:start + batch_size]
            })
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # Handle remaining items if any
        if start < len(test_item_list):
            remaining_items = test_item_list[start:]
            padding_needed = batch_size - len(remaining_items)
            padded_items = remaining_items + [remaining_items[-1]] * padding_needed
            
            items, scores = model.get_scores(sess, {
                model.user_indices: [user] * batch_size,
                model.item_indices: padded_items
            })
            
            for i, (item, score) in enumerate(zip(items, scores)):
                if i < len(remaining_items):  # Only use non-padded results
                    item_score_map[item] = score

        # Sort items by score (descending)
        item_score_pairs = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        sorted_items = [pair[0] for pair in item_score_pairs]
        sorted_scores = [pair[1] for pair in item_score_pairs]
        
        # Get true ratings for all test items (from test_record)
        test_items_user = test_record.get(user, set())
        
        # Create arrays for NDCG calculation
        y_true = np.array([1.0 if item in test_items_user else 0.0 for item in sorted_items])
        y_pred = np.array(sorted_scores)
        
        # Calculate metrics for each k
        for k in k_list:
            if len(sorted_items) >= k:
                # Recall@K
                top_k_items = set(sorted_items[:k])
                relevant_items = test_items_user
                
                if len(relevant_items) > 0:
                    hits = len(top_k_items & relevant_items)
                    recall = hits / len(relevant_items)
                else:
                    recall = 0.0
                
                recall_dict[k].append(recall)
                
                # NDCG@K
                if len(relevant_items) > 0:
                    # Calculate DCG@K
                    dcg = 0.0
                    for i in range(min(k, len(sorted_items))):
                        if sorted_items[i] in relevant_items:
                            dcg += 1.0 / np.log2(i + 2)
                    
                    # Calculate IDCG@K
                    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
                    
                    ndcg = dcg / idcg if idcg > 0 else 0.0
                else:
                    ndcg = 0.0
                
                ndcg_dict[k].append(ndcg)
            else:
                recall_dict[k].append(0.0)
                ndcg_dict[k].append(0.0)

    # Calculate average metrics
    avg_recall = {k: np.mean(recall_dict[k]) if recall_dict[k] else 0.0 for k in k_list}
    avg_ndcg = {k: np.mean(ndcg_dict[k]) if ndcg_dict[k] else 0.0 for k in k_list}

    return avg_recall, avg_ndcg


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label >= 4.0:  # For test, only consider relevant items (rating >= 4)
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
