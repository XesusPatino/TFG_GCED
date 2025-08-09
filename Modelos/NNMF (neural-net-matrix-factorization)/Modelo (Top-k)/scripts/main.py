from __future__ import absolute_import, print_function
from __future__ import absolute_import, print_function
import argparse
import json
import time
import sys
import os
import psutil
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tensorflow as tf
import pandas as pd
import numpy as np
from nnmf.models import NNMF
from nnmf.utils import chunk_df

# Importar CodeCarbon para medir emisiones de CO₂
from codecarbon import EmissionsTracker

# Suppress specific numpy warnings
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# Add imports for top-k evaluation
from collections import defaultdict
import random
from sklearn.preprocessing import LabelEncoder

# Suppress specific numpy warnings
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


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
        
    def end_epoch(self, epoch, train_rmse, valid_rmse=None):
        epoch_time = time.time() - self.epoch_start_time
        self.current_epoch_metrics['epoch_time_sec'] = epoch_time
        self.current_epoch_metrics['train_rmse'] = train_rmse
        if valid_rmse is not None:
            self.current_epoch_metrics['valid_rmse'] = valid_rmse
        self.train_metrics.append(self.current_epoch_metrics)
        
        # Imprimir resumen de época
        print(f"\nEpoch {epoch} Metrics:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Memory: {self.current_epoch_metrics['memory_usage_mb']:.2f}MB")
        print(f"  CPU: {self.current_epoch_metrics['cpu_usage_percent']:.1f}%")
        print(f"  Train RMSE: {train_rmse:.4f}")
        if valid_rmse is not None:
            print(f"  Valid RMSE: {valid_rmse:.4f}")
        
    def end_test(self, test_rmse):
        self.test_metrics = {
            'test_time_sec': time.time() - self.epoch_start_time,
            'total_time_sec': time.time() - self.start_time,
            'final_memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'final_cpu_usage_percent': psutil.cpu_percent(),
            'test_rmse': test_rmse,
        }
        
        # Imprimir métricas finales
        print("\n=== Final Training Metrics ===")
        for m in self.train_metrics:
            metrics_str = f"Epoch {m['epoch']}: Time={m['epoch_time_sec']:.2f}s, Memory={m['memory_usage_mb']:.2f}MB, CPU={m['cpu_usage_percent']:.1f}%, Train RMSE={m['train_rmse']:.4f}"
            if 'valid_rmse' in m:
                metrics_str += f", Valid RMSE={m['valid_rmse']:.4f}"
            print(metrics_str)
        
        print("\n=== Final Test Metrics ===")
        print(f"Total Time: {self.test_metrics['total_time_sec']:.2f}s (Test: {self.test_metrics['test_time_sec']:.2f}s)")
        print(f"Final Memory: {self.test_metrics['final_memory_usage_mb']:.2f}MB")
        print(f"Final CPU: {self.test_metrics['final_cpu_usage_percent']:.1f}%")
        print(f"Test RMSE: {test_rmse:.4f}")
        
        # Guardar métricas en CSV
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        metrics_df = pd.DataFrame(self.train_metrics)
        
        # Create results directory if it doesn't exist
        result_path = "results"
        os.makedirs(result_path, exist_ok=True)
        
        metrics_df.to_csv(f"{result_path}/system_metrics_{timestamp}.csv", index=False)
        print(f"System metrics saved to: {result_path}/system_metrics_{timestamp}.csv")


# Clase para seguimiento de emisiones por época
class EmissionsPerEpochTracker:
    def __init__(self, result_path="results", model_name="NNMF"):
        self.result_path = result_path
        self.model_name = model_name
        self.epoch_emissions = []
        self.cumulative_emissions = []
        self.epoch_train_rmse = []
        self.epoch_valid_rmse = []
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
            print("Main tracker started successfully")
        except Exception as e:
            print(f"Warning: Could not start main tracker: {e}")
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
            print(f"Warning: Could not start tracker for epoch {epoch}: {e}")
            self.trackers[epoch] = None
    
    def end_epoch(self, epoch, train_rmse, valid_rmse=None):
        try:
            epoch_co2 = 0.0
            if epoch in self.trackers and self.trackers[epoch]:
                try:
                    epoch_co2 = self.trackers[epoch].stop() or 0.0
                except Exception as e:
                    print(f"Warning: Error stopping tracker for epoch {epoch}: {e}")
                    epoch_co2 = 0.0
            
            # Acumular emisiones totales
            self.total_emissions += epoch_co2
            
            # Guardar datos de esta época
            self.epoch_emissions.append(epoch_co2)
            self.cumulative_emissions.append(self.total_emissions)
            self.epoch_train_rmse.append(train_rmse)
            if valid_rmse is not None:
                self.epoch_valid_rmse.append(valid_rmse)
            
            print(f"Epoch {epoch} - Emissions: {epoch_co2:.8f} kg, Cumulative: {self.total_emissions:.8f} kg")
            print(f"Train RMSE: {train_rmse:.4f}, Valid RMSE: {valid_rmse:.4f}")
        except Exception as e:
            print(f"Error measuring emissions in epoch {epoch}: {e}")
    
    def end_training(self, final_rmse):
        try:
            # Detener el tracker principal
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
                    self.epoch_valid_rmse = [final_rmse]
            
            # Si no hay datos, salir
            if not self.epoch_emissions:
                print("No emission data to plot")
                return
            
            # Crear dataframe con todos los datos
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            data = {
                'epoch': range(len(self.epoch_emissions)),
                'epoch_emissions_kg': self.epoch_emissions,
                'cumulative_emissions_kg': self.cumulative_emissions,
                'train_rmse': self.epoch_train_rmse
            }
            
            if self.epoch_valid_rmse:
                data['valid_rmse'] = self.epoch_valid_rmse
                
            df = pd.DataFrame(data)
            
            emissions_file = f'{self.result_path}/emissions_reports/emissions_metrics_{self.model_name}_{timestamp}.csv'
            df.to_csv(emissions_file, index=False)
            print(f"Emission metrics saved to: {emissions_file}")
            
            # Graficar las relaciones
            self.plot_emissions_vs_metrics(timestamp, final_rmse)
            
        except Exception as e:
            print(f"Error generating emission graphs: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_emissions_vs_metrics(self, timestamp, final_rmse=None):
        """Generate plots for emissions vs metrics"""
        
        try:
            if self.epoch_valid_rmse:
                # 1. Cumulative emissions vs RMSE
                plt.figure(figsize=(10, 6))
                plt.plot(self.cumulative_emissions, self.epoch_valid_rmse, 'b-', marker='o')
                
                # Add epoch labels
                for i, (emissions, rmse) in enumerate(zip(self.cumulative_emissions, self.epoch_valid_rmse)):
                    plt.annotate(f"{i}", (emissions, rmse), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9)
                    
                plt.xlabel('Cumulative CO2 Emissions (kg)')
                plt.ylabel('Validation RMSE')
                plt.title('Relationship between Cumulative Emissions and RMSE')
                plt.grid(True, alpha=0.3)
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_rmse_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path)
                plt.close()
                print(f"Graph saved to: {file_path}")
            
            # 2. Combined graph: Emissions per epoch and cumulative
            plt.figure(figsize=(12, 10))
            
            plt.subplot(2, 2, 1)
            plt.plot(range(len(self.epoch_emissions)), self.epoch_emissions, 'r-', marker='x')
            plt.title('Emissions per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('CO2 Emissions (kg)')
            
            plt.subplot(2, 2, 2)
            plt.plot(range(len(self.cumulative_emissions)), self.cumulative_emissions, 'r-', marker='o')
            plt.title('Cumulative Emissions per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('CO2 Emissions (kg)')
            
            plt.subplot(2, 2, 3)
            plt.plot(range(len(self.epoch_train_rmse)), self.epoch_train_rmse, 'g-', marker='o')
            plt.title('Train RMSE per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Train RMSE')
            
            if self.epoch_valid_rmse:
                plt.subplot(2, 2, 4)
                plt.plot(range(len(self.epoch_valid_rmse)), self.epoch_valid_rmse, 'b-', marker='o')
                plt.title('Validation RMSE per Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('Validation RMSE')
            
            plt.tight_layout()
            
            file_path = f'{self.result_path}/emissions_plots/metrics_by_epoch_{self.model_name}_{timestamp}.png'
            plt.savefig(file_path)
            plt.close()
            print(f"Graph saved to: {file_path}")
            
            if self.epoch_valid_rmse:
                # 3. Scatter plot of performance vs cumulative emissions
                plt.figure(figsize=(10, 6))
                
                # Adjust point sizes by epoch
                sizes = [(i+1)*20 for i in range(len(self.cumulative_emissions))]
                
                scatter = plt.scatter(self.epoch_valid_rmse, self.cumulative_emissions, 
                            color='blue', marker='o', s=sizes, alpha=0.7)
                
                # Add epoch labels
                for i, (rmse, em) in enumerate(zip(self.epoch_valid_rmse, self.cumulative_emissions)):
                    plt.annotate(f"{i}", (rmse, em), textcoords="offset points", 
                                xytext=(0,5), ha='center', fontsize=9)
                
                plt.ylabel('Cumulative CO2 Emissions (kg)')
                plt.xlabel('Validation RMSE')
                plt.title('Relationship between RMSE and Cumulative Emissions')
                plt.grid(True, alpha=0.3)
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_performance_scatter_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path)
                plt.close()
                print(f"Graph saved to: {file_path}")
        except Exception as e:
            print(f"Error generating plots: {e}")
            import traceback
            traceback.print_exc()


# Function to get positive items for each user
def get_user_positive_items(ratings_df):
    """Creates a dictionary with positive items for each user"""
    user_pos_items = {}
    for user in ratings_df['user_id'].unique():
        user_pos_items[user] = set(ratings_df[ratings_df['user_id'] == user]['item_id'].values)
    return user_pos_items


# Function to generate top-k recommendations
def generate_recommendations(model, user_ids, item_ids, k=10, batch_size=1024):
    """Generates top-k recommendations for a list of users"""
    recommendations = {}
    
    for user_id in user_ids:
        # Create user-item pairs for all items
        pairs = []
        for item_id in item_ids:
            pairs.append((user_id, item_id))
        
        # Process in batches to avoid memory issues
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            batch_users = [p[0] for p in batch_pairs]
            batch_items = [p[1] for p in batch_pairs]
            
            # Convert to TensorFlow tensors
            batch_users_tensor = tf.convert_to_tensor(batch_users, dtype=tf.int32)
            batch_items_tensor = tf.convert_to_tensor(batch_items, dtype=tf.int32)
            
            try:
                # Simple way: make ratings prediction with NNMF model
                # Create a fake ratings tensor to comply with the model's predict method
                fake_ratings = tf.ones_like(batch_users_tensor, dtype=tf.float32)
                
                # Make prediction using model's __call__ method directly
                preds = model(batch_users_tensor, batch_items_tensor)
                
                # If prediction is a tensor, convert to numpy array
                if isinstance(preds, tf.Tensor):
                    preds = preds.numpy()
                    
                # Extend scores with batch predictions
                scores.extend(preds)
            except Exception as e:
                print(f"Prediction failed, using random scores: {e}")
                # Fall back to random scores
                batch_scores = np.random.random(len(batch_pairs))
                scores.extend(batch_scores)
        
        # Convert scores to numpy array
        scores = np.array(scores)
        
        # Get top-k items (largest scores)
        if len(scores) > k:
            top_indices = np.argsort(scores)[-k:][::-1]  # Get indices of top k scores
            recommended_items = [item_ids[idx] for idx in top_indices]
        else:
            # If fewer items than k, return all sorted by score
            top_indices = np.argsort(scores)[::-1]  # Descending order
            recommended_items = [item_ids[idx] for idx in top_indices]
        
        recommendations[user_id] = recommended_items
    
    return recommendations


# Function to calculate top-k metrics
def calculate_topk_metrics(model, test_df, train_df, user_pos_items, all_item_ids, k_values=[5, 10, 20]):
    """Calculates top-k metrics for different k values"""
    # Use only a small subset of users for faster evaluation
    test_users = np.random.choice(test_df['user_id'].unique(), 
                                 size=min(100, len(test_df['user_id'].unique())),
                                 replace=False)
    
    print(f"Calculating top-k metrics for {len(test_users)} users...")
    
    metrics = {f'recall@{k}': [] for k in k_values}
    metrics.update({f'ndcg@{k}': [] for k in k_values})
    
    # Generate recommendations for test users
    recommendations = generate_recommendations(model, test_users, all_item_ids, k=max(k_values))
    
    for user_id in test_users:
        # Relevant items in test (rating >= 4)
        relevant_items = set(test_df[(test_df['user_id'] == user_id) & 
                                   (test_df['rating'] >= 4)]['item_id'].values)
        if not relevant_items:
            continue
            
        # Seen items (to exclude)
        seen_items = user_pos_items.get(user_id, set())
        
        # Get recommendations for this user and filter out seen items
        user_recs = recommendations.get(user_id, [])
        filtered_recs = [item for item in user_recs if item not in seen_items]
        
        for k in k_values:
            # Take first k items
            top_k = filtered_recs[:k]
            
            # Calculate Recall@K
            hits = len(set(top_k) & relevant_items)
            recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
            metrics[f'recall@{k}'].append(recall)
            
            # Calculate NDCG@K
            dcg = 0.0
            for i, item in enumerate(top_k):
                if item in relevant_items:
                    dcg += 1.0 / np.log2(i + 2)
            
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            metrics[f'ndcg@{k}'].append(ndcg)
    
    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items() if v}
    return avg_metrics


def load_data(train_filename, valid_filename, test_filename, delimiter='\t', col_names=['user_id', 'item_id', 'rating']):
    # Load data without modifying the IDs since they're already zero-based from your preprocessing
    train_data = pd.read_csv(train_filename, delimiter=delimiter, header=None, names=col_names)
    valid_data = pd.read_csv(valid_filename, delimiter=delimiter, header=None, names=col_names)
    test_data = pd.read_csv(test_filename, delimiter=delimiter, header=None, names=col_names)
    
    # Print min/max values to verify we have proper ranges
    print(f"Train data - user_id range: {train_data['user_id'].min()} to {train_data['user_id'].max()}")
    print(f"Train data - item_id range: {train_data['item_id'].min()} to {train_data['item_id'].max()}")
    
    return train_data, valid_data, test_data


def train(model, train_data, valid_data, test_data, batch_size, max_epochs, use_early_stop, early_stop_max_epoch, top_k_values=[5, 10, 20], result_path="results"):
    # Initialize trackers
    system_tracker = SystemMetricsTracker()
    emissions_tracker = EmissionsPerEpochTracker(result_path=result_path)
    
    # Lists to store metrics
    train_rmse_list = []
    valid_rmse_list = []
    
    # Top-k metrics
    top_k_metrics = {f'recall@{k}': [] for k in top_k_values}
    top_k_metrics.update({f'ndcg@{k}': [] for k in top_k_values})
    
    # Initial evaluation
    train_rmse = model.eval_rmse(train_data)
    valid_rmse = model.eval_rmse(valid_data)
    print(f"[start] Train RMSE: {train_rmse:.3f}; Valid RMSE: {valid_rmse:.3f}")

    # Get positive items for each user (for top-k evaluation)
    user_pos_items = get_user_positive_items(train_data)
    all_item_ids = sorted(train_data['item_id'].unique())
    
    '''
    # Calculate initial top-k metrics
    initial_metrics = calculate_topk_metrics(
        model, valid_data, train_data, user_pos_items, all_item_ids, top_k_values
    )
    
    # Print initial top-k metrics
    print("[start] Top-K metrics:")
    for k in top_k_values:
        top_k_metrics[f'recall@{k}'].append(initial_metrics.get(f'recall@{k}', 0))
        top_k_metrics[f'ndcg@{k}'].append(initial_metrics.get(f'ndcg@{k}', 0))
        print(f"  Recall@{k}: {initial_metrics.get(f'recall@{k}', 0):.4f}, NDCG@{k}: {initial_metrics.get(f'ndcg@{k}', 0):.4f}")
    '''
    
    prev_valid_rmse = float("inf")
    early_stop_epochs = 0
    best_model_path = None

    for epoch in range(max_epochs):
        # Start tracking for this epoch
        system_tracker.start_epoch(epoch)
        emissions_tracker.start_epoch(epoch)
        
        # Shuffle and batch training data
        shuffled_df = train_data.sample(frac=1)
        batches = chunk_df(shuffled_df, batch_size) if batch_size else [train_data]

        # Process each batch
        batch_rmses = []
        for batch in batches:
            user_ids = tf.convert_to_tensor(batch['user_id'], dtype=tf.int32)
            item_ids = tf.convert_to_tensor(batch['item_id'], dtype=tf.int32)
            ratings = tf.convert_to_tensor(batch['rating'], dtype=tf.float32)

            # Train on batch
            model.train_iteration(user_ids, item_ids, ratings)
            
            # Evaluate batch RMSE
            batch_rmse = model.eval_rmse(batch)
            batch_rmses.append(batch_rmse)
        
        # Calculate epoch metrics
        train_rmse = model.eval_rmse(train_data)
        valid_rmse = model.eval_rmse(valid_data)
        
        # Store metrics
        train_rmse_list.append(train_rmse)
        valid_rmse_list.append(valid_rmse)
        
        # Calculate top-k metrics
        epoch_metrics = calculate_topk_metrics(
            model, valid_data, train_data, user_pos_items, all_item_ids, top_k_values
        )
        
        # Store top-k metrics
        for k in top_k_values:
            top_k_metrics[f'recall@{k}'].append(epoch_metrics.get(f'recall@{k}', 0))
            top_k_metrics[f'ndcg@{k}'].append(epoch_metrics.get(f'ndcg@{k}', 0))
        
        # End tracking for this epoch
        system_tracker.end_epoch(epoch, train_rmse, valid_rmse)
        emissions_tracker.end_epoch(epoch, train_rmse, valid_rmse)
        
        # Print epoch results
        print(f"[{epoch}] Train RMSE: {train_rmse:.3f}; Valid RMSE: {valid_rmse:.3f}")
        print("  Top-K metrics:")
        for k in top_k_values:
            print(f"    Recall@{k}: {epoch_metrics.get(f'recall@{k}', 0):.4f}, NDCG@{k}: {epoch_metrics.get(f'ndcg@{k}', 0):.4f}")

        # Early stopping check
        if use_early_stop:
            early_stop_epochs += 1
            if valid_rmse < prev_valid_rmse:
                prev_valid_rmse = valid_rmse
                early_stop_epochs = 0
                
                # Save best model
                model_dir = os.path.join(result_path, 'models')
                os.makedirs(model_dir, exist_ok=True)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                best_model_path = os.path.join(model_dir, f"nnmf_best_model_{timestamp}.h5")
                model.save_weights(best_model_path)
                print(f"Model checkpoint saved to {best_model_path}")
                
            elif early_stop_epochs == early_stop_max_epoch:
                print("Early stopping...")
                break
    
    # Final evaluation
    system_tracker.start_epoch("test")
    test_rmse = model.eval_rmse(test_data)
    
    # Calculate final top-k metrics on test data
    final_metrics = calculate_topk_metrics(
        model, test_data, train_data, user_pos_items, all_item_ids, top_k_values
    )
    
    system_tracker.end_test(test_rmse)
    
    # Save training history
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Combine all metrics
    history_data = {
        'epoch': range(len(train_rmse_list)),
        'train_rmse': train_rmse_list,
        'valid_rmse': valid_rmse_list,
    }
    # Add top-k metrics
    history_data.update(top_k_metrics)
    
    history_df = pd.DataFrame(history_data)
    
    os.makedirs(result_path, exist_ok=True)
    history_file = os.path.join(result_path, f"training_history_{timestamp}.csv")
    history_df.to_csv(history_file, index=False)
    print(f"Training history saved to {history_file}")
    
    # Save final metrics
    final_results = {
        'test_rmse': [test_rmse],
        **{f'final_{k}': [final_metrics[k]] for k in final_metrics}
    }
    
    final_results_df = pd.DataFrame(final_results)
    final_results_file = os.path.join(result_path, f"final_results_{timestamp}.csv")
    final_results_df.to_csv(final_results_file, index=False)
    print(f"Final results saved to {final_results_file}")
    
    # End emissions tracking
    emissions_tracker.end_training(test_rmse)
    
    # Print final top-k metrics
    print("\n=== Final Top-K Metrics ===")
    for k in top_k_values:
        print(f"Recall@{k}: {final_metrics.get(f'recall@{k}', 0):.4f}, NDCG@{k}: {final_metrics.get(f'ndcg@{k}', 0):.4f}")
    
    return test_rmse, best_model_path, final_metrics


def test(model, test_data):
    test_rmse = model.eval_rmse(test_data)
    print(f"Final test RMSE: {test_rmse:.3f}")
    return test_rmse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains/evaluates NNMF models.')
    parser.add_argument('--model', type=str, choices=['NNMF', 'SVINNMF'], required=True)
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True)
    parser.add_argument('--train', type=str, default='C:/Users/xpati/Documents/TFG/ml-1m/split/u.data.train')
    parser.add_argument('--valid', type=str, default='C:/Users/xpati/Documents/TFG/ml-1m/split/u.data.valid')
    parser.add_argument('--test', type=str, default='C:/Users/xpati/Documents/TFG/ml-1m/split/u.data.test')
    parser.add_argument('--users', type=int, default=6040)
    parser.add_argument('--movies', type=int, default=3706)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--no-early', action='store_true')
    parser.add_argument('--early-stop-max-epoch', type=int, default=40)
    parser.add_argument('--model-params', type=str, default='{}')
    parser.add_argument('--result-path', type=str, default='results')
    parser.add_argument('--top-k', type=str, default='5,10,20', help='Comma-separated list of top-k values for evaluation')

    args = parser.parse_args()

    # Create result directory
    os.makedirs(args.result_path, exist_ok=True)

    model_params = json.loads(args.model_params)
    use_early_stop = not args.no_early
    
    # Parse top-k values
    top_k_values = [int(k) for k in args.top_k.split(',')]

    if args.model == 'NNMF':
        model = NNMF(args.users, args.movies, **model_params)
    else:
        raise NotImplementedError(f"Model '{args.model}' not implemented")

    train_data, valid_data, test_data = load_data(args.train, args.valid, args.test)

    if args.mode == 'train':
        # Train with metrics tracking
        test_rmse, best_model_path, final_metrics = train(  # Add the missing variable here
            model, 
            train_data, 
            valid_data, 
            test_data,
            args.batch, 
            args.max_epochs, 
            use_early_stop, 
            args.early_stop_max_epoch,
            top_k_values=top_k_values,
            result_path=args.result_path
        )
        
        print(f"\nTraining completed!")
        print(f"Test RMSE: {test_rmse:.4f}")
        if best_model_path:
            print(f"Best model saved to: {best_model_path}")
        
        # Print final top-k metrics summary
        print("\n=== Top-K Recommendation Metrics Summary ===")
        for k in top_k_values:
            print(f"Recall@{k}: {final_metrics.get(f'recall@{k}', 0):.4f}, NDCG@{k}: {final_metrics.get(f'ndcg@{k}', 0):.4f}")
        
    elif args.mode == 'test':
        test_rmse = test(model, test_data)
        print(f"Test complete with RMSE: {test_rmse:.4f}")
        
        # Also calculate top-k metrics for test mode
        user_pos_items = get_user_positive_items(train_data)
        all_item_ids = sorted(train_data['item_id'].unique())
        
        test_metrics = calculate_topk_metrics(
            model, test_data, train_data, user_pos_items, all_item_ids, top_k_values
        )
        
        print("\n=== Test Top-K Metrics ===")
        for k in top_k_values:
            print(f"Recall@{k}: {test_metrics.get(f'recall@{k}', 0):.4f}, NDCG@{k}: {test_metrics.get(f'ndcg@{k}', 0):.4f}")