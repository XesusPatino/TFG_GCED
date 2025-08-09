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

# Import our history-based model
from nnmf.models_with_history import NNMFWithHistory

# Importar CodeCarbon para medir emisiones de CO₂
from codecarbon import EmissionsTracker

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
            print(f"Train RMSE: {train_rmse:.4f}" + (f", Valid RMSE: {valid_rmse:.4f}" if valid_rmse is not None else ""))
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


def load_data(train_filename, valid_filename, test_filename, delimiter='\t', col_names=['user_id', 'item_id', 'rating']):
    # Load data without modifying the IDs since they're already zero-based from your preprocessing
    train_data = pd.read_csv(train_filename, delimiter=delimiter, header=None, names=col_names)
    valid_data = pd.read_csv(valid_filename, delimiter=delimiter, header=None, names=col_names)
    test_data = pd.read_csv(test_filename, delimiter=delimiter, header=None, names=col_names)
    
    # Print min/max values to verify we have proper ranges
    print(f"Train data - user_id range: {train_data['user_id'].min()} to {train_data['user_id'].max()}")
    print(f"Train data - item_id range: {train_data['item_id'].min()} to {train_data['item_id'].max()}")
    
    return train_data, valid_data, test_data


def train(model, train_data, valid_data, test_data, batch_size, max_epochs, use_early_stop, early_stop_max_epoch, 
          result_path="results", use_history=False, history_weight=0.3):
    # Initialize trackers
    system_tracker = SystemMetricsTracker()
    emissions_tracker = EmissionsPerEpochTracker(result_path=result_path)
    
    # Lists to store metrics
    train_rmse_list = []
    valid_rmse_list = []
    
    # Prepare data for history-based adjustments if needed
    if use_history and isinstance(model, NNMFWithHistory):
        print("Creating ratings matrix for history-based adjustments...")
        # Create ratings matrices from DataFrames
        max_user_id = int(max(train_data['user_id'].max(), valid_data['user_id'].max(), test_data['user_id'].max()))
        max_item_id = int(max(train_data['item_id'].max(), valid_data['item_id'].max(), test_data['item_id'].max()))
        
        # Create sparse matrices and convert to dense
        train_ratings = np.zeros((max_user_id + 1, max_item_id + 1))
        train_mask = np.zeros((max_user_id + 1, max_item_id + 1))
        
        # Fill in train ratings
        for _, row in train_data.iterrows():
            train_ratings[int(row['user_id']), int(row['item_id'])] = float(row['rating'])
            train_mask[int(row['user_id']), int(row['item_id'])] = 1.0
    
    # Initial evaluation
    train_rmse = model.eval_rmse(train_data)
    valid_rmse = model.eval_rmse(valid_data)
    print(f"[start] Train RMSE: {train_rmse:.3f}; Valid RMSE: {valid_rmse:.3f}")

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
        
        # End tracking for this epoch
        system_tracker.end_epoch(epoch, train_rmse, valid_rmse)
        emissions_tracker.end_epoch(epoch, train_rmse, valid_rmse)
        
        # Print epoch results
        print(f"[{epoch}] Train RMSE: {train_rmse:.3f}; Valid RMSE: {valid_rmse:.3f}")

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
                try:
                    model.save_weights(best_model_path)
                    print(f"Model checkpoint saved to {best_model_path}")
                except Exception as e:
                    print(f"Error saving model: {e}")
                
            elif early_stop_epochs == early_stop_max_epoch:
                print("Early stopping...")
                break
    
    # Set up history adjustments if needed
    if use_history and isinstance(model, NNMFWithHistory):
        print("Setting up user histories and computing item similarities...")
        model.set_user_histories(train_ratings, train_mask)
        model.compute_item_similarities(train_ratings, train_mask)

    # Final evaluation
    system_tracker.start_epoch("test")
    
    # If using history, calculate RMSE with history-based adjustments
    if use_history and isinstance(model, NNMFWithHistory):
        print("\nCalculating RMSE with history-based adjustments...")
        
        # Calculate RMSE with history-based predictions
        squared_error_sum = 0
        test_count = 0
        
        # Group test data by user for more efficient processing
        user_groups = test_data.groupby('user_id')
        total_users = len(user_groups)
        
        # Show progress
        progress_step = max(1, total_users // 10)
        user_idx = 0
        
        for user_id, user_df in user_groups:
            user_idx += 1
            if user_idx % progress_step == 0:
                print(f"  Processing user {user_idx}/{total_users} ({user_idx/total_users*100:.1f}%)")
            
            # Get predictions for this user with history adjustment
            user_predictions = model.predict_for_user(user_id, train_ratings)
            
            # Calculate error for each rating
            for _, row in user_df.iterrows():
                item_id = int(row['item_id'])
                actual_rating = float(row['rating'])
                predicted_rating = user_predictions[item_id]
                
                squared_error = (actual_rating - predicted_rating) ** 2
                squared_error_sum += squared_error
                test_count += 1
        
        test_rmse = np.sqrt(squared_error_sum / max(1, test_count))
        print(f"\nTest RMSE with history adjustment: {test_rmse:.4f}")
        
        # Also calculate standard RMSE for comparison
        standard_rmse = model.eval_rmse(test_data)
        print(f"Standard Test RMSE (without history): {standard_rmse:.4f}")
        print(f"Improvement from history adjustment: {standard_rmse - test_rmse:.4f}")
    else:
        # Standard evaluation
        test_rmse = model.eval_rmse(test_data)
    
    system_tracker.end_test(test_rmse)
    
    # Save training history
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    history_df = pd.DataFrame({
        'epoch': range(len(train_rmse_list)),
        'train_rmse': train_rmse_list,
        'valid_rmse': valid_rmse_list
    })
    
    os.makedirs(result_path, exist_ok=True)
    history_file = os.path.join(result_path, f"training_history_{timestamp}.csv")
    history_df.to_csv(history_file, index=False)
    print(f"Training history saved to {history_file}")
    
    ''''
    # Demonstrate prediction examples
    if use_history and isinstance(model, NNMFWithHistory):
        # Show some examples of history-based predictions
        print("\nExample predictions with history adjustment:")
        sample_size = min(5, len(test_data))
        sample_test = test_data.sample(sample_size)
        
        for _, row in sample_test.iterrows():
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            actual = float(row['rating'])
            
            # Get base prediction
            base_pred = float(model([tf.constant([user_id], dtype=tf.int32), 
                                     tf.constant([item_id], dtype=tf.int32)], 
                                    training=False).numpy()[0])
            
            # Get history-adjusted prediction
            adjusted_pred = model.predict_with_history(user_id, item_id, base_pred)
            
            print(f"\nUser {user_id}, Item {item_id}:")
            print(f"  Actual rating: {actual:.2f}")
            print(f"  Base prediction: {base_pred:.2f}")
            print(f"  History-adjusted prediction: {adjusted_pred:.2f}")
            
            # Show detailed calculation for first example
            if _ == sample_test.index[0]:
                model.demonstrate_prediction(user_id, item_id, train_ratings)
    '''
    
    # End emissions tracking
    emissions_tracker.end_training(test_rmse)
    
    return test_rmse, best_model_path


def test(model, test_data):
    test_rmse = model.eval_rmse(test_data)
    print(f"Final test RMSE: {test_rmse:.3f}")
    return test_rmse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains/evaluates NNMF models.')
    parser.add_argument('--model', type=str, choices=['NNMF', 'NNMFWithHistory'], required=True)
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
    parser.add_argument('--use-history', action='store_true', help='Use history-based prediction adjustments')
    parser.add_argument('--history-weight', type=float, default=0.3, help='Weight for history-based predictions')
    parser.add_argument('--num-similar-items', type=int, default=10, help='Number of similar items for history-based predictions')
    parser.add_argument('--similarity-threshold', type=float, default=0.2, help='Minimum similarity threshold for history comparison')

    args = parser.parse_args()

    # Create result directory
    os.makedirs(args.result_path, exist_ok=True)

    model_params = json.loads(args.model_params)
    use_early_stop = not args.no_early

    # Create appropriate model based on argument
    if args.model == 'NNMF':
        model = NNMF(args.users, args.movies, **model_params)
        use_history = False
    elif args.model == 'NNMFWithHistory':
        # Add history parameters to model parameters
        history_params = {
            'history_weight': args.history_weight,
            'num_similar_items': args.num_similar_items,
            'similarity_threshold': args.similarity_threshold
        }
        model_params.update(history_params)
        
        model = NNMFWithHistory(args.users, args.movies, **model_params)
        use_history = True
    else:
        raise NotImplementedError(f"Model '{args.model}' not implemented")

    train_data, valid_data, test_data = load_data(args.train, args.valid, args.test)

    if args.mode == 'train':
        # Train with metrics tracking
        test_rmse, best_model_path = train(
            model, 
            train_data, 
            valid_data,
            test_data,
            args.batch, 
            args.max_epochs, 
            use_early_stop, 
            args.early_stop_max_epoch,
            args.result_path,
            use_history,
            args.history_weight
        )
        
        print(f"\nTraining completed!")
        print(f"Test RMSE: {test_rmse:.4f}")
        if best_model_path:
            print(f"Best model saved to: {best_model_path}")
        
    elif args.mode == 'test':
        test_rmse = test(model, test_data)
        print(f"Test complete with RMSE: {test_rmse:.4f}")