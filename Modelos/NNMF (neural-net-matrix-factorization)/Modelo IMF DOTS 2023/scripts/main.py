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
        
    def end_epoch(self, epoch, train_rmse, train_mae, valid_rmse=None, valid_mae=None):
        epoch_time = time.time() - self.epoch_start_time
        self.current_epoch_metrics['epoch_time_sec'] = epoch_time
        self.current_epoch_metrics['train_rmse'] = train_rmse
        self.current_epoch_metrics['train_mae'] = train_mae
        if valid_rmse is not None:
            self.current_epoch_metrics['valid_rmse'] = valid_rmse
        if valid_mae is not None:
            self.current_epoch_metrics['valid_mae'] = valid_mae
        self.train_metrics.append(self.current_epoch_metrics)
        
        # Rastrear el mejor RMSE
        if valid_rmse is not None and valid_rmse < self.best_rmse:
            self.best_rmse = valid_rmse
            self.best_rmse_epoch = epoch
            self.best_rmse_metrics = self.current_epoch_metrics.copy()
        
        # Imprimir resumen de época
        print(f"\nEpoch {epoch} Metrics:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Memory: {self.current_epoch_metrics['memory_usage_mb']:.2f}MB")
        print(f"  CPU: {self.current_epoch_metrics['cpu_usage_percent']:.1f}%")
        print(f"  Train RMSE: {train_rmse:.4f}")
        print(f"  Train MAE: {train_mae:.4f}")
        if valid_rmse is not None:
            print(f"  Valid RMSE: {valid_rmse:.4f}")
        if valid_mae is not None:
            print(f"  Valid MAE: {valid_mae:.4f}")
        
    def end_test(self, test_rmse, test_mae):
        self.test_metrics = {
            'test_time_sec': time.time() - self.epoch_start_time,
            'total_time_sec': time.time() - self.start_time,
            'final_memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'final_cpu_usage_percent': psutil.cpu_percent(),
            'test_rmse': test_rmse,
            'test_mae': test_mae,
        }
        
        # Imprimir métricas finales
        print("\n=== Final Training Metrics ===")
        for m in self.train_metrics:
            metrics_str = f"Epoch {m['epoch']}: Time={m['epoch_time_sec']:.2f}s, Memory={m['memory_usage_mb']:.2f}MB, CPU={m['cpu_usage_percent']:.1f}%, RMSE={m['train_rmse']:.4f}, MAE={m['train_mae']:.4f}"
            if 'valid_rmse' in m and 'valid_mae' in m:
                metrics_str += f" (Val RMSE={m['valid_rmse']:.4f}, Val MAE={m['valid_mae']:.4f})"
            print(metrics_str)
        
        print("\n=== Final Test Metrics ===")
        print(f"Total Time: {self.test_metrics['total_time_sec']:.2f}s (Test: {self.test_metrics['test_time_sec']:.2f}s)")
        print(f"Final Memory: {self.test_metrics['final_memory_usage_mb']:.2f}MB")
        print(f"Final CPU: {self.test_metrics['final_cpu_usage_percent']:.1f}%")
        print(f"RMSE: {test_rmse:.4f}")
        print(f"MAE: {test_mae:.4f}")
        
        # Mostrar información del mejor RMSE durante el entrenamiento
        if self.best_rmse_epoch is not None:
            print(f"\n=== Best Training RMSE ===")
            print(f"Best RMSE: {self.best_rmse:.4f} (Epoch {self.best_rmse_epoch})")
            if self.best_rmse_metrics:
                print(f"Time: {self.best_rmse_metrics['epoch_time_sec']:.2f}s")
                print(f"Memory: {self.best_rmse_metrics['memory_usage_mb']:.2f}MB")
                print(f"CPU: {self.best_rmse_metrics['cpu_usage_percent']:.1f}%")
                if 'train_mae' in self.best_rmse_metrics and self.best_rmse_metrics['train_mae'] is not None:
                    print(f"MAE: {self.best_rmse_metrics['valid_mae']:.4f}")
        
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
        self.epoch_train_mae = []
        self.epoch_valid_rmse = []
        self.epoch_valid_mae = []
        self.total_emissions = 0.0
        self.trackers = {}
        self.best_rmse = float('inf')
        self.best_rmse_epoch = None
        self.best_rmse_emissions = None
        self.best_rmse_cumulative_emissions = None
        
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
    
    def end_epoch(self, epoch, train_rmse, train_mae, valid_rmse=None, valid_mae=None):
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
            self.epoch_train_mae.append(train_mae)
            if valid_rmse is not None:
                self.epoch_valid_rmse.append(valid_rmse)
                # Rastrear el mejor RMSE y sus emisiones
                if valid_rmse < self.best_rmse:
                    self.best_rmse = valid_rmse
                    self.best_rmse_epoch = epoch
                    self.best_rmse_emissions = epoch_co2
                    self.best_rmse_cumulative_emissions = self.total_emissions
            if valid_mae is not None:
                self.epoch_valid_mae.append(valid_mae)
            
            print(f"Epoch {epoch} - Emissions: {epoch_co2:.8f} kg, Cumulative: {self.total_emissions:.8f} kg")
            print(f"Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}")
            if valid_rmse is not None and valid_mae is not None:
                print(f"Valid RMSE: {valid_rmse:.4f}, Valid MAE: {valid_mae:.4f}")
        except Exception as e:
            print(f"Error measuring emissions in epoch {epoch}: {e}")
    
    def end_training(self, final_rmse, final_mae):
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
            
            # Información del mejor modelo
            if hasattr(self, 'best_rmse') and self.best_rmse < float('inf'):
                print(f"\nBest model at epoch {self.best_rmse_epoch}:")
                print(f"  RMSE: {self.best_rmse:.4f}")
                print(f"  Epoch emissions: {self.best_rmse_emissions:.8f} kg CO2")
                print(f"  Cumulative emissions: {self.best_rmse_cumulative_emissions:.8f} kg CO2")
            
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
                    self.epoch_train_rmse = [final_rmse]
                if final_mae is not None:
                    self.epoch_valid_mae = [final_mae]
                    self.epoch_train_mae = [final_mae]
            
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
                'train_rmse': self.epoch_train_rmse,
                'train_mae': self.epoch_train_mae
            }
            
            if self.epoch_valid_rmse:
                data['valid_rmse'] = self.epoch_valid_rmse
            if self.epoch_valid_mae:
                data['valid_mae'] = self.epoch_valid_mae
                
            df = pd.DataFrame(data)
            
            emissions_file = f'{self.result_path}/emissions_reports/emissions_metrics_{self.model_name}_{timestamp}.csv'
            df.to_csv(emissions_file, index=False)
            print(f"Emission metrics saved to: {emissions_file}")
            
            # Graficar las relaciones
            self.plot_emissions_vs_metrics(timestamp, final_rmse, final_mae)
            
        except Exception as e:
            print(f"Error generating emission graphs: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_emissions_vs_metrics(self, timestamp, final_rmse=None, final_mae=None):
        """Generate comprehensive plots for emissions vs metrics with MAE support"""
        
        try:
            # Create a comprehensive figure with 2x3 subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Plot 1: Cumulative Emissions vs Validation RMSE
            if self.epoch_valid_rmse:
                axes[0, 0].plot(self.cumulative_emissions, self.epoch_valid_rmse, 'b-', marker='o', linewidth=2, markersize=6)
                for i, (emissions, rmse) in enumerate(zip(self.cumulative_emissions, self.epoch_valid_rmse)):
                    axes[0, 0].annotate(f"{i}", (emissions, rmse), textcoords="offset points", 
                                       xytext=(0,10), ha='center', fontsize=9)
                axes[0, 0].set_xlabel('Cumulative CO2 Emissions (kg)')
                axes[0, 0].set_ylabel('Validation RMSE')
                axes[0, 0].set_title('Cumulative Emissions vs Validation RMSE')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Highlight best RMSE point if available
                if hasattr(self, 'best_rmse_epoch') and self.best_rmse_epoch < len(self.cumulative_emissions):
                    best_emissions = self.cumulative_emissions[self.best_rmse_epoch]
                    axes[0, 0].scatter([best_emissions], [self.best_rmse], color='red', s=100, 
                                      marker='*', label=f'Best RMSE (Epoch {self.best_rmse_epoch})', zorder=5)
                    axes[0, 0].legend()
            
            # Plot 2: Cumulative Emissions vs Validation MAE
            if self.epoch_valid_mae:
                axes[0, 1].plot(self.cumulative_emissions, self.epoch_valid_mae, 'g-', marker='s', linewidth=2, markersize=6)
                for i, (emissions, mae) in enumerate(zip(self.cumulative_emissions, self.epoch_valid_mae)):
                    axes[0, 1].annotate(f"{i}", (emissions, mae), textcoords="offset points", 
                                       xytext=(0,10), ha='center', fontsize=9)
                axes[0, 1].set_xlabel('Cumulative CO2 Emissions (kg)')
                axes[0, 1].set_ylabel('Validation MAE')
                axes[0, 1].set_title('Cumulative Emissions vs Validation MAE')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Epoch Emissions vs Performance (RMSE and MAE)
            ax3 = axes[0, 2]
            if self.epoch_valid_rmse:
                line1 = ax3.plot(range(len(self.epoch_emissions)), self.epoch_valid_rmse, 'b-', marker='o', 
                                linewidth=2, markersize=4, label='Valid RMSE')
                ax3.set_ylabel('Validation RMSE', color='b')
                ax3.tick_params(axis='y', labelcolor='b')
            
            if self.epoch_valid_mae:
                ax3_twin = ax3.twinx()
                line2 = ax3_twin.plot(range(len(self.epoch_emissions)), self.epoch_valid_mae, 'g-', marker='s', 
                                     linewidth=2, markersize=4, label='Valid MAE')
                ax3_twin.set_ylabel('Validation MAE', color='g')
                ax3_twin.tick_params(axis='y', labelcolor='g')
            
            ax3.set_xlabel('Epoch')
            ax3.set_title('Validation Metrics per Epoch')
            ax3.grid(True, alpha=0.3)
            
            # Combine legends
            lines1 = ax3.get_lines() if self.epoch_valid_rmse else []
            lines2 = ax3_twin.get_lines() if self.epoch_valid_mae and 'ax3_twin' in locals() else []
            if lines1 or lines2:
                all_lines = lines1 + lines2
                labels = [l.get_label() for l in all_lines]
                ax3.legend(all_lines, labels, loc='upper right')
            
            # Plot 4: Emissions per Epoch
            axes[1, 0].bar(range(len(self.epoch_emissions)), self.epoch_emissions, alpha=0.7, color='red')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('CO2 Emissions per Epoch (kg)')
            axes[1, 0].set_title('CO2 Emissions per Epoch')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 5: Cumulative Emissions
            axes[1, 1].plot(range(len(self.cumulative_emissions)), self.cumulative_emissions, 'r-', 
                           marker='o', linewidth=2, markersize=6)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Cumulative CO2 Emissions (kg)')
            axes[1, 1].set_title('Cumulative CO2 Emissions')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Plot 6: Training Metrics (RMSE and MAE)
            ax6 = axes[1, 2]
            line1 = ax6.plot(range(len(self.epoch_train_rmse)), self.epoch_train_rmse, 'b-', marker='o', 
                            linewidth=2, markersize=4, label='Train RMSE')
            ax6.set_ylabel('Training RMSE', color='b')
            ax6.tick_params(axis='y', labelcolor='b')
            
            if self.epoch_train_mae:
                ax6_twin = ax6.twinx()
                line2 = ax6_twin.plot(range(len(self.epoch_train_mae)), self.epoch_train_mae, 'g-', marker='s', 
                                     linewidth=2, markersize=4, label='Train MAE')
                ax6_twin.set_ylabel('Training MAE', color='g')
                ax6_twin.tick_params(axis='y', labelcolor='g')
            
            ax6.set_xlabel('Epoch')
            ax6.set_title('Training Metrics per Epoch')
            ax6.grid(True, alpha=0.3)
            
            # Combine legends for plot 6
            lines1 = ax6.get_lines()
            lines2 = ax6_twin.get_lines() if self.epoch_train_mae and 'ax6_twin' in locals() else []
            all_lines = lines1 + lines2
            labels = [l.get_label() for l in all_lines]
            ax6.legend(all_lines, labels, loc='upper right')
            
            plt.tight_layout()
            
            # Save the comprehensive plot
            file_path = f'{self.result_path}/emissions_plots/comprehensive_emissions_metrics_{self.model_name}_{timestamp}.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Comprehensive emissions vs metrics plot saved to: {file_path}")
            
            # Additional scatter plot: RMSE vs MAE with emissions as color scale
            if self.epoch_valid_rmse and self.epoch_valid_mae:
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(self.epoch_valid_rmse, self.epoch_valid_mae, 
                                    c=self.cumulative_emissions, cmap='viridis', s=100, alpha=0.7)
                
                # Add epoch labels
                for i, (rmse, mae, em) in enumerate(zip(self.epoch_valid_rmse, self.epoch_valid_mae, self.cumulative_emissions)):
                    plt.annotate(f"{i}", (rmse, mae), textcoords="offset points", 
                                xytext=(5,5), ha='left', fontsize=9)
                
                plt.colorbar(scatter, label='Cumulative CO2 Emissions (kg)')
                plt.xlabel('Validation RMSE')
                plt.ylabel('Validation MAE')
                plt.title('RMSE vs MAE with Cumulative Emissions')
                plt.grid(True, alpha=0.3)
                
                file_path = f'{self.result_path}/emissions_plots/rmse_vs_mae_emissions_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"RMSE vs MAE scatter plot saved to: {file_path}")
                
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


def train(model, train_data, valid_data, batch_size, max_epochs, use_early_stop, early_stop_max_epoch, result_path="results"):
    # Initialize trackers
    system_tracker = SystemMetricsTracker()
    emissions_tracker = EmissionsPerEpochTracker(result_path=result_path)
    
    # Lists to store metrics
    train_rmse_list = []
    valid_rmse_list = []
    train_mae_list = []
    valid_mae_list = []
    
    # Initial evaluation
    train_rmse = model.eval_rmse(train_data)
    valid_rmse = model.eval_rmse(valid_data)
    train_mae = model.eval_mae(train_data)
    valid_mae = model.eval_mae(valid_data)
    print(f"[start] Train RMSE: {train_rmse:.3f}, Train MAE: {train_mae:.3f}; Valid RMSE: {valid_rmse:.3f}, Valid MAE: {valid_mae:.3f}")

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
        train_mae = model.eval_mae(train_data)
        valid_mae = model.eval_mae(valid_data)
        
        # Store metrics
        train_rmse_list.append(train_rmse)
        valid_rmse_list.append(valid_rmse)
        train_mae_list.append(train_mae)
        valid_mae_list.append(valid_mae)
        
        # End tracking for this epoch
        system_tracker.end_epoch(epoch, train_rmse, train_mae, valid_rmse, valid_mae)
        emissions_tracker.end_epoch(epoch, train_rmse, train_mae, valid_rmse, valid_mae)
        
        # Print epoch results
        print(f"[{epoch}] Train RMSE: {train_rmse:.3f}, Train MAE: {train_mae:.3f}; Valid RMSE: {valid_rmse:.3f}, Valid MAE: {valid_mae:.3f}")

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
    test_mae = model.eval_mae(test_data)
    system_tracker.end_test(test_rmse, test_mae)
    
    # Save training history
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    history_df = pd.DataFrame({
        'epoch': range(len(train_rmse_list)),
        'train_rmse': train_rmse_list,
        'valid_rmse': valid_rmse_list,
        'train_mae': train_mae_list,
        'valid_mae': valid_mae_list
    })
    
    os.makedirs(result_path, exist_ok=True)
    history_file = os.path.join(result_path, f"training_history_{timestamp}.csv")
    history_df.to_csv(history_file, index=False)
    print(f"Training history saved to {history_file}")
    
    # End emissions tracking
    emissions_tracker.end_training(test_rmse, test_mae)
    
    return test_rmse, test_mae, best_model_path


def test(model, test_data):
    test_rmse = model.eval_rmse(test_data)
    test_mae = model.eval_mae(test_data)
    print(f"Final test RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}")
    return test_rmse, test_mae


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains/evaluates NNMF models.')
    parser.add_argument('--model', type=str, choices=['NNMF', 'SVINNMF'], required=True)
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True)
    parser.add_argument('--train', type=str, default='C:/Users/xpati/Documents/TFG/IMF DOTS 2023/split/u.data.train')
    parser.add_argument('--valid', type=str, default='C:/Users/xpati/Documents/TFG/IMF DOTS 2023/split/u.data.valid')
    parser.add_argument('--test', type=str, default='C:/Users/xpati/Documents/TFG/IMF DOTS 2023/split/u.data.test')
    parser.add_argument('--users', type=int, default=1000)  # Update this based on IMF DOTS dataset
    parser.add_argument('--movies', type=int, default=500)  # Update this based on IMF DOTS dataset
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--no-early', action='store_true')
    parser.add_argument('--early-stop-max-epoch', type=int, default=40)
    parser.add_argument('--model-params', type=str, default='{}')
    parser.add_argument('--result-path', type=str, default='results')

    args = parser.parse_args()

    # Create result directory
    os.makedirs(args.result_path, exist_ok=True)

    model_params = json.loads(args.model_params)
    use_early_stop = not args.no_early

    if args.model == 'NNMF':
        model = NNMF(args.users, args.movies, **model_params)
    else:
        raise NotImplementedError(f"Model '{args.model}' not implemented")

    train_data, valid_data, test_data = load_data(args.train, args.valid, args.test)

    if args.mode == 'train':
        # Train with metrics tracking
        test_rmse, test_mae, best_model_path = train(
            model, 
            train_data, 
            valid_data, 
            args.batch, 
            args.max_epochs, 
            use_early_stop, 
            args.early_stop_max_epoch,
            args.result_path
        )
        
        print(f"\nTraining completed!")
        print(f"Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}")
        if best_model_path:
            print(f"Best model saved to: {best_model_path}")
        
    elif args.mode == 'test':
        test_rmse, test_mae = test(model, test_data)
        print(f"Test complete with RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")