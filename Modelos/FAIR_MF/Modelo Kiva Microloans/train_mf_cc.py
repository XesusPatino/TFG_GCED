import os
import time
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import psutil
from codecarbon import EmissionsTracker
import pandas as pd
from os import path
import mf
import mf_with_fair_pretraining as mf_fair
from data import datamodule
from argparse import ArgumentParser
import fairness_metrics as fairness
import matplotlib.pyplot as plt
import logging

logging.getLogger("pytorch_lightning.utilities.distributed").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.gpu").setLevel(logging.WARNING)

DATA_DIR = "C:/Users/xpati/Documents/TFG"

# Create command line arguments
args = ArgumentParser()
args.add_argument("--d", type=int, default=128)
args.add_argument("--dataset", type=str)
args.add_argument("--split", type=int, default=0)  # Hacer split opcional con valor predeterminado
args.add_argument("--use_all_data", action="store_true", help="Usar todos los datos en lugar de splits")
args = args.parse_args()

EMBEDDING_DIM = args.d
DATASET = args.dataset
SPLIT = args.split
USE_ALL_DATA = args.use_all_data

class SystemMetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.train_metrics = []
        self.test_metrics = []
        self.current_epoch_metrics = {}
        self.best_rmse = float('inf')
        self.best_rmse_epoch = None
        self.best_rmse_metrics = None
        
    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.train_start_time = time.time()
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        self.current_epoch_metrics = {
            'epoch': trainer.current_epoch,
            'memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'cpu_usage_percent': psutil.cpu_percent(),
        }
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.current_epoch_metrics['gpu_usage_mb'] = 0
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Simplificado para no requerir dataloader_idx"""
        if torch.cuda.is_available():
            self.current_epoch_metrics['gpu_usage_mb'] = max(
                self.current_epoch_metrics.get('gpu_usage_mb', 0),
                torch.cuda.max_memory_allocated() / 1024 ** 2
            )
    
    def on_train_epoch_end(self, trainer, pl_module):
        self.current_epoch_metrics['epoch_time_sec'] = time.time() - self.epoch_start_time
        
        # Obtener RMSE y MAE de las métricas del trainer (si están disponibles)
        current_rmse = trainer.callback_metrics.get("val_rmse", trainer.callback_metrics.get("train_rmse", torch.tensor(0.0))).item()
        current_mae = trainer.callback_metrics.get("val_mae", trainer.callback_metrics.get("train_mae", torch.tensor(0.0))).item()
        self.current_epoch_metrics['rmse'] = current_rmse
        self.current_epoch_metrics['mae'] = current_mae
        
        # Rastrear el mejor RMSE
        if current_rmse < self.best_rmse:
            self.best_rmse = current_rmse
            self.best_rmse_epoch = trainer.current_epoch
            self.best_rmse_metrics = self.current_epoch_metrics.copy()
        
        self.train_metrics.append(self.current_epoch_metrics)
        
        # Log metrics
        metrics_to_log = {
            'time/epoch_time_sec': self.current_epoch_metrics['epoch_time_sec'],
            'memory/memory_usage_mb': self.current_epoch_metrics['memory_usage_mb'],
            'cpu/cpu_usage_percent': self.current_epoch_metrics['cpu_usage_percent'],
        }
        
        if 'gpu_usage_mb' in self.current_epoch_metrics:
            metrics_to_log['gpu/gpu_usage_mb'] = self.current_epoch_metrics['gpu_usage_mb']
        
        pl_module.log_dict(metrics_to_log)
        
        # Print epoch summary
        print(f"\nEpoch {self.current_epoch_metrics['epoch']} Metrics:")
        print(f"  Time: {self.current_epoch_metrics['epoch_time_sec']:.2f}s")
        print(f"  Memory: {self.current_epoch_metrics['memory_usage_mb']:.2f}MB")
        print(f"  CPU: {self.current_epoch_metrics['cpu_usage_percent']:.1f}%")
        if 'gpu_usage_mb' in self.current_epoch_metrics:
            print(f"  GPU: {self.current_epoch_metrics['gpu_usage_mb']:.2f}MB")
        print(f"  RMSE: {current_rmse:.4f}")
        print(f"  MAE: {current_mae:.4f}")
    
    def on_test_start(self, trainer, pl_module):
        self.test_start_time = time.time()
        self.all_preds = []
        self.all_targets = []
        self.all_indexes = []
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Mantener dataloader_idx como opcional para compatibilidad"""
        user_ids, item_ids, ratings, _ = batch
        preds = pl_module(user_ids, item_ids)
        
        self.all_preds.append(preds)
        self.all_targets.append(ratings)
        self.all_indexes.append(user_ids)
    
    def on_test_end(self, trainer, pl_module):
        test_time = time.time() - self.test_start_time
        total_time = time.time() - self.train_start_time
        
        # Calculate recommendation metrics
        preds = torch.cat(self.all_preds)
        targets = torch.cat(self.all_targets)
        indexes = torch.cat(self.all_indexes)
        
        # Calculate RMSE and MAE
        rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()
        mae = torch.mean(torch.abs(preds - targets)).item()
        
        # Final system metrics
        final_metrics = {
            'time/total_time_sec': total_time,
            'time/test_time_sec': test_time,
            'memory/final_memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'cpu/final_cpu_usage_percent': psutil.cpu_percent(),
            'metrics/rmse': rmse,
            'metrics/mae': mae,
        }
        
        if torch.cuda.is_available():
            final_metrics['gpu/final_gpu_usage_mb'] = torch.cuda.max_memory_allocated() / 1024 ** 2
        
        # Log metrics directly to the logger instead of using pl_module.log_dict()
        if trainer.logger:
            for metric_name, metric_value in final_metrics.items():
                trainer.logger.experiment.add_scalar(metric_name, metric_value)
        
        self.test_metrics.append(final_metrics)
        
        # Print final summary
        print("\n=== Final Training Metrics ===")
        for m in self.train_metrics:
            print(f"Epoch {m['epoch']}: Time={m['epoch_time_sec']:.2f}s, "
                  f"Memory={m['memory_usage_mb']:.2f}MB, CPU={m['cpu_usage_percent']:.1f}%, "
                  f"RMSE={m.get('rmse', 0.0):.4f}, MAE={m.get('mae', 0.0):.4f}", 
                  end='')
            if 'gpu_usage_mb' in m:
                print(f", GPU={m['gpu_usage_mb']:.2f}MB")
            else:
                print()
        
        print("\n=== Final Test Metrics ===")
        print(f"Total Time: {total_time:.2f}s (Test: {test_time:.2f}s)")
        print(f"Final Memory: {final_metrics['memory/final_memory_usage_mb']:.2f}MB")
        print(f"Final CPU: {final_metrics['cpu/final_cpu_usage_percent']:.1f}%")
        if 'gpu/final_gpu_usage_mb' in final_metrics:
            print(f"Final GPU: {final_metrics['gpu/final_gpu_usage_mb']:.2f}MB")
        print(f"RMSE: {rmse:.4f}")
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



class EmissionsPerEpochCallback(Callback):
    def __init__(self):
        super().__init__()
        self.emission_rmse_pairs = []
        self.trackers = {}
        self.epoch_emissions = []      # Emisiones por época
        self.cumulative_emissions = [] # Emisiones acumulativas
        self.epoch_rmse = []           # RMSE por época
        self.epoch_mae = []            # MAE por época
        self.total_emissions = 0.0     # Contador de emisiones totales
        self.best_rmse = float('inf')
        self.best_rmse_epoch = None
        self.best_rmse_emissions = None
        self.best_rmse_cumulative_emissions = None
        
    def on_train_start(self, trainer, pl_module):
        # Crear directorio para emisiones si no existe
        os.makedirs('emissions_reports', exist_ok=True)
        os.makedirs('emissions_plots', exist_ok=True)
        
        # Inicializar tracker general
        self.main_tracker = EmissionsTracker(
            project_name=f"MF_{DATASET}_{'full' if USE_ALL_DATA else f'split{SPLIT}'}_total",
            output_dir="emissions_reports",
            save_to_file=True,
            log_level="error",
            allow_multiple_runs=True
        )
        try:
            self.main_tracker.start()
        except Exception as e:
            print(f"Advertencia: No se pudo iniciar el tracker principal: {e}")
            self.main_tracker = None
        
    def on_train_epoch_start(self, trainer, pl_module):
        # Usar un nombre único para cada tracker y permitir múltiples ejecuciones
        epoch = trainer.current_epoch
        self.trackers[epoch] = EmissionsTracker(
            project_name=f"MF_{DATASET}_{'full' if USE_ALL_DATA else f'split{SPLIT}'}_epoch{epoch}",
            output_dir="emissions_reports",
            save_to_file=True,
            log_level="error",
            allow_multiple_runs=True
        )
        try:
            self.trackers[epoch].start()
        except Exception as e:
            print(f"Advertencia: No se pudo iniciar el tracker para la época {epoch}: {e}")
            self.trackers[epoch] = None
        
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        
        try:
            # Obtener emisiones de esta época
            epoch_co2 = 0.0
            if epoch in self.trackers and self.trackers[epoch]:
                try:
                    epoch_co2 = self.trackers[epoch].stop() or 0.0
                except Exception as e:
                    print(f"Advertencia: Error al detener el tracker para la época {epoch}: {e}")
                    epoch_co2 = 0.0
            
            # Acumular emisiones totales
            self.total_emissions += epoch_co2
            
            # Obtener RMSE y MAE actuales (asegurarse de que sean números Python)
            current_rmse = trainer.callback_metrics.get("val_rmse", torch.tensor(0.0)).item()
            current_mae = trainer.callback_metrics.get("val_mae", trainer.callback_metrics.get("train_mae", torch.tensor(0.0))).item()
            
            # Rastrear el mejor RMSE y sus emisiones asociadas
            if current_rmse < self.best_rmse:
                self.best_rmse = current_rmse
                self.best_rmse_epoch = epoch
                self.best_rmse_emissions = epoch_co2
                self.best_rmse_cumulative_emissions = self.total_emissions
            
            # Guardar datos de esta época
            self.epoch_emissions.append(epoch_co2)
            self.cumulative_emissions.append(self.total_emissions)
            self.epoch_rmse.append(current_rmse)
            self.epoch_mae.append(current_mae)
            self.emission_rmse_pairs.append((self.total_emissions, current_rmse))  # Ahora guardamos emisiones acumulativas
            
            # Registrar métricas (asegurarse de que no sean None)
            pl_module.log_dict({
                'environment/epoch_emissions_kg': float(epoch_co2),
                'environment/cumulative_emissions_kg': float(self.total_emissions)
            })
            
            print(f"Epoch {epoch} - Emisiones: {epoch_co2:.8f} kg, Acumulado: {self.total_emissions:.8f} kg, RMSE: {current_rmse:.4f}, MAE: {current_mae:.4f}")
        except Exception as e:
            print(f"Error al medir emisiones en época {epoch}: {e}")
    
    def on_train_end(self, trainer, pl_module):
        try:
            # Detener el tracker principal
            if hasattr(self, 'main_tracker') and self.main_tracker:
                try:
                    self.main_tracker.stop()
                except Exception as e:
                    print(f"Error al detener el tracker principal: {e}")
            
            # Asegurarse de que todos los trackers estén detenidos
            for epoch, tracker in self.trackers.items():
                if tracker is not None:
                    try:
                        tracker.stop()
                    except:
                        pass
            
            # Si no hay datos, salir
            if not self.emission_rmse_pairs:
                print("No hay datos de emisiones para graficar")
                return
            
            # Crear dataframe con todos los datos
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            df = pd.DataFrame({
                'epoch': range(len(self.epoch_emissions)),
                'epoch_emissions_kg': self.epoch_emissions,
                'cumulative_emissions_kg': self.cumulative_emissions,
                'rmse': self.epoch_rmse,
                'mae': self.epoch_mae
            })
            df.to_csv(f'emissions_reports/emissions_metrics_{DATASET}_{"full" if USE_ALL_DATA else f"split{SPLIT}"}_{timestamp}.csv', index=False)
            
            # Graficar las relaciones
            self.plot_emissions_vs_metrics(timestamp)
            
            # Mostrar información del mejor RMSE y sus emisiones asociadas
            if self.best_rmse_epoch is not None:
                print(f"\n=== Best RMSE and Associated Emissions ===")
                print(f"Best RMSE: {self.best_rmse:.4f} (Epoch {self.best_rmse_epoch})")
                print(f"Emissions at best RMSE: {self.best_rmse_emissions:.8f} kg")
                print(f"Cumulative emissions at best RMSE: {self.best_rmse_cumulative_emissions:.8f} kg")
            
        except Exception as e:
            print(f"Error al generar gráficos de emisiones: {e}")
            
    def plot_emissions_vs_metrics(self, timestamp):
        """Genera gráficos para emisiones vs métricas"""
        
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
        plt.savefig(f'emissions_plots/cumulative_emissions_vs_rmse_{DATASET}_split{SPLIT}_{timestamp}.png')
        plt.close()
        
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
        
        plt.subplot(2, 3, 3)
        plt.plot(range(len(self.epoch_rmse)), self.epoch_rmse, 'b-', marker='o')
        plt.title('RMSE por Época')
        plt.xlabel('Época')
        plt.ylabel('RMSE')
        
        plt.subplot(2, 3, 4)
        plt.plot(range(len(self.epoch_mae)), self.epoch_mae, 'm-', marker='s')
        plt.title('MAE por Época')
        plt.xlabel('Época')
        plt.ylabel('MAE')
        
        plt.subplot(2, 3, 5)
        plt.plot(self.cumulative_emissions, self.epoch_rmse, 'g-', marker='o')
        plt.title('RMSE vs Emisiones Acumuladas')
        plt.xlabel('Emisiones Acumuladas (kg)')
        plt.ylabel('RMSE')
        
        plt.subplot(2, 3, 6)
        plt.plot(self.cumulative_emissions, self.epoch_mae, 'orange', marker='s')
        plt.title('MAE vs Emisiones Acumuladas')
        plt.xlabel('Emisiones Acumuladas (kg)')
        plt.ylabel('MAE')
        
        plt.tight_layout()
        plt.savefig(f'emissions_plots/metrics_by_epoch_{DATASET}_split{SPLIT}_{timestamp}.png')
        plt.close()
        
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
        
        plt.savefig(f'emissions_plots/cumulative_emissions_vs_rmse_{DATASET}_{"full" if USE_ALL_DATA else f"split{SPLIT}"}_{timestamp}.png')
        plt.close()     


def train_MF(
    dataset_name=DATASET,
    embedding_dim=EMBEDDING_DIM,
    batch_size=2048,  # Optimizado para mejor rendimiento
    num_workers=8,    # Optimizado para mejor rendimiento  
    l2_reg=1e-2,
    learning_rate=1e-3,
    verbose=1,
    fair=0,
    use_all_data=False,  # Agregar esta opción
):
    try:
        # Initialize system metrics callback
        system_metrics = SystemMetricsCallback()
        
        # Initialize CodeCarbon tracker with output_dir
        tracker = EmissionsTracker(
            project_name=f"MF_{dataset_name}_{'full' if use_all_data else f'split{SPLIT}'}",
            output_dir="emissions_reports",
            measure_power_secs=10,
            save_to_file=True,
            log_level="error"
        )

        # Start the tracker
        tracker.start()
        
        # Load the dyadic dataset
        data_module = datamodule.DyadicRegressionDataModule(
            dataset_name=dataset_name,
            split=None if use_all_data else SPLIT,  # Pasar None si use_all_data es True
            data_dir=DATA_DIR,  # Usar el directorio base correcto
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose,
            use_all_data=use_all_data,  # Pasar la nueva opción
        )

        MODEL_NAME = "MF" if not fair else "MF_fair"

        if not fair:
            model = mf.CollaborativeFilteringModel(
                num_users=data_module.num_users,
                num_items=data_module.num_items,
                embedding_dim=embedding_dim,
                lr=learning_rate,
                l2_reg=l2_reg,
                rating_range=(data_module.min_rating, data_module.max_rating),
                has_gender=data_module.has_gender,
            )
        else:
            class_weights = data_module.get_class_weights()
            if class_weights is not None:
                model = mf_fair.MFWithFairPretraining(
                    num_users=data_module.num_users,
                    num_items=data_module.num_items,
                    embedding_dim=embedding_dim,
                    lr=learning_rate,
                    l2_reg=l2_reg,
                    rating_range=(data_module.min_rating, data_module.max_rating),
                    class_weights=class_weights,
                )
            else:
                print("Advertencia: No hay datos de género disponibles. Usando modelo estándar MF en lugar del modelo fair.")
                model = mf.CollaborativeFilteringModel(
                    num_users=data_module.num_users,
                    num_items=data_module.num_items,
                    embedding_dim=embedding_dim,
                    lr=learning_rate,
                    l2_reg=l2_reg,
                    rating_range=(data_module.min_rating, data_module.max_rating),
                    has_gender=data_module.has_gender,
                )

        # Clean up old checkpoint if exists
        checkpoint_path = f"models/{MODEL_NAME}/checkpoints/{dataset_name}/split_{SPLIT}/best-model-{EMBEDDING_DIM}.ckpt"
        if path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        # Crear el callback de emisiones
        emissions_callback = EmissionsPerEpochCallback()
        
        # Definir los callbacks básicos
        callbacks = [
            system_metrics,
            emissions_callback,
            pl.callbacks.ModelCheckpoint(
                dirpath=f"models/{MODEL_NAME}/checkpoints/{dataset_name}/split_{SPLIT}",
                filename=f"best-model-{EMBEDDING_DIM}",
                monitor="val_rmse",
                mode="min",
                save_weights_only=True,
                enable_version_counter=False,
            ),
        ]
        
        # Agregar callbacks de fairness solo si hay datos de género disponibles
        if data_module.has_gender:
            callbacks.extend([
                fairness.EOgenreRelationCallback(),
                fairness.EccentricityGCallback(data_module.train_df, SPLIT),
            ])
        else:
            print("Advertencia: No se encontraron datos de género. Se omitirán las métricas de fairness.")

        trainer = pl.Trainer(
            accelerator="auto",
            enable_checkpointing=True,
            callbacks=callbacks,
            logger=True,
            precision=16,
            enable_model_summary=True,
            enable_progress_bar=True,
            max_epochs=50, 
        )
        
        # Check gender distribution (only if gender data is available)
        if verbose and data_module.has_gender:
            print("Distribución de género en entrenamiento:")
            print(data_module.train_df['gender'].value_counts())
            print("\nDistribución de género en validación:")
            print(data_module.val_df['gender'].value_counts())
            print("\nDistribución de género en test:")
            print(data_module.test_df['gender'].value_counts())

        # Training
        trainer.fit(model, data_module)

        # Load best model
        checkpoint_path = f"models/{MODEL_NAME}/checkpoints/{dataset_name}/split_{SPLIT}/best-model-{EMBEDDING_DIM}.ckpt"
        if not fair or data_module.get_class_weights() is None:
            model = mf.CollaborativeFilteringModel.load_from_checkpoint(checkpoint_path)
        else:
            model = mf_fair.MFWithFairPretraining.load_from_checkpoint(checkpoint_path)

        # Testing
        trainer.test(model, datamodule=data_module)
        
        # Sincronizar la información del mejor RMSE entre callbacks
        if hasattr(system_metrics, 'best_rmse_epoch') and system_metrics.best_rmse_epoch is not None:
            # Si el EmissionsCallback no tiene la información del mejor RMSE, sincronizarla
            if (emissions_callback.best_rmse_epoch != system_metrics.best_rmse_epoch and 
                system_metrics.best_rmse_epoch < len(emissions_callback.epoch_emissions)):
                emissions_callback.best_rmse = system_metrics.best_rmse
                emissions_callback.best_rmse_epoch = system_metrics.best_rmse_epoch
                emissions_callback.best_rmse_emissions = emissions_callback.epoch_emissions[system_metrics.best_rmse_epoch]
                emissions_callback.best_rmse_cumulative_emissions = emissions_callback.cumulative_emissions[system_metrics.best_rmse_epoch]
                
                # Mostrar la información sincronizada del mejor RMSE y emisiones
                print(f"\n=== Best RMSE and Associated Emissions ===")
                print(f"Best RMSE: {emissions_callback.best_rmse:.4f} (Epoch {emissions_callback.best_rmse_epoch})")
                print(f"Emissions at best RMSE: {emissions_callback.best_rmse_emissions:.8f} kg")
                print(f"Cumulative emissions at best RMSE: {emissions_callback.best_rmse_cumulative_emissions:.8f} kg")
        
    finally:
        # Stop the tracker and add emissions to metrics
        emissions = tracker.stop()
        if emissions is None:
            emissions = 0.0
        
        if system_metrics.test_metrics:
            system_metrics.test_metrics[-1]['co2_emissions_kg'] = emissions
            print(f"\nTotal CO2 Emissions: {emissions:.6f} kg")
            
            # Log emissions
            if trainer.logger:
                trainer.logger.log_metrics({
                    'environment/co2_emissions_kg': emissions,
                    'environment/energy_consumed_kwh': getattr(tracker, '_total_energy', type('obj', (object,), {'kWh': 0.0})()).kWh
                })
'''
if __name__ == "__main__":
    train_MF(verbose=1, fair=0)
'''

if __name__ == "__main__":
    # Ya no es necesario definir argumentos aquí, pues ya se procesaron arriba
    train_MF(verbose=1, fair=0, use_all_data=USE_ALL_DATA)