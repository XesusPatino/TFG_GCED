import os
import time
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torchmetrics.retrieval import RetrievalRecall, RetrievalNormalizedDCG
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
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
import uuid

logging.getLogger("pytorch_lightning.utilities.distributed").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.gpu").setLevel(logging.WARNING)

DATA_DIR = "C:/Users/xpati/Documents/TFG/Data_Fair_MF"

# Create command line arguments
args = ArgumentParser()
args.add_argument("--d", type=int, default=256)
args.add_argument("--dataset", type=str)
args.add_argument("--split", type=int)
args = args.parse_args()

EMBEDDING_DIM = args.d
DATASET = args.dataset
SPLIT = args.split

class EmissionsPerformanceCallback(pl.Callback):
    """
    Callback para medir el rendimiento del modelo y las emisiones de CO2 en cada época
    durante el entrenamiento de un sistema de recomendación basado en ratings.
    """
    def __init__(self, data_dir: str, dataset_name: str, split: int):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.split = split
        
        # Resultados de test
        self.test_results = {
            'user_ids': [],
            'item_ids': [],
            'ratings': [],
            'predictions': [],
            'genders': []
        }
        
        # Métricas de sistema
        self.train_start_time = None
        self.test_start_time = None
        self.train_metrics = []
        self.test_metrics = []
        
        # Emisiones y rendimiento por época
        self.epoch_emissions = []
        self.cumulative_emissions = []    # Emisiones acumuladas hasta cada época
        self.epoch_rmse = []
        self.total_emissions = 0.0
        
        # Métricas de validación
        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()
        self.test_mae = MeanAbsoluteError()
        
        # Inicializar directorios para reportes
        os.makedirs('emissions_reports', exist_ok=True)
        os.makedirs('emissions_plots', exist_ok=True)
        
        # Trackers por época
        self.trackers = {}
        self.unique_run_id = str(uuid.uuid4())[:8]
        
        # Fairness visualization
        if data_dir:
            self.visualizer = fairness.EccentricityVisualizer(data_dir=data_dir)
    
    def on_train_start(self, trainer, pl_module):
        self.train_start_time = time.time()
        self.train_data = trainer.datamodule.train_df
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        
        # Métricas de sistema
        self.current_epoch_metrics = {
            'epoch': trainer.current_epoch,
            'memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'cpu_usage_percent': psutil.cpu_percent(),
        }
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.current_epoch_metrics['gpu_usage_mb'] = 0
        
        # Tracker de emisiones para esta época
        epoch = trainer.current_epoch
        try:
            # Primero intentamos detener cualquier tracker existente para esta época
            if epoch in self.trackers and self.trackers[epoch]:
                try:
                    self.trackers[epoch].stop()
                except:
                    pass
            
            # Crear nuevo tracker con nombre único para evitar conflictos
            self.trackers[epoch] = EmissionsTracker(
                project_name=f"MF_History_{self.dataset_name}_split{self.split}_epoch{epoch}_{self.unique_run_id}",
                output_dir="emissions_reports",
                save_to_file=True,
                log_level="error"
            )
            self.trackers[epoch].start()
            print(f"Started emissions tracker for epoch {epoch}")
        except Exception as e:
            print(f"Warning: Could not start tracker for epoch {epoch}: {e}")
            self.trackers[epoch] = None
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            self.current_epoch_metrics['gpu_usage_mb'] = max(
                self.current_epoch_metrics.get('gpu_usage_mb', 0),
                torch.cuda.max_memory_allocated() / 1024 ** 2
            )
    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        
        # 1. Recopilar métricas de sistema
        self.current_epoch_metrics['epoch_time_sec'] = time.time() - self.epoch_start_time
        self.train_metrics.append(self.current_epoch_metrics)
        
        # 2. Calcular emisiones para esta época
        epoch_co2 = 0.0
        if epoch in self.trackers and self.trackers[epoch]:
            try:
                epoch_co2 = self.trackers[epoch].stop() or 0.0
                print(f"Stopped emissions tracker for epoch {epoch}, emissions: {epoch_co2:.6f} kg")
            except Exception as e:
                print(f"Warning: Error stopping tracker for epoch {epoch}: {e}")
        
        # Acumular emisiones totales
        self.total_emissions += epoch_co2
        self.current_epoch_metrics['co2_emissions_kg'] = epoch_co2
        self.current_epoch_metrics['cumulative_co2_emissions_kg'] = self.total_emissions
        
        # 3. Ejecutar validación para obtener RMSE
        val_rmse = self._calculate_validation_rmse(trainer, pl_module)
        
        # 4. Almacenar métricas para esta época
        self.epoch_emissions.append(epoch_co2)             # Emisiones de esta época
        self.cumulative_emissions.append(self.total_emissions)  # Emisiones acumuladas
        self.epoch_rmse.append(val_rmse)
        
        # 5. Registrar métricas
        metrics_to_log = {
            'time/epoch_time_sec': self.current_epoch_metrics['epoch_time_sec'],
            'memory/memory_usage_mb': self.current_epoch_metrics['memory_usage_mb'],
            'cpu/cpu_usage_percent': self.current_epoch_metrics['cpu_usage_percent'],
            'emissions/epoch_co2_kg': epoch_co2,
            'emissions/cumulative_co2_kg': self.total_emissions,      # Añadido: emisiones acumuladas
            'metrics/epoch_rmse': val_rmse,
            'val_rmse': val_rmse,  # Para checkpoint
        }
        
        if 'gpu_usage_mb' in self.current_epoch_metrics:
            metrics_to_log['gpu/gpu_usage_mb'] = self.current_epoch_metrics['gpu_usage_mb']
        
        if trainer.logger:
            trainer.logger.log_metrics(metrics_to_log, step=trainer.global_step)
        
        # 6. Imprimir resumen de la época
        print(f"\nEpoch {self.current_epoch_metrics['epoch']} Metrics:")
        print(f"  Time: {self.current_epoch_metrics['epoch_time_sec']:.2f}s")
        print(f"  Memory: {self.current_epoch_metrics['memory_usage_mb']:.2f}MB")
        print(f"  CPU: {self.current_epoch_metrics['cpu_usage_percent']:.1f}%")
        if 'gpu_usage_mb' in self.current_epoch_metrics:
            print(f"  GPU: {self.current_epoch_metrics['gpu_usage_mb']:.2f}MB")
        print(f"  RMSE: {val_rmse:.4f}")
        print(f"  CO2 Emissions: {epoch_co2:.6f} kg")
        print(f"  Cumulative CO2: {self.total_emissions:.6f} kg")
        
        # 7. Generar visualizaciones intermedias
        if epoch > 0:  # Solo si tenemos al menos dos puntos
            self._plot_emissions_vs_performance()
    
    def _calculate_validation_rmse(self, trainer, pl_module):
        """Calcular RMSE en el conjunto de validación"""
        pl_module.eval()
        val_dataloader = trainer.datamodule.val_dataloader()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                user_ids, item_ids, ratings, _ = batch
                user_ids = user_ids.to(pl_module.device)
                item_ids = item_ids.to(pl_module.device)
                ratings = ratings.to(pl_module.device)
                
                # Obtener predicciones
                predictions = pl_module(user_ids, item_ids)
                
                # Asegurar formato correcto
                if predictions.dim() > 1:
                    if predictions.size(1) == 1:
                        predictions = predictions.squeeze(1)
                    else:
                        # Si hay múltiples predicciones por usuario, tomar la primera columna
                        predictions = predictions[:, 0]
                
                all_preds.append(predictions.cpu())
                all_targets.append(ratings.cpu())
        
        # Concatenar resultados
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        
        # Calcular RMSE
        rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()
        
        pl_module.train()
        return rmse
    
    def on_train_end(self, trainer, pl_module):
        # 1. Asegurar que todos los trackers están detenidos
        for epoch, tracker in self.trackers.items():
            if tracker:
                try:
                    tracker.stop()
                except:
                    pass
        
        # 2. Crear dataframe con todas las métricas
        emission_data = pd.DataFrame({
            'epoch': range(len(self.epoch_emissions)),
            'epoch_emissions_kg': self.epoch_emissions,
            'cumulative_emissions_kg': self.cumulative_emissions,   # Añadido: emisiones acumuladas
            'rmse': self.epoch_rmse
        })
        
        # 3. Guardar métricas en CSV
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        emission_data.to_csv(f'emissions_reports/emissions_metrics_{self.dataset_name}_{self.split}_{timestamp}.csv', index=False)
        
        # 4. Generar visualizaciones finales
        self._plot_emissions_vs_performance(is_final=True)
    
    def _plot_emissions_vs_performance(self, is_final=False):
        """Generar gráficos de emisiones vs rendimiento"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        suffix = "final" if is_final else "intermediate"
        
        # 1. Emisiones por época vs RMSE
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        line1 = ax1.plot(range(len(self.epoch_rmse)), self.epoch_rmse, 'b-', marker='o', label='RMSE')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('RMSE', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax1.twinx()
        line2 = ax2.plot(range(len(self.epoch_emissions)), self.epoch_emissions, 'r-', marker='x', label='CO2 Emissions')
        ax2.set_ylabel('CO2 Emissions (kg)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best')
        
        plt.title('RMSE vs CO2 Emissions por Época')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'emissions_plots/emissions_vs_rmse_{self.dataset_name}_{self.split}_{suffix}_{timestamp}.png')
        plt.close()
        
        # 2. Emisiones acumuladas vs RMSE
        plt.figure(figsize=(10, 6))
        plt.plot(self.cumulative_emissions, self.epoch_rmse, 'b-', marker='o')
        
        # Añadir etiquetas a cada punto
        for i, (cum_em, rmse) in enumerate(zip(self.cumulative_emissions, self.epoch_rmse)):
            plt.annotate(f"Epoch {i}", (cum_em, rmse), textcoords="offset points", 
                        xytext=(5,5), ha='center', fontsize=8)
        
        plt.xlabel('Emisiones de CO2 acumuladas (kg)')
        plt.ylabel('RMSE')
        plt.title('Rendimiento vs Emisiones Acumuladas')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'emissions_plots/cumulative_emissions_vs_rmse_{self.dataset_name}_{self.split}_{suffix}_{timestamp}.png')
        plt.close()
        
        # 3. Pareto frontier (si es final)
        if is_final and len(self.epoch_emissions) > 1:
            plt.figure(figsize=(10, 6))
            
            # Scatter plot de todos los puntos
            plt.scatter(self.cumulative_emissions, self.epoch_rmse, color='blue', marker='o', label='Epoch Results')
            
            # Identificar puntos del frente de Pareto (CAMBIO AQUÍ - usar el nombre correcto)
            pareto_points = self._find_pareto_frontier_cumulative()
            pareto_emissions = [self.cumulative_emissions[i] for i in pareto_points]
            pareto_rmse = [self.epoch_rmse[i] for i in pareto_points]
            
            # Destacar frente de Pareto
            plt.scatter(pareto_emissions, pareto_rmse, color='red', marker='*', 
                    s=100, label='Pareto Frontier')
            
            # Conectar puntos del frente
            if len(pareto_emissions) > 1:
                pareto_indices = sorted(range(len(pareto_emissions)), key=lambda i: pareto_emissions[i])
                sorted_pareto_emissions = [pareto_emissions[i] for i in pareto_indices]
                sorted_pareto_rmse = [pareto_rmse[i] for i in pareto_indices]
                plt.plot(sorted_pareto_emissions, sorted_pareto_rmse, 'r--', alpha=0.7)
            
            # Añadir etiquetas a los puntos
            for i, (cum_em, rmse) in enumerate(zip(self.cumulative_emissions, self.epoch_rmse)):
                plt.annotate(f"{i}", (cum_em, rmse), textcoords="offset points", 
                        xytext=(0,5), ha='center', fontsize=8)
            
            plt.xlabel('Emisiones de CO2 acumuladas (kg)')
            plt.ylabel('RMSE')
            plt.title('Pareto Frontier: RMSE vs Emisiones Acumuladas')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'emissions_plots/pareto_frontier_cumulative_{self.dataset_name}_{self.split}_{timestamp}.png')
            plt.close()
            
            # Gráfico adicional: tamaño de puntos proporcional a la época
            plt.figure(figsize=(10, 6))
            
            # Tamaños progresivamente mayores para visualizar mejor la progresión
            sizes = [(i+1)*30 for i in range(len(self.cumulative_emissions))]
            
            plt.scatter(self.cumulative_emissions, self.epoch_rmse, color='blue', 
                    s=sizes, alpha=0.7)
            
            # Añadir etiquetas a los puntos
            for i, (cum_em, rmse) in enumerate(zip(self.cumulative_emissions, self.epoch_rmse)):
                plt.annotate(f"Epoch {i}", (cum_em, rmse), textcoords="offset points", 
                        xytext=(5,5), ha='center', fontsize=8)
            
            plt.xlabel('Emisiones de CO2 acumuladas (kg)')
            plt.ylabel('RMSE')
            plt.title('Progresión de Rendimiento vs Emisiones Acumuladas')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'emissions_plots/progressive_emissions_rmse_{self.dataset_name}_{self.split}_{timestamp}.png')
            plt.close()
    
    def _find_pareto_frontier_cumulative(self):
        """Identificar los puntos que forman el frente de Pareto usando emisiones acumuladas"""
        # En este caso, queremos minimizar RMSE mientras minimizamos emisiones acumuladas
        pareto_points = []
        
        for i in range(len(self.epoch_rmse)):
            is_pareto = True
            
            for j in range(len(self.epoch_rmse)):
                if i != j:
                    # Si j domina a i (mejor en ambas métricas), entonces i no está en el frente
                    if (self.epoch_rmse[j] <= self.epoch_rmse[i] and 
                        self.cumulative_emissions[j] <= self.cumulative_emissions[i]):
                        # Si j es estrictamente mejor en al menos una métrica
                        if (self.epoch_rmse[j] < self.epoch_rmse[i] or 
                            self.cumulative_emissions[j] < self.cumulative_emissions[i]):
                            is_pareto = False
                            break
            
            if is_pareto:
                pareto_points.append(i)
        
        return pareto_points
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        user_ids, item_ids, ratings, genders = batch
        
        if isinstance(outputs, dict):
            predictions = outputs.get("predictions", None)
            if predictions is None and len(outputs) > 0:
                first_key = next(iter(outputs))
                predictions = outputs[first_key]
        else:
            predictions = outputs
        
        # Ensure predictions have the right shape for metrics
        if predictions is not None:
            if predictions.dim() > 1:
                if predictions.size(1) == 1:
                    predictions = predictions.squeeze(1)
                else:
                    predictions = predictions[:, 0]
                    
            # Update metrics
            self.val_mse.update(predictions, ratings)
            self.val_mae.update(predictions, ratings)
    
    def on_validation_end(self, trainer, pl_module):
        # Calculate validation metrics
        val_rmse = torch.sqrt(self.val_mse.compute())
        val_mae = self.val_mae.compute()
        
        # Log metrics directly using the logger - not through pl_module.log()
        if trainer.logger:
            trainer.logger.log_metrics({
                "val_rmse": val_rmse.item(),
                "val_mae": val_mae.item(),
                "val/rmse": val_rmse.item(),
                "val/mae": val_mae.item()
            }, step=trainer.global_step)
        
        # Print metrics
        print(f"\nValidation Metrics:")
        print(f"  RMSE: {val_rmse.item():.4f}")
        print(f"  MAE: {val_mae.item():.4f}")
        
        # Store metrics in the module for reference
        # Use custom attribute names with underscore to avoid conflicts
        setattr(pl_module, "_val_rmse_value", val_rmse.item())
        setattr(pl_module, "_val_mae_value", val_mae.item())
        
        # Reset metrics for next epoch
        self.val_mse.reset()
        self.val_mae.reset()
    
    def on_test_start(self, trainer, pl_module):
        self.test_start_time = time.time()
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        user_ids, item_ids, ratings, genders = batch
        
        # Verificar que outputs sea un diccionario y extraer las predicciones
        if isinstance(outputs, dict):
            predictions = outputs.get("predictions", None)
            if predictions is None and len(outputs) > 0:
                # Si no hay clave 'predictions', usar la primera clave disponible
                first_key = next(iter(outputs))
                predictions = outputs[first_key]
        else:
            # Si no es un diccionario, asumir que es directamente el tensor de predicciones
            predictions = outputs
        
        # Ensure predictions have the right shape for metrics
        if predictions is not None:
            if predictions.dim() > 1:
                if predictions.size(1) == 1:
                    predictions = predictions.squeeze(1)
                else:
                    # If we have multiple predictions per user, take the first column
                    predictions = predictions[:, 0]
                    
            # Update metrics
            self.test_mse.update(predictions, ratings)
            self.test_mae.update(predictions, ratings)
        
            # Store results
            self.test_results['user_ids'].append(user_ids.cpu())
            self.test_results['item_ids'].append(item_ids.cpu())
            self.test_results['ratings'].append(ratings.cpu())
            self.test_results['predictions'].append(predictions.cpu())
            self.test_results['genders'].append(genders.cpu())
    
    def on_test_end(self, trainer, pl_module):
        # Calcular métricas finales
        test_time = time.time() - self.test_start_time
        total_time = time.time() - self.train_start_time
        
        rmse = torch.sqrt(self.test_mse.compute())
        mae = self.test_mae.compute()
        
        # Registrar métricas
        final_metrics = {
            'time/total_time_sec': total_time,
            'time/test_time_sec': test_time,
            'metrics/test_rmse': rmse.item(),
            'metrics/test_mae': mae.item(),
            'emissions/total_co2_kg': self.total_emissions
        }
        
        if torch.cuda.is_available():
            final_metrics['gpu/final_gpu_usage_mb'] = torch.cuda.max_memory_allocated() / 1024 ** 2
        
        self.test_metrics.append(final_metrics)
        
        # Imprimir métricas finales
        print(f"\n=== Final Test Metrics ===")
        print(f"Total Time: {total_time:.2f}s (Test: {test_time:.2f}s)")
        print(f"RMSE: {rmse.item():.4f}")
        print(f"MAE: {mae.item():.4f}")
        print(f"Total CO2 Emissions: {self.total_emissions:.6f} kg")
        
        # Try to visualize fairness
        try:
            test_results = {}
            for k, v in self.test_results.items():
                test_results[k] = torch.cat(v)
            
            self.visualizer.plot(
                train_data=self.train_data,
                test_results=test_results,
                split=self.split
            )
        except Exception as e:
            print(f"Error generando visualización de fairness: {e}")

def train_MF(
    dataset_name=DATASET,
    embedding_dim=EMBEDDING_DIM,
    batch_size=1024,
    num_workers=8,
    l2_reg=1e-2,
    learning_rate=5e-4,
    verbose=1,
    fair=0,
    history_weight=0.3,  # Weight for history-based prediction
):
    
    # Usar la constante global DATA_DIR
    data_dir = DATA_DIR
    
    # Initialize emissions performance callback
    emissions_metrics = EmissionsPerformanceCallback(
        data_dir=data_dir,
        dataset_name=dataset_name or "ml-1m",
        split=SPLIT
    )
    
    try:
        # Load the dyadic dataset
        data_module = datamodule.DyadicRegressionDataModule(
            dataset_name=dataset_name,
            split=SPLIT,
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose,
        )

        MODEL_NAME = "MF_History" if not fair else "MF_History_fair"

        if not fair:
            model = mf.CollaborativeFilteringModel(
                num_users=data_module.num_users,
                num_items=data_module.num_items,
                embedding_dim=embedding_dim,
                lr=learning_rate,
                l2_reg=l2_reg,
                rating_range=(data_module.min_rating, data_module.max_rating),
                history_weight=history_weight,
            )
            
            # Set metadata for history-based prediction
            metadata = data_module.get_user_item_metadata()
            model.set_metadata(metadata)
        else:
            class_weights = data_module.get_class_weights()
            model = mf_fair.MFWithFairPretraining(
                num_users=data_module.num_users,
                num_items=data_module.num_items,
                embedding_dim=embedding_dim,
                lr=learning_rate,
                l2_reg=l2_reg,
                rating_range=(data_module.min_rating, data_module.max_rating),
                class_weights=class_weights,
                history_weight=history_weight,
            )
            
            # Set metadata for history-based prediction
            metadata = data_module.get_user_item_metadata()
            model.set_metadata(metadata)

        # Clean up old checkpoint if exists
        checkpoint_path = f"models/{MODEL_NAME}/checkpoints/{dataset_name}/split_{SPLIT}/best-model-{EMBEDDING_DIM}.ckpt"
        if path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        callbacks = [
            emissions_metrics,
            pl.callbacks.ModelCheckpoint(
                dirpath=f"models/{MODEL_NAME}/checkpoints/{dataset_name}/split_{SPLIT}",
                filename=f"best-model-{EMBEDDING_DIM}",
                monitor="val_rmse",  # Monitor RMSE for rating prediction
                mode="min",         # Minimize RMSE
                save_weights_only=True,
                enable_version_counter=False,
            ),
        ]

        trainer = pl.Trainer(
            accelerator="auto",
            enable_checkpointing=True,
            callbacks=callbacks,
            logger=True,
            precision="16-mixed",  # Usar precisión mixta para acelerar
            max_epochs=15,         # Possibly more epochs for better convergence
            gradient_clip_val=1.0, # Añadir clipping de gradientes
            limit_train_batches=0.50,  # Usar solo 50% del conjunto de datos
            limit_val_batches=0.2,     # Validar en solo 20% de los datos
        )
        
        # Check gender distribution
        if verbose:
            print("Distribución de género en entrenamiento:")
            print(data_module.train_df['gender'].value_counts())
            print("\nDistribución de género en validación:")
            print(data_module.val_df['gender'].value_counts())
            print("\nDistribución de género en test:")
            print(data_module.test_df['gender'].value_counts())

        # Training
        trainer.fit(model, data_module)

        # Load best model
        if not fair:
            model = mf.CollaborativeFilteringModel.load_from_checkpoint(
                f"models/{MODEL_NAME}/checkpoints/{dataset_name}/split_{SPLIT}/best-model-{EMBEDDING_DIM}.ckpt"
            )
            # Set metadata again after loading the checkpoint
            metadata = data_module.get_user_item_metadata()
            model.set_metadata(metadata)
        else:
            model = mf_fair.MFWithFairPretraining.load_from_checkpoint(
                f"models/{MODEL_NAME}/checkpoints/{dataset_name}/split_{SPLIT}/best-model-{EMBEDDING_DIM}.ckpt"
            )
            # Set metadata again after loading the checkpoint
            metadata = data_module.get_user_item_metadata()
            model.set_metadata(metadata)

        # Testing
        trainer.test(model, datamodule=data_module)
        
        # Demonstrate rating prediction for sample users and items
        demonstrate_rating_prediction(model, data_module)
        
    finally:
        # No cleanup needed as the callback handles tracking
        pass

def demonstrate_rating_prediction(model, data_module):
    """Show examples of rating prediction using user history"""
    print("\n=== Rating Prediction Demo ===")
    
    # Get some test users and items
    test_df = data_module.test_df.sample(5)  # Sample 5 random test examples
    
    print("\nSample Rating Predictions:")
    print(f"{'User ID':<10}{'Item ID':<10}{'True Rating':<15}{'Predicted':<15}{'Error':<10}")
    
    for _, row in test_df.iterrows():
        user_id = int(row['user_id'])
        item_id = int(row['item_id'])
        true_rating = row['rating']
        
        # Get prediction
        with torch.no_grad():
            model.eval()
            pred_rating = model.predict_rating(user_id, item_id)
        
        error = abs(true_rating - pred_rating)
        print(f"{user_id:<10}{item_id:<10}{true_rating:<15.2f}{pred_rating:<15.2f}{error:<10.2f}")
    
    # Show a user's history to demonstrate how it's used
    user_id = int(test_df.iloc[0]['user_id'])
    if model.user_histories and user_id in model.user_histories:
        history = model.user_histories[user_id]
        print(f"\nSample of User {user_id}'s Rating History:")
        print(history.head(5))
        
        # Explain the model's approach
        print("\nModel approach:")
        print("The model combines matrix factorization predictions with history-based")
        print("adjustments by finding similar items in the user's history and weighing")
        print(f"them by similarity. The history weight is set to {model.history_weight}.")

if __name__ == "__main__":
    train_MF(verbose=1, fair=0)