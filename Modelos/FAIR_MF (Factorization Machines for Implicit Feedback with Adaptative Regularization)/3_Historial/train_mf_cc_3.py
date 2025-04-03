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

class RatingPredictionCallback(pl.Callback):
    def __init__(self, data_dir: str):
        super().__init__()
        self.visualizer = fairness.EccentricityVisualizer(data_dir=data_dir)
        self.train_data = None
        self.test_results = {
            'user_ids': [],
            'item_ids': [],
            'ratings': [],
            'predictions': [],
            'genders': []
        }
        self.val_results = {  # Diccionario separado para validación
            'user_ids': [],
            'item_ids': [],
            'ratings': [],
            'predictions': [],
            'genders': []
        }
        self.test_metrics = []  # Inicializar test_metrics
        self.train_metrics = []  # Inicializar train_metrics
        self.train_start_time = None  # Inicializar a None
        self.test_start_time = None   # Inicializar a None
        
        # Metrics for rating prediction
        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()
        self.test_mae = MeanAbsoluteError()
        
    def on_train_start(self, trainer, pl_module):
        self.train_start_time = time.time()  # Guardar tiempo de inicio
        self.train_data = trainer.datamodule.train_df
        
        # Load user histories into model if not already done
        if (hasattr(pl_module, 'user_histories') and pl_module.user_histories is None and 
            hasattr(trainer, 'datamodule')):
            metadata = trainer.datamodule.get_user_item_metadata()
            pl_module.set_metadata(metadata)
        
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
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs is None:
            print(f"Warning: outputs is None at batch {batch_idx}")
            return

        user_ids, item_ids, ratings, genders = batch
        
        # Verificar si outputs es un diccionario y extraer las predicciones
        if isinstance(outputs, dict):
            # Si es un diccionario, busca la clave con las predicciones
            if 'predictions' in outputs:
                predictions = outputs['predictions']
            else:
                # Si no encuentras una clave clara, intenta con la primera clave disponible
                print(f"Warning: outputs is a dictionary without 'predictions' key. Keys: {list(outputs.keys())}")
                first_key = next(iter(outputs))
                predictions = outputs[first_key]
        else:
            # Si no es un diccionario, asume que es directamente el tensor de predicciones
            predictions = outputs

        # Verificar dimensiones
        if predictions.size(0) != user_ids.size(0):
            print(f"Warning: Mismatch in batch sizes at batch {batch_idx}")
            return

        # Ensure predictions have the right shape for rating prediction
        if predictions.dim() > 1:
            if predictions.size(1) == 1:
                predictions = predictions.squeeze(1)
            else:
                # If we have multiple predictions per user, take the first column
                # This might happen if the forward method returns all possible item predictions
                predictions = predictions[:, 0]
        
        # Update metrics
        self.val_mse.update(predictions, ratings)
        self.val_mae.update(predictions, ratings)

        # Store results for later analysis
        self.val_results['user_ids'].append(user_ids.cpu())
        self.val_results['item_ids'].append(item_ids.cpu())
        self.val_results['ratings'].append(ratings.cpu())
        self.val_results['predictions'].append(predictions.cpu())
        self.val_results['genders'].append(genders.cpu())

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
        
        # Clear validation results
        self.val_results = {k: [] for k in self.val_results}
    
    def on_test_start(self, trainer, pl_module):
        self.test_start_time = time.time()
        # Si train_start_time no existe, establecerlo ahora
        if not hasattr(self, 'train_start_time') or self.train_start_time is None:
            self.train_start_time = time.time()
        
        # Make sure model has user histories
        if (hasattr(pl_module, 'user_histories') and pl_module.user_histories is None and 
            hasattr(trainer, 'datamodule')):
            metadata = trainer.datamodule.get_user_item_metadata()
            pl_module.set_metadata(metadata)
    
    def on_test_end(self, trainer, pl_module):
        # Verificar si tenemos resultados para procesar
        empty_results = any(len(v) == 0 for k, v in self.test_results.items())
        if empty_results:
            print("No hay resultados de prueba para procesar. Saltando visualización.")
            return
        
        # Convertir listas de tensores a tensores concatenados
        test_results = {}
        for k, v in self.test_results.items():
            if not v:
                print(f"Advertencia: No hay datos para {k}")
                continue
            
            concatenated = torch.cat(v)
            # Para visualización, aplanar todos los tensores
            if k == 'predictions':
                # Si es bidimensional y tiene shape [N, 1], aplanarlo
                if concatenated.dim() > 1 and concatenated.size(1) == 1:
                    test_results[k] = concatenated.squeeze()
                else:
                    # Si tiene varias columnas, usar solo la primera
                    test_results[k] = concatenated[:, 0] if concatenated.dim() > 1 else concatenated
            else:
                # Para otros tensores, aplanar si es necesario
                if concatenated.dim() > 1:
                    test_results[k] = concatenated.squeeze()
                    if test_results[k].dim() > 1:  # Si sigue siendo bidimensional
                        test_results[k] = concatenated[:, 0]
                else:
                    test_results[k] = concatenated
        
        # Si no tenemos resultados después de procesar, salir
        if not test_results:
            print("No hay resultados válidos para procesar después de la concatenación.")
            return
        
        # Asegurarse de que todos los tensores tengan la misma longitud
        length = min(len(v) for v in test_results.values())
        test_results = {k: v[:length] for k, v in test_results.items()}

        # Crear una copia para la visualización (con todos los tensores planos)
        viz_results = {k: v for k, v in test_results.items()}
        
        # Generar y guardar el gráfico de fairness
        try:
            self.visualizer.plot(
                train_data=self.train_data,
                test_results=viz_results,
                split=self.split if hasattr(self, 'split') else trainer.datamodule.split
            )
        except Exception as e:
            print(f"Error en la visualización: {e}")
            # Imprimir formas para depuración
            for k, v in viz_results.items():
                print(f"{k}: shape={v.shape}, type={type(v)}")

        # Calculate rating prediction metrics
        test_time = time.time() - self.test_start_time
        total_time = time.time() - self.train_start_time if self.train_start_time else 0

        # Compute and log final metrics
        rmse = torch.sqrt(self.test_mse.compute())
        mae = self.test_mae.compute()
        
        # Registrar métricas
        final_metrics = {
            'time/total_time_sec': total_time,
            'time/test_time_sec': test_time,
            'metrics/rmse': rmse.item(),
            'metrics/mae': mae.item(),
        }

        if torch.cuda.is_available():
            final_metrics['gpu/final_gpu_usage_mb'] = torch.cuda.max_memory_allocated() / 1024 ** 2

        if trainer.logger:
            for metric_name, metric_value in final_metrics.items():
                trainer.logger.experiment.add_scalar(metric_name, metric_value)

        self.test_metrics.append(final_metrics)

        # Imprimir métricas finales
        print(f"\n=== Final Test Metrics ===")
        print(f"Total Time: {total_time:.2f}s (Test: {test_time:.2f}s)")
        print(f"RMSE: {rmse.item():.4f}")
        print(f"MAE: {mae.item():.4f}")
        
        # Reset metrics
        self.test_mse.reset()
        self.test_mae.reset()
        
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
    
    # Initialize metrics callback
    system_metrics = RatingPredictionCallback(data_dir=data_dir)
    
    # Try to get any existing trackers and stop them first
    try:
        import codecarbon.emissions_tracker
        if hasattr(codecarbon.emissions_tracker, "_singleton_tracker"):
            if codecarbon.emissions_tracker._singleton_tracker:
                codecarbon.emissions_tracker._singleton_tracker.stop()
                codecarbon.emissions_tracker._singleton_tracker = None
    except:
        pass

    # Now initialize a new tracker
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    tracker = EmissionsTracker(
        project_name=f"MF_History_{dataset_name}_split{SPLIT}_{unique_id}",
        output_dir="emissions_reports",
        measure_power_secs=10,
        save_to_file=True,
        log_level="error"
    )
    
    try:
        tracker.start()
        
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
            system_metrics,
            pl.callbacks.ModelCheckpoint(
                dirpath=f"models/{MODEL_NAME}/checkpoints/{dataset_name}/split_{SPLIT}",
                filename=f"best-model-{EMBEDDING_DIM}",
                monitor="train_loss",  # Change to train_loss which is logged by the model
                mode="min",         # Minimize loss
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
        # Stop the tracker and add emissions to metrics
        emissions = tracker.stop()
        if system_metrics.test_metrics:
            # Check if emissions is None and handle it gracefully
            if emissions is not None:
                system_metrics.test_metrics[-1]['co2_emissions_kg'] = emissions
                print(f"\nTotal CO2 Emissions: {emissions:.6f} kg")
                
                # Log emissions
                if trainer.logger:
                    trainer.logger.log_metrics({
                        'environment/co2_emissions_kg': emissions,
                        'environment/energy_consumed_kwh': tracker._total_energy.kWh
                    })
            else:
                print("\nCO2 Emissions tracking failed - no data available")
                # Add a placeholder value
                system_metrics.test_metrics[-1]['co2_emissions_kg'] = 0.0

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