import os
import time
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torchmetrics.retrieval import RetrievalRecall, RetrievalNormalizedDCG
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

class EccentricityCallback(pl.Callback):
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
        
    def generate_predictions_for_all_items(model, user_ids, num_items):
        all_item_ids = torch.arange(num_items, device=model.device)
        preds = []
        for user_id in user_ids:
            user_id_tensor = torch.tensor([user_id], device=model.device)
            preds.append(model(user_id_tensor, all_item_ids).detach())
        return torch.stack(preds)
        
    def on_train_start(self, trainer, pl_module):
        self.train_start_time = time.time()  # Guardar tiempo de inicio
        self.train_data = trainer.datamodule.train_df
        
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

        self.val_results['user_ids'].append(user_ids.cpu())
        self.val_results['item_ids'].append(item_ids.cpu())
        self.val_results['ratings'].append(ratings.cpu())
        self.val_results['predictions'].append(predictions.cpu())
        self.val_results['genders'].append(genders.cpu())
            
    def on_validation_end(self, trainer, pl_module):
        # Normalizar dimensiones
        max_len = max(v[0].size(0) for v in self.val_results.values() if len(v) > 0)
        for k in self.val_results:
            self.val_results[k] = [
                torch.cat([v, torch.zeros(max_len - v.size(0), *v.shape[1:], device=v.device)]) if v.size(0) < max_len else v
                for v in self.val_results[k]
            ]

        val_results = {
            k: torch.cat(v) for k, v in self.val_results.items()
        }

        preds = val_results['predictions']
        targets = val_results['ratings']
        user_ids = val_results['user_ids']

        print(f"Shape of preds: {preds.shape}")  # Depuración
        print(f"Shape of targets: {targets.shape}")  # Depuración

        # Asegurarse de que preds sea bidimensional
        if preds.dim() == 1:
            # Si preds es unidimensional, probablemente necesites preds por usuario para cada ítem
            # En este caso, deberías recoger predicciones para todos los ítems por usuario
            user_ids_unique = torch.unique(user_ids)
            print(f"Number of unique users: {len(user_ids_unique)}")
            
            # Opción 1: Simplemente reformatear para tener una matriz de predicciones por usuario
            # Esta es una implementación simple que puede no ser exactamente lo que necesitas
            preds = preds.unsqueeze(1)  # Ahora tiene forma [num_users, 1]
            
            # También puedes considerar generar predicciones para todos los ítems aquí:
            # preds = pl_module.forward(user_ids)  # Esto debería generar [num_users, num_items]
        
        threshold = 3.0
        binary_targets = (targets >= threshold).float()
        if binary_targets.dim() == 1:
            binary_targets = binary_targets.unsqueeze(1)  # Asegurar que sea bidimensional

        print(f"Shape of binary_targets: {binary_targets.shape}")  # Depuración
        print(f"Shape of preds after reshape: {preds.shape}")  # Depuración adicional

        # Ahora podemos usar topk con dim=1
        k = min(5, preds.size(1))  # Asegurarse de que k no exceda el número de columnas
        _, top_k_indices = torch.topk(preds, k=k, dim=1)
        print(f"Shape of top_k_indices: {top_k_indices.shape}")  # Depuración

        # Si binary_targets solo tiene una columna pero necesitamos varias para gather
        if binary_targets.size(1) == 1 and preds.size(1) > 1:
            # Expandir binary_targets para que coincida con preds
            binary_targets = binary_targets.expand(-1, preds.size(1))
        
        print(f"Expanded binary_targets shape: {binary_targets.shape}")  # Depuración

        # Ahora podemos realizar gather con seguridad
        preds_top_k = torch.gather(preds, dim=1, index=top_k_indices)
        targets_top_k = torch.gather(binary_targets, dim=1, index=top_k_indices)

        # Expandir user_ids para que coincida con preds_top_k y targets_top_k
        user_ids_expanded = user_ids.unsqueeze(1).expand(-1, preds_top_k.size(1))

        recall_at_k = RetrievalRecall(top_k=k)
        ndcg_at_k = RetrievalNormalizedDCG(top_k=k)
        
        # Calcular métricas (usando user_ids como indexes)
        recall_k = recall_at_k(preds_top_k, targets_top_k, indexes=user_ids_expanded)
        ndcg_k = ndcg_at_k(preds_top_k, targets_top_k, indexes=user_ids_expanded)

        # QUITAR ESTAS DOS LÍNEAS:
        # pl_module.log("val_recall@10", recall_k.item(), on_epoch=True, prog_bar=True, sync_dist=True)
        # pl_module.log("val_ndcg@10", ndcg_k.item(), on_epoch=True, prog_bar=True, sync_dist=True)

        # En su lugar, loggear directamente al logger
        if trainer.logger:
            trainer.logger.log_metrics({
                "val_recall@10": recall_k.item(),  # Este es el nombre que el ModelCheckpoint está monitoreando
                "val_ndcg@10": ndcg_k.item(),
                "val/recall@10": recall_k.item(),
                "val/ndcg@10": ndcg_k.item(),
            }, step=trainer.global_step)
        
        # También imprimir las métricas
        print(f"\nValidation Metrics:")
        print(f"  Recall@{k}: {recall_k.item():.4f}")
        print(f"  NDCG@{k}: {ndcg_k.item():.4f}")
        
        # Almacenar métricas en el módulo para que el ModelCheckpoint pueda acceder a ellas
        # Esta es una solución alternativa ya que no podemos usar self.log
        pl_module.val_recall = recall_k.item()
        pl_module.val_ndcg = ndcg_k.item()
        
        # Limpiar resultados de validación para la próxima época
        self.val_results = {k: [] for k in self.val_results}

        '''
        recall_k = recall_at_k(preds_top_k, targets_top_k, indexes=user_ids_expanded)
        ndcg_k = ndcg_at_k(preds_top_k, targets_top_k, indexes=user_ids_expanded)

        # Loggear métricas directamente al logger
        if trainer.logger:
            trainer.logger.log_metrics({
                "val_recall@10": recall_k.item(),
                "val_ndcg@10": ndcg_k.item(),
            }, step=trainer.global_step)

        # Limpiar resultados de validación para la próxima época
        self.val_results = {k: [] for k in self.val_results}
        '''
    
    ''' 
    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, ratings, genders = batch
        preds = self(user_ids, item_ids)

        # Calcular métricas aquí
        threshold = 3.0
        binary_targets = (ratings >= threshold).float()
        _, top_k_indices = torch.topk(preds, k=5, dim=1)
        preds_top_k = torch.gather(preds, dim=1, index=top_k_indices)
        targets_top_k = torch.gather(binary_targets.unsqueeze(1), dim=1, index=top_k_indices)

        recall_at_k = RetrievalRecall(top_k=10)
        ndcg_at_k = RetrievalNormalizedDCG(top_k=10)

        recall_k = recall_at_k(preds_top_k, targets_top_k, indexes=user_ids.unsqueeze(1))
        ndcg_k = ndcg_at_k(preds_top_k, targets_top_k, indexes=user_ids.unsqueeze(1))

        # Loggear métricas
        self.log("val_recall@10", recall_k, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_ndcg@10", ndcg_k, on_step=False, on_epoch=True, prog_bar=True)

        return preds
    '''
    
    def on_test_start(self, trainer, pl_module):
        self.test_start_time = time.time()
        # Si train_start_time no existe, establecerlo ahora
        if not hasattr(self, 'train_start_time') or self.train_start_time is None:
            self.train_start_time = time.time()
        self.all_preds = []
        self.all_targets = []
        self.all_indexes = []
    
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

        # Calcular métricas de recomendación
        test_time = time.time() - self.test_start_time
        total_time = time.time() - self.train_start_time if self.train_start_time else 0

        # Obtener predicciones originales del test_results (antes de ser aplanadas)
        # Necesitamos la versión original para topk
        preds_orig = torch.cat(self.test_results['predictions'])
        targets = test_results['ratings']
        user_ids = test_results['user_ids']
        
        # Convertir ratings a relevancia binaria
        threshold = 3.0  
        binary_targets = (targets >= threshold).float()

        # Verificar dimensionalidad de preds_orig
        if preds_orig.dim() == 1:
            # Si solo tenemos predicciones unidimensionales, no podemos hacer top-k real
            # Usar una solución simplificada
            print("ADVERTENCIA: Predicciones unidimensionales, usando una aproximación simplificada para top-k")
            rmse = torch.sqrt(torch.mean((preds_orig - targets) ** 2))
            recall_k = torch.tensor(0.0)  # No podemos calcular recall genuino
            ndcg_k = torch.tensor(0.0)    # No podemos calcular ndcg genuino
            top_k = 1
        else:
            # Calcular top-k normalmente con predicciones bidimensionales
            # Asegurarse de que preds_orig sea bidimensional
            if preds_orig.dim() > 2:
                preds_orig = preds_orig.view(preds_orig.size(0), -1)  # Aplanar dimensiones extra
                
            top_k = min(5, preds_orig.size(1))  # Limitar k al número de columnas
            
            # Asegurarse de que binary_targets sea bidimensional para gather
            if binary_targets.dim() == 1:
                # Expandir binary_targets para que coincida con preds_orig
                binary_targets = binary_targets.unsqueeze(1)
                if preds_orig.size(1) > 1:
                    binary_targets = binary_targets.expand(-1, preds_orig.size(1))
            
            # Calcular top-k
            _, top_k_indices = torch.topk(preds_orig, k=top_k, dim=1)
            
            # Crear métricas
            recall_at_k = RetrievalRecall(top_k=top_k)
            ndcg_at_k = RetrievalNormalizedDCG(top_k=top_k)
            
            # Filtrar las predicciones y los targets según los Top-K
            preds_top_k = torch.gather(preds_orig, dim=1, index=top_k_indices)
            targets_top_k = torch.gather(binary_targets, dim=1, index=top_k_indices)
            
            # Expandir user_ids para índices
            user_ids_expanded = user_ids.unsqueeze(1).expand(-1, top_k)
            
            # Calcular métricas
            recall_k = recall_at_k(preds_top_k, targets_top_k, indexes=user_ids_expanded)
            ndcg_k = ndcg_at_k(preds_top_k, targets_top_k, indexes=user_ids_expanded)
            
            # RMSE
            rmse = torch.sqrt(torch.mean((preds_orig.squeeze() - targets) ** 2))
        
        # Registrar métricas
        final_metrics = {
            'time/total_time_sec': total_time,
            'time/test_time_sec': test_time,
            'metrics/rmse': rmse.item(),
            'metrics/recall@5': recall_k.item(),
            'metrics/ndcg@5': ndcg_k.item(),
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
        print(f"Recall@{top_k}: {recall_k.item():.4f}")
        print(f"NDCG@{top_k}: {ndcg_k.item():.4f}")
        
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
        
        # Asegurarse de que predictions es un tensor
        if predictions is not None:
            self.test_results['user_ids'].append(user_ids.cpu())
            self.test_results['item_ids'].append(item_ids.cpu())
            self.test_results['ratings'].append(ratings.cpu())
            self.test_results['predictions'].append(predictions.cpu())
            self.test_results['genders'].append(genders.cpu())

def train_MF(
    dataset_name=DATASET,
    embedding_dim=EMBEDDING_DIM,
    #batch_size=2**10,
    batch_size=1024,
    num_workers=8,
    l2_reg=1e-2,
    learning_rate=5e-4,
    verbose=1,
    fair=0,
):
    
    # Usar la constante global DATA_DIR
    data_dir = DATA_DIR
    
    # Initialize system metrics callback CON el data_dir
    system_metrics = EccentricityCallback(data_dir=data_dir)
    
    # Initialize CodeCarbon tracker
    tracker = EmissionsTracker(
        project_name=f"MF_{dataset_name}_split{SPLIT}",
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

        MODEL_NAME = "MF" if not fair else "MF_fair"

        if not fair:
            model = mf.CollaborativeFilteringModel(
                max_items=4096,
                num_users=data_module.num_users,
                num_items=data_module.num_items,
                embedding_dim=embedding_dim,
                lr=learning_rate,
                l2_reg=l2_reg,
                rating_range=(data_module.min_rating, data_module.max_rating),
            )
        else:
            class_weights = data_module.get_class_weights()
            model = mf_fair.MFWithFairPretraining(
                max_items=4096,
                num_users=data_module.num_users,
                num_items=data_module.num_items,
                embedding_dim=embedding_dim,
                lr=learning_rate,
                l2_reg=l2_reg,
                rating_range=(data_module.min_rating, data_module.max_rating),
                class_weights=class_weights,
            )

        # Clean up old checkpoint if exists
        checkpoint_path = f"models/{MODEL_NAME}/checkpoints/{dataset_name}/split_{SPLIT}/best-model-{EMBEDDING_DIM}.ckpt"
        if path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        callbacks = [
            system_metrics,  # Usa solo esta instancia de EccentricityCallback
            pl.callbacks.ModelCheckpoint(
                dirpath=f"models/{MODEL_NAME}/checkpoints/{dataset_name}/split_{SPLIT}",
                filename=f"best-model-{EMBEDDING_DIM}",
                monitor="train_loss",  # Cambiar a una métrica disponible (como train_loss)
                mode="min",            # Cambiar a "min" para minimizar la pérdida
                save_weights_only=True,
                enable_version_counter=False,
            ),
            # Eliminar esta línea: EccentricityCallback(data_dir=DATA_DIR),
            ]

        trainer = pl.Trainer(
            accelerator="auto",
            enable_checkpointing=True,
            callbacks=callbacks,
            logger=True,
            precision="16-mixed",  # Usar precisión mixta para acelerar
            max_epochs=15,
            gradient_clip_val=1.0,  # Añadir clipping de gradientes
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
        else:
            model = mf_fair.MFWithFairPretraining.load_from_checkpoint(
                f"models/{MODEL_NAME}/checkpoints/{dataset_name}/split_{SPLIT}/best-model-{EMBEDDING_DIM}.ckpt"
            )

        # Testing
        trainer.test(model, datamodule=data_module)
        
    finally:
        # Stop the tracker and add emissions to metrics
        emissions = tracker.stop()
        if system_metrics.test_metrics:
            system_metrics.test_metrics[-1]['co2_emissions_kg'] = emissions
            print(f"\nTotal CO2 Emissions: {emissions:.6f} kg")
            
            # Log emissions
            if trainer.logger:
                trainer.logger.log_metrics({
                    'environment/co2_emissions_kg': emissions,
                    'environment/energy_consumed_kwh': tracker._total_energy.kWh
                })

if __name__ == "__main__":
    train_MF(verbose=1, fair=0)