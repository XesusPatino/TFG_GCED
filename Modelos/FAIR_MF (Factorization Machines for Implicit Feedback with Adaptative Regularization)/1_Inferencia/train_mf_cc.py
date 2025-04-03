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

class SystemMetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.train_metrics = []
        self.test_metrics = []
        self.current_epoch_metrics = {}
        
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
        
        # Convert ratings to binary relevance (1 if rating >= threshold, else 0)
        threshold = 3.0  # Example threshold for 1-5 rating scale
        binary_targets = (targets >= threshold).float()
        
        # Create metric instances
        recall_at_10 = RetrievalRecall(top_k=10)
        recall_at_50 = RetrievalRecall(top_k=50)
        ndcg_at_10 = RetrievalNormalizedDCG(top_k=10)
        ndcg_at_50 = RetrievalNormalizedDCG(top_k=50)
        
        # Calculate metrics
        recall_10 = recall_at_10(preds, binary_targets, indexes=indexes)
        recall_50 = recall_at_50(preds, binary_targets, indexes=indexes)
        ndcg_10 = ndcg_at_10(preds, targets, indexes=indexes)
        ndcg_50 = ndcg_at_50(preds, targets, indexes=indexes)
        
        # Final system metrics
        final_metrics = {
            'time/total_time_sec': total_time,
            'time/test_time_sec': test_time,
            'memory/final_memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'cpu/final_cpu_usage_percent': psutil.cpu_percent(),
            'metrics/recall@10': recall_10.item(),
            'metrics/recall@50': recall_50.item(),
            'metrics/ndcg@10': ndcg_10.item(),
            'metrics/ndcg@50': ndcg_50.item(),
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
                  f"Memory={m['memory_usage_mb']:.2f}MB, CPU={m['cpu_usage_percent']:.1f}%", 
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
        print(f"Recall@10: {recall_10:.4f}")
        print(f"Recall@50: {recall_50:.4f}")
        print(f"NDCG@10: {ndcg_10:.4f}")
        print(f"NDCG@50: {ndcg_50:.4f}")

def train_MF(
    dataset_name=DATASET,
    embedding_dim=EMBEDDING_DIM,
    batch_size=2**12,
    num_workers=4,
    l2_reg=1e-2,
    learning_rate=5e-4,
    verbose=1,
    fair=0,
):
    # Initialize system metrics callback
    system_metrics = SystemMetricsCallback()
    
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
            system_metrics,
            pl.callbacks.ModelCheckpoint(
                dirpath=f"models/{MODEL_NAME}/checkpoints/{dataset_name}/split_{SPLIT}",
                filename=f"best-model-{EMBEDDING_DIM}",
                monitor="val_rmse",
                mode="min",
                save_weights_only=True,
                enable_version_counter=False,
            ),
            fairness.EOgenreRelationCallback(),
            fairness.EccentricityGCallback(data_module.train_df, SPLIT),
        ]

        trainer = pl.Trainer(
            accelerator="auto",
            enable_checkpointing=True,
            callbacks=callbacks,
            logger=True,
            precision=16,
            enable_model_summary=True,
            enable_progress_bar=True,
            max_epochs=15,
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