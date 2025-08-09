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

DATA_DIR = "C:/Users/xpati/Documents/TFG/ml-1m"

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

class EmissionsMetricsCallback(pl.Callback):
    def __init__(self, data_dir=None):
        super().__init__()
        # Fairness visualization related
        if data_dir:
            self.visualizer = fairness.EccentricityVisualizer(data_dir=data_dir)
        self.train_data = None
        self.test_results = {
            'user_ids': [],
            'item_ids': [],
            'ratings': [],
            'predictions': [],
            'genders': []
        }
        
        # System metrics related
        self.train_start_time = None
        self.test_start_time = None
        self.train_metrics = []
        self.test_metrics = []
        
        # Emissions related
        self.epoch_emissions = []
        self.cumulative_emissions = []  # Nueva lista para emisiones acumulativas
        self.epoch_rmse = []
        self.epoch_recall = []
        self.epoch_ndcg = []
        self.total_emissions = 0.0
        self.emission_tracker = None
        self.trackers = {}
        
        # Val data collection
        self.val_results = {
            'user_ids': [],
            'item_ids': [],
            'ratings': [],
            'predictions': [],
            'genders': []
        }
        
    def on_train_start(self, trainer, pl_module):
        self.train_start_time = time.time()
        self.train_data = trainer.datamodule.train_df
        
        # Crear directorio para emisiones
        os.makedirs('emissions_reports', exist_ok=True)
        os.makedirs('emissions_plots', exist_ok=True)
        
        # Usar split_str para el nombre del proyecto
        split_str = "full" if getattr(trainer.datamodule, 'use_all_data', False) else f"split{SPLIT}"
        
        # Inicializar tracker general
        self.emission_tracker = EmissionsTracker(
            project_name=f"MF_{DATASET}_{split_str}_total",
            output_dir="emissions_reports",
            save_to_file=True,
            log_level="error",
            allow_multiple_runs=True
        )
        try:
            self.emission_tracker.start()
        except Exception as e:
            print(f"Warning: Could not start main emissions tracker: {e}")
            self.emission_tracker = None
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        
        # System metrics tracking
        self.current_epoch_metrics = {
            'epoch': trainer.current_epoch,
            'memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'cpu_usage_percent': psutil.cpu_percent(),
        }
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.current_epoch_metrics['gpu_usage_mb'] = 0
        
        # Usar split_str para el nombre del proyecto
        split_str = "full" if getattr(trainer.datamodule, 'use_all_data', False) else f"split{SPLIT}"
        
        # Emissions tracking per epoch
        epoch = trainer.current_epoch
        self.trackers[epoch] = EmissionsTracker(
            project_name=f"MF_{DATASET}_{split_str}_epoch{epoch}",
            output_dir="emissions_reports",
            save_to_file=True,
            log_level="error",
            allow_multiple_runs=True
        )
        try:
            self.trackers[epoch].start()
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
        
        # 1. Collect system metrics
        self.current_epoch_metrics['epoch_time_sec'] = time.time() - self.epoch_start_time
        self.train_metrics.append(self.current_epoch_metrics)
        
        # 2. Get emissions for this epoch
        epoch_co2 = 0.0
        if epoch in self.trackers and self.trackers[epoch]:
            try:
                epoch_co2 = self.trackers[epoch].stop() or 0.0
            except Exception as e:
                print(f"Warning: Error stopping tracker for epoch {epoch}: {e}")
                epoch_co2 = 0.0
        
        # Acumular emisiones totales
        self.total_emissions += epoch_co2
        self.current_epoch_metrics['co2_emissions_kg'] = epoch_co2
        self.current_epoch_metrics['cumulative_co2_emissions_kg'] = self.total_emissions
        
        # 3. Calculate RMSE and recommendation metrics on validation set
        val_metrics = self.calculate_epoch_metrics(trainer, pl_module)
        val_rmse = val_metrics['rmse']
        val_recall = val_metrics['recall']
        val_ndcg = val_metrics['ndcg']
        
        # 4. Store metrics for this epoch
        self.epoch_emissions.append(epoch_co2)  # Emisiones de esta época
        self.cumulative_emissions.append(self.total_emissions)  # Emisiones acumuladas hasta esta época
        self.epoch_rmse.append(val_rmse)
        self.epoch_recall.append(val_recall)
        self.epoch_ndcg.append(val_ndcg)
        
        # 5. Log all metrics
        metrics_to_log = {
            'time/epoch_time_sec': self.current_epoch_metrics['epoch_time_sec'],
            'memory/memory_usage_mb': self.current_epoch_metrics['memory_usage_mb'],
            'cpu/cpu_usage_percent': self.current_epoch_metrics['cpu_usage_percent'],
            'emissions/epoch_co2_kg': epoch_co2,
            'emissions/cumulative_co2_kg': self.total_emissions,  # Log de emisiones acumuladas
            'metrics/epoch_rmse': val_rmse,
            'metrics/epoch_recall': val_recall,
            'metrics/epoch_ndcg': val_ndcg,
            # También log con nombres para checkpointing
            'val_rmse': val_rmse,
            'val_recall': val_recall,
            'val_ndcg': val_ndcg,
        }
        
        if 'gpu_usage_mb' in self.current_epoch_metrics:
            metrics_to_log['gpu/gpu_usage_mb'] = self.current_epoch_metrics['gpu_usage_mb']
        
        pl_module.log_dict(metrics_to_log)
        
        # 6. Print epoch summary
        print(f"\nEpoch {self.current_epoch_metrics['epoch']} Metrics:")
        print(f"  Time: {self.current_epoch_metrics['epoch_time_sec']:.2f}s")
        print(f"  Memory: {self.current_epoch_metrics['memory_usage_mb']:.2f}MB")
        print(f"  CPU: {self.current_epoch_metrics['cpu_usage_percent']:.1f}%")
        print(f"  RMSE: {val_rmse:.4f}")
        print(f"  Recall@5: {val_recall:.4f}")
        print(f"  NDCG@5: {val_ndcg:.4f}")
        print(f"  Epoch CO2: {epoch_co2:.6f} kg")
        print(f"  Total CO2: {self.total_emissions:.6f} kg")  # Mostrar emisiones acumuladas
        if 'gpu_usage_mb' in self.current_epoch_metrics:
            print(f"  GPU: {self.current_epoch_metrics['gpu_usage_mb']:.2f}MB")
            
    
    def calculate_epoch_metrics(self, trainer, pl_module):
        """Calculate RMSE, Recall@K, and NDCG@K on validation set for current epoch"""
        pl_module.eval()
        dataloader = trainer.datamodule.val_dataloader()
        
        all_preds = []
        all_targets = []
        all_user_ids = []
        
        with torch.no_grad():
            for batch in dataloader:
                user_ids, item_ids, ratings, _ = batch
                user_ids = user_ids.to(pl_module.device)
                item_ids = item_ids.to(pl_module.device)
                ratings = ratings.to(pl_module.device)
                
                preds = pl_module(user_ids, item_ids)
                
                all_preds.append(preds.cpu())
                all_targets.append(ratings.cpu())
                all_user_ids.append(user_ids.cpu())
        
        # Concatenate results
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        user_ids = torch.cat(all_user_ids)
        
        # Calculate RMSE
        if preds.dim() > 1 and preds.size(1) == 1:
            preds_flat = preds.squeeze()
        else:
            preds_flat = preds
            
        rmse = torch.sqrt(torch.mean((preds_flat - targets) ** 2)).item()
        
        # Calculate top-k metrics
        threshold = 3.0  # Umbral para convertir ratings a relevancia binaria
        binary_targets = (targets >= threshold).float()
        
        # Prepare for top-k calculations
        if preds.dim() == 1:
            preds = preds.unsqueeze(1)
        
        if binary_targets.dim() == 1:
            binary_targets = binary_targets.unsqueeze(1)
        
        # Ensure sizes match for top-k operations
        k = min(5, preds.size(1))
        
        # If binary_targets only has one column but we need multiple for gather
        if binary_targets.size(1) == 1 and preds.size(1) > 1:
            binary_targets = binary_targets.expand(-1, preds.size(1))
        
        # Calculate top-k
        _, top_k_indices = torch.topk(preds, k=k, dim=1)
        
        # Gather predictions and targets for top-k items
        preds_top_k = torch.gather(preds, dim=1, index=top_k_indices)
        targets_top_k = torch.gather(binary_targets, dim=1, index=top_k_indices)
        
        # Expand user_ids for indexes
        user_ids_expanded = user_ids.unsqueeze(1).expand(-1, preds_top_k.size(1))
        
        # Calculate metrics
        recall_at_k = RetrievalRecall(top_k=k)
        ndcg_at_k = RetrievalNormalizedDCG(top_k=k)
        
        recall = recall_at_k(preds_top_k, targets_top_k, indexes=user_ids_expanded).item()
        ndcg = ndcg_at_k(preds_top_k, targets_top_k, indexes=user_ids_expanded).item()
        
        pl_module.train()
        
        return {
            'rmse': rmse,
            'recall': recall,
            'ndcg': ndcg
        }
    
    def on_train_end(self, trainer, pl_module):
        # 1. Stop main emission tracker
        if self.emission_tracker:
            try:
                self.emission_tracker.stop()
            except Exception as e:
                print(f"Warning: Error stopping main tracker: {e}")
        
        # 2. Make sure all epoch trackers are stopped
        for epoch, tracker in self.trackers.items():
            if tracker:
                try:
                    tracker.stop()
                except:
                    pass
        
        # 3. Create dataframe with all metrics
        emission_data = pd.DataFrame({
            'epoch': range(len(self.epoch_emissions)),
            'epoch_emissions_kg': self.epoch_emissions,
            'cumulative_emissions_kg': self.cumulative_emissions,  # Emisiones acumuladas
            'rmse': self.epoch_rmse,
            'recall': self.epoch_recall,
            'ndcg': self.epoch_ndcg
        })
        
        # 4. Save metrics to CSV
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        emission_data.to_csv(f'emissions_reports/emissions_metrics_{timestamp}.csv', index=False)
        
        # 5. Generate visualization plots
        self.plot_emissions_vs_metrics(timestamp)
    
    def plot_emissions_vs_metrics(self, timestamp):
        """Generate plots for emissions vs different metrics"""
        
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
        plt.savefig(f'emissions_plots/cumulative_emissions_vs_rmse_{timestamp}.png')
        plt.close()
        
        # 2. Emisiones acumulativas vs Recall
        plt.figure(figsize=(10, 6))
        plt.plot(self.cumulative_emissions, self.epoch_recall, 'g-', marker='o')
        
        # Añadir etiquetas con el número de época
        for i, (emissions, recall) in enumerate(zip(self.cumulative_emissions, self.epoch_recall)):
            plt.annotate(f"{i}", (emissions, recall), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
            
        plt.xlabel('Emisiones de CO2 acumuladas (kg)')
        plt.ylabel('Recall@5')
        plt.title('Relación entre Emisiones Acumuladas y Recall')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'emissions_plots/cumulative_emissions_vs_recall_{timestamp}.png')
        plt.close()
        
        # 3. Emisiones acumulativas vs NDCG
        plt.figure(figsize=(10, 6))
        plt.plot(self.cumulative_emissions, self.epoch_ndcg, 'c-', marker='o')
        
        # Añadir etiquetas con el número de época
        for i, (emissions, ndcg) in enumerate(zip(self.cumulative_emissions, self.epoch_ndcg)):
            plt.annotate(f"{i}", (emissions, ndcg), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
            
        plt.xlabel('Emisiones de CO2 acumuladas (kg)')
        plt.ylabel('NDCG@5')
        plt.title('Relación entre Emisiones Acumuladas y NDCG')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'emissions_plots/cumulative_emissions_vs_ndcg_{timestamp}.png')
        plt.close()
        
        # 4. Gráfica combinada: Todas las métricas vs emisiones acumuladas
        plt.figure(figsize=(12, 10))
        
        # Normalizar los valores para que se puedan comparar en la misma escala
        rmse_norm = [r / max(self.epoch_rmse) for r in self.epoch_rmse]
        recall_norm = [r / max(self.epoch_recall) if max(self.epoch_recall) > 0 else 0 for r in self.epoch_recall]
        ndcg_norm = [n / max(self.epoch_ndcg) if max(self.epoch_ndcg) > 0 else 0 for n in self.epoch_ndcg]
        
        plt.plot(self.cumulative_emissions, rmse_norm, 'b-', marker='o', label='RMSE (normalizado)')
        plt.plot(self.cumulative_emissions, recall_norm, 'g-', marker='^', label='Recall@5 (normalizado)')
        plt.plot(self.cumulative_emissions, ndcg_norm, 'c-', marker='s', label='NDCG@5 (normalizado)')
        
        # Añadir etiquetas con el número de época
        for i, emissions in enumerate(self.cumulative_emissions):
            plt.annotate(f"{i}", (emissions, rmse_norm[i]), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
        
        plt.xlabel('Emisiones de CO2 acumuladas (kg)')
        plt.ylabel('Métricas Normalizadas')
        plt.title('Relación entre Emisiones Acumuladas y Métricas de Rendimiento')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'emissions_plots/cumulative_emissions_vs_all_metrics_{timestamp}.png')
        plt.close()
        
        # 5. También mantener los gráficos originales por época para completitud
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        plt.plot(range(len(self.epoch_emissions)), self.epoch_emissions, 'r-', marker='x')
        plt.title('Emisiones por Época')
        plt.xlabel('Época')
        plt.ylabel('CO2 Emissions (kg)')
        
        plt.subplot(2, 2, 2)
        plt.plot(range(len(self.cumulative_emissions)), self.cumulative_emissions, 'r-', marker='o')
        plt.title('Emisiones Acumuladas por Época')
        plt.xlabel('Época')
        plt.ylabel('CO2 Emissions (kg)')
        
        plt.subplot(2, 2, 3)
        plt.plot(range(len(self.epoch_rmse)), self.epoch_rmse, 'b-', marker='o')
        plt.title('RMSE por Época')
        plt.xlabel('Época')
        plt.ylabel('RMSE')
        
        plt.subplot(2, 2, 4)
        plt.plot(range(len(self.epoch_ndcg)), self.epoch_ndcg, 'c-', marker='o')
        plt.title('NDCG@5 por Época')
        plt.xlabel('Época')
        plt.ylabel('NDCG@5')
        
        plt.tight_layout()
        plt.savefig(f'emissions_plots/metrics_by_epoch_{timestamp}.png')
        plt.close()
        
        # 6. Scatter plot de rendimiento frente a emisiones acumulativas
        plt.figure(figsize=(10, 6))
        
        # Ajustar tamaño de los puntos según la época
        sizes = [(i+1)*20 for i in range(len(self.cumulative_emissions))]
        
        scatter1 = plt.scatter(self.epoch_rmse, self.cumulative_emissions, 
                    label='RMSE', color='blue', marker='o', s=sizes, alpha=0.7)
        scatter2 = plt.scatter(self.epoch_recall, self.cumulative_emissions, 
                    label='Recall@5', color='green', marker='^', s=sizes, alpha=0.7)
        scatter3 = plt.scatter(self.epoch_ndcg, self.cumulative_emissions, 
                    label='NDCG@5', color='cyan', marker='s', s=sizes, alpha=0.7)
        
        # Añadir etiquetas de época
        for i, (rmse, em) in enumerate(zip(self.epoch_rmse, self.cumulative_emissions)):
            plt.annotate(f"{i}", (rmse, em), textcoords="offset points", 
                        xytext=(0,5), ha='center', fontsize=9)
        
        plt.ylabel('Emisiones de CO2 acumuladas (kg)')
        plt.xlabel('Métrica de Rendimiento')
        plt.title('Relación entre Métricas de Rendimiento y Emisiones Acumuladas')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Añadir leyenda para tamaño de puntos
        handles, labels = plt.gca().get_legend_handles_labels()
        legend1 = plt.legend(handles, labels, loc='upper left')
        plt.gca().add_artist(legend1)
        
        # Crear leyenda personalizada para las épocas
        plt.savefig(f'emissions_plots/cumulative_emissions_performance_scatter_{timestamp}.png')
        plt.close()
    
    def on_test_start(self, trainer, pl_module):
        self.test_start_time = time.time()
        if not hasattr(self, 'train_start_time') or self.train_start_time is None:
            self.train_start_time = time.time()
        
        # Reset test results
        self.test_results = {k: [] for k in self.test_results}
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        user_ids, item_ids, ratings, genders = batch
        
        # Ensure we have predictions
        if isinstance(outputs, dict):
            predictions = outputs.get("predictions", None)
            if predictions is None and len(outputs) > 0:
                first_key = next(iter(outputs))
                predictions = outputs[first_key]
        else:
            predictions = outputs
        
        if predictions is not None:
            self.test_results['user_ids'].append(user_ids.cpu())
            self.test_results['item_ids'].append(item_ids.cpu())
            self.test_results['ratings'].append(ratings.cpu())
            self.test_results['predictions'].append(predictions.cpu())
            self.test_results['genders'].append(genders.cpu())
    
    def on_test_end(self, trainer, pl_module):
        # Process test results
        if any(len(v) == 0 for k, v in self.test_results.items()):
            print("No hay resultados de prueba para procesar.")
            return
        
        # Concatenate results
        test_results = {k: torch.cat(v) for k, v in self.test_results.items()}
        
        # Calculate test metrics
        preds = test_results['predictions']
        targets = test_results['ratings']
        user_ids = test_results['user_ids']
        
        # Calculate RMSE
        if preds.dim() > 1 and preds.size(1) == 1:
            preds_flat = preds.squeeze()
        else:
            preds_flat = preds
            
        rmse = torch.sqrt(torch.mean((preds_flat - targets) ** 2))
        
        # Calculate top-k metrics
        threshold = 3.0
        binary_targets = (targets >= threshold).float()
        
        top_k = 5  # You can change this to match your requirements
        
        # Prepare data for top-k calculations
        if preds.dim() == 1:
            preds = preds.unsqueeze(1)
        
        if binary_targets.dim() == 1:
            binary_targets = binary_targets.unsqueeze(1)
        
        if binary_targets.size(1) == 1 and preds.size(1) > 1:
            binary_targets = binary_targets.expand(-1, preds.size(1))
        
        # Calculate top-k indices
        _, top_k_indices = torch.topk(preds, k=min(top_k, preds.size(1)), dim=1)
        
        # Gather predictions and targets for top-k items
        preds_top_k = torch.gather(preds, dim=1, index=top_k_indices)
        targets_top_k = torch.gather(binary_targets, dim=1, index=top_k_indices)
        
        # Expand user_ids for indexes
        user_ids_expanded = user_ids.unsqueeze(1).expand(-1, preds_top_k.size(1))
        
        # Calculate metrics
        recall_at_k = RetrievalRecall(top_k=top_k)
        ndcg_at_k = RetrievalNormalizedDCG(top_k=top_k)
        
        recall = recall_at_k(preds_top_k, targets_top_k, indexes=user_ids_expanded)
        ndcg = ndcg_at_k(preds_top_k, targets_top_k, indexes=user_ids_expanded)
        
        # Calculate timing
        test_time = time.time() - self.test_start_time
        total_time = time.time() - self.train_start_time
        
        # Log final metrics
        final_metrics = {
            'time/total_time_sec': total_time,
            'time/test_time_sec': test_time,
            'metrics/test_rmse': rmse.item(),
            'metrics/test_recall@5': recall.item(),
            'metrics/test_ndcg@5': ndcg.item(),
        }
        
        if torch.cuda.is_available():
            final_metrics['gpu/final_gpu_usage_mb'] = torch.cuda.max_memory_allocated() / 1024 ** 2
        
        self.test_metrics.append(final_metrics)
        
        # Print final summary
        print(f"\n=== Final Test Metrics ===")
        print(f"Total Time: {total_time:.2f}s (Test: {test_time:.2f}s)")
        print(f"RMSE: {rmse.item():.4f}")
        print(f"Recall@{top_k}: {recall.item():.4f}")
        print(f"NDCG@{top_k}: {ndcg.item():.4f}")
        
        # Log metrics to the logger
        if trainer.logger:
            trainer.logger.log_metrics(final_metrics)

def train_MF(
    dataset_name=DATASET,
    embedding_dim=EMBEDDING_DIM,
    batch_size=1024,
    num_workers=4,
    l2_reg=1e-2,
    learning_rate=1e-3,
    verbose=1,
    fair=0,
    use_all_data=USE_ALL_DATA,
):
    # Use the global DATA_DIR
    data_dir = DATA_DIR
    
    # Si dataset_name contiene una ruta completa y use_all_data es True,
    # extraemos el nombre del dataset y ajustamos el data_dir
    if use_all_data and os.path.isabs(dataset_name):
        # Usar el directorio que contiene los datos como data_dir
        data_dir = os.path.dirname(dataset_name)
        # Usar solo el nombre del directorio como nombre del dataset
        dataset_name = os.path.basename(dataset_name)
    
    # Definir nombre del split para directorios y nombres de proyecto
    split_str = "full" if use_all_data else f"split_{SPLIT}"
    
    # Initialize combined metrics callback
    emissions_metrics = EmissionsMetricsCallback(data_dir=data_dir)
    
    # Initialize main CodeCarbon tracker
    tracker = EmissionsTracker(
        project_name=f"MF_{dataset_name}_{split_str}",
        output_dir="emissions_reports",
        measure_power_secs=10,
        save_to_file=True,
        log_level="error",
        allow_multiple_runs=True
    )
    
    try:
        # Start main tracker
        try:
            tracker.start()
        except Exception as e:
            print(f"Warning: Could not start main tracker: {e}")
            tracker = None
        
        # Load the dataset
        data_module = datamodule.DyadicRegressionDataModule(
            dataset_name=dataset_name,
            split=None if use_all_data else SPLIT,  # Pasar None si use_all_data es True
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose,
            data_dir=data_dir,
            use_all_data=use_all_data,  # Pasar la nueva opción
        )

        MODEL_NAME = "MF" if not fair else "MF_fair"

        # Create the model
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
        checkpoint_path = f"models/{MODEL_NAME}/checkpoints/{dataset_name}/{split_str}/best-model-{EMBEDDING_DIM}.ckpt"
        if path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        # Configure callbacks
        callbacks = [
            emissions_metrics,
            pl.callbacks.ModelCheckpoint(
                dirpath=f"models/{MODEL_NAME}/checkpoints/{dataset_name}/{split_str}",
                filename=f"best-model-{EMBEDDING_DIM}",
                monitor="val_rmse",
                mode="min",
                save_weights_only=True,
                enable_version_counter=False,
            ),
        ]

        # Configure trainer
        trainer = pl.Trainer(
            accelerator="auto",
            enable_checkpointing=True,
            callbacks=callbacks,
            logger=True,
            precision="16-mixed",
            max_epochs=50,
            gradient_clip_val=1.0,
            limit_train_batches=0.50,
            limit_val_batches=0.2,
        )
        
        # Show gender distribution if verbose
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
                f"models/{MODEL_NAME}/checkpoints/{dataset_name}/{split_str}/best-model-{EMBEDDING_DIM}.ckpt"
            )
        else:
            model = mf_fair.MFWithFairPretraining.load_from_checkpoint(
                f"models/{MODEL_NAME}/checkpoints/{dataset_name}/{split_str}/best-model-{EMBEDDING_DIM}.ckpt"
            )

        # Testing
        trainer.test(model, datamodule=data_module)
        
    finally:
        # Stop the tracker and add emissions to metrics
        emissions = 0.0
        if tracker:
            try:
                emissions = tracker.stop() or 0.0
            except Exception as e:
                print(f"Warning: Error stopping main tracker: {e}")
                emissions = 0.0
        
        if emissions_metrics.test_metrics:
            emissions_metrics.test_metrics[-1]['co2_emissions_kg'] = emissions
            print(f"\nTotal CO2 Emissions: {emissions:.6f} kg")
            
            # Log emissions
            if trainer.logger:
                trainer.logger.log_metrics({
                    'environment/co2_emissions_kg': emissions,
                    'environment/energy_consumed_kwh': tracker._total_energy.kWh if hasattr(tracker, '_total_energy') else 0.0
                })

if __name__ == "__main__":
    train_MF(verbose=1, fair=0, use_all_data=USE_ALL_DATA)