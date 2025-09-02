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
        self.current_epoch_metrics = {}
        
        # Best metrics tracking
        self.best_rmse = float('inf')
        self.best_rmse_epoch = None
        self.best_rmse_metrics = None
        
        # Emissions related
        self.epoch_emissions = []
        self.cumulative_emissions = []  # Nueva lista para emisiones acumulativas
        self.epoch_rmse = []
        self.epoch_recall_5 = []
        self.epoch_recall_10 = []
        self.epoch_recall_20 = []
        self.epoch_recall_50 = []
        self.epoch_ndcg_5 = []
        self.epoch_ndcg_10 = []
        self.epoch_ndcg_20 = []
        self.epoch_ndcg_50 = []
        self.total_emissions = 0.0
        self.emission_tracker = None
        self.trackers = {}
        self.best_rmse_emissions = None
        self.best_rmse_cumulative_emissions = None
        
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
        
        # Track best RMSE for later reporting
        if val_rmse < self.best_rmse:
            self.best_rmse = val_rmse
            self.best_rmse_epoch = epoch
            self.best_rmse_metrics = self.current_epoch_metrics.copy()
            self.best_rmse_emissions = epoch_co2
            self.best_rmse_cumulative_emissions = self.total_emissions
        
        # 4. Store metrics for this epoch
        self.epoch_emissions.append(epoch_co2)  # Emisiones de esta época
        self.cumulative_emissions.append(self.total_emissions)  # Emisiones acumuladas hasta esta época
        self.epoch_rmse.append(val_rmse)
        self.epoch_recall_5.append(val_metrics['recall_5'])
        self.epoch_recall_10.append(val_metrics['recall_10'])
        self.epoch_recall_20.append(val_metrics['recall_20'])
        self.epoch_recall_50.append(val_metrics['recall_50'])
        self.epoch_ndcg_5.append(val_metrics['ndcg_5'])
        self.epoch_ndcg_10.append(val_metrics['ndcg_10'])
        self.epoch_ndcg_20.append(val_metrics['ndcg_20'])
        self.epoch_ndcg_50.append(val_metrics['ndcg_50'])
        
        # 5. Log all metrics
        metrics_to_log = {
            'time/epoch_time_sec': self.current_epoch_metrics['epoch_time_sec'],
            'memory/memory_usage_mb': self.current_epoch_metrics['memory_usage_mb'],
            'cpu/cpu_usage_percent': self.current_epoch_metrics['cpu_usage_percent'],
            'emissions/epoch_co2_kg': epoch_co2,
            'emissions/cumulative_co2_kg': self.total_emissions,  # Log de emisiones acumuladas
            'metrics/epoch_rmse': val_rmse,
            'metrics/epoch_recall_5': val_metrics['recall_5'],
            'metrics/epoch_recall_10': val_metrics['recall_10'],
            'metrics/epoch_recall_20': val_metrics['recall_20'],
            'metrics/epoch_recall_50': val_metrics['recall_50'],
            'metrics/epoch_ndcg_5': val_metrics['ndcg_5'],
            'metrics/epoch_ndcg_10': val_metrics['ndcg_10'],
            'metrics/epoch_ndcg_20': val_metrics['ndcg_20'],
            'metrics/epoch_ndcg_50': val_metrics['ndcg_50'],
            # También log con nombres para checkpointing
            'val_rmse': val_rmse,
            'val_recall': val_metrics['recall_5'],  # Use recall@5 for checkpointing
            'val_ndcg': val_metrics['ndcg_5'],      # Use ndcg@5 for checkpointing
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
        print(f"  Recall@5: {val_metrics['recall_5']:.4f}")
        print(f"  Recall@10: {val_metrics['recall_10']:.4f}")
        print(f"  NDCG@5: {val_metrics['ndcg_5']:.4f}")
        print(f"  NDCG@10: {val_metrics['ndcg_10']:.4f}")
        print(f"  Epoch CO2: {epoch_co2:.6f} kg")
        print(f"  Total CO2: {self.total_emissions:.6f} kg")  # Mostrar emisiones acumuladas
        if 'gpu_usage_mb' in self.current_epoch_metrics:
            print(f"  GPU: {self.current_epoch_metrics['gpu_usage_mb']:.2f}MB")
            
    
    def calculate_epoch_metrics(self, trainer, pl_module):
        """Calculate RMSE, Recall@K, and NDCG@K on validation set for current epoch"""
        pl_module.eval()
        dataloader = trainer.datamodule.val_dataloader()
        
        # Collect predictions by user for proper top-k calculation
        user_predictions = {}
        user_ratings = {}
        
        with torch.no_grad():
            for batch in dataloader:
                user_ids, item_ids, ratings, _ = batch
                user_ids = user_ids.to(pl_module.device)
                item_ids = item_ids.to(pl_module.device)
                ratings = ratings.to(pl_module.device)
                
                preds = pl_module(user_ids, item_ids)
                
                # Flatten predictions if needed
                if preds.dim() > 1 and preds.size(1) == 1:
                    preds = preds.squeeze()
                
                # Group by user
                for i in range(len(user_ids)):
                    uid = user_ids[i].item()
                    iid = item_ids[i].item()
                    pred = preds[i].item()
                    rating = ratings[i].item()
                    
                    if uid not in user_predictions:
                        user_predictions[uid] = []
                        user_ratings[uid] = []
                    
                    user_predictions[uid].append((iid, pred))
                    user_ratings[uid].append((iid, rating))
        
        # Calculate RMSE first
        all_preds = []
        all_targets = []
        for uid in user_predictions:
            for (iid, pred), (_, rating) in zip(user_predictions[uid], user_ratings[uid]):
                all_preds.append(pred)
                all_targets.append(rating)
        
        rmse = np.sqrt(np.mean([(p - t) ** 2 for p, t in zip(all_preds, all_targets)]))
        
        # Calculate top-k metrics for multiple k values
        k_values = [5, 10, 20, 50]
        threshold = 3.0  # Threshold for binary relevance
        
        recall_results = {}
        ndcg_results = {}
        
        for k in k_values:
            recalls = []
            ndcgs = []
            
            for uid in user_predictions:
                # Sort items by prediction score (descending)
                user_items = sorted(user_predictions[uid], key=lambda x: x[1], reverse=True)
                user_ratings_dict = {iid: rating for iid, rating in user_ratings[uid]}
                
                # Get top-k items
                top_k_items = user_items[:min(k, len(user_items))]
                
                if len(top_k_items) == 0:
                    continue
                
                # Calculate recall@k
                relevant_items = [iid for iid, rating in user_ratings[uid] if rating >= threshold]
                if len(relevant_items) > 0:
                    recommended_relevant = [iid for iid, _ in top_k_items if iid in relevant_items]
                    recall = len(recommended_relevant) / len(relevant_items)
                else:
                    recall = 0.0
                recalls.append(recall)
                
                # Calculate NDCG@k
                dcg = 0.0
                idcg = 0.0
                
                # DCG calculation
                for i, (iid, _) in enumerate(top_k_items):
                    if iid in user_ratings_dict:
                        rating = user_ratings_dict[iid]
                        if rating >= threshold:
                            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
                
                # IDCG calculation (ideal ranking)
                ideal_ratings = sorted([rating for _, rating in user_ratings[uid]], reverse=True)
                for i, rating in enumerate(ideal_ratings[:k]):
                    if rating >= threshold:
                        idcg += 1.0 / np.log2(i + 2)
                
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcgs.append(ndcg)
            
            recall_results[k] = np.mean(recalls) if recalls else 0.0
            ndcg_results[k] = np.mean(ndcgs) if ndcgs else 0.0
        
        pl_module.train()
        
        return {
            'rmse': rmse,
            'recall_5': recall_results[5],
            'recall_10': recall_results[10],
            'recall_20': recall_results[20],
            'recall_50': recall_results[50],
            'ndcg_5': ndcg_results[5],
            'ndcg_10': ndcg_results[10],
            'ndcg_20': ndcg_results[20],
            'ndcg_50': ndcg_results[50]
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
            'recall_5': self.epoch_recall_5,
            'recall_10': self.epoch_recall_10,
            'recall_20': self.epoch_recall_20,
            'recall_50': self.epoch_recall_50,
            'ndcg_5': self.epoch_ndcg_5,
            'ndcg_10': self.epoch_ndcg_10,
            'ndcg_20': self.epoch_ndcg_20,
            'ndcg_50': self.epoch_ndcg_50
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
        plt.plot(self.cumulative_emissions, self.epoch_recall_5, 'g-', marker='o')
        
        # Añadir etiquetas con el número de época
        for i, (emissions, recall) in enumerate(zip(self.cumulative_emissions, self.epoch_recall_5)):
            plt.annotate(f"{i}", (emissions, recall), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
            
        plt.xlabel('Emisiones de CO2 acumuladas (kg)')
        plt.ylabel('Recall@5')
        plt.title('Relación entre Emisiones Acumuladas y Recall@5')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'emissions_plots/cumulative_emissions_vs_recall_{timestamp}.png')
        plt.close()
        
        # 3. Emisiones acumulativas vs NDCG
        plt.figure(figsize=(10, 6))
        plt.plot(self.cumulative_emissions, self.epoch_ndcg_5, 'c-', marker='o')
        
        # Añadir etiquetas con el número de época
        for i, (emissions, ndcg) in enumerate(zip(self.cumulative_emissions, self.epoch_ndcg_5)):
            plt.annotate(f"{i}", (emissions, ndcg), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
            
        plt.xlabel('Emisiones de CO2 acumuladas (kg)')
        plt.ylabel('NDCG@5')
        plt.title('Relación entre Emisiones Acumuladas y NDCG@5')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'emissions_plots/cumulative_emissions_vs_ndcg_{timestamp}.png')
        plt.close()
        
        # 4. Gráfica combinada: Todas las métricas vs emisiones acumuladas
        plt.figure(figsize=(12, 10))
        
        # Normalizar los valores para que se puedan comparar en la misma escala
        rmse_norm = [r / max(self.epoch_rmse) for r in self.epoch_rmse]
        recall_norm = [r / max(self.epoch_recall_5) if max(self.epoch_recall_5) > 0 else 0 for r in self.epoch_recall_5]
        ndcg_norm = [n / max(self.epoch_ndcg_5) if max(self.epoch_ndcg_5) > 0 else 0 for n in self.epoch_ndcg_5]
        
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
        plt.plot(range(len(self.epoch_ndcg_5)), self.epoch_ndcg_5, 'c-', marker='o')
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
        scatter2 = plt.scatter(self.epoch_recall_5, self.cumulative_emissions, 
                    label='Recall@5', color='green', marker='^', s=sizes, alpha=0.7)
        scatter3 = plt.scatter(self.epoch_ndcg_5, self.cumulative_emissions, 
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
        
        # Calculate comprehensive test metrics using the same approach as validation
        user_predictions = {}
        user_ratings = {}
        
        # Group test results by user
        for i in range(len(test_results['user_ids'])):
            uid = test_results['user_ids'][i].item()
            iid = test_results['item_ids'][i].item()
            pred = test_results['predictions'][i].item() if test_results['predictions'][i].dim() == 0 else test_results['predictions'][i].squeeze().item()
            rating = test_results['ratings'][i].item()
            
            if uid not in user_predictions:
                user_predictions[uid] = []
                user_ratings[uid] = []
            
            user_predictions[uid].append((iid, pred))
            user_ratings[uid].append((iid, rating))
        
        # Calculate RMSE
        all_preds = []
        all_targets = []
        for uid in user_predictions:
            for (iid, pred), (_, rating) in zip(user_predictions[uid], user_ratings[uid]):
                all_preds.append(pred)
                all_targets.append(rating)
        
        rmse = np.sqrt(np.mean([(p - t) ** 2 for p, t in zip(all_preds, all_targets)]))
        
        # Calculate top-k metrics for multiple k values
        k_values = [5, 10, 20, 50]
        threshold = 3.0
        
        test_recall_results = {}
        test_ndcg_results = {}
        
        for k in k_values:
            recalls = []
            ndcgs = []
            
            for uid in user_predictions:
                # Sort items by prediction score (descending)
                user_items = sorted(user_predictions[uid], key=lambda x: x[1], reverse=True)
                user_ratings_dict = {iid: rating for iid, rating in user_ratings[uid]}
                
                # Get top-k items
                top_k_items = user_items[:min(k, len(user_items))]
                
                if len(top_k_items) == 0:
                    continue
                
                # Calculate recall@k
                relevant_items = [iid for iid, rating in user_ratings[uid] if rating >= threshold]
                if len(relevant_items) > 0:
                    recommended_relevant = [iid for iid, _ in top_k_items if iid in relevant_items]
                    recall = len(recommended_relevant) / len(relevant_items)
                else:
                    recall = 0.0
                recalls.append(recall)
                
                # Calculate NDCG@k
                dcg = 0.0
                idcg = 0.0
                
                # DCG calculation
                for i, (iid, _) in enumerate(top_k_items):
                    if iid in user_ratings_dict:
                        rating = user_ratings_dict[iid]
                        if rating >= threshold:
                            dcg += 1.0 / np.log2(i + 2)
                
                # IDCG calculation (ideal ranking)
                ideal_ratings = sorted([rating for _, rating in user_ratings[uid]], reverse=True)
                for i, rating in enumerate(ideal_ratings[:k]):
                    if rating >= threshold:
                        idcg += 1.0 / np.log2(i + 2)
                
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcgs.append(ndcg)
            
            test_recall_results[k] = np.mean(recalls) if recalls else 0.0
            test_ndcg_results[k] = np.mean(ndcgs) if ndcgs else 0.0
        
        # Calculate timing
        test_time = time.time() - self.test_start_time
        total_time = time.time() - self.train_start_time
        
        # Log final metrics
        final_metrics = {
            'time/total_time_sec': total_time,
            'time/test_time_sec': test_time,
            'memory/final_memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'cpu/final_cpu_usage_percent': psutil.cpu_percent(),
            'metrics/test_rmse': rmse,
            'metrics/test_recall@5': test_recall_results[5],
            'metrics/test_recall@10': test_recall_results[10],
            'metrics/test_recall@20': test_recall_results[20],
            'metrics/test_recall@50': test_recall_results[50],
            'metrics/test_ndcg@5': test_ndcg_results[5],
            'metrics/test_ndcg@10': test_ndcg_results[10],
            'metrics/test_ndcg@20': test_ndcg_results[20],
            'metrics/test_ndcg@50': test_ndcg_results[50],
        }
        
        if torch.cuda.is_available():
            final_metrics['gpu/final_gpu_usage_mb'] = torch.cuda.max_memory_allocated() / 1024 ** 2
        
        self.test_metrics.append(final_metrics)
        
        # Print comprehensive final summaries
        print("\n=== Final Training Metrics ===")
        for i, m in enumerate(self.train_metrics):
            print(f"Epoch {m['epoch']}: Time={m['epoch_time_sec']:.2f}s, "
                  f"Memory={m['memory_usage_mb']:.2f}MB, CPU={m['cpu_usage_percent']:.1f}%, "
                  f"RMSE={self.epoch_rmse[i]:.4f}, Recall@5={self.epoch_recall_5[i]:.4f}, "
                  f"Recall@10={self.epoch_recall_10[i]:.4f}, NDCG@5={self.epoch_ndcg_5[i]:.4f}, "
                  f"NDCG@10={self.epoch_ndcg_10[i]:.4f}", end='')
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
        print(f"Recall@5: {test_recall_results[5]:.4f}")
        print(f"Recall@10: {test_recall_results[10]:.4f}")
        print(f"Recall@20: {test_recall_results[20]:.4f}")
        print(f"Recall@50: {test_recall_results[50]:.4f}")
        print(f"NDCG@5: {test_ndcg_results[5]:.4f}")
        print(f"NDCG@10: {test_ndcg_results[10]:.4f}")
        print(f"NDCG@20: {test_ndcg_results[20]:.4f}")
        print(f"NDCG@50: {test_ndcg_results[50]:.4f}")
        
        # Show best training RMSE information
        if self.best_rmse_epoch is not None:
            print(f"\n=== Best Training RMSE ===")
            print(f"Best RMSE: {self.best_rmse:.4f} (Epoch {self.best_rmse_epoch})")
            if self.best_rmse_metrics:
                print(f"Time: {self.best_rmse_metrics['epoch_time_sec']:.2f}s")
                print(f"Memory: {self.best_rmse_metrics['memory_usage_mb']:.2f}MB")
                print(f"CPU: {self.best_rmse_metrics['cpu_usage_percent']:.1f}%")
                if 'gpu_usage_mb' in self.best_rmse_metrics:
                    print(f"GPU: {self.best_rmse_metrics['gpu_usage_mb']:.2f}MB")
            
            print(f"\n=== Best RMSE and Associated Emissions ===")
            print(f"Best RMSE: {self.best_rmse:.4f} (Epoch {self.best_rmse_epoch})")
            if self.best_rmse_emissions is not None:
                print(f"Emissions at best RMSE: {self.best_rmse_emissions:.8f} kg")
            if self.best_rmse_cumulative_emissions is not None:
                print(f"Cumulative emissions at best RMSE: {self.best_rmse_cumulative_emissions:.8f} kg")
        
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