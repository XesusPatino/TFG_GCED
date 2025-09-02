import json
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import random
from tensorflow.keras.utils import Progbar
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import time
import psutil
import pandas as pd
from codecarbon import EmissionsTracker
import math

# Crear directorios para resultados
result_path = "results_lgcn_topk"
os.makedirs(result_path, exist_ok=True)
os.makedirs(f"{result_path}/emissions_reports", exist_ok=True)
os.makedirs(f"{result_path}/emissions_plots", exist_ok=True)

# --- Funciones de Métricas Top-K ---

def recall_at_k(recs, test_data, k=10):
    user_recs = defaultdict(list)
    for (u, i, s) in recs:
        user_recs[u].append(i)
    recalls = []
    for u, items_pred in user_recs.items():
        if len(test_data[u]) == 0:
            continue
        if k > 0:
            items_pred = items_pred[:k]
        gt = test_data[u]
        num_hit = len(set(items_pred).intersection(gt))
        recalls.append(num_hit / float(len(gt)))
    return np.mean(recalls) if recalls else 0.0

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(recommended, ground_truth, k=10):
    rel = [1 if i in ground_truth else 0 for i in recommended[:k]]
    ideal_rel = sorted(rel, reverse=True)
    dcg = dcg_at_k(rel, k)
    idcg = dcg_at_k(ideal_rel, k)
    return (dcg / idcg) if idcg > 0 else 0.0

def ndcg(recs, test_data, k=10):
    user_recs = defaultdict(list)
    for (u, i, s) in recs:
        user_recs[u].append(i)
    ndcgs = []
    for u, items_pred in user_recs.items():
        gt = test_data[u]
        if len(gt) == 0:
            continue
        ndcgs.append(ndcg_at_k(items_pred, gt, k))
    return np.mean(ndcgs) if ndcgs else 0.0

def calculate_rmse(model, users, data, batch_size=2048):
    """Calcula el RMSE para las interacciones positivas (calificación verdadera = 1.0)."""
    all_scores = []
    user_final, item_final, _, _, _ = model((model.user_embedding, model.item_embedding), mask_prob=0.0)
    
    for u in users:
        pos_items = list(data[u])
        if not pos_items:
            continue
        
        u_emb = tf.gather(user_final, [u])
        pos_item_embs = tf.gather(item_final, pos_items)
        
        # Calcula las puntuaciones (producto escalar)
        scores = tf.reduce_sum(u_emb * pos_item_embs, axis=1).numpy()
        all_scores.extend(scores)
        
    if not all_scores:
        return 0.0
        
    # El error es (1.0 - puntuación), ya que la calificación verdadera es 1
    squared_errors = [(1.0 - score)**2 for score in all_scores]
    return np.sqrt(np.mean(squared_errors))

def evaluate_model(model, users, data, ks=[5, 10, 20, 50], batch_size=2048):
    """Evalúa el modelo para un conjunto de usuarios y devuelve Recall y NDCG para cada k."""
    all_recs = []
    idx_start = 0
    max_k = max(ks)
    
    # Generar recomendaciones en lotes para no agotar la memoria
    while idx_start < len(users):
        idx_end = min(idx_start + batch_size, len(users))
        user_batch = users[idx_start:idx_end]
        recs_chunk = model.recommend(user_batch, k=max_k)
        all_recs.extend(recs_chunk)
        idx_start = idx_end

    results = {}
    # Añadir cálculo de RMSE a la evaluación
    rmse = calculate_rmse(model, users, data, batch_size)
    results['RMSE'] = rmse

    for k in ks:
        rec_k = recall_at_k(all_recs, data, k=k)
        ndcg_k = ndcg(all_recs, data, k=k)
        results[f'Recall@{k}'] = rec_k
        results[f'NDCG@{k}'] = ndcg_k
        
    return results

# Clases para seguimiento de métricas
class SystemMetricsTracker:
    def __init__(self):
        self.train_metrics = []
        self.test_metrics = {}
        self.start_time = time.time()
        self.best_ndcg10 = -1.0
        self.best_ndcg10_epoch = None
        self.best_ndcg10_metrics = None
        
    def start_epoch(self, epoch):
        self.epoch_start_time = time.time()
        self.current_epoch_metrics = {
            'epoch': epoch,
            'memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'cpu_usage_percent': psutil.cpu_percent(),
        }
        
    def end_epoch(self, epoch, loss, metrics):
        epoch_time = time.time() - self.epoch_start_time
        self.current_epoch_metrics['epoch_time_sec'] = epoch_time
        self.current_epoch_metrics['loss'] = loss
        self.current_epoch_metrics.update(metrics)
        self.train_metrics.append(self.current_epoch_metrics)
        
        # Rastrear el mejor NDCG@10
        current_ndcg10 = metrics.get('NDCG@10', -1.0)
        if current_ndcg10 > self.best_ndcg10:
            self.best_ndcg10 = current_ndcg10
            self.best_ndcg10_epoch = epoch
            self.best_ndcg10_metrics = self.current_epoch_metrics.copy()
        
        # Imprimir resumen de época con el formato deseado
        print(f"\nEpoch {epoch} Metrics:")
        metrics_str = (
            f"  Time: {epoch_time:.2f}s, "
            f"Memory: {self.current_epoch_metrics['memory_usage_mb']:.2f}MB, "
            f"CPU: {self.current_epoch_metrics['cpu_usage_percent']:.1f}%, "
            f"Loss: {loss:.4f}"
        )
        # Añadir RMSE si está presente
        if 'RMSE' in metrics:
            metrics_str += f", RMSE={metrics['RMSE']:.4f}"
            
        for k, v in metrics.items():
            if k != 'RMSE': # Evitar duplicar la métrica
                metrics_str += f", {k}={v:.4f}"
        print(metrics_str)
        
    def end_test(self, test_metrics_dict):
        self.test_metrics = {
            'test_time_sec': time.time() - self.epoch_start_time,
            'total_time_sec': time.time() - self.start_time,
            'final_memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'final_cpu_usage_percent': psutil.cpu_percent(),
        }
        self.test_metrics.update(test_metrics_dict)
        
        print("\n=== Final Training Metrics ===")
        for m in self.train_metrics:
            metrics_str = (
                f"Epoch {m['epoch']}: Time={m['epoch_time_sec']:.2f}s, "
                f"Memory={m['memory_usage_mb']:.2f}MB, "
                f"CPU={m['cpu_usage_percent']:.1f}%"
            )
            # Añadir RMSE y métricas de top-k si existen
            if 'RMSE' in m:
                metrics_str += f", RMSE={m['RMSE']:.4f}"
            for k in [5, 10, 20, 50]:
                if f'Recall@{k}' in m:
                    metrics_str += f", Recall@{k}={m[f'Recall@{k}']:.4f}"
                if f'NDCG@{k}' in m:
                    metrics_str += f", NDCG@{k}={m[f'NDCG@{k}']:.4f}"
            print(metrics_str)
        
        print("\n=== Final Test Metrics ===")
        print(f"Total Time: {self.test_metrics['total_time_sec']:.2f}s (Test: {self.test_metrics['test_time_sec']:.2f}s)")
        print(f"Final Memory: {self.test_metrics['final_memory_usage_mb']:.2f}MB")
        print(f"Final CPU: {self.test_metrics['final_cpu_usage_percent']:.1f}%")
        if 'RMSE' in self.test_metrics:
            print(f"RMSE: {self.test_metrics['RMSE']:.4f}")
        for k in [5, 10, 20, 50]:
            if f'Recall@{k}' in self.test_metrics:
                print(f"Recall@{k}: {self.test_metrics[f'Recall@{k}']:.4f}")
            if f'NDCG@{k}' in self.test_metrics:
                print(f"NDCG@{k}: {self.test_metrics[f'NDCG@{k}']:.4f}")

        if self.best_ndcg10_epoch is not None:
            print(f"\n=== Best Training NDCG@10 ===")
            print(f"Best NDCG@10: {self.best_ndcg10:.4f} (Epoch {self.best_ndcg10_epoch})")
            if self.best_ndcg10_metrics:
                print(f"Time: {self.best_ndcg10_metrics['epoch_time_sec']:.2f}s")
                print(f"Memory: {self.best_ndcg10_metrics['memory_usage_mb']:.2f}MB")
                print(f"CPU: {self.best_ndcg10_metrics['cpu_usage_percent']:.1f}%")
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        metrics_df = pd.DataFrame(self.train_metrics)
        metrics_df.to_csv(f"{result_path}/system_metrics_{timestamp}.csv", index=False)
        print(f"\nSystem metrics saved to: {result_path}/system_metrics_{timestamp}.csv")

    def get_best_ndcg_info(self):
        if self.best_ndcg10_epoch is not None:
            return {
                'epoch': self.best_ndcg10_epoch,
                'ndcg10': self.best_ndcg10,
                'metrics': self.best_ndcg10_metrics
            }
        return None

class EmissionsPerEpochTracker:
    def __init__(self, result_path, model_name="LightGCN_TopK"):
        self.result_path = result_path
        self.model_name = model_name
        self.epoch_emissions = []
        self.cumulative_emissions = []
        self.epoch_metrics = []
        self.epoch_loss = []
        self.total_emissions = 0.0
        self.trackers = {}
        
        self.main_tracker = EmissionsTracker(
            project_name=f"{model_name}_total",
            output_dir=f"{result_path}/emissions_reports",
            save_to_file=True, log_level="error", save_to_api=False, tracking_mode="process"
        )
        try:
            self.main_tracker.start()
        except Exception as e:
            print(f"Advertencia: No se pudo iniciar el tracker principal: {e}")
            self.main_tracker = None
    
    def start_epoch(self, epoch):
        timestamp = int(time.time())
        tracker_name = f"{self.model_name}_epoch{epoch}_{timestamp}"
        self.trackers[epoch] = EmissionsTracker(
            project_name=tracker_name, output_dir=f"{self.result_path}/emissions_reports",
            save_to_file=True, log_level="error", save_to_api=False, tracking_mode="process",
            measure_power_secs=1, allow_multiple_runs=True
        )
        try:
            self.trackers[epoch].start()
        except Exception as e:
            print(f"Advertencia: No se pudo iniciar el tracker para la época {epoch}: {e}")
            self.trackers[epoch] = None
    
    def end_epoch(self, epoch, loss, metrics):
        epoch_co2 = 0.0
        if epoch in self.trackers and self.trackers[epoch]:
            try:
                epoch_co2 = self.trackers[epoch].stop() or 0.0
            except Exception as e:
                print(f"Advertencia: Error al detener el tracker para la época {epoch}: {e}")
        
        self.total_emissions += epoch_co2
        self.epoch_emissions.append(epoch_co2)
        self.cumulative_emissions.append(self.total_emissions)
        self.epoch_loss.append(loss)
        self.epoch_metrics.append(metrics)
        
        print(f"Epoch {epoch} - Emisiones: {epoch_co2:.8f} kg, Acumulado: {self.total_emissions:.8f} kg")
            
    def end_training(self, final_metrics, best_ndcg_info):
        final_emissions = self.total_emissions
        if hasattr(self, 'main_tracker') and self.main_tracker:
            try:
                final_emissions = self.main_tracker.stop() or self.total_emissions
                print(f"\nTotal CO2 Emissions: {final_emissions:.6f} kg")
            except Exception as e:
                print(f"Error al detener el tracker principal: {e}")
        
        for tracker in self.trackers.values():
            if tracker:
                try: tracker.stop()
                except: pass
                    
        if (best_ndcg_info and best_ndcg_info['epoch'] is not None and 
            best_ndcg_info['epoch'] < len(self.epoch_emissions)):
            epoch = best_ndcg_info['epoch']
            print(f"\n=== Best NDCG@10 and Associated Emissions ===")
            print(f"Best NDCG@10: {best_ndcg_info['ndcg10']:.4f} (Epoch {epoch})")
            print(f"Emissions at best NDCG@10: {self.epoch_emissions[epoch]:.8f} kg")
            print(f"Cumulative emissions at best NDCG@10: {self.cumulative_emissions[epoch]:.8f} kg")
        
        if not self.epoch_emissions:
            print("No hay datos de emisiones para graficar.")
            return
            
        timestamp = time.strftime("%Y%m%d-%H%MS")
        df_data = {
            'epoch': range(len(self.epoch_emissions)),
            'epoch_emissions_kg': self.epoch_emissions,
            'cumulative_emissions_kg': self.cumulative_emissions,
            'loss': self.epoch_loss,
        }
        # Aplanar las métricas de cada época en columnas
        metrics_by_epoch = pd.DataFrame(self.epoch_metrics)
        df = pd.concat([pd.DataFrame(df_data), metrics_by_epoch], axis=1)
        
        emissions_file = f'{self.result_path}/emissions_reports/emissions_metrics_{self.model_name}_{timestamp}.csv'
        df.to_csv(emissions_file, index=False)
        print(f"Métricas de emisiones guardadas en: {emissions_file}")
        
        self.plot_emissions_vs_metrics(df, timestamp)
    
    def plot_emissions_vs_metrics(self, df, timestamp):
        plt.style.use('default')
        
        # Gráfico de emisiones acumulativas vs NDCG@10 y RMSE
        if 'NDCG@10' in df.columns:
            fig, ax1 = plt.subplots(figsize=(10, 6), facecolor='white')
            
            # Eje izquierdo para NDCG@10
            color = 'tab:blue'
            ax1.set_xlabel('Emisiones de CO2 acumuladas (kg)', color='black')
            ax1.set_ylabel('NDCG@10', color=color)
            ax1.plot(df['cumulative_emissions_kg'].to_numpy(), df['NDCG@10'].to_numpy(), color=color, marker='o', label='NDCG@10')
            ax1.tick_params(axis='y', labelcolor=color)
            
            # Eje derecho para RMSE
            if 'RMSE' in df.columns:
                ax2 = ax1.twinx()
                color = 'tab:red'
                ax2.set_ylabel('RMSE', color=color)
                ax2.plot(df['cumulative_emissions_kg'].to_numpy(), df['RMSE'].to_numpy(), color=color, marker='x', linestyle='--', label='RMSE')
                ax2.tick_params(axis='y', labelcolor=color)

            for i, row in df.iterrows():
                if 'NDCG@10' in df.columns:
                    ax1.annotate(f"{i}", (row['cumulative_emissions_kg'], row['NDCG@10']), textcoords="offset points", 
                                 xytext=(0,10), ha='center', fontsize=9, color='black')

            plt.title('Emisiones Acumuladas vs. NDCG@10 y RMSE', color='black')
            fig.tight_layout()
            plt.grid(True, alpha=0.3)
            file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_metrics_{self.model_name}_{timestamp}.png'
            plt.savefig(file_path, facecolor='white')
            plt.close()
            print(f"Gráfico guardado en: {file_path}")

        # Gráficos de métricas por época
        fig, axes = plt.subplots(3, 4, figsize=(20, 15), facecolor='white')
        axes = axes.flatten()
        
        plot_map = {
            'Loss': ('loss', 'g-'),
            'RMSE': ('RMSE', 'orange'),
            'Recall@5': ('Recall@5', 'b-'),
            'NDCG@5': ('NDCG@5', 'r-'),
            'Recall@10': ('Recall@10', 'b--'),
            'NDCG@10': ('NDCG@10', 'r--'),
            'Recall@20': ('Recall@20', 'b:'),
            'NDCG@20': ('NDCG@20', 'r:'),
            'Recall@50': ('Recall@50', 'b-.' ),
            'NDCG@50': ('NDCG@50', 'r-.'),
            'Emisiones por Época': ('epoch_emissions_kg', 'c-'),
            'Emisiones Acumuladas': ('cumulative_emissions_kg', 'm-')
        }
        
        for i, (title, (col, style)) in enumerate(plot_map.items()):
            if col in df.columns and i < len(axes):
                ax = axes[i]
                ax.plot(df['epoch'].to_numpy(), df[col].to_numpy(), style, marker='o')
                ax.set_title(title, color='black')
                ax.set_xlabel('Época', color='black')
                ax.set_ylabel(col, color='black')
                ax.tick_params(colors='black')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        file_path = f'{self.result_path}/emissions_plots/metrics_by_epoch_{self.model_name}_{timestamp}.png'
        plt.savefig(file_path, facecolor='white')
        plt.close()
        print(f"Gráfico de métricas por época guardado en: {file_path}")

# Funciones originales de LightGCN
def load_mydataset(train_file, test_file, val_file):
    def read_json(path):
        with open(path, 'r') as f:
            return [set(x) for x in json.load(f)]

    train_list = read_json(train_file)
    test_list = read_json(test_file)
    val_list = read_json(val_file)

    train_items = {item for items in train_list for item in items}

    def filter_orphans(data_list, valid_items):
        return [{item for item in items if item in valid_items} for items in data_list]

    test_list = filter_orphans(test_list, train_items)
    val_list = filter_orphans(val_list, train_items)

    n_users = len(train_list)
    n_items = max(train_items) + 1 if train_items else 0

    return train_list, test_list, val_list, n_users, n_items

def build_adjacency_matrix(train_data, n_users, n_items):
    R_dok = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    for u, items in enumerate(train_data):
        for i in items:
            R_dok[u, i] = 1.0
    R_csr = R_dok.tocsr()

    adj_size = n_users + n_items
    adj_dok = sp.dok_matrix((adj_size, adj_size), dtype=np.float32)
    adj_dok[:n_users, n_users:] = R_csr
    adj_dok[n_users:, :n_users] = R_csr.transpose()
    return adj_dok.tocsr()


def normalize_adj_sym(adj_mat):
    rowsum = np.array(adj_mat.sum(axis=1)).flatten() + 1e-9
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt.dot(adj_mat).dot(D_inv_sqrt)

class LightGCNModel(tf.keras.Model):
    def __init__(self, n_users, n_items, adj_mat, n_layers=3, emb_dim=64, decay=1e-4, use_personalized_alpha=False):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.adj_mat = adj_mat
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.decay = decay
        self.use_personalized_alpha = use_personalized_alpha

        initializer = tf.initializers.GlorotUniform(seed=42)
        self.user_embedding = self.add_weight(
            name='user_embedding', shape=(n_users, emb_dim), initializer=initializer, trainable=True
        )
        item_initializer = tf.initializers.GlorotUniform(seed=43)
        self.item_embedding = self.add_weight(
            name='item_embedding', shape=(n_items, emb_dim), initializer=item_initializer, trainable=True
        )

        if use_personalized_alpha:
            alpha_initializer = tf.initializers.GlorotUniform(seed=44)
            self.alpha_mlp = tf.keras.Sequential([
                tf.keras.layers.Dense(n_layers + 1, activation='softmax', kernel_initializer=alpha_initializer)
            ])
        
        attr_initializer = tf.initializers.GlorotUniform(seed=45)
        self.attribute_predictor = tf.keras.layers.Dense(emb_dim, activation='relu', 
                                                       kernel_initializer=attr_initializer,
                                                       name="attribute_predictor")

    def call(self, embeddings, mask_prob=0.2):
        user_emb, item_emb = embeddings
        all_emb = tf.concat([user_emb, item_emb], axis=0)
        emb_list = [all_emb]
    
        for _ in range(self.n_layers):
            all_emb = tf.sparse.sparse_dense_matmul(self.adj_mat, all_emb)
            emb_list.append(all_emb)
    
        if not self.use_personalized_alpha:
            alpha_k = 1.0 / (self.n_layers + 1)
            alpha_weights = [alpha_k] * (self.n_layers + 1)
            alpha_weights = tf.convert_to_tensor(alpha_weights, dtype=tf.float32)
            alpha_weights = tf.reshape(alpha_weights, (-1, 1, 1))
            stacked_emb = tf.stack(emb_list, axis=0)
            combined_emb = tf.reduce_sum(stacked_emb * alpha_weights, axis=0)
        else:
            alpha = self.alpha_mlp(emb_list[0])
            alpha = tf.expand_dims(alpha, axis=-1)
            stacked_emb = tf.stack(emb_list, axis=0)
            combined_emb = tf.reduce_sum(stacked_emb * alpha, axis=0)
    
        user_final, item_final = tf.split(combined_emb, [self.n_users, self.n_items], axis=0)
    
        masked_user_emb, mask = mask_embeddings(user_final, mask_prob)
        predicted_attributes = self.attribute_predictor(masked_user_emb)
        return user_final, item_final, masked_user_emb, predicted_attributes, mask

    def recommend(self, user_ids, k=10):
        user_final, item_final, _, _, _ = self((self.user_embedding, self.item_embedding), mask_prob=0.0)
        user_vecs = tf.gather(user_final, user_ids)
    
        all_recs = []
        scores = tf.matmul(user_vecs, item_final, transpose_b=True)
        scores_np = scores.numpy()
        
        for i, uid in enumerate(user_ids):
            user_scores = scores_np[i]
            idx_topk = np.argpartition(user_scores, -k)[-k:]
            topk_scores = user_scores[idx_topk]
            # Sort the top-k items by score
            sorted_idx = np.argsort(topk_scores)[::-1]
            topk_items = idx_topk[sorted_idx]
            
            for item_id in topk_items:
                all_recs.append((int(uid), int(item_id), float(user_scores[item_id])))
        return all_recs

def mask_embeddings(embeddings, mask_prob=0.2):
    mask = tf.cast(tf.random.uniform(embeddings.shape) > mask_prob, tf.float32)
    masked_embeddings = embeddings * mask
    return masked_embeddings, mask

def sample_neg(pos_items, n_items, strategy='random'):
    neg_item = random.randint(0, n_items - 1)
    while neg_item in pos_items:
        neg_item = random.randint(0, n_items - 1)
    return neg_item

def train_lightgcn_with_metrics(model, train_data, val_data, test_data, n_users, n_items, batch_size=1024, epochs=10, initial_lr=1e-2):
    system_tracker = SystemMetricsTracker()
    emissions_tracker = EmissionsPerEpochTracker(result_path, "LightGCN_TopK")
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr, decay_steps=1000, decay_rate=0.96, staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    train_pairs = [(u, i) for u in range(n_users) for i in train_data[u]]
    steps_per_epoch = len(train_pairs) // batch_size + (len(train_pairs) % batch_size != 0)
    
    val_users = [u for u in range(n_users) if val_data[u]]
    test_users = [u for u in range(n_users) if test_data[u]]

    for epoch in range(epochs):
        system_tracker.start_epoch(epoch)
        emissions_tracker.start_epoch(epoch)
        
        random.shuffle(train_pairs)
        progbar = Progbar(steps_per_epoch)
        epoch_loss = 0

        for step in range(steps_per_epoch):
            batch_slice = train_pairs[step * batch_size:(step + 1) * batch_size]
            users = [u for (u, _) in batch_slice]
            pos_items = [i for (_, i) in batch_slice]
            neg_items = [sample_neg(train_data[u], n_items) for (u, _) in batch_slice]

            with tf.GradientTape() as tape:
                user_emb, item_emb, masked_user_emb, predicted_attributes, mask = model(
                    (model.user_embedding, model.item_embedding)
                )
                u_emb = tf.nn.embedding_lookup(user_emb, users)
                pos_emb = tf.nn.embedding_lookup(item_emb, pos_items)
                neg_emb = tf.nn.embedding_lookup(item_emb, neg_items)
            
                pos_scores = tf.reduce_sum(u_emb * pos_emb, axis=1)
                neg_scores = tf.reduce_sum(u_emb * neg_emb, axis=1)
                mf_loss = tf.reduce_mean(tf.nn.softplus(neg_scores - pos_scores))
            
                attribute_loss = tf.reduce_mean(tf.square(masked_user_emb - predicted_attributes) * mask)
            
                reg_loss = model.decay * (
                    tf.nn.l2_loss(u_emb) + tf.nn.l2_loss(pos_emb) + tf.nn.l2_loss(neg_emb)
                ) / batch_size
            
                loss = mf_loss + reg_loss + 0.1 * attribute_loss

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += loss.numpy()
            progbar.add(1, values=[('loss', float(loss))])

        avg_epoch_loss = epoch_loss / steps_per_epoch
        
        # Evaluar en validación (incluye ahora RMSE)
        val_metrics = evaluate_model(model, val_users, val_data, ks=[5, 10, 20, 50])
        
        system_tracker.end_epoch(epoch, avg_epoch_loss, val_metrics)
        emissions_tracker.end_epoch(epoch, avg_epoch_loss, val_metrics)
        
        print(f"\nEpoch {epoch}/{epochs-1} completed. Average loss: {avg_epoch_loss:.6f}")

    # Evaluación final en el conjunto de pruebas
    print("\nEvaluating on final test set...")
    system_tracker.start_epoch("test")
    final_test_metrics = evaluate_model(model, test_users, test_data, ks=[5, 10, 20, 50])
    
    system_tracker.end_test(final_test_metrics)
    best_ndcg_info = system_tracker.get_best_ndcg_info()
    emissions_tracker.end_training(final_test_metrics, best_ndcg_info)
    
    print("\nTraining and evaluation completed!")
    return final_test_metrics

def run_lightgcn_with_metrics():
    train_file = 'C:/Users/xpati/Documents/TFG/ml-1m/train_data.json'
    test_file = 'C:/Users/xpati/Documents/TFG/ml-1m/test_data.json'
    val_file = 'C:/Users/xpati/Documents/TFG/ml-1m/validation_data.json'
    
    train_data, test_data, val_data, n_users, n_items = load_mydataset(
        train_file, test_file, val_file
    )
    print(f"Number of Users: {n_users}, Number of Items: {n_items}")

    adj_csr = build_adjacency_matrix(train_data, n_users, n_items)
    norm_adj_csr = normalize_adj_sym(adj_csr)

    coo = norm_adj_csr.tocoo().astype(np.float32)
    indices = np.vstack((coo.row, coo.col)).transpose()
    A_tilde = tf.sparse.SparseTensor(indices=indices, values=coo.data, dense_shape=coo.shape)
    A_tilde = tf.sparse.reorder(A_tilde)

    N_LAYERS = 2
    EMBED_DIM = 64
    DECAY = 1e-4
    INITIAL_LR = 1e-3
    EPOCHS = 50
    BATCH_SIZE = 2048

    model = LightGCNModel(
        n_users=n_users, n_items=n_items, adj_mat=A_tilde,
        n_layers=N_LAYERS, emb_dim=EMBED_DIM, decay=DECAY,
        use_personalized_alpha=False
    )

    print("\nStarting LightGCN training for Top-K...")
    train_lightgcn_with_metrics(
        model=model, train_data=train_data, val_data=val_data, test_data=test_data,
        n_users=n_users, n_items=n_items, batch_size=BATCH_SIZE, epochs=EPOCHS,
        initial_lr=INITIAL_LR
    )

if __name__ == "__main__":
    run_lightgcn_with_metrics()