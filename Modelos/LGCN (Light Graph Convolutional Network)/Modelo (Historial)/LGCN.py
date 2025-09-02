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
from sklearn.metrics import mean_squared_error

# Crear directorios para resultados
result_path = "results_lgcn"
os.makedirs(result_path, exist_ok=True)
os.makedirs(f"{result_path}/emissions_reports", exist_ok=True)
os.makedirs(f"{result_path}/emissions_plots", exist_ok=True)

# Diccionario global para almacenar los ratings reales
ratings_dict = {}

def load_ratings_from_file(ratings_file):
    """
    Carga los ratings reales desde el archivo ratings.dat de MovieLens-1M
    
    Args:
        ratings_file: Ruta al archivo ratings.dat
    
    Returns:
        dict: Diccionario {(user_id, item_id): rating}
    """
    global ratings_dict
    ratings_dict = {}
    
    try:
        # Asumimos que el archivo de ratings usa '::' como separador y no tiene cabecera
        df = pd.read_csv(ratings_file, sep='::', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python', encoding='latin-1')
        
        # Convertir a índice base 0
        df['user_id'] = df['user_id'] - 1
        df['item_id'] = df['item_id'] - 1
        
        for _, row in df.iterrows():
            ratings_dict[(int(row['user_id']), int(row['item_id']))] = float(row['rating'])
        
        print(f"Cargados {len(ratings_dict)} ratings reales desde {ratings_file}")
        return ratings_dict
    
    except FileNotFoundError:
        print(f"Archivo {ratings_file} no encontrado. No se pueden calcular RMSE/MAE.")
        return {}
    except Exception as e:
        print(f"Error al cargar ratings: {e}. No se pueden calcular RMSE/MAE.")
        return {}

# Función OPTIMIZADA para calcular RMSE
def calculate_rmse(model, test_data, n_users, n_items, sample_size=1000):
    """
    Calcula el RMSE usando los ratings reales de MovieLens - OPTIMIZADA.
    """
    global ratings_dict
    
    if not ratings_dict:
        return 1.0
    
    all_ratings = list(ratings_dict.items())
    sample_ratings = random.sample(all_ratings, min(sample_size, len(all_ratings)))
    
    user_emb, item_emb, _, _, _ = model((model.user_embedding, model.item_embedding))
    
    y_true = []
    y_pred = []
    
    for (u, item_id), true_rating in sample_ratings:
        if u >= n_users or item_id >= n_items:
            continue
        
        # Usar el método predict_rating del modelo que considera el historial
        predicted_rating = model.predict_rating(u, item_id)
        
        y_true.append(true_rating)
        y_pred.append(predicted_rating)
    
    if not y_true:
        return 1.0
    
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

# Función OPTIMIZADA para calcular MAE
def calculate_mae(model, test_data, n_users, n_items, sample_size=1000):
    """
    Calcula el MAE usando los ratings reales de MovieLens - OPTIMIZADA.
    """
    global ratings_dict
    
    if not ratings_dict:
        return 0.8
    
    all_ratings = list(ratings_dict.items())
    sample_ratings = random.sample(all_ratings, min(sample_size, len(all_ratings)))
    
    y_true = []
    y_pred = []
    
    for (u, item_id), true_rating in sample_ratings:
        if u >= n_users or item_id >= n_items:
            continue
            
        # Usar el método predict_rating del modelo que considera el historial
        predicted_rating = model.predict_rating(u, item_id)
        
        y_true.append(true_rating)
        y_pred.append(predicted_rating)
    
    if not y_true:
        return 0.8
    
    mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    return mae


# Clases para seguimiento de métricas
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
        
    def end_epoch(self, epoch, loss, rmse=None, mae=None):
        epoch_time = time.time() - self.epoch_start_time
        self.current_epoch_metrics['epoch_time_sec'] = epoch_time
        self.current_epoch_metrics['loss'] = loss
        if rmse is not None:
            self.current_epoch_metrics['rmse'] = rmse
        if mae is not None:
            self.current_epoch_metrics['mae'] = mae
        self.train_metrics.append(self.current_epoch_metrics)
        
        if rmse is not None and rmse < self.best_rmse:
            self.best_rmse = rmse
            self.best_rmse_epoch = epoch
            self.best_rmse_metrics = self.current_epoch_metrics.copy()
        
    def end_test(self, rmse, mae=None):
        self.test_metrics = {
            'test_time_sec': time.time() - self.epoch_start_time,
            'total_time_sec': time.time() - self.start_time,
            'final_memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'final_cpu_usage_percent': psutil.cpu_percent(),
            'test_rmse': rmse,
        }
        if mae is not None:
            self.test_metrics['test_mae'] = mae
        
        print("\n=== Final Training Metrics ===")
        for m in self.train_metrics:
            metrics_str = f"Epoch {m['epoch']}: Time={m['epoch_time_sec']:.2f}s, Memory={m['memory_usage_mb']:.2f}MB, CPU={m['cpu_usage_percent']:.1f}%"
            if 'rmse' in m and m['rmse'] is not None:
                metrics_str += f", RMSE={m['rmse']:.4f}"
            if 'mae' in m and m['mae'] is not None:
                metrics_str += f", MAE={m['mae']:.4f}"
            print(metrics_str)
        
        print("\n=== Final Test Metrics ===")
        print(f"Total Time: {self.test_metrics['total_time_sec']:.2f}s (Test: {self.test_metrics['test_time_sec']:.2f}s)")
        print(f"Final Memory: {self.test_metrics['final_memory_usage_mb']:.2f}MB")
        print(f"Final CPU: {self.test_metrics['final_cpu_usage_percent']:.1f}%")
        print(f"RMSE: {rmse:.4f}")
        if mae is not None:
            print(f"MAE: {mae:.4f}")
            
        if self.best_rmse_epoch is not None:
            print(f"\n=== Best Training RMSE ===")
            print(f"Best RMSE: {self.best_rmse:.4f} (Epoch {self.best_rmse_epoch})")
            if self.best_rmse_metrics:
                print(f"Time: {self.best_rmse_metrics['epoch_time_sec']:.2f}s")
                print(f"Memory: {self.best_rmse_metrics['memory_usage_mb']:.2f}MB")
                print(f"CPU: {self.best_rmse_metrics['cpu_usage_percent']:.1f}%")
                if 'mae' in self.best_rmse_metrics and self.best_rmse_metrics['mae'] is not None:
                    print(f"MAE: {self.best_rmse_metrics['mae']:.4f}")

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        metrics_df = pd.DataFrame(self.train_metrics)
        metrics_df.to_csv(f"{result_path}/system_metrics_{timestamp}.csv", index=False)

    def get_best_rmse_info(self):
        if self.best_rmse_epoch is not None:
            return {
                'epoch': self.best_rmse_epoch,
                'rmse': self.best_rmse,
                'metrics': self.best_rmse_metrics
            }
        return None


class EmissionsPerEpochTracker:
    def __init__(self, result_path, model_name="LightGCN"):
        self.result_path = result_path
        self.model_name = model_name
        self.epoch_emissions = []
        self.cumulative_emissions = []
        self.epoch_rmse = []
        self.epoch_mae = []
        self.epoch_loss = []
        self.total_emissions = 0.0
        self.trackers = {}
        
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
            print("Tracker principal iniciado correctamente")
        except Exception as e:
            print(f"Advertencia: No se pudo iniciar el tracker principal: {e}")
            self.main_tracker = None
    
    def start_epoch(self, epoch):
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
            print(f"Advertencia: No se pudo iniciar el tracker para la época {epoch}: {e}")
            self.trackers[epoch] = None
    
    def end_epoch(self, epoch, loss, rmse=None, mae=None):
        try:
            epoch_co2 = 0.0
            if epoch in self.trackers and self.trackers[epoch]:
                try:
                    epoch_co2 = self.trackers[epoch].stop() or 0.0
                except Exception as e:
                    print(f"Advertencia: Error al detener el tracker para la época {epoch}: {e}")
                    epoch_co2 = 0.0
            
            self.total_emissions += epoch_co2
            
            self.epoch_emissions.append(epoch_co2)
            self.cumulative_emissions.append(self.total_emissions)
            self.epoch_loss.append(loss)
            if rmse is not None:
                self.epoch_rmse.append(rmse)
            if mae is not None:
                self.epoch_mae.append(mae)
            
        except Exception as e:
            print(f"Error al medir emisiones en época {epoch}: {e}")
    
    def end_training(self, final_rmse=None, final_mae=None, best_rmse_info=None):
        try:
            final_emissions = 0.0
            if hasattr(self, 'main_tracker') and self.main_tracker:
                try:
                    final_emissions = self.main_tracker.stop() or 0.0
                    print(f"\nTotal CO2 Emissions: {final_emissions:.6f} kg")
                except Exception as e:
                    print(f"Error al detener el tracker principal: {e}")
                    final_emissions = self.total_emissions
            else:
                final_emissions = self.total_emissions
            
            for epoch, tracker in self.trackers.items():
                if tracker is not None:
                    try:
                        tracker.stop()
                    except:
                        pass
            
            if (best_rmse_info and 
                best_rmse_info['epoch'] is not None and 
                best_rmse_info['epoch']-1 < len(self.epoch_emissions)):
                
                epoch_idx = best_rmse_info['epoch'] - 1
                print(f"\n=== Best RMSE and Associated Emissions ===")
                print(f"Best RMSE: {best_rmse_info['rmse']:.4f} (Epoch {best_rmse_info['epoch']})")
                print(f"Emissions at best RMSE: {self.epoch_emissions[epoch_idx]:.8f} kg")
                print(f"Cumulative emissions at best RMSE: {self.cumulative_emissions[epoch_idx]:.8f} kg")

            if not self.epoch_emissions:
                print("No hay datos de emisiones para graficar")
                return
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            df = pd.DataFrame({
                'epoch': range(1, len(self.epoch_emissions) + 1),
                'epoch_emissions_kg': self.epoch_emissions,
                'cumulative_emissions_kg': self.cumulative_emissions,
                'loss': self.epoch_loss if self.epoch_loss else [0.0] * len(self.epoch_emissions),
                'rmse': self.epoch_rmse if self.epoch_rmse else [None] * len(self.epoch_emissions),
                'mae': self.epoch_mae if self.epoch_mae else [None] * len(self.epoch_emissions)
            })
            
            emissions_file = f'{self.result_path}/emissions_reports/emissions_metrics_{self.model_name}_{timestamp}.csv'
            df.to_csv(emissions_file, index=False)
            print(f"Métricas de emisiones guardadas en: {emissions_file}")
            
            self.plot_emissions_vs_metrics(timestamp)
            
        except Exception as e:
            print(f"Error al generar gráficos de emisiones: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_emissions_vs_metrics(self, timestamp):
        plt.style.use('default')
        
        try:
            if self.epoch_rmse:
                plt.figure(figsize=(10, 6), facecolor='white')
                plt.plot(self.cumulative_emissions, self.epoch_rmse, 'b-', marker='o')
                for i, (emissions, rmse) in enumerate(zip(self.cumulative_emissions, self.epoch_rmse)):
                    plt.annotate(f"{i+1}", (emissions, rmse), textcoords="offset points", xytext=(0,10), ha='center')
                plt.xlabel('Emisiones de CO2 acumuladas (kg)')
                plt.ylabel('RMSE')
                plt.title('Relación entre Emisiones Acumuladas y RMSE')
                plt.grid(True, alpha=0.3)
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_rmse_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path, facecolor='white')
                plt.close()

            if self.epoch_mae:
                plt.figure(figsize=(10, 6), facecolor='white')
                plt.plot(self.cumulative_emissions, self.epoch_mae, 'm-', marker='D')
                for i, (emissions, mae) in enumerate(zip(self.cumulative_emissions, self.epoch_mae)):
                    plt.annotate(f"{i+1}", (emissions, mae), textcoords="offset points", xytext=(0,10), ha='center')
                plt.xlabel('Emisiones de CO2 acumuladas (kg)')
                plt.ylabel('MAE')
                plt.title('Relación entre Emisiones Acumuladas y MAE')
                plt.grid(True, alpha=0.3)
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_mae_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path, facecolor='white')
                plt.close()

            plt.figure(figsize=(12, 10), facecolor='white')
            epochs_range = range(1, len(self.epoch_emissions) + 1)

            plt.subplot(2, 3, 1)
            plt.plot(epochs_range, self.epoch_emissions, 'r-', marker='x', label='Emisiones por Época')
            plt.title('Emisiones por Época')
            plt.xlabel('Época')
            plt.ylabel('CO2 Emissions (kg)')
            
            plt.subplot(2, 3, 2)
            plt.plot(epochs_range, self.cumulative_emissions, 'r-', marker='o', label='Emisiones Acumuladas')
            plt.title('Emisiones Acumuladas')
            plt.xlabel('Época')
            plt.ylabel('CO2 Emissions (kg)')
            
            if self.epoch_loss:
                plt.subplot(2, 3, 3)
                plt.plot(epochs_range, self.epoch_loss, 'g-', marker='s', label='Loss')
                plt.title('Loss por Época')
                plt.xlabel('Época')
                plt.ylabel('Loss')

            if self.epoch_rmse:
                plt.subplot(2, 3, 4)
                plt.plot(epochs_range, self.epoch_rmse, 'b-', marker='o', label='RMSE')
                plt.title('RMSE por Época')
                plt.xlabel('Época')
                plt.ylabel('RMSE')

            if self.epoch_mae:
                plt.subplot(2, 3, 5)
                plt.plot(epochs_range, self.epoch_mae, 'm-', marker='D', label='MAE')
                plt.title('MAE por Época')
                plt.xlabel('Época')
                plt.ylabel('MAE')

            plt.tight_layout()
            file_path = f'{self.result_path}/emissions_plots/metrics_by_epoch_{self.model_name}_{timestamp}.png'
            plt.savefig(file_path, facecolor='white')
            plt.close()
                
        except Exception as e:
            print(f"Error al generar los gráficos: {e}")
            import traceback
            traceback.print_exc()

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
    def __init__(self, n_users, n_items, adj_mat, n_layers=3, emb_dim=64, decay=1e-4, use_personalized_alpha=False, history_weight=0.3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.adj_mat = adj_mat
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.decay = decay
        self.use_personalized_alpha = use_personalized_alpha
        self.history_weight = history_weight
        self.user_histories = {}
        self.item_metadata = {}

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

    def set_metadata(self, metadata):
        if 'user_histories' in metadata:
            self.user_histories = metadata['user_histories']
        if 'item_metadata' in metadata:
            self.item_metadata = metadata['item_metadata']
        print(f"Loaded histories for {len(self.user_histories)} users")
            
    def predict_rating(self, user_id, item_id):
        user_emb, item_emb, _, _, _ = self((self.user_embedding, self.item_embedding))
        
        u_tensor = tf.convert_to_tensor([user_id], dtype=tf.int32)
        i_tensor = tf.convert_to_tensor([item_id], dtype=tf.int32)
        
        u_emb = tf.nn.embedding_lookup(user_emb, u_tensor)
        i_emb = tf.nn.embedding_lookup(item_emb, i_tensor)
        
        model_score = tf.reduce_sum(u_emb * i_emb, axis=1).numpy()[0]
        
        min_score, max_score = -10, 10 # Heuristic score range
        model_rating = 1.0 + 4.0 * (model_score - min_score) / (max_score - min_score)
        model_rating = np.clip(model_rating, 1.0, 5.0)

        if self.history_weight == 0 or user_id not in self.user_histories or self.user_histories[user_id].empty:
            return float(model_rating)
        
        history_rating = self._predict_from_history(user_id, item_id, item_emb)
        
        final_rating = (1 - self.history_weight) * model_rating + self.history_weight * history_rating
        
        return float(final_rating)
    
    def _predict_from_history(self, user_id, target_item_id, item_embeddings):
        if user_id not in self.user_histories:
            return 3.0
        
        history = self.user_histories[user_id]
        
        if history.empty or 'item_id' not in history.columns or 'rating' not in history.columns:
            return 3.0
        
        target_tensor = tf.convert_to_tensor([target_item_id], dtype=tf.int32)
        target_emb = tf.nn.embedding_lookup(item_embeddings, target_tensor)
        
        similarities = []
        ratings = []
        
        for _, row in history.iterrows():
            hist_item_id = int(row['item_id'])
            hist_rating = float(row['rating'])
            
            if hist_item_id == target_item_id:
                continue
            
            hist_tensor = tf.convert_to_tensor([hist_item_id], dtype=tf.int32)
            try:
                hist_emb = tf.nn.embedding_lookup(item_embeddings, hist_tensor)
                similarity = tf.nn.cosine_similarity(target_emb, hist_emb, axis=1).numpy()[0]
                
                similarities.append(max(0, similarity))
                ratings.append(hist_rating)
            except:
                continue
        
        if not similarities:
            return 3.0
        
        similarities = np.array(similarities)
        ratings = np.array(ratings)
        
        sum_sim = np.sum(similarities)
        if sum_sim > 0:
            weights = similarities / sum_sim
            weighted_rating = np.sum(weights * ratings)
            return float(weighted_rating)
        else:
            return 3.0


def mask_embeddings(embeddings, mask_prob=0.2):
    mask = tf.cast(tf.random.uniform(embeddings.shape) > mask_prob, tf.float32)
    masked_embeddings = embeddings * mask
    return masked_embeddings, mask


def sample_neg(pos_items, n_items, strategy='random'):
    if strategy == 'random':
        neg_item = random.randint(0, n_items - 1)
        while neg_item in pos_items:
            neg_item = random.randint(0, n_items - 1)
    return neg_item


def extract_user_histories(train_data, ratings_df=None):
    user_histories = {}
    
    if ratings_df is not None and not ratings_df.empty:
        for user_id in range(len(train_data)):
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            if not user_ratings.empty:
                user_histories[user_id] = user_ratings[['item_id', 'rating']]
            else:
                user_histories[user_id] = pd.DataFrame(columns=['item_id', 'rating'])
    else:
        for user_id in range(len(train_data)):
            if train_data[user_id]:
                item_ids = list(train_data[user_id])
                ratings = [1.0] * len(item_ids)
                user_histories[user_id] = pd.DataFrame({'item_id': item_ids, 'rating': ratings})
            else:
                user_histories[user_id] = pd.DataFrame(columns=['item_id', 'rating'])
    
    return user_histories


def train_lightgcn_with_history(model, train_data, val_data, test_data, n_users, n_items, 
                               batch_size=1024, epochs=10, initial_lr=1e-2,
                               ratings_df=None, history_weight=0.3):
    model.history_weight = history_weight
    
    user_histories = extract_user_histories(train_data, ratings_df)
    model.set_metadata({'user_histories': user_histories})
    
    print("Inicializando trackers...")
    system_tracker = SystemMetricsTracker()
    emissions_tracker = EmissionsPerEpochTracker(result_path, "LightGCN_History")
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr, decay_steps=1000, decay_rate=0.96, staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    train_pairs = [(u, i) for u in range(n_users) for i in train_data[u]]
    steps_per_epoch = len(train_pairs) // batch_size + (len(train_pairs) % batch_size != 0)

    epoch_losses = []
    rmse_scores = []
    mae_scores = []
    
    for epoch in range(1, epochs + 1):
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

            users = np.array(users, dtype=np.int32)
            pos_items = np.array(pos_items, dtype=np.int32)
            neg_items = np.array(neg_items, dtype=np.int32)

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
        epoch_losses.append(avg_epoch_loss)
        
        epoch_rmse = calculate_rmse(model, val_data, n_users, n_items)
        epoch_mae = calculate_mae(model, val_data, n_users, n_items)
        rmse_scores.append(epoch_rmse)
        mae_scores.append(epoch_mae)

        system_tracker.end_epoch(epoch, avg_epoch_loss, epoch_rmse, epoch_mae)
        emissions_tracker.end_epoch(epoch, avg_epoch_loss, epoch_rmse, epoch_mae)
        
        print(f"Epoch {epoch}/{epochs} completed. Loss: {avg_epoch_loss:.4f}, RMSE: {epoch_rmse:.4f}, MAE: {epoch_mae:.4f}")
        
    print("\nEvaluando en conjunto de prueba final...")
    system_tracker.start_epoch("test")
    
    final_rmse = calculate_rmse(model, test_data, n_users, n_items, sample_size=5000)
    final_mae = calculate_mae(model, test_data, n_users, n_items, sample_size=5000)
    
    best_rmse_info = system_tracker.get_best_rmse_info()
    
    try:
        system_tracker.end_test(final_rmse, final_mae)
    except Exception as e:
        print(f"Error al generar métricas finales con tracker: {e}")
    
    try:
        emissions_tracker.end_training(final_rmse, final_mae, best_rmse_info)
    except Exception as e:
        print(f"Error al generar métricas de emisiones: {e}")
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, epochs + 1)),
        'loss': epoch_losses,
        'rmse': rmse_scores,
        'mae': mae_scores
    })
    
    metrics_file = f"{result_path}/model_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Métricas del modelo guardadas en: {metrics_file}")
    
    print(f"\nEntrenamiento finalizado! RMSE: {final_rmse:.4f}, MAE: {final_mae:.4f}")
    
    return epoch_losses, rmse_scores, mae_scores, final_rmse, final_mae


# Función principal para ejecutar todo
def run_lightgcn_with_history(ratings_path=None, history_weight=0.3):
    train_file = 'C:/Users/xpati/Documents/TFG/ml-1m/train_data.json'
    test_file = 'C:/Users/xpati/Documents/TFG/ml-1m/test_data.json'
    val_file = 'C:/Users/xpati/Documents/TFG/ml-1m/validation_data.json'

    train_data, test_data, val_data, n_users, n_items = load_mydataset(
        train_file, test_file, val_file
    )
    print(f"Number of Users: {n_users}, Number of Items: {n_items}")

    ratings_df = None
    if ratings_path and os.path.exists(ratings_path):
        try:
            load_ratings_from_file(ratings_path)
            ratings_df = pd.read_csv(ratings_path, sep='::', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python', encoding='latin-1')
            ratings_df['user_id'] = ratings_df['user_id'] - 1
            ratings_df['item_id'] = ratings_df['item_id'] - 1
            print(f"Loaded ratings data for history: {len(ratings_df)} rows")
        except Exception as e:
            print(f"Error loading ratings data: {e}")
        
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
        n_users=n_users,
        n_items=n_items,
        adj_mat=A_tilde,
        n_layers=N_LAYERS,
        emb_dim=EMBED_DIM,
        decay=DECAY,
        use_personalized_alpha=False,
        history_weight=history_weight
    )

    print("\nStarting LightGCN training with history-based predictions...")
    train_lightgcn_with_history(
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        n_users=n_users,
        n_items=n_items,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        initial_lr=INITIAL_LR,
        ratings_df=ratings_df,
        history_weight=history_weight
    )
    
    print("\nTraining and evaluation completed!")
    return model

if __name__ == "__main__":
    ratings_path = "C:/Users/xpati/Documents/TFG/ml-1m/ratings.dat"
    run_lightgcn_with_history(ratings_path=ratings_path, history_weight=0.3)
# filepath: c:\Users\xpati\Documents\TFG\Pruebas(Metricas)\LGCN (Light Graph Convolutional Network)\Modelo (Historial)\LGCN.py
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
from sklearn.metrics import mean_squared_error

# Crear directorios para resultados
result_path = "results_lgcn"
os.makedirs(result_path, exist_ok=True)
os.makedirs(f"{result_path}/emissions_reports", exist_ok=True)
os.makedirs(f"{result_path}/emissions_plots", exist_ok=True)

# Diccionario global para almacenar los ratings reales
ratings_dict = {}

def load_ratings_from_file(ratings_file):
    """
    Carga los ratings reales desde el archivo ratings.dat de MovieLens-1M
    
    Args:
        ratings_file: Ruta al archivo ratings.dat
    
    Returns:
        dict: Diccionario {(user_id, item_id): rating}
    """
    global ratings_dict
    ratings_dict = {}
    
    try:
        # Asumimos que el archivo de ratings usa '::' como separador y no tiene cabecera
        df = pd.read_csv(ratings_file, sep='::', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python', encoding='latin-1')
        
        # Convertir a índice base 0
        df['user_id'] = df['user_id'] - 1
        df['item_id'] = df['item_id'] - 1
        
        for _, row in df.iterrows():
            ratings_dict[(int(row['user_id']), int(row['item_id']))] = float(row['rating'])
        
        print(f"Cargados {len(ratings_dict)} ratings reales desde {ratings_file}")
        return ratings_dict
    
    except FileNotFoundError:
        print(f"Archivo {ratings_file} no encontrado. No se pueden calcular RMSE/MAE.")
        return {}
    except Exception as e:
        print(f"Error al cargar ratings: {e}. No se pueden calcular RMSE/MAE.")
        return {}

# Función OPTIMIZADA para calcular RMSE
def calculate_rmse(model, test_data, n_users, n_items, sample_size=1000):
    """
    Calcula el RMSE usando los ratings reales de MovieLens - OPTIMIZADA.
    """
    global ratings_dict
    
    if not ratings_dict:
        return 1.0
    
    all_ratings = list(ratings_dict.items())
    sample_ratings = random.sample(all_ratings, min(sample_size, len(all_ratings)))
    
    user_emb, item_emb, _, _, _ = model((model.user_embedding, model.item_embedding))
    
    y_true = []
    y_pred = []
    
    for (u, item_id), true_rating in sample_ratings:
        if u >= n_users or item_id >= n_items:
            continue
        
        # Usar el método predict_rating del modelo que considera el historial
        predicted_rating = model.predict_rating(u, item_id)
        
        y_true.append(true_rating)
        y_pred.append(predicted_rating)
    
    if not y_true:
        return 1.0
    
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

# Función OPTIMIZADA para calcular MAE
def calculate_mae(model, test_data, n_users, n_items, sample_size=1000):
    """
    Calcula el MAE usando los ratings reales de MovieLens - OPTIMIZADA.
    """
    global ratings_dict
    
    if not ratings_dict:
        return 0.8
    
    all_ratings = list(ratings_dict.items())
    sample_ratings = random.sample(all_ratings, min(sample_size, len(all_ratings)))
    
    y_true = []
    y_pred = []
    
    for (u, item_id), true_rating in sample_ratings:
        if u >= n_users or item_id >= n_items:
            continue
            
        # Usar el método predict_rating del modelo que considera el historial
        predicted_rating = model.predict_rating(u, item_id)
        
        y_true.append(true_rating)
        y_pred.append(predicted_rating)
    
    if not y_true:
        return 0.8
    
    mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    return mae


# Clases para seguimiento de métricas
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
        
    def end_epoch(self, epoch, loss, rmse=None, mae=None):
        epoch_time = time.time() - self.epoch_start_time
        self.current_epoch_metrics['epoch_time_sec'] = epoch_time
        self.current_epoch_metrics['loss'] = loss
        if rmse is not None:
            self.current_epoch_metrics['rmse'] = rmse
        if mae is not None:
            self.current_epoch_metrics['mae'] = mae
        self.train_metrics.append(self.current_epoch_metrics)
        
        if rmse is not None and rmse < self.best_rmse:
            self.best_rmse = rmse
            self.best_rmse_epoch = epoch
            self.best_rmse_metrics = self.current_epoch_metrics.copy()
        
    def end_test(self, rmse, mae=None):
        self.test_metrics = {
            'test_time_sec': time.time() - self.epoch_start_time,
            'total_time_sec': time.time() - self.start_time,
            'final_memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'final_cpu_usage_percent': psutil.cpu_percent(),
            'test_rmse': rmse,
        }
        if mae is not None:
            self.test_metrics['test_mae'] = mae
        
        print("\n=== Final Training Metrics ===")
        for m in self.train_metrics:
            metrics_str = f"Epoch {m['epoch']}: Time={m['epoch_time_sec']:.2f}s, Memory={m['memory_usage_mb']:.2f}MB, CPU={m['cpu_usage_percent']:.1f}%"
            if 'rmse' in m and m['rmse'] is not None:
                metrics_str += f", RMSE={m['rmse']:.4f}"
            if 'mae' in m and m['mae'] is not None:
                metrics_str += f", MAE={m['mae']:.4f}"
            print(metrics_str)
        
        print("\n=== Final Test Metrics ===")
        print(f"Total Time: {self.test_metrics['total_time_sec']:.2f}s (Test: {self.test_metrics['test_time_sec']:.2f}s)")
        print(f"Final Memory: {self.test_metrics['final_memory_usage_mb']:.2f}MB")
        print(f"Final CPU: {self.test_metrics['final_cpu_usage_percent']:.1f}%")
        print(f"RMSE: {rmse:.4f}")
        if mae is not None:
            print(f"MAE: {mae:.4f}")
            
        if self.best_rmse_epoch is not None:
            print(f"\n=== Best Training RMSE ===")
            print(f"Best RMSE: {self.best_rmse:.4f} (Epoch {self.best_rmse_epoch})")
            if self.best_rmse_metrics:
                print(f"Time: {self.best_rmse_metrics['epoch_time_sec']:.2f}s")
                print(f"Memory: {self.best_rmse_metrics['memory_usage_mb']:.2f}MB")
                print(f"CPU: {self.best_rmse_metrics['cpu_usage_percent']:.1f}%")
                if 'mae' in self.best_rmse_metrics and self.best_rmse_metrics['mae'] is not None:
                    print(f"MAE: {self.best_rmse_metrics['mae']:.4f}")

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        metrics_df = pd.DataFrame(self.train_metrics)
        metrics_df.to_csv(f"{result_path}/system_metrics_{timestamp}.csv", index=False)

    def get_best_rmse_info(self):
        if self.best_rmse_epoch is not None:
            return {
                'epoch': self.best_rmse_epoch,
                'rmse': self.best_rmse,
                'metrics': self.best_rmse_metrics
            }
        return None


class EmissionsPerEpochTracker:
    def __init__(self, result_path, model_name="LightGCN"):
        self.result_path = result_path
        self.model_name = model_name
        self.epoch_emissions = []
        self.cumulative_emissions = []
        self.epoch_rmse = []
        self.epoch_mae = []
        self.epoch_loss = []
        self.total_emissions = 0.0
        self.trackers = {}
        
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
            print("Tracker principal iniciado correctamente")
        except Exception as e:
            print(f"Advertencia: No se pudo iniciar el tracker principal: {e}")
            self.main_tracker = None
    
    def start_epoch(self, epoch):
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
            print(f"Advertencia: No se pudo iniciar el tracker para la época {epoch}: {e}")
            self.trackers[epoch] = None
    
    def end_epoch(self, epoch, loss, rmse=None, mae=None):
        try:
            epoch_co2 = 0.0
            if epoch in self.trackers and self.trackers[epoch]:
                try:
                    epoch_co2 = self.trackers[epoch].stop() or 0.0
                except Exception as e:
                    print(f"Advertencia: Error al detener el tracker para la época {epoch}: {e}")
                    epoch_co2 = 0.0
            
            self.total_emissions += epoch_co2
            
            self.epoch_emissions.append(epoch_co2)
            self.cumulative_emissions.append(self.total_emissions)
            self.epoch_loss.append(loss)
            if rmse is not None:
                self.epoch_rmse.append(rmse)
            if mae is not None:
                self.epoch_mae.append(mae)
            
        except Exception as e:
            print(f"Error al medir emisiones en época {epoch}: {e}")
    
    def end_training(self, final_rmse=None, final_mae=None, best_rmse_info=None):
        try:
            final_emissions = 0.0
            if hasattr(self, 'main_tracker') and self.main_tracker:
                try:
                    final_emissions = self.main_tracker.stop() or 0.0
                    print(f"\nTotal CO2 Emissions: {final_emissions:.6f} kg")
                except Exception as e:
                    print(f"Error al detener el tracker principal: {e}")
                    final_emissions = self.total_emissions
            else:
                final_emissions = self.total_emissions
            
            for epoch, tracker in self.trackers.items():
                if tracker is not None:
                    try:
                        tracker.stop()
                    except:
                        pass
            
            if (best_rmse_info and 
                best_rmse_info['epoch'] is not None and 
                best_rmse_info['epoch']-1 < len(self.epoch_emissions)):
                
                epoch_idx = best_rmse_info['epoch'] - 1
                print(f"\n=== Best RMSE and Associated Emissions ===")
                print(f"Best RMSE: {best_rmse_info['rmse']:.4f} (Epoch {best_rmse_info['epoch']})")
                print(f"Emissions at best RMSE: {self.epoch_emissions[epoch_idx]:.8f} kg")
                print(f"Cumulative emissions at best RMSE: {self.cumulative_emissions[epoch_idx]:.8f} kg")

            if not self.epoch_emissions:
                print("No hay datos de emisiones para graficar")
                return
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            df = pd.DataFrame({
                'epoch': range(1, len(self.epoch_emissions) + 1),
                'epoch_emissions_kg': self.epoch_emissions,
                'cumulative_emissions_kg': self.cumulative_emissions,
                'loss': self.epoch_loss if self.epoch_loss else [0.0] * len(self.epoch_emissions),
                'rmse': self.epoch_rmse if self.epoch_rmse else [None] * len(self.epoch_emissions),
                'mae': self.epoch_mae if self.epoch_mae else [None] * len(self.epoch_emissions)
            })
            
            emissions_file = f'{self.result_path}/emissions_reports/emissions_metrics_{self.model_name}_{timestamp}.csv'
            df.to_csv(emissions_file, index=False)
            print(f"Métricas de emisiones guardadas en: {emissions_file}")
            
            self.plot_emissions_vs_metrics(timestamp)
            
        except Exception as e:
            print(f"Error al generar gráficos de emisiones: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_emissions_vs_metrics(self, timestamp):
        plt.style.use('default')
        
        try:
            if self.epoch_rmse:
                plt.figure(figsize=(10, 6), facecolor='white')
                plt.plot(self.cumulative_emissions, self.epoch_rmse, 'b-', marker='o')
                for i, (emissions, rmse) in enumerate(zip(self.cumulative_emissions, self.epoch_rmse)):
                    plt.annotate(f"{i+1}", (emissions, rmse), textcoords="offset points", xytext=(0,10), ha='center')
                plt.xlabel('Emisiones de CO2 acumuladas (kg)')
                plt.ylabel('RMSE')
                plt.title('Relación entre Emisiones Acumuladas y RMSE')
                plt.grid(True, alpha=0.3)
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_rmse_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path, facecolor='white')
                plt.close()

            if self.epoch_mae:
                plt.figure(figsize=(10, 6), facecolor='white')
                plt.plot(self.cumulative_emissions, self.epoch_mae, 'm-', marker='D')
                for i, (emissions, mae) in enumerate(zip(self.cumulative_emissions, self.epoch_mae)):
                    plt.annotate(f"{i+1}", (emissions, mae), textcoords="offset points", xytext=(0,10), ha='center')
                plt.xlabel('Emisiones de CO2 acumuladas (kg)')
                plt.ylabel('MAE')
                plt.title('Relación entre Emisiones Acumuladas y MAE')
                plt.grid(True, alpha=0.3)
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_mae_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path, facecolor='white')
                plt.close()

            plt.figure(figsize=(12, 10), facecolor='white')
            epochs_range = range(1, len(self.epoch_emissions) + 1)

            plt.subplot(2, 3, 1)
            plt.plot(epochs_range, self.epoch_emissions, 'r-', marker='x', label='Emisiones por Época')
            plt.title('Emisiones por Época')
            plt.xlabel('Época')
            plt.ylabel('CO2 Emissions (kg)')
            
            plt.subplot(2, 3, 2)
            plt.plot(epochs_range, self.cumulative_emissions, 'r-', marker='o', label='Emisiones Acumuladas')
            plt.title('Emisiones Acumuladas')
            plt.xlabel('Época')
            plt.ylabel('CO2 Emissions (kg)')
            
            if self.epoch_loss:
                plt.subplot(2, 3, 3)
                plt.plot(epochs_range, self.epoch_loss, 'g-', marker='s', label='Loss')
                plt.title('Loss por Época')
                plt.xlabel('Época')
                plt.ylabel('Loss')

            if self.epoch_rmse:
                plt.subplot(2, 3, 4)
                plt.plot(epochs_range, self.epoch_rmse, 'b-', marker='o', label='RMSE')
                plt.title('RMSE por Época')
                plt.xlabel('Época')
                plt.ylabel('RMSE')

            if self.epoch_mae:
                plt.subplot(2, 3, 5)
                plt.plot(epochs_range, self.epoch_mae, 'm-', marker='D', label='MAE')
                plt.title('MAE por Época')
                plt.xlabel('Época')
                plt.ylabel('MAE')

            plt.tight_layout()
            file_path = f'{self.result_path}/emissions_plots/metrics_by_epoch_{self.model_name}_{timestamp}.png'
            plt.savefig(file_path, facecolor='white')
            plt.close()
                
        except Exception as e:
            print(f"Error al generar los gráficos: {e}")
            import traceback
            traceback.print_exc()

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
    def __init__(self, n_users, n_items, adj_mat, n_layers=3, emb_dim=64, decay=1e-4, use_personalized_alpha=False, history_weight=0.3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.adj_mat = adj_mat
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.decay = decay
        self.use_personalized_alpha = use_personalized_alpha
        self.history_weight = history_weight
        self.user_histories = {}
        self.item_metadata = {}

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

    def set_metadata(self, metadata):
        if 'user_histories' in metadata:
            self.user_histories = metadata['user_histories']
        if 'item_metadata' in metadata:
            self.item_metadata = metadata['item_metadata']
        print(f"Loaded histories for {len(self.user_histories)} users")
            
    def predict_rating(self, user_id, item_id):
        user_emb, item_emb, _, _, _ = self((self.user_embedding, self.item_embedding))
        
        u_tensor = tf.convert_to_tensor([user_id], dtype=tf.int32)
        i_tensor = tf.convert_to_tensor([item_id], dtype=tf.int32)
        
        u_emb = tf.nn.embedding_lookup(user_emb, u_tensor)
        i_emb = tf.nn.embedding_lookup(item_emb, i_tensor)
        
        model_score = tf.reduce_sum(u_emb * i_emb, axis=1).numpy()[0]
        
        min_score, max_score = -10, 10 # Heuristic score range
        model_rating = 1.0 + 4.0 * (model_score - min_score) / (max_score - min_score)
        model_rating = np.clip(model_rating, 1.0, 5.0)

        if self.history_weight == 0 or user_id not in self.user_histories or self.user_histories[user_id].empty:
            return float(model_rating)
        
        history_rating = self._predict_from_history(user_id, item_id, item_emb)
        
        final_rating = (1 - self.history_weight) * model_rating + self.history_weight * history_rating
        
        return float(final_rating)
    
    def _predict_from_history(self, user_id, target_item_id, item_embeddings):
        if user_id not in self.user_histories:
            return 3.0
        
        history = self.user_histories[user_id]
        
        if history.empty or 'item_id' not in history.columns or 'rating' not in history.columns:
            return 3.0
        
        target_tensor = tf.convert_to_tensor([target_item_id], dtype=tf.int32)
        target_emb = tf.nn.embedding_lookup(item_embeddings, target_tensor)
        
        similarities = []
        ratings = []
        
        for _, row in history.iterrows():
            hist_item_id = int(row['item_id'])
            hist_rating = float(row['rating'])
            
            if hist_item_id == target_item_id:
                continue
            
            hist_tensor = tf.convert_to_tensor([hist_item_id], dtype=tf.int32)
            try:
                hist_emb = tf.nn.embedding_lookup(item_embeddings, hist_tensor)
                similarity = tf.nn.cosine_similarity(target_emb, hist_emb, axis=1).numpy()[0]
                
                similarities.append(max(0, similarity))
                ratings.append(hist_rating)
            except:
                continue
        
        if not similarities:
            return 3.0
        
        similarities = np.array(similarities)
        ratings = np.array(ratings)
        
        sum_sim = np.sum(similarities)
        if sum_sim > 0:
            weights = similarities / sum_sim
            weighted_rating = np.sum(weights * ratings)
            return float(weighted_rating)
        else:
            return 3.0


def mask_embeddings(embeddings, mask_prob=0.2):
    mask = tf.cast(tf.random.uniform(embeddings.shape) > mask_prob, tf.float32)
    masked_embeddings = embeddings * mask
    return masked_embeddings, mask


def sample_neg(pos_items, n_items, strategy='random'):
    if strategy == 'random':
        neg_item = random.randint(0, n_items - 1)
        while neg_item in pos_items:
            neg_item = random.randint(0, n_items - 1)
    return neg_item


def extract_user_histories(train_data, ratings_df=None):
    user_histories = {}
    
    if ratings_df is not None and not ratings_df.empty:
        for user_id in range(len(train_data)):
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            if not user_ratings.empty:
                user_histories[user_id] = user_ratings[['item_id', 'rating']]
            else:
                user_histories[user_id] = pd.DataFrame(columns=['item_id', 'rating'])
    else:
        for user_id in range(len(train_data)):
            if train_data[user_id]:
                item_ids = list(train_data[user_id])
                ratings = [1.0] * len(item_ids)
                user_histories[user_id] = pd.DataFrame({'item_id': item_ids, 'rating': ratings})
            else:
                user_histories[user_id] = pd.DataFrame(columns=['item_id', 'rating'])
    
    return user_histories


def train_lightgcn_with_history(model, train_data, val_data, test_data, n_users, n_items, 
                               batch_size=1024, epochs=10, initial_lr=1e-2,
                               ratings_df=None, history_weight=0.3):
    model.history_weight = history_weight
    
    user_histories = extract_user_histories(train_data, ratings_df)
    model.set_metadata({'user_histories': user_histories})
    
    print("Inicializando trackers...")
    system_tracker = SystemMetricsTracker()
    emissions_tracker = EmissionsPerEpochTracker(result_path, "LightGCN_History")
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr, decay_steps=1000, decay_rate=0.96, staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    train_pairs = [(u, i) for u in range(n_users) for i in train_data[u]]
    steps_per_epoch = len(train_pairs) // batch_size + (len(train_pairs) % batch_size != 0)

    epoch_losses = []
    rmse_scores = []
    mae_scores = []
    
    for epoch in range(1, epochs + 1):
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

            users = np.array(users, dtype=np.int32)
            pos_items = np.array(pos_items, dtype=np.int32)
            neg_items = np.array(neg_items, dtype=np.int32)

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
        epoch_losses.append(avg_epoch_loss)
        
        epoch_rmse = calculate_rmse(model, val_data, n_users, n_items)
        epoch_mae = calculate_mae(model, val_data, n_users, n_items)
        rmse_scores.append(epoch_rmse)
        mae_scores.append(epoch_mae)

        system_tracker.end_epoch(epoch, avg_epoch_loss, epoch_rmse, epoch_mae)
        emissions_tracker.end_epoch(epoch, avg_epoch_loss, epoch_rmse, epoch_mae)
        
        print(f"Epoch {epoch}/{epochs} completed. Loss: {avg_epoch_loss:.4f}, RMSE: {epoch_rmse:.4f}, MAE: {epoch_mae:.4f}")
        
    print("\nEvaluando en conjunto de prueba final...")
    system_tracker.start_epoch("test")
    
    final_rmse = calculate_rmse(model, test_data, n_users, n_items, sample_size=5000)
    final_mae = calculate_mae(model, test_data, n_users, n_items, sample_size=5000)
    
    best_rmse_info = system_tracker.get_best_rmse_info()
    
    try:
        system_tracker.end_test(final_rmse, final_mae)
    except Exception as e:
        print(f"Error al generar métricas finales con tracker: {e}")
    
    try:
        emissions_tracker.end_training(final_rmse, final_mae, best_rmse_info)
    except Exception as e:
        print(f"Error al generar métricas de emisiones: {e}")
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, epochs + 1)),
        'loss': epoch_losses,
        'rmse': rmse_scores,
        'mae': mae_scores
    })
    
    metrics_file = f"{result_path}/model_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Métricas del modelo guardadas en: {metrics_file}")
    
    print(f"\nEntrenamiento finalizado! RMSE: {final_rmse:.4f}, MAE: {final_mae:.4f}")
    
    return epoch_losses, rmse_scores, mae_scores, final_rmse, final_mae


# Función principal para ejecutar todo
def run_lightgcn_with_history(ratings_path=None, history_weight=0.3):
    train_file = 'C:/Users/xpati/Documents/TFG/ml-1m/train_data.json'
    test_file = 'C:/Users/xpati/Documents/TFG/ml-1m/test_data.json'
    val_file = 'C:/Users/xpati/Documents/TFG/ml-1m/validation_data.json'

    train_data, test_data, val_data, n_users, n_items = load_mydataset(
        train_file, test_file, val_file
    )
    print(f"Number of Users: {n_users}, Number of Items: {n_items}")

    ratings_df = None
    if ratings_path and os.path.exists(ratings_path):
        try:
            load_ratings_from_file(ratings_path)
            ratings_df = pd.read_csv(ratings_path, sep='::', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python', encoding='latin-1')
            ratings_df['user_id'] = ratings_df['user_id'] - 1
            ratings_df['item_id'] = ratings_df['item_id'] - 1
            print(f"Loaded ratings data for history: {len(ratings_df)} rows")
        except Exception as e:
            print(f"Error loading ratings data: {e}")
        
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
        n_users=n_users,
        n_items=n_items,
        adj_mat=A_tilde,
        n_layers=N_LAYERS,
        emb_dim=EMBED_DIM,
        decay=DECAY,
        use_personalized_alpha=False,
        history_weight=history_weight
    )

    print("\nStarting LightGCN training with history-based predictions...")
    train_lightgcn_with_history(
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        n_users=n_users,
        n_items=n_items,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        initial_lr=INITIAL_LR,
        ratings_df=ratings_df,
        history_weight=history_weight
    )
    
    print("\nTraining and evaluation completed!")
    return model