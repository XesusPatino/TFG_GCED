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

# Función para calcular RMSE para LightGCN
def calculate_rmse(model, test_data, n_users, n_items, sample_size=10000):
    """
    Calcula el RMSE entre las calificaciones reales y las predicciones del modelo LightGCN.
    
    Args:
        model: Modelo LightGCN entrenado
        test_data: Datos de prueba con interacciones usuario-item
        n_users: Número total de usuarios
        n_items: Número total de ítems
        sample_size: Tamaño máximo de muestra para cálculo (para eficiencia)
    
    Returns:
        float: RMSE calculado
    """
    user_emb, item_emb, _, _, _ = model((model.user_embedding, model.item_embedding))
    
    # Recolectar pares usuario-item de los datos de prueba
    y_true = []  # Valores reales (1 para interacciones positivas)
    y_scores = []  # Puntuaciones predichas
    
    # Limitar la cantidad de muestras para eficiencia
    count = 0
    test_pairs = []
    neg_pairs = []
    
    # Crear pares positivos (usuario-item que interactuaron)
    for u in range(n_users):
        if len(test_data[u]) > 0:
            for i in test_data[u]:
                test_pairs.append((u, i, 1.0))  # 1.0 indica interacción positiva
                count += 1
                if count >= sample_size // 2:
                    break
        if count >= sample_size // 2:
            break
    
    # Crear algunos pares negativos para comparación
    count = 0
    for u in range(n_users):
        if count >= sample_size // 2:
            break
        if len(test_data[u]) > 0:
            for _ in range(min(5, len(test_data[u]))):  # Generar algunos negativos por usuario
                neg_item = random.randint(0, n_items - 1)
                while neg_item in test_data[u]:
                    neg_item = random.randint(0, n_items - 1)
                neg_pairs.append((u, neg_item, 0.0))  # 0.0 indica que no hay interacción
                count += 1
                if count >= sample_size // 2:
                    break
    
    # Combinar y mezclar pares positivos y negativos
    all_pairs = test_pairs + neg_pairs
    random.shuffle(all_pairs)
    
    # Calcular puntuaciones predichas
    for u, i, true_val in all_pairs:
        u_tensor = tf.convert_to_tensor([u], dtype=tf.int32)
        i_tensor = tf.convert_to_tensor([i], dtype=tf.int32)
        
        u_emb_lookup = tf.nn.embedding_lookup(user_emb, u_tensor)
        i_emb_lookup = tf.nn.embedding_lookup(item_emb, i_tensor)
        
        # Calcular similitud de coseno como predicción (normalizada entre 0 y 1)
        pred = tf.reduce_sum(u_emb_lookup * i_emb_lookup, axis=1)
        pred_val = tf.sigmoid(pred).numpy()[0]  # Convertir a valor entre 0 y 1
        
        y_true.append(true_val)
        y_scores.append(pred_val)
    
    # Calcular RMSE
    rmse = math.sqrt(mean_squared_error(y_true, y_scores))
    return rmse

# Clases para seguimiento de métricas
class SystemMetricsTracker:
    def __init__(self):
        self.train_metrics = []
        self.test_metrics = {}
        self.start_time = time.time()
        
    def start_epoch(self, epoch):
        self.epoch_start_time = time.time()
        self.current_epoch_metrics = {
            'epoch': epoch,
            'memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'cpu_usage_percent': psutil.cpu_percent(),
        }
        
    def end_epoch(self, epoch, loss, recall=None, ndcg=None, rmse=None):
        epoch_time = time.time() - self.epoch_start_time
        self.current_epoch_metrics['epoch_time_sec'] = epoch_time
        self.current_epoch_metrics['loss'] = loss
        if recall is not None:
            self.current_epoch_metrics['recall'] = recall
        if ndcg is not None:
            self.current_epoch_metrics['ndcg'] = ndcg
        if rmse is not None:
            self.current_epoch_metrics['rmse'] = rmse
        self.train_metrics.append(self.current_epoch_metrics)
        
        # Imprimir resumen de época
        print(f"\nEpoch {epoch} Metrics:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Memory: {self.current_epoch_metrics['memory_usage_mb']:.2f}MB")
        print(f"  CPU: {self.current_epoch_metrics['cpu_usage_percent']:.1f}%")
        print(f"  Loss: {loss:.4f}")
        if recall is not None:
            print(f"  Recall: {recall:.4f}")
        if ndcg is not None:
            print(f"  NDCG: {ndcg:.4f}")
        if rmse is not None:
            print(f"  RMSE: {rmse:.4f}")
        
    def end_test(self, recall, ndcg=None, rmse=None):
        self.test_metrics = {
            'test_time_sec': time.time() - self.epoch_start_time,
            'total_time_sec': time.time() - self.start_time,
            'final_memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            'final_cpu_usage_percent': psutil.cpu_percent(),
            'test_recall': recall,
        }
        if ndcg is not None:
            self.test_metrics['test_ndcg'] = ndcg
        if rmse is not None:
            self.test_metrics['test_rmse'] = rmse
        
        # Imprimir métricas finales
        print("\n=== Final Training Metrics ===")
        for m in self.train_metrics:
            metrics_str = f"Epoch {m['epoch']}: Time={m['epoch_time_sec']:.2f}s, Memory={m['memory_usage_mb']:.2f}MB, CPU={m['cpu_usage_percent']:.1f}%, Loss={m['loss']:.4f}"
            if 'recall' in m:
                metrics_str += f", Recall={m['recall']:.4f}"
            if 'ndcg' in m:
                metrics_str += f", NDCG={m['ndcg']:.4f}"
            if 'rmse' in m:
                metrics_str += f", RMSE={m['rmse']:.4f}"
            print(metrics_str)
        
        print("\n=== Final Test Metrics ===")
        print(f"Total Time: {self.test_metrics['total_time_sec']:.2f}s (Test: {self.test_metrics['test_time_sec']:.2f}s)")
        print(f"Final Memory: {self.test_metrics['final_memory_usage_mb']:.2f}MB")
        print(f"Final CPU: {self.test_metrics['final_cpu_usage_percent']:.1f}%")
        print(f"Test Recall: {recall:.4f}")
        if ndcg is not None:
            print(f"Test NDCG: {ndcg:.4f}")
        if rmse is not None:
            print(f"Test RMSE: {rmse:.4f}")
        
        # Guardar métricas en CSV
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        metrics_df = pd.DataFrame(self.train_metrics)
        metrics_df.to_csv(f"{result_path}/system_metrics_{timestamp}.csv", index=False)


class EmissionsPerEpochTracker:
    def __init__(self, result_path, model_name="LightGCN"):
        self.result_path = result_path
        self.model_name = model_name
        self.epoch_emissions = []
        self.cumulative_emissions = []
        self.epoch_recall = []
        self.epoch_ndcg = []
        self.epoch_rmse = []  # Añadir lista para RMSE
        self.epoch_loss = []
        self.total_emissions = 0.0
        self.trackers = {}
        
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
            print("Tracker principal iniciado correctamente")
        except Exception as e:
            print(f"Advertencia: No se pudo iniciar el tracker principal: {e}")
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
            print(f"Advertencia: No se pudo iniciar el tracker para la época {epoch}: {e}")
            self.trackers[epoch] = None
    
    def end_epoch(self, epoch, loss, recall=None, ndcg=None, rmse=None):
        try:
            epoch_co2 = 0.0
            if epoch in self.trackers and self.trackers[epoch]:
                try:
                    epoch_co2 = self.trackers[epoch].stop() or 0.0
                except Exception as e:
                    print(f"Advertencia: Error al detener el tracker para la época {epoch}: {e}")
                    epoch_co2 = 0.0
            
            # Acumular emisiones totales
            self.total_emissions += epoch_co2
            
            # Guardar datos de esta época
            self.epoch_emissions.append(epoch_co2)
            self.cumulative_emissions.append(self.total_emissions)
            self.epoch_loss.append(loss)
            if recall is not None:
                self.epoch_recall.append(recall)
            if ndcg is not None:
                self.epoch_ndcg.append(ndcg)
            if rmse is not None:
                self.epoch_rmse.append(rmse)  # Guardar RMSE
            
            print(f"Epoch {epoch} - Emisiones: {epoch_co2:.8f} kg, Acumulado: {self.total_emissions:.8f} kg, Loss: {loss:.4f}")
            if recall is not None:
                print(f"Recall: {recall:.4f}")
            if ndcg is not None:
                print(f"NDCG: {ndcg:.4f}")
            if rmse is not None:
                print(f"RMSE: {rmse:.4f}")
        except Exception as e:
            print(f"Error al medir emisiones en época {epoch}: {e}")
    
    def end_training(self, final_recall, final_ndcg=None, final_rmse=None):
        try:
            # Detener el tracker principal
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
                if final_recall is not None:
                    self.epoch_recall = [final_recall]
                if final_ndcg is not None:
                    self.epoch_ndcg = [final_ndcg]
                if final_rmse is not None:
                    self.epoch_rmse = [final_rmse]
            
            # Si no hay datos, salir
            if not self.epoch_emissions:
                print("No hay datos de emisiones para graficar")
                return
            
            # Asegurarse de que tengamos un Recall final si no se rastreó por época
            if not self.epoch_recall and final_recall is not None:
                self.epoch_recall = [final_recall] * len(self.epoch_emissions)
            if not self.epoch_rmse and final_rmse is not None:
                self.epoch_rmse = [final_rmse] * len(self.epoch_emissions)
                
            # Crear dataframe con todos los datos
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            df = pd.DataFrame({
                'epoch': range(len(self.epoch_emissions)),
                'epoch_emissions_kg': self.epoch_emissions,
                'cumulative_emissions_kg': self.cumulative_emissions,
                'loss': self.epoch_loss if self.epoch_loss else [0.0] * len(self.epoch_emissions),
                'recall': self.epoch_recall if self.epoch_recall else [None] * len(self.epoch_emissions),
                'ndcg': self.epoch_ndcg if self.epoch_ndcg else [None] * len(self.epoch_emissions),
                'rmse': self.epoch_rmse if self.epoch_rmse else [None] * len(self.epoch_emissions)
            })
            
            emissions_file = f'{self.result_path}/emissions_reports/emissions_metrics_{self.model_name}_{timestamp}.csv'
            df.to_csv(emissions_file, index=False)
            print(f"Métricas de emisiones guardadas en: {emissions_file}")
            
            # Graficar las relaciones
            self.plot_emissions_vs_metrics(timestamp, final_recall, final_ndcg, final_rmse)
            
        except Exception as e:
            print(f"Error al generar gráficos de emisiones: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_emissions_vs_metrics(self, timestamp, final_recall=None, final_ndcg=None, final_rmse=None):
        """Genera gráficos para emisiones vs métricas"""
        
        # Configurar estilo para fondo blanco y texto negro (más legible)
        plt.style.use('default')
        
        try:
            # Añadir gráfico de emisiones acumulativas vs RMSE
            if self.epoch_rmse:
                plt.figure(figsize=(10, 6), facecolor='white')
                plt.plot(self.cumulative_emissions, self.epoch_rmse, 'm-', marker='D')
                
                # Añadir etiquetas con el número de época
                for i, (emissions, rmse) in enumerate(zip(self.cumulative_emissions, self.epoch_rmse)):
                    plt.annotate(f"{i}", (emissions, rmse), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9, color='black')
                    
                plt.xlabel('Emisiones de CO2 acumuladas (kg)', color='black')
                plt.ylabel('RMSE', color='black')
                plt.title('Relación entre Emisiones Acumuladas y RMSE', color='black')
                plt.grid(True, alpha=0.3)
                plt.tick_params(colors='black')
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_rmse_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path, facecolor='white')
                plt.close()
                print(f"Gráfico guardado en: {file_path}")
            
            # Resto de los gráficos existentes
            if self.epoch_recall:
                # 1. Emisiones acumulativas vs Recall
                plt.figure(figsize=(10, 6), facecolor='white')
                plt.plot(self.cumulative_emissions, self.epoch_recall, 'b-', marker='o')
                
                # Añadir etiquetas con el número de época
                for i, (emissions, recall) in enumerate(zip(self.cumulative_emissions, self.epoch_recall)):
                    plt.annotate(f"{i}", (emissions, recall), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9, color='black')
                    
                plt.xlabel('Emisiones de CO2 acumuladas (kg)', color='black')
                plt.ylabel('Recall@20', color='black')
                plt.title('Relación entre Emisiones Acumuladas y Recall', color='black')
                plt.grid(True, alpha=0.3)
                plt.tick_params(colors='black')
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_recall_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path, facecolor='white')
                plt.close()
                print(f"Gráfico guardado en: {file_path}")
            
            
            # 2. Gráfico combinado: Emisiones por época y acumulativas
            plt.figure(figsize=(12, 10), facecolor='white')
            
            plt.subplot(2, 3, 1)
            plt.plot(range(len(self.epoch_emissions)), self.epoch_emissions, 'r-', marker='x')
            plt.title('Emisiones por Época', color='black')
            plt.xlabel('Época', color='black')
            plt.ylabel('CO2 Emissions (kg)', color='black')
            plt.tick_params(colors='black')
            
            plt.subplot(2, 3, 2)
            plt.plot(range(len(self.cumulative_emissions)), self.cumulative_emissions, 'r-', marker='o')
            plt.title('Emisiones Acumuladas por Época', color='black')
            plt.xlabel('Época', color='black')
            plt.ylabel('CO2 Emissions (kg)', color='black')
            plt.tick_params(colors='black')
            
            if self.epoch_loss:
                plt.subplot(2, 3, 3)
                plt.plot(range(len(self.epoch_loss)), self.epoch_loss, 'g-', marker='o')
                plt.title('Loss por Época', color='black')
                plt.xlabel('Época', color='black')
                plt.ylabel('Loss', color='black')
                plt.tick_params(colors='black')
            
            if self.epoch_recall:
                plt.subplot(2, 3, 4)
                plt.plot(range(len(self.epoch_recall)), self.epoch_recall, 'b-', marker='o')
                plt.title('Recall@20 por Época', color='black')
                plt.xlabel('Época', color='black')
                plt.ylabel('Recall', color='black')
                plt.tick_params(colors='black')
            
            if self.epoch_rmse:
                plt.subplot(2, 3, 5)
                plt.plot(range(len(self.epoch_rmse)), self.epoch_rmse, 'm-', marker='D')
                plt.title('RMSE por Época', color='black')
                plt.xlabel('Época', color='black')
                plt.ylabel('RMSE', color='black')
                plt.tick_params(colors='black')
            
            plt.tight_layout()
            
            file_path = f'{self.result_path}/emissions_plots/metrics_by_epoch_{self.model_name}_{timestamp}.png'
            plt.savefig(file_path, facecolor='white')
            plt.close()
            print(f"Gráfico guardado en: {file_path}")
            
            # Gráfico comparativo de varias métricas
            if self.epoch_ndcg and self.epoch_recall and self.epoch_rmse:
                plt.figure(figsize=(10, 6), facecolor='white')
                plt.plot(range(len(self.epoch_recall)), self.epoch_recall, 'b-', marker='o', label='Recall@20')
                plt.plot(range(len(self.epoch_ndcg)), self.epoch_ndcg, 'g-', marker='s', label='NDCG@20')
                
                # Para RMSE, usar un segundo eje Y debido a la diferencia de escala
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                ax2.plot(range(len(self.epoch_rmse)), self.epoch_rmse, 'm-', marker='D', label='RMSE')
                ax2.set_ylabel('RMSE', color='m')
                ax2.tick_params(axis='y', colors='m')
                
                # Añadir las leyendas de ambos ejes
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                lines = lines1 + lines2
                labels = labels1 + labels2
                ax1.legend(lines, labels, loc='best')
                
                plt.title('Comparación de Métricas por Época', color='black')
                plt.xlabel('Época', color='black')
                ax1.set_ylabel('Recall/NDCG', color='black')
                plt.grid(True, alpha=0.3)
                ax1.tick_params(colors='black')
                
                file_path = f'{self.result_path}/emissions_plots/metrics_comparison_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path, facecolor='white')
                plt.close()
                print(f"Gráfico comparativo guardado en: {file_path}")
                
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
        # remove items not appear in train set
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
    # R in the upper-right block
    adj_dok[:n_users, n_users:] = R_csr
    # R^T in the lower-left block
    adj_dok[n_users:, :n_users] = R_csr.transpose()
    return adj_dok.tocsr()


def normalize_adj_sym(adj_mat):
    # symmetric normalization: D^-1/2 * A * D^-1/2.
    rowsum = np.array(adj_mat.sum(axis=1)).flatten() + 1e-9
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt.dot(adj_mat).dot(D_inv_sqrt)

class LightGCNModel(tf.keras.Model):
    def __init__(self, n_users, n_items, adj_mat, n_layers=3, emb_dim=64, decay=1e-4, 
                use_personalized_alpha=False, history_weight=0.3):
        super().__init__()
        # Existing initialization code...
        self.n_users = n_users
        self.n_items = n_items
        self.adj_mat = adj_mat  # TF SparseTensor
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.decay = decay
        self.use_personalized_alpha = use_personalized_alpha
        self.history_weight = history_weight  # Weight for history-based prediction
        
        # User histories and item metadata will be stored here
        self.user_histories = {}
        self.item_metadata = {}

        # Añadir semilla a los inicializadores
        initializer = tf.initializers.GlorotUniform(seed=42)
        self.user_embedding = self.add_weight(
            name='user_embedding',
            shape=(n_users, emb_dim),
            initializer=initializer,
            trainable=True
        )
        # Usar semilla diferente para cada inicialización
        item_initializer = tf.initializers.GlorotUniform(seed=43)
        self.item_embedding = self.add_weight(
            name='item_embedding',
            shape=(n_items, emb_dim),
            initializer=item_initializer,
            trainable=True
        )

        if use_personalized_alpha:
            alpha_initializer = tf.initializers.GlorotUniform(seed=44)
            self.alpha_mlp = tf.keras.Sequential([
                tf.keras.layers.Dense(n_layers + 1, activation='softmax', 
                                     kernel_initializer=alpha_initializer)
            ])
        
        # node attribute prediction (auxiliary task)
        attr_initializer = tf.initializers.GlorotUniform(seed=45)
        self.attribute_predictor = tf.keras.layers.Dense(emb_dim, activation='relu', 
                                                       kernel_initializer=attr_initializer,
                                                       name="attribute_predictor")


    def call(self, embeddings, mask_prob=0.2):
        user_emb, item_emb = embeddings
        all_emb = tf.concat([user_emb, item_emb], axis=0)
        emb_list = [all_emb]
    
        # propagation layers
        for _ in range(self.n_layers):
            all_emb = tf.sparse.sparse_dense_matmul(self.adj_mat, all_emb)
            emb_list.append(all_emb)
    
        # combine embeddings from different layers
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
    
        # node attribute prediction
        masked_user_emb, mask = mask_embeddings(user_final, mask_prob)
        predicted_attributes = self.attribute_predictor(masked_user_emb)
        return user_final, item_final, masked_user_emb, predicted_attributes, mask


    def recommend(self, user_ids, k=10):
        user_final, item_final, _, _, _ = self((self.user_embedding, self.item_embedding))
        user_vecs = tf.gather(user_final, user_ids)
    
        all_recs = []
        for idx, uid in enumerate(user_ids):
            u_vec = user_vecs[idx:idx + 1]
            scores = tf.matmul(u_vec, item_final, transpose_b=True)  # (1, n_items)
            scores_np = scores.numpy().flatten()
            idx_topk = np.argsort(scores_np)[::-1][:k]
            score_topk = scores_np[idx_topk]
            for item_id, sc in zip(idx_topk, score_topk):
                all_recs.append((int(uid), int(item_id), float(sc)))
        return all_recs
        
    # Add method to set user histories
    def set_metadata(self, metadata):
        """
        Set user histories and item metadata for history-based predictions.
        
        Args:
            metadata: Dictionary containing:
                - 'user_histories': Dict mapping user_id to DataFrame of their rating history
                - 'item_metadata': Dict with item metadata (optional)
        """
        if 'user_histories' in metadata:
            self.user_histories = metadata['user_histories']
        if 'item_metadata' in metadata:
            self.item_metadata = metadata['item_metadata']
        print(f"Loaded histories for {len(self.user_histories)} users")
            
    # Add method to predict rating based on embeddings and history
    def predict_rating(self, user_id, item_id):
        """
        Predict rating for a user-item pair using both model embeddings and history.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating (float)
        """
        # Get model-based prediction
        user_emb, item_emb, _, _, _ = self((self.user_embedding, self.item_embedding))
        
        # Convert IDs to tensors
        u_tensor = tf.convert_to_tensor([user_id], dtype=tf.int32)
        i_tensor = tf.convert_to_tensor([item_id], dtype=tf.int32)
        
        # Get embeddings
        u_emb = tf.nn.embedding_lookup(user_emb, u_tensor)
        i_emb = tf.nn.embedding_lookup(item_emb, i_tensor)
        
        # Calculate similarity score (dot product)
        model_score = tf.reduce_sum(u_emb * i_emb, axis=1).numpy()[0]
        
        # Convert to rating scale (assuming 1-5 rating)
        model_rating = 1.0 + 4.0 * tf.sigmoid(model_score).numpy()
        
        # If no history or history weight is 0, return model prediction
        if self.history_weight == 0 or user_id not in self.user_histories or len(self.user_histories[user_id]) == 0:
            return float(model_rating)
        
        # Get history-based prediction
        history_rating = self._predict_from_history(user_id, item_id, item_emb)
        
        # Combine model and history predictions
        final_rating = (1 - self.history_weight) * model_rating + self.history_weight * history_rating
        
        return float(final_rating)
    
    def _predict_from_history(self, user_id, target_item_id, item_embeddings):
        """
        Predict rating based on user's rating history and item similarities.
        
        Args:
            user_id: User ID
            target_item_id: Target item ID
            item_embeddings: Tensor of all item embeddings
            
        Returns:
            Predicted rating based on history
        """
        if user_id not in self.user_histories:
            return 3.0  # Default rating if no history
        
        history = self.user_histories[user_id]
        
        # If history is empty or doesn't have necessary columns
        if len(history) == 0 or 'item_id' not in history.columns or 'rating' not in history.columns:
            return 3.0
        
        # Get target item embedding
        target_tensor = tf.convert_to_tensor([target_item_id], dtype=tf.int32)
        target_emb = tf.nn.embedding_lookup(item_embeddings, target_tensor)
        
        similarities = []
        ratings = []
        
        # Calculate similarity with each item in history
        for _, row in history.iterrows():
            hist_item_id = int(row['item_id'])
            hist_rating = float(row['rating'])
            
            # Skip if it's the same item
            if hist_item_id == target_item_id:
                continue
            
            # Get item embedding
            hist_tensor = tf.convert_to_tensor([hist_item_id], dtype=tf.int32)
            try:
                hist_emb = tf.nn.embedding_lookup(item_embeddings, hist_tensor)
                
                # Calculate cosine similarity
                similarity = tf.nn.cosine_similarity(target_emb, hist_emb, axis=1).numpy()[0]
                
                similarities.append(max(0, similarity))  # Only use positive similarities
                ratings.append(hist_rating)
            except:
                continue
        
        # If no similar items found
        if len(similarities) == 0:
            return 3.0
        
        # Weighted average of ratings based on similarities
        similarities = np.array(similarities)
        ratings = np.array(ratings)
        
        # Normalize similarities
        sum_sim = np.sum(similarities)
        if sum_sim > 0:
            weights = similarities / sum_sim
            weighted_rating = np.sum(weights * ratings)
            return float(weighted_rating)
        else:
            return 3.0  # Default if no positive similarities


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
    """
    Extract user rating histories from training data.
    
    Args:
        train_data: Training data in the format used by LightGCN
        ratings_df: Optional DataFrame containing actual ratings (user_id, item_id, rating)
        
    Returns:
        Dictionary mapping user_id to DataFrame of their rating history
    """
    user_histories = {}
    
    if ratings_df is not None and not ratings_df.empty:
        # If we have a DataFrame with actual ratings
        for user_id in range(len(train_data)):
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            if len(user_ratings) > 0:
                user_histories[user_id] = user_ratings[['item_id', 'rating']]
    else:
        # If we only have interaction data (no explicit ratings)
        # Create dummy ratings (all 1.0 for positive interactions)
        for user_id in range(len(train_data)):
            if len(train_data[user_id]) > 0:
                item_ids = list(train_data[user_id])
                ratings = [1.0] * len(item_ids)  # Assume all positive interactions
                user_histories[user_id] = pd.DataFrame({
                    'item_id': item_ids,
                    'rating': ratings
                })
    
    return user_histories


def train_lightgcn_with_history(model, train_data, val_data, test_data, n_users, n_items, 
                               batch_size=1024, epochs=10, initial_lr=1e-2, k=20,
                               ratings_df=None, history_weight=0.3):
    """
    Train LightGCN model with history-based rating prediction.
    
    Args:
        model: LightGCN model
        train_data, val_data, test_data: Training, validation and test data
        n_users, n_items: Number of users and items
        batch_size, epochs, initial_lr: Training parameters
        k: Top-k items for evaluation
        ratings_df: DataFrame with ratings (user_id, item_id, rating)
        history_weight: Weight for history-based predictions
    """
    # Set history weight
    model.history_weight = history_weight
    
    # Extract and set user histories
    user_histories = extract_user_histories(train_data, ratings_df)
    model.set_metadata({'user_histories': user_histories})
    
    # Initialize trackers (existing code)
    print("Inicializando trackers...")
    system_tracker = SystemMetricsTracker()
    emissions_tracker = EmissionsPerEpochTracker(result_path, "LightGCN")
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # (user, item) pairs from train_data
    train_pairs = [(u, i) for u in range(n_users) for i in train_data[u]]
    steps_per_epoch = len(train_pairs) // batch_size + (len(train_pairs) % batch_size != 0)

    epoch_losses = []
    recall_scores = []
    ndcg_scores = []
    rmse_scores = []  # Añadir lista para RMSE
    
    # Para métricas finales
    tiempo_inicio = time.time()
    
    for epoch in range(1, epochs + 1):
        # Iniciar seguimiento de época
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
            
                # BPR loss: mean( softplus(neg_score - pos_score) )
                pos_scores = tf.reduce_sum(u_emb * pos_emb, axis=1)
                neg_scores = tf.reduce_sum(u_emb * neg_emb, axis=1)
                mf_loss = tf.reduce_mean(tf.nn.softplus(neg_scores - pos_scores))
            
                # node attribute prediction loss
                attribute_loss = tf.reduce_mean(tf.square(masked_user_emb - predicted_attributes) * mask)
            
                # L2 Regularization
                reg_loss = model.decay * (
                    tf.nn.l2_loss(u_emb) + tf.nn.l2_loss(pos_emb) + tf.nn.l2_loss(neg_emb)
                ) / batch_size
            
                # total loss
                loss = mf_loss + reg_loss + 0.1 * attribute_loss  # Weighted auxiliary loss

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss += loss.numpy()
            progbar.add(1, values=[('loss', float(loss))])

        avg_epoch_loss = epoch_loss / steps_per_epoch
        epoch_losses.append(avg_epoch_loss)
        
        # evaluate on validation set
        val_users = [u for u in range(n_users) if len(val_data[u]) > 0]
        val_recs = model.recommend(val_users, k=k)
        epoch_recall = recall_at_k(val_recs, val_data, k=k)
        epoch_ndcg = ndcg(val_recs, val_data, k=k)
        
        # Calcular RMSE en cada época
        epoch_rmse = calculate_rmse(model, val_data, n_users, n_items)
        rmse_scores.append(epoch_rmse)

        recall_scores.append(epoch_recall)
        ndcg_scores.append(epoch_ndcg)
        
        # Actualizar trackers con las métricas
        system_tracker.end_epoch(epoch, avg_epoch_loss, epoch_recall, epoch_ndcg, epoch_rmse)
        emissions_tracker.end_epoch(epoch, avg_epoch_loss, epoch_recall, epoch_ndcg, epoch_rmse)
        
        print(f"Epoch {epoch}/{epochs} completed. Average loss: {avg_epoch_loss:.6f}")
        print(f"Epoch {epoch}: Recall@{k}: {epoch_recall:.6f}, NDCG@{k}: {epoch_ndcg:.6f}, RMSE: {epoch_rmse:.6f}")
        
        # Add demonstration of rating prediction after training
        demonstrate_rating_prediction(model, test_data, ratings_df)

    # Evaluación final en el conjunto de pruebas
    print("\nEvaluando en conjunto de prueba final...")
    system_tracker.start_epoch("test")
    
    test_users = [u for u in range(n_users) if len(test_data[u]) > 0]
    final_metrics = evaluate_lightgcn(model, test_users, test_data, ks=[k])
    final_recall = final_metrics[k][0]  # Recall@k
    final_ndcg = final_metrics[k][1]    # NDCG@k
    
    # Calcular RMSE final en conjunto de prueba
    final_rmse = calculate_rmse(model, test_data, n_users, n_items)
    
    # Finalizar seguimiento de sistemas
    try:
        print("\nGenerando métricas finales del sistema...")
        system_tracker.end_test(final_recall, final_ndcg, final_rmse)
    except Exception as e:
        print(f"Error al generar métricas finales con tracker: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print("\nGenerando gráficos y métricas de emisiones...")
        emissions_tracker.end_training(final_recall, final_ndcg, final_rmse)
    except Exception as e:
        print(f"Error al generar métricas de emisiones: {e}")
        import traceback
        traceback.print_exc()
    
    # Guardar métricas de entrenamiento
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, epochs + 1)),
        'loss': epoch_losses,
        'recall': recall_scores,
        'ndcg': ndcg_scores,
        'rmse': rmse_scores  # Añadir RMSE a las métricas guardadas
    })
    
    metrics_file = f"{result_path}/model_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Métricas del modelo guardadas en: {metrics_file}")
    
    # Mostrar métricas finales (independientes)
    memoria_final = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    cpu_final = psutil.cpu_percent(interval=1.0)
    tiempo_total = time.time() - tiempo_inicio
    
    print("\n" + "="*60)
    print("MÉTRICAS FINALES DEL SISTEMA")
    print("="*60)
    print(f"Memoria final: {memoria_final:.2f} MB")
    print(f"CPU final: {cpu_final:.2f}%")
    print(f"Tiempo total de ejecución: {tiempo_total:.2f} segundos")
    print(f"Recall@{k} final: {final_recall:.4f}")
    print(f"NDCG@{k} final: {final_ndcg:.4f}")
    print(f"RMSE final: {final_rmse:.4f}")
    print("="*60)
    
    # Guardar las métricas finales
    final_metrics_dict = {
        'final_memory_mb': memoria_final,
        'final_cpu_percent': cpu_final,
        'total_time_sec': tiempo_total,
        'final_recall': final_recall,
        'final_ndcg': final_ndcg,
        'final_rmse': final_rmse,
        'timestamp': timestamp
    }
    
    final_metrics_df = pd.DataFrame([final_metrics_dict])
    final_metrics_file = f"{result_path}/final_metrics_{timestamp}.csv"
    final_metrics_df.to_csv(final_metrics_file, index=False)
    print(f"Métricas finales guardadas en: {final_metrics_file}")
    
    # Graficar resultados de entrenamiento incluyendo RMSE
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', linestyle='-', color='b', label="Loss")
    plt.title("LightGCN - Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{result_path}/training_loss_{timestamp}.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(recall_scores) + 1), recall_scores, marker='o', linestyle='-', color='g', label=f"Recall@{k}")
    plt.title(f"LightGCN - Recall@{k}")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{result_path}/recall_{timestamp}.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(ndcg_scores) + 1), ndcg_scores, marker='o', linestyle='-', color='r', label=f"NDCG@{k}")
    plt.title(f"LightGCN - NDCG@{k}")
    plt.xlabel("Epoch")
    plt.ylabel("NDCG")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{result_path}/ndcg_{timestamp}.png")
    plt.close()
    
    # Graficar RMSE
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(rmse_scores) + 1), rmse_scores, marker='D', linestyle='-', color='m', label="RMSE")
    plt.title("LightGCN - RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{result_path}/rmse_{timestamp}.png")
    plt.close()

    print(f"\nEntrenamiento finalizado! Recall@{k}: {final_recall:.4f}, NDCG@{k}: {final_ndcg:.4f}, RMSE: {final_rmse:.4f}")
    
    return epoch_losses, recall_scores, ndcg_scores, rmse_scores, final_recall, final_ndcg, final_rmse


def demonstrate_rating_prediction(model, test_data, ratings_df=None):
    """
    Demonstrate history-based rating prediction with LightGCN.
    
    Args:
        model: Trained LightGCN model
        test_data: Test data
        ratings_df: DataFrame with ratings
    """
    print("\n=== Rating Prediction Demo ===")
    
    # Find some users with test data
    test_users = [u for u in range(len(test_data)) if len(test_data[u]) > 0]
    if not test_users:
        print("No test users found!")
        return
    
    # Sample up to 5 users
    sample_users = random.sample(test_users, min(5, len(test_users)))
    
    for user_id in sample_users:
        # Get an item from test data
        if len(test_data[user_id]) == 0:
            continue
            
        item_id = random.choice(list(test_data[user_id]))
        
        # Get true rating if available
        true_rating = None
        if ratings_df is not None:
            user_item_ratings = ratings_df[(ratings_df['user_id'] == user_id) & 
                                          (ratings_df['item_id'] == item_id)]
            if not user_item_ratings.empty:
                true_rating = user_item_ratings.iloc[0]['rating']
        
        # Get model prediction
        model_only_rating = model.predict_rating(user_id, item_id)
        
        # Temporarily set history weight to 0 to get model-only prediction
        orig_weight = model.history_weight
        model.history_weight = 0
        model_only_rating = model.predict_rating(user_id, item_id)
        model.history_weight = orig_weight
        
        # Get combined prediction
        combined_rating = model.predict_rating(user_id, item_id)
        
        # Print predictions
        print(f"\nUser {user_id}, Item {item_id}:")
        if true_rating is not None:
            print(f"  True Rating: {true_rating:.2f}")
        print(f"  Model-Only Rating: {model_only_rating:.2f}")
        print(f"  Combined Rating: {combined_rating:.2f}")
        
        # Show history information
        if user_id in model.user_histories and len(model.user_histories[user_id]) > 0:
            history = model.user_histories[user_id]
            print(f"  User has rated {len(history)} items in their history")
            print("  Sample of rating history:")
            print(history.head(3))
        else:
            print("  No rating history available for this user")
    
    print("\nHistory-Based Rating Prediction:")
    print(f"The model uses a weighted combination of embeddings-based predictions")
    print(f"and history-based predictions, with history weight = {model.history_weight}")

# Funciones de evaluación existentes
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

def evaluate_lightgcn(model, users, test_data, ks=[5, 10, 20], batch_size=2000):
    all_recs = []
    idx_start = 0
    max_k = max(ks)
    while idx_start < len(users):
        idx_end = min(idx_start + batch_size, len(users))
        user_batch = users[idx_start:idx_end]
        recs_chunk = model.recommend(user_batch, k=max_k)
        all_recs.extend(recs_chunk)
        idx_start = idx_end

    results = {}
    for k in ks:
        rec = recall_at_k(all_recs, test_data, k=k)
        ndcg_ = ndcg(all_recs, test_data, k=k)
        results[k] = (rec, ndcg_)
        print(f"\nEvaluation Results (k={k}):")
        print(f"  Recall@{k}:    {rec:.6f}")
        print(f"  NDCG@{k}:      {ndcg_:.6f}")

    return results

# Función principal para ejecutar todo
def run_lightgcn_with_history(ratings_path=None, history_weight=0.3):
    """
    Run LightGCN with history-based rating prediction.
    
    Args:
        ratings_path: Path to ratings CSV file (user_id, item_id, rating)
        history_weight: Weight for history-based predictions
    """
    train_file = 'C:/Users/xpati/Documents/TFG/ml-1m/train_data.json'
    test_file = 'C:/Users/xpati/Documents/TFG/ml-1m/test_data.json'
    val_file = 'C:/Users/xpati/Documents/TFG/ml-1m/validation_data.json'

    train_data, test_data, val_data, n_users, n_items = load_mydataset(
        train_file, test_file, val_file
    )
    print(f"Number of Users: {n_users}, Number of Items: {n_items}")

    # Load ratings data if available
    ratings_df = None
    if ratings_path and os.path.exists(ratings_path):
        try:
            ratings_df = pd.read_csv(ratings_path)
            print(f"Loaded ratings data: {len(ratings_df)} rows")
        except Exception as e:
            print(f"Error loading ratings data: {e}")
        
    adj_csr = build_adjacency_matrix(train_data, n_users, n_items)
    norm_adj_csr = normalize_adj_sym(adj_csr)

    # convert to TensorFlow SparseTensor
    coo = norm_adj_csr.tocoo().astype(np.float32)
    indices = np.vstack((coo.row, coo.col)).transpose()
    A_tilde = tf.sparse.SparseTensor(indices=indices, values=coo.data, dense_shape=coo.shape)
    A_tilde = tf.sparse.reorder(A_tilde)

    N_LAYERS = 2
    EMBED_DIM = 128
    DECAY = 1e-2
    INITIAL_LR = 1e-3
    EPOCHS = 50
    BATCH_SIZE = 1024

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
        k=20,
        ratings_df=ratings_df,
        history_weight=history_weight
    )
    
    print("\nTraining and evaluation completed!")
    return model

if __name__ == "__main__":
    # Example: Run with history weight of 0.3
    # You can specify the path to a ratings CSV file
    ratings_path = "C:/Users/xpati/Documents/TFG/ml-1m/ratings.csv"  # Path to ratings file if available
    run_lightgcn_with_history(ratings_path=ratings_path, history_weight=0.3)