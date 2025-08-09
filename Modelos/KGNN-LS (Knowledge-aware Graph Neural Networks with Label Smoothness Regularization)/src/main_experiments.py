import argparse
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_data
from train_Inf import train
import os
import datetime
import random
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error

# Configuración de reproducibilidad
np.random.seed(555)
random.seed(555)
tf.random.set_seed(555)

# Función para crear un modelo baseline simple (sin KG)
def matrix_factorization_model(n_users, n_items, dim):
    """
    Crea un modelo simple de factorización de matrices sin usar el grafo de conocimiento
    """
    from tensorflow.keras.layers import Input, Embedding, Flatten, Dot
    from tensorflow.keras.models import Model
    
    # Input layers
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')
    
    # Embedding layers
    user_embedding = Embedding(input_dim=n_users, output_dim=dim, name='user_embedding')(user_input)
    item_embedding = Embedding(input_dim=n_items, output_dim=dim, name='item_embedding')(item_input)
    
    # Flatten embeddings
    user_vector = Flatten()(user_embedding)
    item_vector = Flatten()(item_embedding)
    
    # Dot product for prediction
    prediction = Dot(axes=1)([user_vector, item_vector])
    
    # Build model
    model = Model(inputs=[user_input, item_input], outputs=prediction)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Función para crear datos con etiquetas aleatorizadas
def randomize_labels(train_data, percentage=0.5):
    """
    Aleatoriza un porcentaje de las etiquetas en los datos de entrenamiento
    """
    randomized_data = train_data.copy()
    n_samples = int(len(train_data) * percentage)
    indices = np.random.choice(len(train_data), n_samples, replace=False)
    
    # Invertir las etiquetas (0->1, 1->0)
    randomized_data[indices, 2] = 1 - randomized_data[indices, 2]
    
    return randomized_data

# Función para evaluar modelo baseline
def evaluate_baseline(train_data, test_data, n_users, n_items, dim, epochs=5, batch_size=1024):
    """
    Entrena y evalúa un modelo simple de factorización de matrices
    """
    # Crear modelo
    model = matrix_factorization_model(n_users, n_items, dim)
    
    # Preparar datos
    X_train = [train_data[:, 0], train_data[:, 1]]
    y_train = train_data[:, 2]
    X_test = [test_data[:, 0], test_data[:, 1]]
    y_test = test_data[:, 2]
    
    # Entrenar modelo
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluar modelo
    y_pred = model.predict(X_test)
    test_auc = roc_auc_score(y_test, y_pred)
    test_f1 = f1_score(y_test, np.round(y_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Baseline MF Model Results:")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  F1: {test_f1:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    
    return test_auc, test_f1, test_rmse, history

# Función para ejecutar el experimento principal con KGNN-LS
def run_kgnn_experiment(args, data, show_loss=False, show_topk=False):
    """
    Ejecuta el experimento estándar con KGNN-LS y devuelve los resultados
    """
    # Reiniciar grafo de TensorFlow
    tf.compat.v1.reset_default_graph()
    
    # Usar un scope de variables único
    with tf.compat.v1.variable_scope(f"kgnn_standard_{int(time.time())}"):
        results = train(args, data, show_loss, show_topk)
    
    # Extraer métricas finales para compatibilidad con otros métodos
    if results is None:
        # Si train no devuelve resultados, crear un diccionario con valores por defecto
        return {
            'test_auc': 0.0,
            'test_f1': 0.0,
            'test_rmse': 0.0
        }
    return results

# Función para ejecutar el experimento con etiquetas aleatorias
def run_randomized_experiment(args, data, percentage=0.5, show_loss=False, show_topk=False):
    """
    Ejecuta KGNN-LS con un porcentaje de etiquetas aleatorizadas
    """
    # Reiniciar grafo de TensorFlow
    tf.compat.v1.reset_default_graph()
    
    # Hacer copia de los datos
    randomized_data = list(data)
    # Aleatorizar etiquetas en los datos de entrenamiento
    randomized_data[4] = randomize_labels(data[4], percentage)
    
    print(f"\n=== Running KGNN-LS with {percentage*100:.0f}% randomized labels ===\n")
    
    # Usar un scope de variables único para evitar conflictos
    with tf.compat.v1.variable_scope(f"kgnn_random_{percentage}_{int(time.time())}"):
        results = train(args, randomized_data, show_loss, show_topk)
    
    # Extraer métricas finales para compatibilidad con otros métodos
    if results is None:
        # Si train no devuelve resultados, crear un diccionario con valores por defecto
        return {
            'test_auc': 0.0,
            'test_f1': 0.0,
            'test_rmse': 0.0
        }
    return results

# Función para reducir dimensionalidad del modelo
def run_smaller_model_experiment(args, data, dim_reduction=0.5, show_loss=False, show_topk=False):
    """
    Ejecuta KGNN-LS con un modelo más pequeño
    """
    # Reiniciar grafo de TensorFlow
    tf.compat.v1.reset_default_graph()
    
    # Hacer copia de los argumentos
    smaller_args = argparse.Namespace(**vars(args))
    # Reducir dimensionalidad de embeddings
    smaller_args.dim = max(4, int(args.dim * dim_reduction))
    
    print(f"\n=== Running KGNN-LS with smaller embeddings (dim={smaller_args.dim}) ===\n")
    
    # Usar un scope de variables único
    with tf.compat.v1.variable_scope(f"kgnn_small_{smaller_args.dim}_{int(time.time())}"):
        results = train(smaller_args, data, show_loss, show_topk)
    
    # Extraer métricas finales para compatibilidad con otros métodos
    if results is None:
        # Si train no devuelve resultados, crear un diccionario con valores por defecto
        return {
            'test_auc': 0.0,
            'test_f1': 0.0,
            'test_rmse': 0.0
        }
    return results

# Función para configurar los experimentos
def run_experiments():
    parser = argparse.ArgumentParser()
    
    # Configuración básica
    parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
    parser.add_argument('--n_epochs', type=int, default=15, help='the number of epochs')
    parser.add_argument('--neighbor_sample_size', type=int, default=16, help='the number of neighbors to be sampled')
    parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
    parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
    parser.add_argument('--ls_weight', type=float, default=1.0, help='weight of LS regularization')
    parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
    parser.add_argument('--track_emissions', type=bool, default=True, help='track CO2 emissions')
    parser.add_argument('--experiments', type=str, default='all', 
                       help='which experiments to run: all, standard, baseline, randomized, smaller')
    parser.add_argument('--short_epochs', type=int, default=5, 
                       help='use fewer epochs for experiments to save time')
    
    args = parser.parse_args()
    
    # Crear directorio para resultados
    os.makedirs("./experiment_results", exist_ok=True)
    
    # Cargar datos
    print("Loading data...")
    data = load_data(args)
    n_user, n_item = data[0], data[1]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    
    # Para experimentos más rápidos
    if args.short_epochs > 0 and args.short_epochs < args.n_epochs:
        short_args = argparse.Namespace(**vars(args))
        short_args.n_epochs = args.short_epochs
    else:
        short_args = args
    
    results = {}
    
    try:
        # Ejecutar experimentos
        if args.experiments == 'all' or args.experiments == 'standard':
            print("\n=== Running Standard KGNN-LS Model ===\n")
            results['standard'] = run_kgnn_experiment(args, data)
        
        if args.experiments == 'all' or args.experiments == 'baseline':
            print("\n=== Running Simple MF Baseline ===\n")
            results['baseline'] = evaluate_baseline(train_data, test_data, n_user, n_item, args.dim)
        
        if args.experiments == 'all' or args.experiments == 'randomized':
            print("\n=== Running KGNN-LS with Randomized Labels ===\n")
            # Usar menos épocas para los experimentos secundarios
            results['randomized_25'] = run_randomized_experiment(short_args, data, percentage=0.25)
            results['randomized_50'] = run_randomized_experiment(short_args, data, percentage=0.50)
        
        if args.experiments == 'all' or args.experiments == 'smaller':
            print("\n=== Running Smaller KGNN-LS Model ===\n")
            results['smaller'] = run_smaller_model_experiment(short_args, data, dim_reduction=0.5)
        
        # Guardar resultados
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(f"./experiment_results/results_summary_{timestamp}.txt", "w") as f:
            for exp_name, exp_results in results.items():
                f.write(f"=== {exp_name.upper()} ===\n")
                f.write(str(exp_results))
                f.write("\n\n")
        
        # Visualizar comparación
        if len(results) > 1:
            plot_comparison(results, timestamp)
        
        return results
    
    except Exception as e:
        print(f"Error during experiments: {e}")
        import traceback
        traceback.print_exc()
        return results

# Función para visualizar comparación
def plot_comparison(results, timestamp):
    try:
        # Extraer métricas para comparar
        experiments = list(results.keys())
        
        # Función para extraer valores de manera segura
        def extract_value(res, key, default=0.0):
            if isinstance(res, dict):
                return res.get(key, default)
            elif isinstance(res, tuple) and len(res) >= 3:
                if key == 'test_auc':
                    return res[0]
                elif key == 'test_f1':
                    return res[1]
                elif key == 'test_rmse':
                    return res[2]
            return default
        
        auc_values = [extract_value(results[exp], 'test_auc') for exp in experiments]
        f1_values = [extract_value(results[exp], 'test_f1') for exp in experiments]
        rmse_values = [extract_value(results[exp], 'test_rmse') for exp in experiments]
        
        # Crear gráficos de comparación
        plt.figure(figsize=(15, 5))
        
        # AUC
        plt.subplot(1, 3, 1)
        bars = plt.bar(experiments, auc_values, color='skyblue')
        plt.title('AUC Comparison')
        plt.ylim(0, 1)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}', ha='center', va='bottom')
        
        # F1
        plt.subplot(1, 3, 2)
        bars = plt.bar(experiments, f1_values, color='lightgreen')
        plt.title('F1 Comparison')
        plt.ylim(0, 1)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}', ha='center', va='bottom')
        
        # RMSE
        plt.subplot(1, 3, 3)
        bars = plt.bar(experiments, rmse_values, color='salmon')
        plt.title('RMSE Comparison (lower is better)')
        plt.ylim(0, max(rmse_values) * 1.2)  # Ajustar escala
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"./experiment_results/metrics_comparison_{timestamp}.png")
        plt.close()
        
        print(f"Comparison plot saved to: ./experiment_results/metrics_comparison_{timestamp}.png")
    
    except Exception as e:
        print(f"Error generating comparison plot: {e}")
        import traceback
        traceback.print_exc()
        
# Añade este código al final de tu script main_experiments.py o ejecútalo en un notebook
def analyze_dataset_balance(data):
    """Analiza el balance de clases en el dataset"""
    train_data = data[4]
    eval_data = data[5]
    test_data = data[6]
    
    train_pos = np.sum(train_data[:, 2] == 1)
    train_neg = np.sum(train_data[:, 2] == 0)
    eval_pos = np.sum(eval_data[:, 2] == 1)
    eval_neg = np.sum(eval_data[:, 2] == 0)
    test_pos = np.sum(test_data[:, 2] == 1)
    test_neg = np.sum(test_data[:, 2] == 0)
    
    print("Dataset balance analysis:")
    print(f"  Train: {train_pos} positives ({train_pos/(train_pos+train_neg)*100:.2f}%), {train_neg} negatives")
    print(f"  Eval:  {eval_pos} positives ({eval_pos/(eval_pos+eval_neg)*100:.2f}%), {eval_neg} negatives")
    print(f"  Test:  {test_pos} positives ({test_pos/(test_pos+test_neg)*100:.2f}%), {test_neg} negatives")

if __name__ == "__main__":
    results = run_experiments()
    print("\nExperiments completed! Check the 'experiment_results' directory for detailed results.")