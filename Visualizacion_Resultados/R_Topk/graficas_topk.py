import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from pathlib import Path
import re

# Configurar el estilo para gráficos profesionales
plt.style.use('default')
sns.set_palette("husl")

# =============================================================================
# CONFIGURACIÓN - Modifica esta sección según tus necesidades
# =============================================================================

# Lista específica de archivos con nombres personalizados
files_config = [
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Topk\Modelos_Matrices\emissions_metrics_MF_20250817-202556.csv",
        'name': 'MF',
        'category': 'Modelos_Matrices',
        'color': '#2E86AB'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Topk\Modelos_Matrices\emissions_metrics_NNMF_20250828-130213.csv",
        'name': 'NNMF',
        'category': 'Modelos_Matrices',
        'color': '#A23B72'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Topk\Modelos_Redes_Profundas\emissions_metrics_AutoRec_20250818-181708.csv",
        'name': 'AutoRec',
        'category': 'Modelos_Redes_Profundas',
        'color': '#D2B48C'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Topk\Modelos_Redes_Profundas\emissions_metrics_NCF_20250828-132012.csv",
        'name': 'NCF',
        'category': 'Modelos_Redes_Profundas',
        'color': '#C73E1D'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Topk\Modelos_Redes_Profundas\emissions_metrics_LRML_TopK_20250819-111619.csv",
        'name': 'LRML',
        'category': 'Modelos_Redes_Profundas',
        'color': '#6A994E'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Topk\Modelos_Redes_Grafos\emissions_metrics_GHRS_20250829-011122.csv",
        'name': 'GHRS',
        'category': 'Modelos_Redes_Grafos',
        'color': '#7209B7'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Topk\Modelos_Redes_Grafos\emissions_metrics_KGNN_LS_20250825-195206.csv",
        'name': 'KGNN-LS',
        'category': 'Modelos_Redes_Grafos',
        'color': '#F77F00'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Topk\Modelos_Redes_Grafos\emissions_metrics_LightGCN_TopK_20250828-0505S.csv",
        'name': 'LightGCN',
        'category': 'Modelos_Redes_Grafos',
        'color': '#003566'
    },
]

# Métricas que vamos a analizar
recall_metrics = ['recall_5', 'recall_10', 'recall_20', 'recall_50']
ndcg_metrics = ['ndcg_5', 'ndcg_10', 'ndcg_20', 'ndcg_50']
all_metrics = recall_metrics + ndcg_metrics

# =============================================================================
# FUNCIONES
# =============================================================================

def clean_tensor_values(df):
    """Limpia los valores que están en formato tf.Tensor()"""
    for col in df.columns:
        if df[col].dtype == 'object':
            # Buscar patrones como tf.Tensor(value, ...)
            pattern = r'tf\.Tensor\(([^,]+)'
            df[col] = df[col].astype(str).str.extract(pattern)[0]
            # Convertir a float si es posible
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    return df

def standardize_columns(df):
    """Estandariza las columnas del DataFrame"""
    # Limpiar valores tensor
    df = clean_tensor_values(df)
    
    # Mapear nombres de columnas inconsistentes
    column_mapping = {
        'recall@5': 'recall_5',
        'Recall@5': 'recall_5',
        'recall@10': 'recall_10', 
        'Recall@10': 'recall_10',
        'recall@20': 'recall_20',
        'Recall@20': 'recall_20',
        'recall@50': 'recall_50',
        'Recall@50': 'recall_50',
        'ndcg@5': 'ndcg_5',
        'NDCG@5': 'ndcg_5',
        'ndcg@10': 'ndcg_10',
        'NDCG@10': 'ndcg_10',
        'ndcg@20': 'ndcg_20',
        'NDCG@20': 'ndcg_20',
        'ndcg@50': 'ndcg_50',
        'NDCG@50': 'ndcg_50'
    }
    
    # Renombrar columnas
    df = df.rename(columns=column_mapping)
    
    # Convertir columnas numéricas
    numeric_columns = ['epoch', 'epoch_emissions_kg', 'cumulative_emissions_kg'] + all_metrics
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def load_csv_data(file_config):
    """Carga y procesa un archivo CSV"""
    try:
        df = pd.read_csv(file_config['path'])
        print(f"Columnas originales en {file_config['name']}: {list(df.columns)}")
        
        df = standardize_columns(df)
        
        print(f"Columnas después de procesamiento: {list(df.columns)}")
        print(f"✓ Cargado: {file_config['name']} ({len(df)} filas)")
        return df
    except Exception as e:
        print(f"✗ Error cargando {file_config['name']}: {e}")
        return None

def plot_single_metric_vs_emissions(df, model_name, metric, color, ax):
    """Plotea una métrica específica vs emisiones"""
    if metric not in df.columns:
        print(f"Advertencia: {model_name} no tiene columna '{metric}'. Columnas disponibles: {list(df.columns)}")
        return ax, None
    
    # Encontrar el mejor valor (máximo para recall)
    best_idx = df[metric].idxmax()
    best_value = df.loc[best_idx, metric]
    best_emissions = df.loc[best_idx, 'cumulative_emissions_kg']
    best_epoch = df.loc[best_idx, 'epoch']

    # Plotear la línea
    ax.plot(df['cumulative_emissions_kg'].values, df[metric].values, 
           color=color, linewidth=2, alpha=0.8, marker='o', markersize=2, 
           label=model_name, linestyle='-')

    # Marcar el mejor punto con un diamante
    ax.scatter(best_emissions, best_value, 
              color=color, s=40, zorder=5, 
              edgecolors='white', linewidth=1, marker='D')

    return ax, (best_value, best_emissions, best_epoch)

def create_individual_plots(dataframes):
    """Crea gráficos individuales para cada modelo con todas las métricas de recall y NDCG"""
    for model_name, model_data in dataframes.items():
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'{model_name}: Métricas TopK vs Emisiones de CO₂', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        axes = axes.flatten()
        df = model_data['data']
        config = model_data['config']
        color = config['color']
        
        # Plotear cada métrica (primero recall, luego ndcg)
        for idx, metric in enumerate(all_metrics):
            ax = axes[idx]
            _, best_results = plot_single_metric_vs_emissions(df, model_name, metric, color, ax)
            
            if best_results:
                best_value, best_emissions, best_epoch = best_results
                ax.set_title(f'{metric.replace("_", "@").upper()}\nMejor: {best_value:.4f} (Epoch {best_epoch})', 
                            fontsize=11, fontweight='bold')
            else:
                ax.set_title(f'{metric.replace("_", "@").upper()}\nDatos no disponibles', 
                            fontsize=11, fontweight='bold')
            
            ax.set_xlabel('Emisiones Acumuladas (kg CO₂)', fontsize=9)
            ax.set_ylabel(metric.replace('_', '@').upper(), fontsize=9)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=8)
        
        plt.tight_layout()
        individual_path = f"C:\\Users\\xpati\\Documents\\TFG\\RESULTADOS_FINAL\\R_Topk\\grafico_{model_name.replace(' ', '_').replace('-', '_')}_topk.png"
        plt.savefig(individual_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Gráfico individual guardado: {individual_path}")
        plt.close()

def create_combined_collage_recall(dataframes):
    """Crea un collage con 4 gráficos combinados (uno por métrica de recall)"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Comparación de Métricas Recall vs Emisiones de CO₂', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    axes = axes.flatten()
    
    # Para cada métrica de recall, crear un gráfico combinado
    for metric_idx, metric in enumerate(recall_metrics):
        ax = axes[metric_idx]
        
        # Plotear todos los modelos en esta métrica
        for model_name, model_data in dataframes.items():
            df = model_data['data']
            config = model_data['config']
            color = config['color']
            
            plot_single_metric_vs_emissions(df, model_name, metric, color, ax)
        
        # Configurar el subplot
        ax.set_title(f'{metric.replace("_", "@").upper()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Emisiones Acumuladas (kg CO₂)', fontsize=12)
        ax.set_ylabel(metric.replace('_', '@').upper(), fontsize=12)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    collage_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Topk\collage_recall_comparacion.png"
    plt.savefig(collage_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Collage de comparación Recall guardado: {collage_path}")
    plt.show()

def create_combined_collage_ndcg(dataframes):
    """Crea un collage con 4 gráficos combinados (uno por métrica de NDCG)"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Comparación de Métricas NDCG vs Emisiones de CO₂', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    axes = axes.flatten()
    
    # Para cada métrica de NDCG, crear un gráfico combinado
    for metric_idx, metric in enumerate(ndcg_metrics):
        ax = axes[metric_idx]
        
        # Plotear todos los modelos en esta métrica
        for model_name, model_data in dataframes.items():
            df = model_data['data']
            config = model_data['config']
            color = config['color']
            
            plot_single_metric_vs_emissions(df, model_name, metric, color, ax)
        
        # Configurar el subplot
        ax.set_title(f'{metric.replace("_", "@").upper()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Emisiones Acumuladas (kg CO₂)', fontsize=12)
        ax.set_ylabel(metric.replace('_', '@').upper(), fontsize=12)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    collage_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Topk\collage_ndcg_comparacion.png"
    plt.savefig(collage_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Collage de comparación NDCG guardado: {collage_path}")
    plt.show()

def create_best_values_bar_charts_recall(dataframes):
    """Crea un collage con 4 gráficos de barras (uno por métrica de recall)"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Mejores Valores de Recall por Modelo', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    axes = axes.flatten()
    
    # Para cada métrica de recall, crear un gráfico de barras
    for metric_idx, metric in enumerate(recall_metrics):
        ax = axes[metric_idx]
        
        model_names = []
        best_values = []
        colors = []
        
        # Recopilar el mejor valor de cada modelo para esta métrica
        for model_name, model_data in dataframes.items():
            df = model_data['data']
            config = model_data['config']
            
            if metric in df.columns:
                best_value = df[metric].max()
                best_epoch = df.loc[df[metric].idxmax(), 'epoch']
                
                model_names.append(model_name)
                best_values.append(best_value)
                colors.append(config['color'])
                
                print(f"{model_name}: Mejor {metric} = {best_value:.4f} (Epoch {best_epoch})")
        
        # Crear el gráfico de barras
        if model_names:
            bars = ax.bar(model_names, best_values, color=colors, alpha=0.8, 
                         edgecolor='white', linewidth=1.5)
            
            # Añadir valores en las barras
            for bar, value in zip(bars, best_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            # Configurar el gráfico
            ax.set_title(f'Mejor {metric.replace("_", "@").upper()}', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric.replace('_', '@').upper(), fontsize=12)
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Ajustar límites del eje Y
            if best_values:
                y_min = min(best_values)
                y_max = max(best_values)
                y_range = y_max - y_min
                ax.set_ylim(y_min - y_range*0.1, y_max + y_range*0.15)
    
    plt.tight_layout()
    bars_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Topk\collage_barras_recall.png"
    plt.savefig(bars_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Collage de barras Recall guardado: {bars_path}")
    plt.show()

def create_best_values_bar_charts_ndcg(dataframes):
    """Crea un collage con 4 gráficos de barras (uno por métrica de NDCG)"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Mejores Valores de NDCG por Modelo', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    axes = axes.flatten()
    
    # Para cada métrica de NDCG, crear un gráfico de barras
    for metric_idx, metric in enumerate(ndcg_metrics):
        ax = axes[metric_idx]
        
        model_names = []
        best_values = []
        colors = []
        
        # Recopilar el mejor valor de cada modelo para esta métrica
        for model_name, model_data in dataframes.items():
            df = model_data['data']
            config = model_data['config']
            
            if metric in df.columns:
                best_value = df[metric].max()
                best_epoch = df.loc[df[metric].idxmax(), 'epoch']
                
                model_names.append(model_name)
                best_values.append(best_value)
                colors.append(config['color'])
                
                print(f"{model_name}: Mejor {metric} = {best_value:.4f} (Epoch {best_epoch})")
        
        # Crear el gráfico de barras
        if model_names:
            bars = ax.bar(model_names, best_values, color=colors, alpha=0.8, 
                         edgecolor='white', linewidth=1.5)
            
            # Añadir valores en las barras
            for bar, value in zip(bars, best_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            # Configurar el gráfico
            ax.set_title(f'Mejor {metric.replace("_", "@").upper()}', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric.replace('_', '@').upper(), fontsize=12)
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Ajustar límites del eje Y
            if best_values:
                y_min = min(best_values)
                y_max = max(best_values)
                y_range = y_max - y_min
                ax.set_ylim(y_min - y_range*0.1, y_max + y_range*0.15)
    
    plt.tight_layout()
    bars_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Topk\collage_barras_ndcg.png"
    plt.savefig(bars_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Collage de barras NDCG guardado: {bars_path}")
    plt.show()

def create_individual_collage_recall_horizontal(dataframes):
    """Crea un collage horizontal con gráficas individuales de Recall@5 y Recall@50 (2x4)"""
    # Configurar el layout: 2 filas x 4 columnas
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Aplanar el array de axes para facilitar la iteración
    axes = axes.flatten()
    
    # Colores para cada modelo
    model_colors = ['#2E86AB', '#A23B72', '#D2B48C', '#C73E1D', '#6A994E', '#7209B7', '#F77F00', '#003566']
    
    # Iterar por cada modelo y crear su gráfico individual
    for idx, (model_name, model_data) in enumerate(dataframes.items()):
        ax = axes[idx]
        df = model_data['data']
        config = model_data['config']
        
        # Usar el color del modelo
        color = model_colors[idx % len(model_colors)]
        
        # Plotear Recall@5 y Recall@50 si existen las columnas
        best_recall5_value = best_recall50_value = 0
        best_recall5_epoch = best_recall50_epoch = 0
        best_recall5_emissions = best_recall50_emissions = 0
        
        if 'recall_5' in df.columns:
            # Encontrar el mejor Recall@5
            best_recall5_idx = df['recall_5'].idxmax()
            best_recall5_value = df.loc[best_recall5_idx, 'recall_5']
            best_recall5_emissions = df.loc[best_recall5_idx, 'cumulative_emissions_kg']
            best_recall5_epoch = df.loc[best_recall5_idx, 'epoch']
            
            # Plotear la línea de Recall@5 (línea sólida)
            ax.plot(df['cumulative_emissions_kg'].values, df['recall_5'].values, 
                   color=color, linewidth=2, alpha=0.8, marker='o', markersize=2, 
                   label='Recall@5', linestyle='-')
            
            # Marcar el mejor punto Recall@5
            ax.scatter(best_recall5_emissions, best_recall5_value, 
                      color=color, s=40, zorder=5, 
                      edgecolors='white', linewidth=1, marker='D')
        
        # Plotear Recall@50 si existe la columna
        if 'recall_50' in df.columns:
            # Encontrar el mejor Recall@50
            best_recall50_idx = df['recall_50'].idxmax()
            best_recall50_value = df.loc[best_recall50_idx, 'recall_50']
            best_recall50_emissions = df.loc[best_recall50_idx, 'cumulative_emissions_kg']
            best_recall50_epoch = df.loc[best_recall50_idx, 'epoch']
            
            # Crear un color más claro para Recall@50
            import matplotlib.colors as mcolors
            recall50_color = mcolors.to_rgba(color, alpha=0.7)
            
            # Plotear la línea de Recall@50 (línea punteada)
            ax.plot(df['cumulative_emissions_kg'].values, df['recall_50'].values, 
                   color=recall50_color, linewidth=1.5, alpha=0.6, marker='.', markersize=1.5, 
                   label='Recall@50', linestyle=':', markevery=3)
            
            # Marcar el mejor punto Recall@50
            ax.scatter(best_recall50_emissions, best_recall50_value, 
                      color=recall50_color, s=40, zorder=4, 
                      edgecolors='white', linewidth=1, marker='s', alpha=0.8)
        
        # Configurar el subplot con información completa
        if 'recall_5' in df.columns and 'recall_50' in df.columns:
            title_text = f'{model_name}\nRecall@5: {best_recall5_value:.3f} (Ep.{best_recall5_epoch})\nRecall@50: {best_recall50_value:.3f} (Ep.{best_recall50_epoch})'
        elif 'recall_5' in df.columns:
            title_text = f'{model_name}\nRecall@5: {best_recall5_value:.3f} (Ep.{best_recall5_epoch})'
        else:
            title_text = f'{model_name}\nDatos no disponibles'
            
        ax.set_title(title_text, fontsize=9, fontweight='bold', pad=10)
        ax.set_xlabel('Emisiones Acumuladas (kg CO₂)', fontsize=8)
        ax.set_ylabel('Recall', fontsize=8)
        
        # Configurar escala logarítmica en X para mejor visualización
        ax.set_xscale('log')
        
        # Grid ligero
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Ajustar tamaño de los ticks
        ax.tick_params(axis='both', which='major', labelsize=7)
        
        # Añadir leyenda pequeña
        if 'recall_5' in df.columns and 'recall_50' in df.columns:
            ax.legend(fontsize=7, loc='lower right', framealpha=0.8)
    
    # Ajustar layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Guardar el collage horizontal
    collage_horizontal_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Topk\collage_recall_individuales_horizontal.png"
    plt.savefig(collage_horizontal_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Collage horizontal de Recall individuales guardado: {collage_horizontal_path}")
    plt.show()

def create_individual_collage_recall_vertical(dataframes):
    """Crea un collage vertical con gráficas individuales de Recall@5 y Recall@50 (4x2)"""
    # Configurar el layout: 4 filas x 2 columnas
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    
    # Aplanar el array de axes para facilitar la iteración
    axes = axes.flatten()
    
    # Colores para cada modelo
    model_colors = ['#2E86AB', '#A23B72', '#D2B48C', '#C73E1D', '#6A994E', '#7209B7', '#F77F00', '#003566']
    
    # Iterar por cada modelo y crear su gráfico individual
    for idx, (model_name, model_data) in enumerate(dataframes.items()):
        ax = axes[idx]
        df = model_data['data']
        config = model_data['config']
        
        # Usar el color del modelo
        color = model_colors[idx % len(model_colors)]
        
        # Plotear Recall@5 y Recall@50 si existen las columnas
        best_recall5_value = best_recall50_value = 0
        best_recall5_epoch = best_recall50_epoch = 0
        best_recall5_emissions = best_recall50_emissions = 0
        
        if 'recall_5' in df.columns:
            # Encontrar el mejor Recall@5
            best_recall5_idx = df['recall_5'].idxmax()
            best_recall5_value = df.loc[best_recall5_idx, 'recall_5']
            best_recall5_emissions = df.loc[best_recall5_idx, 'cumulative_emissions_kg']
            best_recall5_epoch = df.loc[best_recall5_idx, 'epoch']
            
            # Plotear la línea de Recall@5 (línea sólida)
            ax.plot(df['cumulative_emissions_kg'].values, df['recall_5'].values, 
                   color=color, linewidth=2.5, alpha=0.8, marker='o', markersize=3, 
                   label='Recall@5', linestyle='-')
            
            # Marcar el mejor punto Recall@5
            ax.scatter(best_recall5_emissions, best_recall5_value, 
                      color=color, s=60, zorder=5, 
                      edgecolors='white', linewidth=1.5, marker='D')
        
        # Plotear Recall@50 si existe la columna
        if 'recall_50' in df.columns:
            # Encontrar el mejor Recall@50
            best_recall50_idx = df['recall_50'].idxmax()
            best_recall50_value = df.loc[best_recall50_idx, 'recall_50']
            best_recall50_emissions = df.loc[best_recall50_idx, 'cumulative_emissions_kg']
            best_recall50_epoch = df.loc[best_recall50_idx, 'epoch']
            
            # Crear un color más claro para Recall@50
            import matplotlib.colors as mcolors
            recall50_color = mcolors.to_rgba(color, alpha=0.7)
            
            # Plotear la línea de Recall@50 (línea punteada)
            ax.plot(df['cumulative_emissions_kg'].values, df['recall_50'].values, 
                   color=recall50_color, linewidth=2, alpha=0.6, marker='.', markersize=2, 
                   label='Recall@50', linestyle=':', markevery=2)
            
            # Marcar el mejor punto Recall@50
            ax.scatter(best_recall50_emissions, best_recall50_value, 
                      color=recall50_color, s=60, zorder=4, 
                      edgecolors='white', linewidth=1.5, marker='s', alpha=0.8)
        
        # Configurar el subplot con información completa
        if 'recall_5' in df.columns and 'recall_50' in df.columns:
            title_text = f'{model_name}\nRecall@5: {best_recall5_value:.3f} (Ep.{best_recall5_epoch})\nRecall@50: {best_recall50_value:.3f} (Ep.{best_recall50_epoch})'
        elif 'recall_5' in df.columns:
            title_text = f'{model_name}\nRecall@5: {best_recall5_value:.3f} (Ep.{best_recall5_epoch})'
        else:
            title_text = f'{model_name}\nDatos no disponibles'
            
        ax.set_title(title_text, fontsize=11, fontweight='bold', pad=15)
        ax.set_xlabel('Emisiones Acumuladas (kg CO₂)', fontsize=10)
        ax.set_ylabel('Recall', fontsize=10)
        
        # Configurar escala logarítmica en X para mejor visualización
        ax.set_xscale('log')
        
        # Grid ligero
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Ajustar tamaño de los ticks
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        # Añadir leyenda pequeña
        if 'recall_5' in df.columns and 'recall_50' in df.columns:
            ax.legend(fontsize=8, loc='lower right', framealpha=0.8)
    
    # Ajustar layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    
    # Guardar el collage vertical
    collage_vertical_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Topk\collage_recall_individuales_vertical.png"
    plt.savefig(collage_vertical_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Collage vertical de Recall individuales guardado: {collage_vertical_path}")
    plt.show()

def create_individual_collage_ndcg_horizontal(dataframes):
    """Crea un collage horizontal con gráficas individuales de NDCG@5 y NDCG@50 (2x4)"""
    # Configurar el layout: 2 filas x 4 columnas
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Aplanar el array de axes para facilitar la iteración
    axes = axes.flatten()
    
    # Colores para cada modelo
    model_colors = ['#2E86AB', '#A23B72', '#D2B48C', '#C73E1D', '#6A994E', '#7209B7', '#F77F00', '#003566']
    
    # Iterar por cada modelo y crear su gráfico individual
    for idx, (model_name, model_data) in enumerate(dataframes.items()):
        ax = axes[idx]
        df = model_data['data']
        config = model_data['config']
        
        # Usar el color del modelo
        color = model_colors[idx % len(model_colors)]
        
        # Plotear NDCG@5 y NDCG@50 si existen las columnas
        best_ndcg5_value = best_ndcg50_value = 0
        best_ndcg5_epoch = best_ndcg50_epoch = 0
        best_ndcg5_emissions = best_ndcg50_emissions = 0
        
        if 'ndcg_5' in df.columns:
            # Encontrar el mejor NDCG@5
            best_ndcg5_idx = df['ndcg_5'].idxmax()
            best_ndcg5_value = df.loc[best_ndcg5_idx, 'ndcg_5']
            best_ndcg5_emissions = df.loc[best_ndcg5_idx, 'cumulative_emissions_kg']
            best_ndcg5_epoch = df.loc[best_ndcg5_idx, 'epoch']
            
            # Plotear la línea de NDCG@5 (línea sólida)
            ax.plot(df['cumulative_emissions_kg'].values, df['ndcg_5'].values, 
                   color=color, linewidth=2, alpha=0.8, marker='o', markersize=2, 
                   label='NDCG@5', linestyle='-')
            
            # Marcar el mejor punto NDCG@5
            ax.scatter(best_ndcg5_emissions, best_ndcg5_value, 
                      color=color, s=40, zorder=5, 
                      edgecolors='white', linewidth=1, marker='D')
        
        # Plotear NDCG@50 si existe la columna
        if 'ndcg_50' in df.columns:
            # Encontrar el mejor NDCG@50
            best_ndcg50_idx = df['ndcg_50'].idxmax()
            best_ndcg50_value = df.loc[best_ndcg50_idx, 'ndcg_50']
            best_ndcg50_emissions = df.loc[best_ndcg50_idx, 'cumulative_emissions_kg']
            best_ndcg50_epoch = df.loc[best_ndcg50_idx, 'epoch']
            
            # Crear un color más claro para NDCG@50
            import matplotlib.colors as mcolors
            ndcg50_color = mcolors.to_rgba(color, alpha=0.7)
            
            # Plotear la línea de NDCG@50 (línea punteada)
            ax.plot(df['cumulative_emissions_kg'].values, df['ndcg_50'].values, 
                   color=ndcg50_color, linewidth=1.5, alpha=0.6, marker='.', markersize=1.5, 
                   label='NDCG@50', linestyle=':', markevery=3)
            
            # Marcar el mejor punto NDCG@50
            ax.scatter(best_ndcg50_emissions, best_ndcg50_value, 
                      color=ndcg50_color, s=40, zorder=4, 
                      edgecolors='white', linewidth=1, marker='s', alpha=0.8)
        
        # Configurar el subplot con información completa
        if 'ndcg_5' in df.columns and 'ndcg_50' in df.columns:
            title_text = f'{model_name}\nNDCG@5: {best_ndcg5_value:.3f} (Ep.{best_ndcg5_epoch})\nNDCG@50: {best_ndcg50_value:.3f} (Ep.{best_ndcg50_epoch})'
        elif 'ndcg_5' in df.columns:
            title_text = f'{model_name}\nNDCG@5: {best_ndcg5_value:.3f} (Ep.{best_ndcg5_epoch})'
        else:
            title_text = f'{model_name}\nDatos no disponibles'
            
        ax.set_title(title_text, fontsize=9, fontweight='bold', pad=10)
        ax.set_xlabel('Emisiones Acumuladas (kg CO₂)', fontsize=8)
        ax.set_ylabel('NDCG', fontsize=8)
        
        # Configurar escala logarítmica en X para mejor visualización
        ax.set_xscale('log')
        
        # Grid ligero
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Ajustar tamaño de los ticks
        ax.tick_params(axis='both', which='major', labelsize=7)
        
        # Añadir leyenda pequeña
        if 'ndcg_5' in df.columns and 'ndcg_50' in df.columns:
            ax.legend(fontsize=7, loc='lower right', framealpha=0.8)
    
    # Ajustar layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Guardar el collage horizontal
    collage_horizontal_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Topk\collage_ndcg_individuales_horizontal.png"
    plt.savefig(collage_horizontal_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Collage horizontal de NDCG individuales guardado: {collage_horizontal_path}")
    plt.show()

def create_individual_collage_ndcg_vertical(dataframes):
    """Crea un collage vertical con gráficas individuales de NDCG@5 y NDCG@50 (4x2)"""
    # Configurar el layout: 4 filas x 2 columnas
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    
    # Aplanar el array de axes para facilitar la iteración
    axes = axes.flatten()
    
    # Colores para cada modelo
    model_colors = ['#2E86AB', '#A23B72', '#D2B48C', '#C73E1D', '#6A994E', '#7209B7', '#F77F00', '#003566']
    
    # Iterar por cada modelo y crear su gráfico individual
    for idx, (model_name, model_data) in enumerate(dataframes.items()):
        ax = axes[idx]
        df = model_data['data']
        config = model_data['config']
        
        # Usar el color del modelo
        color = model_colors[idx % len(model_colors)]
        
        # Plotear NDCG@5 y NDCG@50 si existen las columnas
        best_ndcg5_value = best_ndcg50_value = 0
        best_ndcg5_epoch = best_ndcg50_epoch = 0
        best_ndcg5_emissions = best_ndcg50_emissions = 0
        
        if 'ndcg_5' in df.columns:
            # Encontrar el mejor NDCG@5
            best_ndcg5_idx = df['ndcg_5'].idxmax()
            best_ndcg5_value = df.loc[best_ndcg5_idx, 'ndcg_5']
            best_ndcg5_emissions = df.loc[best_ndcg5_idx, 'cumulative_emissions_kg']
            best_ndcg5_epoch = df.loc[best_ndcg5_idx, 'epoch']
            
            # Plotear la línea de NDCG@5 (línea sólida)
            ax.plot(df['cumulative_emissions_kg'].values, df['ndcg_5'].values, 
                   color=color, linewidth=2.5, alpha=0.8, marker='o', markersize=3, 
                   label='NDCG@5', linestyle='-')
            
            # Marcar el mejor punto NDCG@5
            ax.scatter(best_ndcg5_emissions, best_ndcg5_value, 
                      color=color, s=60, zorder=5, 
                      edgecolors='white', linewidth=1.5, marker='D')
        
        # Plotear NDCG@50 si existe la columna
        if 'ndcg_50' in df.columns:
            # Encontrar el mejor NDCG@50
            best_ndcg50_idx = df['ndcg_50'].idxmax()
            best_ndcg50_value = df.loc[best_ndcg50_idx, 'ndcg_50']
            best_ndcg50_emissions = df.loc[best_ndcg50_idx, 'cumulative_emissions_kg']
            best_ndcg50_epoch = df.loc[best_ndcg50_idx, 'epoch']
            
            # Crear un color más claro para NDCG@50
            import matplotlib.colors as mcolors
            ndcg50_color = mcolors.to_rgba(color, alpha=0.7)
            
            # Plotear la línea de NDCG@50 (línea punteada)
            ax.plot(df['cumulative_emissions_kg'].values, df['ndcg_50'].values, 
                   color=ndcg50_color, linewidth=2, alpha=0.6, marker='.', markersize=2, 
                   label='NDCG@50', linestyle=':', markevery=2)
            
            # Marcar el mejor punto NDCG@50
            ax.scatter(best_ndcg50_emissions, best_ndcg50_value, 
                      color=ndcg50_color, s=60, zorder=4, 
                      edgecolors='white', linewidth=1.5, marker='s', alpha=0.8)
        
        # Configurar el subplot con información completa
        if 'ndcg_5' in df.columns and 'ndcg_50' in df.columns:
            title_text = f'{model_name}\nNDCG@5: {best_ndcg5_value:.3f} (Ep.{best_ndcg5_epoch})\nNDCG@50: {best_ndcg50_value:.3f} (Ep.{best_ndcg50_epoch})'
        elif 'ndcg_5' in df.columns:
            title_text = f'{model_name}\nNDCG@5: {best_ndcg5_value:.3f} (Ep.{best_ndcg5_epoch})'
        else:
            title_text = f'{model_name}\nDatos no disponibles'
            
        ax.set_title(title_text, fontsize=11, fontweight='bold', pad=15)
        ax.set_xlabel('Emisiones Acumuladas (kg CO₂)', fontsize=10)
        ax.set_ylabel('NDCG', fontsize=10)
        
        # Configurar escala logarítmica en X para mejor visualización
        ax.set_xscale('log')
        
        # Grid ligero
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Ajustar tamaño de los ticks
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        # Añadir leyenda pequeña
        if 'ndcg_5' in df.columns and 'ndcg_50' in df.columns:
            ax.legend(fontsize=8, loc='lower right', framealpha=0.8)
    
    # Ajustar layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    
    # Guardar el collage vertical
    collage_vertical_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Topk\collage_ndcg_individuales_vertical.png"
    plt.savefig(collage_vertical_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Collage vertical de NDCG individuales guardado: {collage_vertical_path}")
    plt.show()

# =============================================================================
# PROCESAMIENTO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("Procesando archivos TopK...")
    print(f"Procesando {len(files_config)} archivos...")
    
    # Cargar todos los datos
    dataframes = {}
    for file_config in files_config:
        if os.path.exists(file_config['path']):
            print(f"✓ {file_config['name']}: {file_config['path']}")
            df = load_csv_data(file_config)
            if df is not None:
                dataframes[file_config['name']] = {
                    'data': df,
                    'config': file_config
                }
        else:
            print(f"✗ Archivo no encontrado: {file_config['path']}")
    
    if not dataframes:
        print("No se pudieron cargar datos.")
        exit(1)
    
    print(f"\n--- Generando gráficos individuales ---")
    create_individual_plots(dataframes)
    
    print(f"\n--- Generando collages de comparación ---")
    create_combined_collage_recall(dataframes)
    create_combined_collage_ndcg(dataframes)
    
    print(f"\n--- Generando collages de barras ---")
    create_best_values_bar_charts_recall(dataframes)
    create_best_values_bar_charts_ndcg(dataframes)
    
    print(f"\n--- Generando collages individuales Recall@5 y Recall@50 ---")
    create_individual_collage_recall_horizontal(dataframes)
    create_individual_collage_recall_vertical(dataframes)
    
    print(f"\n--- Generando collages individuales NDCG@5 y NDCG@50 ---")
    create_individual_collage_ndcg_horizontal(dataframes)
    create_individual_collage_ndcg_vertical(dataframes)
    
    print("\n¡Proceso completado!")
