import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from pathlib import Path
import re

# Configurar el estilo
plt.style.use('default')
sns.set_palette("husl")

# Opción 1: Lista específica de archivos con nombres personalizados
files_config = [
    # Dataset Movielens 1M (Resultados_Completo)
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\Resultados_Completo\Modelos_Matrices\emissions_metrics_MF_20250813-101838.csv",
        'name': 'MF',
        'category': 'Modelos_Matrices',
        'color_rmse': '#2E86AB',
        'color_mae': '#1E5F8B'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\Resultados_Completo\Modelos_Matrices\emissions_metrics_NNMF_20250814-011801.csv",
        'name': 'NNMF',
        'category': 'Modelos_Matrices',
        'color_rmse': '#A23B72',
        'color_mae': '#822B52'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\Resultados_Completo\Modelos_Redes_Profundas\emissions_metrics_AutoRec_20250814-001313.csv",
        'name': 'AutoRec',
        'category': 'Modelos_Redes_Profundas',
        'color_rmse': '#D2B48C',
        'color_mae': '#B8986B'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\Resultados_Completo\Modelos_Redes_Profundas\emissions_metrics_NCF_20250811-104345.csv",
        'name': 'NCF',
        'category': 'Modelos_Redes_Profundas',
        'color_rmse': '#C73E1D',
        'color_mae': '#A7321A'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\Resultados_Completo\Modelos_Redes_Profundas\emissions_metrics_LRML_20250814-013839.csv",
        'name': 'LRML',
        'category': 'Modelos_Redes_Profundas',
        'color_rmse': '#6A994E',
        'color_mae': '#5A7F42'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\Resultados_Completo\Modelos_Redes_Grafos\emissions_metrics_GHRS_20250814-003416.csv",
        'name': 'GHRS',
        'category': 'Modelos_Redes_Grafos',
        'color_rmse': '#7209B7',
        'color_mae': '#5F0799'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\Resultados_Completo\Modelos_Redes_Grafos\emissions_metrics_KGNN_LS_20250811-173339.csv",
        'name': 'KGNN-LS',
        'category': 'Modelos_Redes_Grafos',
        'color_rmse': '#F77F00',
        'color_mae': '#D96A00'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\Resultados_Completo\Modelos_Redes_Grafos\emissions_metrics_LightGCN_20250826-153939.csv",
        'name': 'LightGCN',
        'category': 'Modelos_Redes_Grafos',
        'color_rmse': '#003566',
        'color_mae': '#002145'
    },
    
    # Dataset CTRPv2
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_CTRPv2\emissions_metrics_AutoRec_20250814-114500.csv",
        'name': 'AutoRec (CTRPv2)',
        'category': 'Dataset_CTRPv2',
        'color_rmse': '#D2B48C',
        'color_mae': '#B8986B'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_CTRPv2\emissions_metrics_CTRPV_split0_20250814-235108.csv",
        'name': 'MF (CTRPv2)',
        'category': 'Dataset_CTRPv2',
        'color_rmse': '#2E86AB',
        'color_mae': '#1E5F8B'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_CTRPv2\emissions_metrics_GHRS_CTRPv2_20250814-130928.csv",
        'name': 'GHRS (CTRPv2)',
        'category': 'Dataset_CTRPv2',
        'color_rmse': '#7209B7',
        'color_mae': '#5F0799'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_CTRPv2\emissions_metrics_KGNN_LS_20250817-004034.csv",
        'name': 'KGNN-LS (CTRPv2)',
        'category': 'Dataset_CTRPv2',
        'color_rmse': '#F77F00',
        'color_mae': '#D96A00'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_CTRPv2\emissions_metrics_LightGCN_CTRPv2_20250826-161420.csv",
        'name': 'LightGCN (CTRPv2)',
        'category': 'Dataset_CTRPv2',
        'color_rmse': '#003566',
        'color_mae': '#002145'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_CTRPv2\emissions_metrics_LRML_20250814-150035.csv",
        'name': 'LRML (CTRPv2)',
        'category': 'Dataset_CTRPv2',
        'color_rmse': '#6A994E',
        'color_mae': '#5A7F42'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_CTRPv2\emissions_metrics_NCF_20250815-105558.csv",
        'name': 'NCF (CTRPv2)',
        'category': 'Dataset_CTRPv2',
        'color_rmse': '#C73E1D',
        'color_mae': '#A7321A'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_CTRPv2\emissions_metrics_NNMF_20250814-221012.csv",
        'name': 'NNMF (CTRPv2)',
        'category': 'Dataset_CTRPv2',
        'color_rmse': '#A23B72',
        'color_mae': '#822B52'
    },
    
    # Dataset GDSC1
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_GDSC1\emissions_metrics_AutoRec_20250814-120047.csv",
        'name': 'AutoRec (GDSC1)',
        'category': 'Dataset_GDSC1',
        'color_rmse': '#D2B48C',
        'color_mae': '#B8986B'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_GDSC1\emissions_metrics_GDSC1_20250814-132356.csv",
        'name': 'GHRS (GDSC1)',
        'category': 'Dataset_GDSC1',
        'color_rmse': '#7209B7',
        'color_mae': '#5F0799'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_GDSC1\emissions_metrics_GDSC1_split0_20250815-011821.csv",
        'name': 'MF (GDSC1)',
        'category': 'Dataset_GDSC1',
        'color_rmse': '#2E86AB',
        'color_mae': '#1E5F8B'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_GDSC1\emissions_metrics_KGNN_LS_20250816-202545.csv",
        'name': 'KGNN-LS (GDSC1)',
        'category': 'Dataset_GDSC1',
        'color_rmse': '#F77F00',
        'color_mae': '#D96A00'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_GDSC1\emissions_metrics_LightGCN_GDSC1_20250827-003000.csv",
        'name': 'LightGCN (GDSC1)',
        'category': 'Dataset_GDSC1',
        'color_rmse': '#003566',
        'color_mae': '#002145'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_GDSC1\emissions_metrics_LRML_20250814-151721.csv",
        'name': 'LRML (GDSC1)',
        'category': 'Dataset_GDSC1',
        'color_rmse': '#6A994E',
        'color_mae': '#5A7F42'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_GDSC1\emissions_metrics_NCF_20250815-111506.csv",
        'name': 'NCF (GDSC1)',
        'category': 'Dataset_GDSC1',
        'color_rmse': '#C73E1D',
        'color_mae': '#A7321A'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_GDSC1\emissions_metrics_NNMF_20250814-222928.csv",
        'name': 'NNMF (GDSC1)',
        'category': 'Dataset_GDSC1',
        'color_rmse': '#A23B72',
        'color_mae': '#822B52'
    },
    
    # Dataset IMF_DOTS_2023
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_IMF_DOTS_2023\emissions_metrics_AutoRec_20250814-120621.csv",
        'name': 'AutoRec (IMF_DOTS)',
        'category': 'Dataset_IMF_DOTS',
        'color_rmse': '#D2B48C',
        'color_mae': '#B8986B'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_IMF_DOTS_2023\emissions_metrics_GHRS_IMF_DOTS_20250814-133332.csv",
        'name': 'GHRS (IMF_DOTS)',
        'category': 'Dataset_IMF_DOTS',
        'color_rmse': '#7209B7',
        'color_mae': '#5F0799'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_IMF_DOTS_2023\emissions_metrics_IMF_DOTS_2023_split0_20250815-100601.csv",
        'name': 'MF (IMF_DOTS)',
        'category': 'Dataset_IMF_DOTS',
        'color_rmse': '#2E86AB',
        'color_mae': '#1E5F8B'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_IMF_DOTS_2023\emissions_metrics_KGNN_LS_20250816-231234.csv",
        'name': 'KGNN-LS (IMF_DOTS)',
        'category': 'Dataset_IMF_DOTS',
        'color_rmse': '#F77F00',
        'color_mae': '#D96A00'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_IMF_DOTS_2023\emissions_metrics_LightGCN_DOTS2023_20250826-173718.csv",
        'name': 'LightGCN (IMF_DOTS)',
        'category': 'Dataset_IMF_DOTS',
        'color_rmse': '#003566',
        'color_mae': '#002145'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_IMF_DOTS_2023\emissions_metrics_LRML_20250814-153443.csv",
        'name': 'LRML (IMF_DOTS)',
        'category': 'Dataset_IMF_DOTS',
        'color_rmse': '#6A994E',
        'color_mae': '#5A7F42'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_IMF_DOTS_2023\emissions_metrics_NCF_20250815-112124.csv",
        'name': 'NCF (IMF_DOTS)',
        'category': 'Dataset_IMF_DOTS',
        'color_rmse': '#C73E1D',
        'color_mae': '#A7321A'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_IMF_DOTS_2023\emissions_metrics_NNMF_20250814-223616.csv",
        'name': 'NNMF (IMF_DOTS)',
        'category': 'Dataset_IMF_DOTS',
        'color_rmse': '#A23B72',
        'color_mae': '#822B52'
    },
    
    # Dataset Kiva_Microloans
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_Kiva_Microloans\emissions_metrics_AutoRec_20250814-121421.csv",
        'name': 'AutoRec (Kiva)',
        'category': 'Dataset_Kiva',
        'color_rmse': '#D2B48C',
        'color_mae': '#B8986B'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_Kiva_Microloans\emissions_metrics_GHRS_Kiva_20250814-134816.csv",
        'name': 'GHRS (Kiva)',
        'category': 'Dataset_Kiva',
        'color_rmse': '#7209B7',
        'color_mae': '#5F0799'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_Kiva_Microloans\emissions_metrics_KGNN_LS_20250816-235923.csv",
        'name': 'KGNN-LS (Kiva)',
        'category': 'Dataset_Kiva',
        'color_rmse': '#F77F00',
        'color_mae': '#D96A00'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_Kiva_Microloans\emissions_metrics_KIVA_MICROLOANS_split0_20250815-102433.csv",
        'name': 'MF (Kiva)',
        'category': 'Dataset_Kiva',
        'color_rmse': '#2E86AB',
        'color_mae': '#1E5F8B'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_Kiva_Microloans\emissions_metrics_LightGCN_Kiva_20250826-172327.csv",
        'name': 'LightGCN (Kiva)',
        'category': 'Dataset_Kiva',
        'color_rmse': '#003566',
        'color_mae': '#002145'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_Kiva_Microloans\emissions_metrics_LRML_20250814-155747.csv",
        'name': 'LRML (Kiva)',
        'category': 'Dataset_Kiva',
        'color_rmse': '#6A994E',
        'color_mae': '#5A7F42'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_Kiva_Microloans\emissions_metrics_NCF_20250815-113316.csv",
        'name': 'NCF (Kiva)',
        'category': 'Dataset_Kiva',
        'color_rmse': '#C73E1D',
        'color_mae': '#A7321A'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_Kiva_Microloans\emissions_metrics_NNMF_20250814-224917.csv",
        'name': 'NNMF (Kiva)',
        'category': 'Dataset_Kiva',
        'color_rmse': '#A23B72',
        'color_mae': '#822B52'
    },
    
    # Dataset Netflix_Prize
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Datasets\Resultados_Netflix_Prize\emissions_metrics_AutoRec_20250815-091221.csv",
        'name': 'AutoRec (Netflix)',
        'category': 'Dataset_Netflix',
        'color_rmse': '#D2B48C',
        'color_mae': '#B8986B'
    },
]

# Opción 2: Procesar todos los archivos de una carpeta
folder_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\Resultados_Completo\Modelos_Matrices"
process_all_in_folder = False  # Cambia a True para usar esta opción

# Configuración del gráfico
plot_type = "combined"  # "combined", "separate", "subplots"
save_individual = True  # Guardar gráficos individuales además del combinado

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
    """Estandariza los nombres de las columnas"""
    # Si tiene columnas de validación, usar esas; si no, usar las de entrenamiento
    if 'valid_rmse' in df.columns and 'valid_mae' in df.columns:
        df = df.rename(columns={'valid_rmse': 'rmse', 'valid_mae': 'mae'})
    elif 'test_rmse' in df.columns and 'test_mae' in df.columns:
        df = df.rename(columns={'test_rmse': 'rmse', 'test_mae': 'mae'})
    elif 'train_rmse' in df.columns and 'train_mae' in df.columns:
        df = df.rename(columns={'train_rmse': 'rmse', 'train_mae': 'mae'})
    
    return df

def extract_model_name(filepath):
    """Extrae el nombre del modelo del nombre del archivo"""
    filename = os.path.basename(filepath)
    if 'MF' in filename.lower():
        return 'MF'
    elif 'nnmf' in filename.lower():
        return 'NNMF'
    elif 'autorec' in filename.lower():
        return 'AutoRec'
    elif 'ncf' in filename.lower():
        return 'NCF'
    elif 'lrml' in filename.lower():
        return 'LRML'
    elif 'lightgcn' in filename.lower():
        return 'LightGCN'
    elif 'kgnn' in filename.lower():
        return 'KGNN-LS'
    elif 'ghrs' in filename.lower():
        return 'GHRS'
    else:
        return filename.replace('emissions_metrics_', '').replace('.csv', '')

# Mapeo de colores por modelo base (sin dataset)
MODEL_COLOR_MAP = {
    'MF': '#2E86AB',
    'NNMF': '#A23B72', 
    'AutoRec': '#D2B48C',
    'NCF': '#C73E1D',
    'LRML': '#6A994E',
    'GHRS': '#7209B7',
    'KGNN-LS': '#F77F00',
    'LightGCN': '#003566'
}

# Mapeo de colores por dataset
DATASET_COLOR_MAP = {
    'Movielens 1M': '#2E86AB',
    'CTRPv2': '#A23B72',
    'GDSC1': '#D2B48C', 
    'IMF_DOTS': '#C73E1D',
    'Kiva': '#6A994E',
    'Netflix': '#7209B7'
}

def get_model_color(model_name):
    """Obtiene el color específico para un modelo"""
    # Extraer el nombre base del modelo (sin dataset)
    base_model = model_name.split(' (')[0]  # Ej: "AutoRec (CTRPv2)" -> "AutoRec"
    return MODEL_COLOR_MAP.get(base_model, '#666666')

def get_dataset_color(dataset_name):
    """Obtiene el color específico para un dataset"""
    return DATASET_COLOR_MAP.get(dataset_name, '#666666')

def get_colors(index, total):
    """Genera colores profesionales para cada serie"""
    colors = [
        ('#2E86AB', '#1E5F8B'),  # Azules
        ('#A23B72', '#822B52'),  # Magentas
        ('#F18F01', '#D17000'),  # Naranjas
        ('#C73E1D', '#A7321A'),  # Rojos
        ('#6A994E', '#5A7F42'),  # Verdes
        ('#7209B7', '#5F0799'),  # Morados
        ('#F77F00', '#D96A00'),  # Naranjas oscuros
        ('#003566', '#002145'),  # Azules oscuros
    ]
    return colors[index % len(colors)]

def plot_single_csv(df, model_name, colors, ax=None, show_info_box=True):
    """Plotea un solo CSV"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    color_rmse, color_mae = colors
    
    # Verificar que las columnas necesarias existen
    if 'rmse' not in df.columns or 'mae' not in df.columns:
        print(f"Advertencia: {model_name} no tiene columnas 'rmse' y 'mae'. Columnas disponibles: {list(df.columns)}")
        return ax, None
    
    # Encontrar los mejores valores
    best_rmse_idx = df['rmse'].idxmin()
    best_mae_idx = df['mae'].idxmin()

    best_rmse_value = df.loc[best_rmse_idx, 'rmse']
    best_rmse_emissions = df.loc[best_rmse_idx, 'cumulative_emissions_kg']
    best_rmse_epoch = df.loc[best_rmse_idx, 'epoch']

    best_mae_value = df.loc[best_mae_idx, 'mae']
    best_mae_emissions = df.loc[best_mae_idx, 'cumulative_emissions_kg']
    best_mae_epoch = df.loc[best_mae_idx, 'epoch']

    # Plotear las líneas principales
    line_rmse = ax.plot(df['cumulative_emissions_kg'].values, df['rmse'].values, 
                       color=color_rmse, linewidth=2.5, label=f'{model_name} - RMSE', alpha=0.8)
    line_mae = ax.plot(df['cumulative_emissions_kg'].values, df['mae'].values, 
                      color=color_mae, linewidth=2.5, label=f'{model_name} - MAE', alpha=0.8, linestyle='--')

    # Marcar los mejores puntos
    ax.scatter(best_rmse_emissions, best_rmse_value, 
              color=color_rmse, s=100, zorder=5, 
              edgecolors='white', linewidth=2, marker='o')
    
    ax.scatter(best_mae_emissions, best_mae_value, 
              color=color_mae, s=100, zorder=5, 
              edgecolors='white', linewidth=2, marker='s')

    if show_info_box:
        # Añadir caja de información
        info_text = f"""{model_name} - Mejores Resultados:
RMSE: {best_rmse_value:.4f} (Epoch {best_rmse_epoch})
Emisiones: {best_rmse_emissions:.6f} kg

MAE: {best_mae_value:.4f} (Epoch {best_mae_epoch})
Emisiones: {best_mae_emissions:.6f} kg"""

        bbox_props = dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8, edgecolor='gray')
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=bbox_props, family='monospace')

    return ax, (best_rmse_value, best_rmse_emissions, best_rmse_epoch, best_mae_value, best_mae_emissions, best_mae_epoch)

def plot_rmse_only_combined(df, model_name, color, ax):
    """Plotea solo RMSE para el gráfico combinado simplificado"""
    # Verificar que las columnas necesarias existen
    if 'rmse' not in df.columns:
        print(f"Advertencia: {model_name} no tiene columna 'rmse'. Columnas disponibles: {list(df.columns)}")
        return ax, None
    
    # Encontrar el mejor RMSE
    best_rmse_idx = df['rmse'].idxmin()
    best_rmse_value = df.loc[best_rmse_idx, 'rmse']
    best_rmse_emissions = df.loc[best_rmse_idx, 'cumulative_emissions_kg']
    best_rmse_epoch = df.loc[best_rmse_idx, 'epoch']

    # Plotear solo los datos reales sin puntos artificiales
    emissions_data = df['cumulative_emissions_kg'].values
    rmse_data = df['rmse'].values

    # Plotear la línea principal
    line = ax.plot(emissions_data, rmse_data, 
                  color=color, linewidth=3, label=f'{model_name}', alpha=0.8, marker='o', markersize=4)

    # Marcar el mejor punto con un diamante
    ax.scatter(best_rmse_emissions, best_rmse_value, 
              color=color, s=40, zorder=5, 
              edgecolors='white', linewidth=1, marker='D')

    return ax, (best_rmse_value, best_rmse_emissions, best_rmse_epoch)

def setup_combined_plot_formatting(ax, title):
    """Configura el formato del gráfico combinado simplificado"""
    ax.set_xlabel('Emisiones Acumuladas (kg CO₂) - escala logarítmica', fontsize=14, fontweight='bold')
    ax.set_ylabel('RMSE', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Configurar escala logarítmica para el eje X (emisiones)
    ax.set_xscale('log')
    
    # Configurar la leyenda
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                      shadow=True, fontsize=11, framealpha=0.95, ncol=2)
    legend.get_frame().set_facecolor('white')
    
    # Formato de ejes y grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.3, which='minor')

def create_category_plot(dataframes, category_name, category_models):
    """Crea un gráfico combinado para una categoría específica de modelos"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Colores específicos para cada categoría
    if category_name == 'Modelos_Matrices':
        colors = ['#2E86AB', '#A23B72']
        title_suffix = 'Modelos de Matrices'
    elif category_name == 'Modelos_Redes_Profundas':
        colors = ['#F18F01', '#C73E1D', '#6A994E']
        title_suffix = 'Modelos de Redes Profundas'
    elif category_name == 'Modelos_Redes_Grafos':
        colors = ['#7209B7', '#F77F00', '#003566']
        title_suffix = 'Modelos de Redes de Grafos'
    else:
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7', '#F77F00', '#003566']
        title_suffix = category_name
    
    color_index = 0
    all_best_results = []
    
    for model_name in category_models:
        if model_name in dataframes:
            df = dataframes[model_name]['data']
            color = colors[color_index % len(colors)]
            
            _, best_results = plot_rmse_only_combined(df, model_name, color, ax)
            if best_results:
                all_best_results.append((model_name, best_results))
            color_index += 1
    
    setup_combined_plot_formatting(ax, f'Comparación {title_suffix}: RMSE vs Emisiones de CO₂')
    
    plt.tight_layout()
    filename = f"grafico_combinado_{category_name.lower()}.png"
    category_path = f"C:\\Users\\xpati\\Documents\\TFG\\RESULTADOS_FINAL\\{filename}"
    plt.savefig(category_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Gráfico de {title_suffix} guardado: {category_path}")
    plt.show()
    
    return all_best_results

def plot_rmse_vs_epoch(df, model_name, color, ax):
    """Plotea RMSE vs Epoch"""
    if 'rmse' not in df.columns:
        print(f"Advertencia: {model_name} no tiene columna 'rmse'. Columnas disponibles: {list(df.columns)}")
        return ax, None
    
    # Encontrar el mejor RMSE
    best_rmse_idx = df['rmse'].idxmin()
    best_rmse_value = df.loc[best_rmse_idx, 'rmse']
    best_rmse_epoch = df.loc[best_rmse_idx, 'epoch']

    # Plotear RMSE vs Epoch
    line = ax.plot(df['epoch'].values, df['rmse'].values, 
                  color=color, linewidth=2.5, label=f'{model_name}', alpha=0.8, marker='o', markersize=3)

    # Marcar el mejor punto
    ax.scatter(best_rmse_epoch, best_rmse_value, 
              color=color, s=40, zorder=5, 
              edgecolors='white', linewidth=1, marker='D')

    return ax, (best_rmse_value, best_rmse_epoch)

def plot_emissions_vs_epoch(df, model_name, color, ax):
    """Plotea Emisiones Acumuladas vs Epoch"""
    if 'cumulative_emissions_kg' not in df.columns:
        print(f"Advertencia: {model_name} no tiene columna 'cumulative_emissions_kg'. Columnas disponibles: {list(df.columns)}")
        return ax, None

    # Plotear Emisiones vs Epoch
    line = ax.plot(df['epoch'].values, df['cumulative_emissions_kg'].values, 
                  color=color, linewidth=2.5, label=f'{model_name}', alpha=0.8, marker='o', markersize=3)

    # Marcar el punto final (máximas emisiones)
    final_epoch = df['epoch'].iloc[-1]
    final_emissions = df['cumulative_emissions_kg'].iloc[-1]
    ax.scatter(final_epoch, final_emissions, 
              color=color, s=100, zorder=5, 
              edgecolors='white', linewidth=2, marker='s')

    return ax, final_emissions

def create_rmse_epoch_plots(dataframes):
    """Crea gráficos de RMSE vs Epoch"""
    # Gráfico combinado RMSE vs Epoch
    fig, ax = plt.subplots(figsize=(12, 8))
    
    model_colors = ['#2E86AB', '#A23B72', '#D2B48C', '#C73E1D', '#6A994E', '#7209B7', '#F77F00', '#003566']
    color_index = 0
    
    for model_name, model_data in dataframes.items():
        df = model_data['data']
        color = model_colors[color_index % len(model_colors)]
        
        plot_rmse_vs_epoch(df, model_name, color, ax)
        color_index += 1
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('RMSE', fontsize=14, fontweight='bold')
    ax.set_title('Evolución del RMSE por Epoch', fontsize=16, fontweight='bold', pad=20)
    
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                      shadow=True, fontsize=10, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    rmse_epoch_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\grafico_rmse_vs_epoch.png"
    plt.savefig(rmse_epoch_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Gráfico RMSE vs Epoch guardado: {rmse_epoch_path}")
    plt.show()

def create_emissions_epoch_plots(dataframes):
    """Crea gráficos de Emisiones vs Epoch"""
    # Gráfico combinado Emisiones vs Epoch
    fig, ax = plt.subplots(figsize=(12, 8))
    
    model_colors = ['#2E86AB', '#A23B72', '#D2B48C', '#C73E1D', '#6A994E', '#7209B7', '#F77F00', '#003566']
    color_index = 0
    
    for model_name, model_data in dataframes.items():
        df = model_data['data']
        color = model_colors[color_index % len(model_colors)]
        
        plot_emissions_vs_epoch(df, model_name, color, ax)
        color_index += 1
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Emisiones Acumuladas (kg CO₂)', fontsize=14, fontweight='bold')
    ax.set_title('Evolución de las Emisiones Acumuladas por Epoch', fontsize=16, fontweight='bold', pad=20)
    
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, 
                      shadow=True, fontsize=10, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    emissions_epoch_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\grafico_emisiones_vs_epoch.png"
    plt.savefig(emissions_epoch_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Gráfico Emisiones vs Epoch guardado: {emissions_epoch_path}")
    plt.show()

def create_best_rmse_bar_chart(dataframes):
    """Crea un gráfico de barras con el mejor RMSE de cada modelo"""
    model_names = []
    best_rmse_values = []
    model_colors = ['#2E86AB', '#A23B72', '#D2B48C', '#C73E1D', '#6A994E', '#7209B7', '#F77F00', '#003566']
    
    # Recopilar el mejor RMSE de cada modelo
    for model_name, model_data in dataframes.items():
        df = model_data['data']
        if 'rmse' in df.columns:
            best_rmse = df['rmse'].min()
            best_epoch = df.loc[df['rmse'].idxmin(), 'epoch']
            
            model_names.append(model_name)
            best_rmse_values.append(best_rmse)
            
            print(f"{model_name}: Mejor RMSE = {best_rmse:.4f} (Epoch {best_epoch})")
    
    # Crear el gráfico de barras
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Crear las barras
    bars = ax.bar(model_names, best_rmse_values, 
                  color=model_colors[:len(model_names)], alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Añadir valores en las barras
    for i, (bar, value) in enumerate(zip(bars, best_rmse_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Configurar el gráfico
    ax.set_xlabel('Modelos', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mejor RMSE', fontsize=14, fontweight='bold')
    ax.set_title('Comparación del Mejor RMSE por Modelo', fontsize=16, fontweight='bold', pad=20)
    
    # Rotar etiquetas del eje X para mejor legibilidad
    plt.xticks(rotation=45, ha='right')
    
    # Añadir grid horizontal
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    # Ajustar límites del eje Y para que se vean mejor las diferencias
    y_min = min(best_rmse_values)
    y_max = max(best_rmse_values)
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range*0.1, y_max + y_range*0.15)
    
    plt.tight_layout()
    bar_chart_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\grafico_mejor_rmse_barras.png"
    plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Gráfico de barras del mejor RMSE guardado: {bar_chart_path}")
    plt.show()

def create_individual_collage(dataframes):
    """Crea un collage con todas las gráficas individuales"""
    # Calcular el layout dinámicamente basado en el número de modelos
    num_models = len(dataframes)
    
    # Calcular filas y columnas para un layout vertical (más filas que columnas)
    rows = int(np.ceil(np.sqrt(num_models)))
    cols = int(np.ceil(num_models / rows))
    
    # Ajustar el layout para que sea más manejable visualmente en vertical
    if num_models > 20:
        rows = 7  # Máximo 7 filas para mantener legibilidad
        cols = int(np.ceil(num_models / rows))
    
    print(f"Creando collage de {rows}x{cols} para {num_models} modelos...")
    
    # Configurar el layout
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    
    # Aplanar el array de axes para facilitar la iteración
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Colores extendidos para muchos modelos
    base_colors = ['#2E86AB', '#A23B72', '#D2B48C', '#C73E1D', '#6A994E', '#7209B7', '#F77F00', '#003566']
    extended_colors = base_colors * 10  # Repetir colores si es necesario
    
    # Iterar por cada modelo y crear su gráfico individual
    for idx, (model_name, model_data) in enumerate(dataframes.items()):
        ax = axes[idx]
        df = model_data['data']
        config = model_data['config']
        
        # Usar el color del modelo
        color = extended_colors[idx % len(extended_colors)]
        
        # Plotear RMSE y MAE si existen las columnas
        if 'rmse' in df.columns:
            # Encontrar el mejor RMSE
            best_rmse_idx = df['rmse'].idxmin()
            best_rmse_value = df.loc[best_rmse_idx, 'rmse']
            best_rmse_emissions = df.loc[best_rmse_idx, 'cumulative_emissions_kg']
            best_rmse_epoch = df.loc[best_rmse_idx, 'epoch']
            
            # Plotear la línea de RMSE
            ax.plot(df['cumulative_emissions_kg'].values, df['rmse'].values, 
                   color=color, linewidth=2, alpha=0.8, marker='o', markersize=2, 
                   label='RMSE', linestyle='-')
            
            # Marcar el mejor punto RMSE con un diamante del mismo tamaño que el cuadrado MAE
            ax.scatter(best_rmse_emissions, best_rmse_value, 
                      color=color, s=40, zorder=5, 
                      edgecolors='white', linewidth=1, marker='D')
        
        # Plotear MAE si existe la columna
        if 'mae' in df.columns:
            # Encontrar el mejor MAE
            best_mae_idx = df['mae'].idxmin()
            best_mae_value = df.loc[best_mae_idx, 'mae']
            best_mae_emissions = df.loc[best_mae_idx, 'cumulative_emissions_kg']
            best_mae_epoch = df.loc[best_mae_idx, 'epoch']
            
            # Crear un color más claro para MAE
            import matplotlib.colors as mcolors
            mae_color = mcolors.to_rgba(color, alpha=0.7)
            
            # Plotear la línea de MAE con puntos
            ax.plot(df['cumulative_emissions_kg'].values, df['mae'].values, 
                   color=mae_color, linewidth=1.5, alpha=0.6, marker='.', markersize=1.5, 
                   label='MAE', linestyle=':', markevery=3)
            
            # Marcar el mejor punto MAE
            ax.scatter(best_mae_emissions, best_mae_value, 
                      color=mae_color, s=40, zorder=4, 
                      edgecolors='white', linewidth=1, marker='s', alpha=0.8)
        
        # Configurar el subplot con información completa
        if 'rmse' in df.columns:
            title_text = f'{model_name}\nMejor RMSE: {best_rmse_value:.4f}\nEpoch {best_rmse_epoch} | Emisiones: {best_rmse_emissions:.6f} kg'
        else:
            title_text = f'{model_name}\nDatos no disponibles'
            
        ax.set_title(title_text, fontsize=9, fontweight='bold', pad=10)
        ax.set_xlabel('Emisiones Acumuladas (kg CO₂)', fontsize=8)
        ax.set_ylabel('Error', fontsize=8)
        
        # Configurar escala logarítmica en X para mejor visualización
        ax.set_xscale('log')
        
        # Grid ligero
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Ajustar tamaño de los ticks
        ax.tick_params(axis='both', which='major', labelsize=7)
        
        # Añadir leyenda pequeña
        if 'rmse' in df.columns and 'mae' in df.columns:
            ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    
    # Ocultar ejes no utilizados
    total_subplots = rows * cols
    for idx in range(num_models, total_subplots):
        axes[idx].set_visible(False)
    
    # Ajustar layout sin título superior
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Guardar el collage
    collage_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\collage_graficos_individuales.png"
    plt.savefig(collage_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Collage de gráficos individuales guardado: {collage_path}")
    plt.show()

def setup_plot_formatting(ax, title):
    """Configura el formato del gráfico"""
    ax.set_xlabel('Emisiones Acumuladas (kg CO₂)', fontsize=14, fontweight='bold')
    ax.set_ylabel('RMSE', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Configurar la leyenda
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                      shadow=True, fontsize=10, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    
    # Formato de ejes y grid
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# =============================================================================
# PROCESAMIENTO PRINCIPAL
# =============================================================================

# Determinar qué archivos procesar
if process_all_in_folder:
    csv_files = list(Path(folder_path).glob("emissions_metrics*.csv"))
    files_to_process = []
    for i, file_path in enumerate(csv_files):
        model_name = extract_model_name(str(file_path))
        colors = get_colors(i, len(csv_files))
        files_to_process.append({
            'path': str(file_path),
            'name': model_name,
            'color_rmse': colors[0],
            'color_mae': colors[1]
        })
else:
    files_to_process = files_config

print(f"Procesando {len(files_to_process)} archivos...")

# Verificar que los archivos existen
valid_files = []
for file_config in files_to_process:
    if os.path.exists(file_config['path']):
        valid_files.append(file_config)
        print(f"✓ {file_config['name']}: {file_config['path']}")
    else:
        print(f"✗ {file_config['name']}: Archivo no encontrado - {file_config['path']}")

if not valid_files:
    print("No se encontraron archivos válidos.")
    exit()

# Cargar todos los DataFrames
dataframes = {}
for file_config in valid_files:
    try:
        df = pd.read_csv(file_config['path'])
        print(f"Columnas originales en {file_config['name']}: {list(df.columns)}")
        
        # Limpiar valores de tensor si existen
        df = clean_tensor_values(df)
        
        # Estandarizar nombres de columnas
        df = standardize_columns(df)
        
        print(f"Columnas después de procesamiento: {list(df.columns)}")
        
        dataframes[file_config['name']] = {
            'data': df,
            'config': file_config
        }
        print(f"✓ Cargado: {file_config['name']} ({len(df)} filas)")
    except Exception as e:
        print(f"✗ Error cargando {file_config['name']}: {e}")

# =============================================================================
# NUEVAS FUNCIONALIDADES SOLICITADAS
# =============================================================================

def create_model_comparison_collage(dataframes):
    """Crea un collage comparando cada modelo entre diferentes datasets"""
    print("--- Generando collage de comparación de modelos por dataset ---")
    
    # Obtener lista de modelos únicos (sin dataset en el nombre)
    unique_models = {}
    for model_name in dataframes.keys():
        if '(' in model_name:
            base_model = model_name.split(' (')[0]
        else:
            base_model = model_name
        
        if base_model not in unique_models:
            unique_models[base_model] = []
        unique_models[base_model].append(model_name)
    
    # Calcular layout específico: 4 filas x 2 columnas para 8 modelos
    num_models = len(unique_models)
    rows = 4
    cols = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, (base_model, model_variants) in enumerate(unique_models.items()):
        ax = axes[idx]
        
        for variant_idx, model_name in enumerate(model_variants):
            if model_name in dataframes:
                df = dataframes[model_name]['data']
                
                # Determinar dataset del modelo
                if '(' in model_name:
                    dataset = model_name.split('(')[1].replace(')', '')
                else:
                    dataset = 'Movielens 1M'
                
                # Obtener color específico del dataset
                color = get_dataset_color(dataset)
                
                if 'rmse' in df.columns:
                    # Encontrar mejor RMSE
                    best_idx = df['rmse'].idxmin()
                    best_rmse = df.loc[best_idx, 'rmse']
                    best_emissions = df.loc[best_idx, 'cumulative_emissions_kg']
                    
                    # Plotear línea
                    ax.plot(df['cumulative_emissions_kg'].values, df['rmse'].values, 
                           color=color, linewidth=2, alpha=0.8, 
                           label=model_name, marker='o', markersize=3)
                    
                    # Marcar mejor punto
                    ax.scatter(best_emissions, best_rmse, 
                             color=color, s=60, zorder=5, 
                             edgecolors='white', linewidth=2, marker='D')
        
        ax.set_title(f'{base_model}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Emisiones Acumuladas (kg CO₂)', fontsize=10)
        ax.set_ylabel('RMSE', fontsize=10)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
    
    # Ocultar ejes no utilizados
    for idx in range(num_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    collage_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\collage_comparacion_modelos_por_dataset.png"
    plt.savefig(collage_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Collage de comparación de modelos guardado: {collage_path}")
    plt.close()

def create_dataset_comparison_collage(dataframes):
    """Crea un collage comparando cada dataset entre diferentes modelos"""
    print("--- Generando collage de comparación de datasets por modelo ---")
    
    # Obtener datasets únicos
    datasets = {}
    for model_name in dataframes.keys():
        if '(' in model_name:
            dataset = model_name.split(' (')[1].replace(')', '')
            if dataset not in datasets:
                datasets[dataset] = []
            datasets[dataset].append(model_name)
        else:
            # Modelos originales (sin dataset específico)
            if 'Movielens 1M' not in datasets:
                datasets['Movielens 1M'] = []
            datasets['Movielens 1M'].append(model_name)
    
    # Calcular layout (cambiado a vertical: más filas que columnas)
    num_datasets = len(datasets)
    rows = min(3, num_datasets)
    cols = int(np.ceil(num_datasets / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, (dataset, models) in enumerate(datasets.items()):
        ax = axes[idx]
        
        for model_idx, model_name in enumerate(models):
            if model_name in dataframes:
                df = dataframes[model_name]['data']
                
                # Obtener color específico del modelo
                color = get_model_color(model_name)
                
                if 'rmse' in df.columns:
                    # Encontrar mejor RMSE
                    best_idx = df['rmse'].idxmin()
                    best_rmse = df.loc[best_idx, 'rmse']
                    best_emissions = df.loc[best_idx, 'cumulative_emissions_kg']
                    
                    # Plotear línea
                    ax.plot(df['cumulative_emissions_kg'].values, df['rmse'].values, 
                           color=color, linewidth=2, alpha=0.8, 
                           label=model_name.split(' (')[0], marker='o', markersize=3)
                    
                    # Marcar mejor punto
                    ax.scatter(best_emissions, best_rmse, 
                             color=color, s=60, zorder=5, 
                             edgecolors='white', linewidth=2, marker='D')
        
        ax.set_title(f'Dataset: {dataset}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Emisiones Acumuladas (kg CO₂)', fontsize=10)
        ax.set_ylabel('RMSE', fontsize=10)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
    
    # Ocultar ejes no utilizados
    for idx in range(num_datasets, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    collage_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\collage_comparacion_datasets_por_modelo.png"
    plt.savefig(collage_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Collage de comparación de datasets guardado: {collage_path}")
    plt.close()

def create_dataset_bar_charts(dataframes):
    """Crea gráficos de barras para cada dataset con los mejores RMSE"""
    print("--- Generando gráficos de barras por dataset ---")
    
    # Organizar por datasets
    datasets = {}
    for model_name in dataframes.keys():
        if '(' in model_name:
            dataset = model_name.split(' (')[1].replace(')', '')
            model_base = model_name.split(' (')[0]
        else:
            dataset = 'Movielens 1M'
            model_base = model_name
        
        if dataset not in datasets:
            datasets[dataset] = {}
        
        df = dataframes[model_name]['data']
        if 'rmse' in df.columns:
            best_idx = df['rmse'].idxmin()
            best_rmse = df.loc[best_idx, 'rmse']
            best_epoch = df.loc[best_idx, 'epoch']
            datasets[dataset][model_base] = {'rmse': best_rmse, 'epoch': best_epoch}
    
    # Crear gráfico individual para cada dataset
    bar_chart_paths = []
    for dataset, models_data in datasets.items():
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(models_data.keys())
        rmse_values = [models_data[model]['rmse'] for model in models]
        epochs = [models_data[model]['epoch'] for model in models]
        
        # Usar colores específicos por modelo
        colors = [get_model_color(model) for model in models]
        bars = ax.bar(models, rmse_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Añadir valores en las barras
        for bar, rmse, epoch in zip(bars, rmse_values, epochs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{rmse:.4f}\n(Epoch {epoch})', ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f'Mejores RMSE por Modelo - Dataset: {dataset}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Modelos', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mejor RMSE', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        path = f"C:\\Users\\xpati\\Documents\\TFG\\RESULTADOS_FINAL\\barras_rmse_dataset_{dataset.lower()}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Gráfico de barras para {dataset} guardado: {path}")
        bar_chart_paths.append(path)
        plt.close()
    
    return bar_chart_paths

def create_bar_charts_collage(dataframes):
    """Crea un collage con todos los gráficos de barras por dataset"""
    print("--- Generando collage de gráficos de barras ---")
    
    # Organizar por datasets
    datasets = {}
    for model_name in dataframes.keys():
        if '(' in model_name:
            dataset = model_name.split(' (')[1].replace(')', '')
            model_base = model_name.split(' (')[0]
        else:
            dataset = 'Movielens 1M'
            model_base = model_name
        
        if dataset not in datasets:
            datasets[dataset] = {}
        
        df = dataframes[model_name]['data']
        if 'rmse' in df.columns:
            best_idx = df['rmse'].idxmin()
            best_rmse = df.loc[best_idx, 'rmse']
            best_epoch = df.loc[best_idx, 'epoch']
            datasets[dataset][model_base] = {'rmse': best_rmse, 'epoch': best_epoch}
    
    # Calcular layout (cambiado a vertical: más filas que columnas)
    num_datasets = len(datasets)
    rows = min(3, num_datasets)
    cols = int(np.ceil(num_datasets / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, (dataset, models_data) in enumerate(datasets.items()):
        ax = axes[idx]
        
        models = list(models_data.keys())
        rmse_values = [models_data[model]['rmse'] for model in models]
        epochs = [models_data[model]['epoch'] for model in models]
        
        # Usar colores específicos por modelo
        colors = [get_model_color(model) for model in models]
        bars = ax.bar(models, rmse_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Añadir valores en las barras
        for bar, rmse, epoch in zip(bars, rmse_values, epochs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{rmse:.3f}\n(E{epoch})', ha='center', va='bottom', fontsize=7)
        
        ax.set_title(f'Dataset: {dataset}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Modelos', fontsize=9)
        ax.set_ylabel('Mejor RMSE', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
    
    # Ocultar ejes no utilizados
    for idx in range(num_datasets, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    collage_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\collage_barras_por_dataset.png"
    plt.savefig(collage_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Collage de gráficos de barras por dataset guardado: {collage_path}")
    plt.close()

def create_model_bar_charts(dataframes):
    """Crea gráficos de barras para cada modelo con los mejores RMSE en diferentes datasets"""
    print("--- Generando gráficos de barras por modelo ---")
    
    # Organizar por modelos
    models = {}
    for model_name in dataframes.keys():
        if '(' in model_name:
            dataset = model_name.split(' (')[1].replace(')', '')
            model_base = model_name.split(' (')[0]
        else:
            dataset = 'Movielens 1M'
            model_base = model_name
        
        if model_base not in models:
            models[model_base] = {}
        
        df = dataframes[model_name]['data']
        if 'rmse' in df.columns:
            best_idx = df['rmse'].idxmin()
            best_rmse = df.loc[best_idx, 'rmse']
            best_epoch = df.loc[best_idx, 'epoch']
            models[model_base][dataset] = {'rmse': best_rmse, 'epoch': best_epoch}
    
    # Crear gráfico individual para cada modelo
    bar_chart_paths = []
    for model, datasets_data in models.items():
        if len(datasets_data) > 1:  # Solo crear gráfico si el modelo tiene más de un dataset
            fig, ax = plt.subplots(figsize=(12, 6))
            
            datasets = list(datasets_data.keys())
            rmse_values = [datasets_data[dataset]['rmse'] for dataset in datasets]
            epochs = [datasets_data[dataset]['epoch'] for dataset in datasets]
            
            # Usar colores específicos por dataset
            colors = [get_dataset_color(dataset) for dataset in datasets]
            bars = ax.bar(datasets, rmse_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Añadir valores en las barras
            for bar, rmse, epoch in zip(bars, rmse_values, epochs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{rmse:.4f}\n(Epoch {epoch})', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(f'Mejores RMSE por Dataset - Modelo: {model}', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Datasets', fontsize=12, fontweight='bold')
            ax.set_ylabel('Mejor RMSE', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            path = f"C:\\Users\\xpati\\Documents\\TFG\\RESULTADOS_FINAL\\barras_rmse_modelo_{model.lower().replace('-', '_')}.png"
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Gráfico de barras para {model} guardado: {path}")
            bar_chart_paths.append(path)
            plt.close()
    
    return bar_chart_paths

def create_model_bar_charts_collage(dataframes):
    """Crea un collage con todos los gráficos de barras por modelo"""
    print("--- Generando collage de gráficos de barras por modelo ---")
    
    # Organizar por modelos
    models = {}
    for model_name in dataframes.keys():
        if '(' in model_name:
            dataset = model_name.split(' (')[1].replace(')', '')
            model_base = model_name.split(' (')[0]
        else:
            dataset = 'Movielens 1M'
            model_base = model_name
        
        if model_base not in models:
            models[model_base] = {}
        
        df = dataframes[model_name]['data']
        if 'rmse' in df.columns:
            best_idx = df['rmse'].idxmin()
            best_rmse = df.loc[best_idx, 'rmse']
            best_epoch = df.loc[best_idx, 'epoch']
            models[model_base][dataset] = {'rmse': best_rmse, 'epoch': best_epoch}
    
    # Filtrar modelos que tienen más de un dataset
    models_with_multiple_datasets = {model: data for model, data in models.items() if len(data) > 1}
    
    # Calcular layout específico: 4 filas x 2 columnas para 8 modelos
    num_models = len(models_with_multiple_datasets)
    if num_models == 0:
        print("No hay modelos con múltiples datasets para crear el collage.")
        return
    
    rows = 4
    cols = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, (model, datasets_data) in enumerate(models_with_multiple_datasets.items()):
        ax = axes[idx]
        
        datasets = list(datasets_data.keys())
        rmse_values = [datasets_data[dataset]['rmse'] for dataset in datasets]
        epochs = [datasets_data[dataset]['epoch'] for dataset in datasets]
        
        # Usar colores específicos por dataset
        colors = [get_dataset_color(dataset) for dataset in datasets]
        bars = ax.bar(datasets, rmse_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Añadir valores en las barras
        for bar, rmse, epoch in zip(bars, rmse_values, epochs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{rmse:.3f}\n(E{epoch})', ha='center', va='bottom', fontsize=7)
        
        ax.set_title(f'Modelo: {model}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Datasets', fontsize=9)
        ax.set_ylabel('Mejor RMSE', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
    
    # Ocultar ejes no utilizados
    for idx in range(num_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    collage_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\collage_barras_por_modelo.png"
    plt.savefig(collage_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Collage de gráficos de barras por modelo guardado: {collage_path}")
    plt.close()

# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

# Determinar qué archivos procesar
if process_all_in_folder:
    csv_files = list(Path(folder_path).glob("emissions_metrics*.csv"))
    files_to_process = []
    for i, file_path in enumerate(csv_files):
        model_name = extract_model_name(str(file_path))
        colors = get_colors(i, len(csv_files))
        files_to_process.append({
            'path': str(file_path),
            'name': model_name,
            'color_rmse': colors[0],
            'color_mae': colors[1]
        })
else:
    files_to_process = files_config

print(f"Procesando {len(files_to_process)} archivos...")

# Verificar que los archivos existen
valid_files = []
for file_config in files_to_process:
    if os.path.exists(file_config['path']):
        valid_files.append(file_config)
        print(f"✓ {file_config['name']}: {file_config['path']}")
    else:
        print(f"✗ {file_config['name']}: Archivo no encontrado - {file_config['path']}")

if not valid_files:
    print("No se encontraron archivos válidos.")
    exit()

# Cargar todos los DataFrames
dataframes = {}
for file_config in valid_files:
    try:
        df = pd.read_csv(file_config['path'])
        print(f"Columnas originales en {file_config['name']}: {list(df.columns)}")
        
        # Limpiar valores de tensor si existen
        df = clean_tensor_values(df)
        
        # Estandarizar nombres de columnas
        df = standardize_columns(df)
        
        print(f"Columnas después de procesamiento: {list(df.columns)}")
        
        dataframes[file_config['name']] = {
            'data': df,
            'config': file_config
        }
        print(f"✓ Cargado: {file_config['name']} ({len(df)} filas)")
    except Exception as e:
        print(f"✗ Error cargando {file_config['name']}: {e}")

# Ejecutar solo las funciones solicitadas
create_model_comparison_collage(dataframes)
create_dataset_comparison_collage(dataframes)
create_dataset_bar_charts(dataframes)
create_bar_charts_collage(dataframes)
create_model_bar_charts(dataframes)
create_model_bar_charts_collage(dataframes)

print("\n¡Proceso completado! Se han generado solo los gráficos solicitados.")
