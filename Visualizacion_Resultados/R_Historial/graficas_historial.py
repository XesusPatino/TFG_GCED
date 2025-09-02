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

# Opción 1: Lista específica de archivos con nombres personalizados
files_config = [
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Historial\Modelos_Matrices\emissions_metrics_MF_20250817-055915.csv",
        'name': 'MF',
        'category': 'Modelos_Matrices',
        'color_rmse': '#2E86AB',
        'color_mae': '#1E5F8B'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Historial\Modelos_Matrices\emissions_metrics_NNMF_20250825-130457.csv",
        'name': 'NNMF',
        'category': 'Modelos_Matrices',
        'color_rmse': '#A23B72',
        'color_mae': '#822B52'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Historial\Modelos_Redes_Profundas\emissions_metrics_AutoRec_20250819-032250.csv",
        'name': 'AutoRec',
        'category': 'Modelos_Redes_Profundas',
        'color_rmse': '#D2B48C',
        'color_mae': '#B8986B'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Historial\Modelos_Redes_Profundas\emissions_metrics_NCF_20250825-154226.csv",
        'name': 'NCF',
        'category': 'Modelos_Redes_Profundas',
        'color_rmse': '#C73E1D',
        'color_mae': '#A7321A'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Historial\Modelos_Redes_Profundas\emissions_metrics_LRML_History_20250819-093040.csv",
        'name': 'LRML',
        'category': 'Modelos_Redes_Profundas',
        'color_rmse': '#6A994E',
        'color_mae': '#5A7F42'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Historial\Modelos_Redes_Grafos\emissions_metrics_GHRS_History_20250819-113502.csv",
        'name': 'GHRS',
        'category': 'Modelos_Redes_Grafos',
        'color_rmse': '#7209B7',
        'color_mae': '#5F0799'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Historial\Modelos_Redes_Grafos\emissions_metrics_KGNN_LS_20250825-144539.csv",
        'name': 'KGNN-LS',
        'category': 'Modelos_Redes_Grafos',
        'color_rmse': '#F77F00',
        'color_mae': '#D96A00'
    },
    {
        'path': r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Historial\Modelos_Redes_Grafos\emissions_metrics_LightGCN_History_20250827-205350.csv",
        'name': 'LightGCN',
        'category': 'Modelos_Redes_Grafos',
        'color_rmse': '#003566',
        'color_mae': '#002145'
    },
]

# Opción 2: Procesar todos los archivos de una carpeta
folder_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Historial\Modelos_Matrices"
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
    if 'ml-1m' in filename.lower():
        return 'ML-1M Full'
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

def create_individual_collage_horizontal(dataframes):
    """Crea un collage horizontal con todas las gráficas individuales (2x4)"""
    # Configurar el layout: 2 filas x 4 columnas
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Aplanar el array de axes para facilitar la iteración
    axes = axes.flatten()
    
    # Colores para cada modelo (mismo orden que en el procesamiento)
    model_colors = ['#2E86AB', '#A23B72', '#D2B48C', '#C73E1D', '#6A994E', '#7209B7', '#F77F00', '#003566']
    
    # Iterar por cada modelo y crear su gráfico individual
    for idx, (model_name, model_data) in enumerate(dataframes.items()):
        ax = axes[idx]
        df = model_data['data']
        config = model_data['config']
        
        # Usar el color del modelo
        color = model_colors[idx % len(model_colors)]
        
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
    
    # Ajustar layout sin título superior
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Guardar el collage horizontal
    collage_horizontal_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Historial\collage_graficos_individuales_horizontal.png"
    plt.savefig(collage_horizontal_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Collage horizontal de gráficos individuales guardado: {collage_horizontal_path}")
    plt.show()

def create_individual_collage_vertical(dataframes):
    """Crea un collage vertical con todas las gráficas individuales (4x2)"""
    # Configurar el layout: 4 filas x 2 columnas
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    
    # Aplanar el array de axes para facilitar la iteración
    axes = axes.flatten()
    
    # Colores para cada modelo (mismo orden que en el procesamiento)
    model_colors = ['#2E86AB', '#A23B72', '#D2B48C', '#C73E1D', '#6A994E', '#7209B7', '#F77F00', '#003566']
    
    # Iterar por cada modelo y crear su gráfico individual
    for idx, (model_name, model_data) in enumerate(dataframes.items()):
        ax = axes[idx]
        df = model_data['data']
        config = model_data['config']
        
        # Usar el color del modelo
        color = model_colors[idx % len(model_colors)]
        
        # Plotear RMSE y MAE si existen las columnas
        if 'rmse' in df.columns:
            # Encontrar el mejor RMSE
            best_rmse_idx = df['rmse'].idxmin()
            best_rmse_value = df.loc[best_rmse_idx, 'rmse']
            best_rmse_emissions = df.loc[best_rmse_idx, 'cumulative_emissions_kg']
            best_rmse_epoch = df.loc[best_rmse_idx, 'epoch']
            
            # Plotear la línea de RMSE
            ax.plot(df['cumulative_emissions_kg'].values, df['rmse'].values, 
                   color=color, linewidth=2.5, alpha=0.8, marker='o', markersize=3, 
                   label='RMSE', linestyle='-')
            
            # Marcar el mejor punto RMSE
            ax.scatter(best_rmse_emissions, best_rmse_value, 
                      color=color, s=60, zorder=5, 
                      edgecolors='white', linewidth=1.5, marker='D')
        
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
            
            # Plotear la línea de MAE
            ax.plot(df['cumulative_emissions_kg'].values, df['mae'].values, 
                   color=mae_color, linewidth=2, alpha=0.6, marker='.', markersize=2, 
                   label='MAE', linestyle=':', markevery=2)
            
            # Marcar el mejor punto MAE
            ax.scatter(best_mae_emissions, best_mae_value, 
                      color=mae_color, s=60, zorder=4, 
                      edgecolors='white', linewidth=1.5, marker='s', alpha=0.8)
        
        # Configurar el subplot con información completa
        if 'rmse' in df.columns:
            title_text = f'{model_name}\nMejor RMSE: {best_rmse_value:.4f}\nEpoch {best_rmse_epoch} | Emisiones: {best_rmse_emissions:.6f} kg'
        else:
            title_text = f'{model_name}\nDatos no disponibles'
            
        ax.set_title(title_text, fontsize=11, fontweight='bold', pad=15)
        ax.set_xlabel('Emisiones Acumuladas (kg CO₂)', fontsize=10)
        ax.set_ylabel('Error', fontsize=10)
        
        # Configurar escala logarítmica en X para mejor visualización
        ax.set_xscale('log')
        
        # Grid ligero
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Ajustar tamaño de los ticks
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        # Añadir leyenda pequeña
        if 'rmse' in df.columns and 'mae' in df.columns:
            ax.legend(fontsize=8, loc='upper right', framealpha=0.8)
    
    # Ajustar layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    
    # Guardar el collage vertical
    collage_vertical_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\R_Historial\collage_graficos_individuales_vertical.png"
    plt.savefig(collage_vertical_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Collage vertical de gráficos individuales guardado: {collage_vertical_path}")
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
# GENERAR GRÁFICOS
# =============================================================================

if plot_type == "combined":
    # Gráfico combinado simplificado (solo RMSE)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Colores profesionales para cada modelo
    model_colors = ['#2E86AB', '#A23B72', '#D2B48C', '#C73E1D', '#6A994E', '#7209B7', '#F77F00', '#003566']
    
    all_best_results = []
    color_index = 0
    
    for model_name, model_data in dataframes.items():
        df = model_data['data']
        color = model_colors[color_index % len(model_colors)]
        
        _, best_results = plot_rmse_only_combined(df, model_name, color, ax)
        if best_results:
            all_best_results.append((model_name, best_results))
        color_index += 1
    
    setup_combined_plot_formatting(ax, 'Comparación de Modelos: RMSE vs Emisiones de CO₂')
    
    plt.tight_layout()
    combined_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\grafico_combinado_rmse_simple.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Gráfico combinado simplificado guardado: {combined_path}")
    plt.show()

    # Crear gráficos por categoría
    print("\n--- Generando gráficos por categoría ---")
    
    # Organizar modelos por categoría
    categories = {}
    for model_name, model_data in dataframes.items():
        config = model_data['config']
        category = config.get('category', 'Other')
        if category not in categories:
            categories[category] = []
        categories[category].append(model_name)
    
    # Generar un gráfico para cada categoría
    for category_name, models_in_category in categories.items():
        print(f"\nGenerando gráfico para categoría: {category_name}")
        print(f"Modelos incluidos: {', '.join(models_in_category)}")
        create_category_plot(dataframes, category_name, models_in_category)

elif plot_type == "subplots":
    # Subgráficos
    n_models = len(dataframes)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, (model_name, model_data) in enumerate(dataframes.items()):
        df = model_data['data']
        config = model_data['config']
        colors = (config['color_rmse'], config['color_mae'])
        
        ax = axes[i] if n_models > 1 else axes[0]
        plot_single_csv(df, model_name, colors, ax, show_info_box=True)
        setup_plot_formatting(ax, f'{model_name}')
    
    # Ocultar ejes no utilizados
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    subplots_path = r"C:\Users\xpati\Documents\TFG\RESULTADOS_FINAL\graficos_subplots_modelos.png"
    plt.savefig(subplots_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Subgráficos guardados: {subplots_path}")
    plt.show()

# Gráficos individuales (si está habilitado)
if save_individual:
    for model_name, model_data in dataframes.items():
        fig, ax = plt.subplots(figsize=(12, 8))
        df = model_data['data']
        config = model_data['config']
        colors = (config['color_rmse'], config['color_mae'])
        
        plot_single_csv(df, model_name, colors, ax, show_info_box=True)
        setup_plot_formatting(ax, f'{model_name}: RMSE vs Emisiones de CO₂')
        
        plt.tight_layout()
        individual_path = f"C:\\Users\\xpati\\Documents\\TFG\\RESULTADOS_FINAL\\grafico_{model_name.replace(' ', '_').replace('-', '_')}.png"
        plt.savefig(individual_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Gráfico individual guardado: {individual_path}")
        plt.close()

# Generar gráficos adicionales: RMSE vs Epoch y Emisiones vs Epoch
print("\n--- Generando gráficos adicionales ---")
create_rmse_epoch_plots(dataframes)
create_emissions_epoch_plots(dataframes)
create_best_rmse_bar_chart(dataframes)

# Crear ambos collages: horizontal y vertical
create_individual_collage_horizontal(dataframes)
create_individual_collage_vertical(dataframes)

print("\n¡Proceso completado!")
