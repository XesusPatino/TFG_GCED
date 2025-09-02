import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Carga y Preparación de Datos ---
# Leer el archivo con delimitador :: y sin encabezados
try:
    df = pd.read_csv("ratings.csv", 
                     sep='::', 
                     header=None, 
                     names=['user_id', 'item_id', 'rating', 'timestamp'],
                     engine='python')
except FileNotFoundError:
    print("Error: El archivo 'ratings.csv' no se encontró.")
    # Se detiene la ejecución si el archivo no se encuentra.
    exit()

# Renombrar las columnas para mantener consistencia con el resto del código
df.rename(columns={
    'user_id': 'userId',
    'item_id': 'itemId'
}, inplace=True)

# --- 2. Estadísticas Generales ---
n_users = df['userId'].nunique()
n_items = df['itemId'].nunique()
n_ratings = df.shape[0]
density = (n_ratings / (n_users * n_items)) * 100
avg_rating = df['rating'].mean()
min_rating = df['rating'].min()
max_rating = df['rating'].max()

print("--- Análisis Exploratorio del Dataset de Ratings ---")
print(f"Usuarios únicos: {n_users}")
print(f"Ítems únicos: {n_items}")
print(f"Total de valoraciones: {n_ratings}")
print(f"Rating promedio global: {avg_rating:.2f}")
print(f"Rating mínimo: {min_rating:.2f}")
print(f"Rating máximo: {max_rating:.2f}")
print(f"Densidad de la matriz: {density:.4f}% (muy dispersa, como es habitual)")
print("-" * 50)

# --- 3. Visualizaciones en Collage ---

# Configuración de estilo visual para todas las gráficas
sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 18,
    'figure.titleweight': 'bold'
})

# Crear figura con subplots (2x2)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Análisis Exploratorio del Dataset de Ratings', fontsize=20, fontweight='bold', y=0.98)

# Gráfica 1: Distribución de las Valoraciones (Ratings)
sns.histplot(data=df, x='rating', bins=30, color='purple', ax=axes[0, 0])
axes[0, 0].set_title('Distribución de las Valoraciones')
axes[0, 0].set_xlabel('Rating')
axes[0, 0].set_ylabel('Frecuencia')

# Gráfica 2: Distribución de Interacciones por Usuario (con escala logarítmica)
user_interaction_counts = df['userId'].value_counts()
sns.histplot(user_interaction_counts, bins=50, color="skyblue", log_scale=(True, False), ax=axes[0, 1])
axes[0, 1].set_title('Distribución de Interacciones por Usuario')
axes[0, 1].set_xlabel('Número de Interacciones (Escala Log)')
axes[0, 1].set_ylabel('Número de Usuarios')
median_interactions = user_interaction_counts.median()
axes[0, 1].axvline(median_interactions, color='red', linestyle='--', 
                   label=f'Mediana: {median_interactions:.0f}')
axes[0, 1].legend()

# Gráfica 3: Distribución de Interacciones por Ítem (con escala logarítmica)
item_interaction_counts = df['itemId'].value_counts()
sns.histplot(item_interaction_counts, bins=50, color="salmon", log_scale=(True, False), ax=axes[1, 0])
axes[1, 0].set_title('Distribución de Interacciones por Ítem')
axes[1, 0].set_xlabel('Número de Interacciones (Escala Log)')
axes[1, 0].set_ylabel('Número de Ítems')
median_interactions_item = item_interaction_counts.median()
axes[1, 0].axvline(median_interactions_item, color='red', linestyle='--', 
                   label=f'Mediana: {median_interactions_item:.0f}')
axes[1, 0].legend()

# Gráfica 4: Distribución de la Calificación Promedio por Usuario
user_avg_ratings = df.groupby('userId')['rating'].mean()
sns.histplot(user_avg_ratings, bins=40, color="lightgreen", ax=axes[1, 1])
axes[1, 1].set_title('Calificación Promedio por Usuario')
axes[1, 1].set_xlabel('Rating Promedio')
axes[1, 1].set_ylabel('Número de Usuarios')
avg_rating_overall = df['rating'].mean()
axes[1, 1].axvline(avg_rating_overall, color='blue', linestyle='--', 
                   label=f'Promedio Global: {avg_rating_overall:.2f}')
axes[1, 1].legend()

# Ajustar espaciado entre subplots
plt.tight_layout()
plt.subplots_adjust(top=0.94)

# Guardar el collage
plt.savefig('ratings_analisis_exploratorio.png', dpi=300, bbox_inches='tight')
plt.show()