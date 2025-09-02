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


# --- 3. Visualizaciones ---

# Configuración de estilo visual para todas las gráficas
sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 22,
    'figure.titleweight': 'bold'
})

# Gráfica 1: Distribución de las Valoraciones (Ratings)
# Como 'rating' es continuo, un histograma es más adecuado que un countplot.
plt.figure(figsize=(12, 7))
ax = sns.histplot(data=df, x='rating', bins=30, color='purple')
plt.title('Distribución de las Valoraciones')
plt.xlabel('Rating')
plt.ylabel('Frecuencia')
# plt.suptitle('¿Cómo se distribuyen las calificaciones?', y=1.02)
plt.tight_layout()
plt.show()


# Gráfica 2: Distribución de Interacciones por Usuario (con escala logarítmica)
plt.figure(figsize=(12, 7))
user_interaction_counts = df['userId'].value_counts()
sns.histplot(user_interaction_counts, bins=50, color="skyblue", log_scale=(True, False))
plt.title('Distribución de Interacciones por Usuario')
plt.xlabel('Número de Interacciones (Escala Logarítmica)')
plt.ylabel('Número de Usuarios')
median_interactions = user_interaction_counts.median()
plt.axvline(median_interactions, color='red', linestyle='--', label=f'Mediana: {median_interactions:.0f} interacciones')
plt.legend()
# plt.suptitle('¿Cuántas calificaciones da cada usuario?', y=1.02)
plt.tight_layout()
plt.show()

'''
# --- 2.1. Distribución de interacciones por usuario (escala normal) ---
plt.figure(figsize=(10, 6))
user_interaction_counts = df['userId'].value_counts()
sns.histplot(user_interaction_counts, bins=40, color="skyblue", edgecolor='black')
plt.title('Distribución de Interacciones por Usuario')
plt.xlabel('Número de Interacciones')
plt.ylabel('Número de Usuarios')
plt.tight_layout()
plt.show()
'''

# Gráfica 3: Distribución de Interacciones por Ítem (con escala logarítmica)
plt.figure(figsize=(12, 7))
item_interaction_counts = df['itemId'].value_counts()
sns.histplot(item_interaction_counts, bins=50, color="salmon", log_scale=(True, False))
plt.title('Distribución de Interacciones por Ítem')
plt.xlabel('Número de Interacciones (Escala Logarítmica)')
plt.ylabel('Número de Ítems')
median_interactions_item = item_interaction_counts.median()
plt.axvline(median_interactions_item, color='red', linestyle='--', label=f'Mediana: {median_interactions_item:.0f} interacciones')
plt.legend()
# plt.suptitle('¿Cuántas calificaciones recibe cada ítem?', y=1.02)
plt.tight_layout()
plt.show()

'''
# --- 3.1. Distribución de interacciones por ítem (escala normal) ---
plt.figure(figsize=(10, 6))
item_interaction_counts = df['itemId'].value_counts()
sns.histplot(item_interaction_counts, bins=40, color="salmon", edgecolor='black')
plt.title('Distribución de Interacciones por Ítem')
plt.xlabel('Número de Interacciones')
plt.ylabel('Número de Ítems')
plt.tight_layout()
plt.show()
'''

# Gráfica 4: Distribución de la Calificación Promedio por Usuario
plt.figure(figsize=(12, 7))
user_avg_ratings = df.groupby('userId')['rating'].mean()
sns.histplot(user_avg_ratings, bins=40, color="lightgreen")
plt.title('Distribución de la Calificación Promedio por Usuario')
plt.xlabel('Rating Promedio')
plt.ylabel('Número de Usuarios')
avg_rating_overall = df['rating'].mean()
plt.axvline(avg_rating_overall, color='blue', linestyle='--', label=f'Promedio Global: {avg_rating_overall:.2f}')
plt.legend()
# plt.suptitle('¿Los usuarios tienden a dar calificaciones altas o bajas?', y=1.02)
plt.tight_layout()
plt.show()