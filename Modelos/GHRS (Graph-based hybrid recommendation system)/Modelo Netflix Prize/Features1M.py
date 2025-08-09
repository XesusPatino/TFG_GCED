import numpy as np
import pandas as pd
import itertools
import collections
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import networkx as nx
import os

dataPath = 'C:/Users/xpati/Documents/TFG/'

# Asegurarse de que exista el directorio para guardar resultados
if not os.path.exists('data1m'):
    os.makedirs('data1m')

# Lectura del archivo netflix.csv
df = pd.read_csv(dataPath + 'netflix.csv')

# Renombrar columnas para mantener consistencia con el código original
df = df.rename(columns={'user_id': 'UID', 'movie_id': 'MID', 'rating': 'rate'})

# Crear df_user a partir de los usuarios únicos en el dataset
unique_users = df['UID'].unique()
df_user = pd.DataFrame({'UID': unique_users})

# Información del dataset
print(f"Total de usuarios únicos: {len(unique_users)}")
print(f"Total de items únicos: {df['MID'].nunique()}")
print(f"Total de calificaciones: {len(df)}")

# Probar con valores más bajos de alpha_coef
alpha_coefs = [0.08]  # Valores más bajos para datasets pequeños

for alpha_coef in alpha_coefs:
    print(f"\nProcesando con alpha_coef = {alpha_coef}")
    pairs = []
    grouped = df.groupby(['MID', 'rate'])
    for key, group in grouped:
        pairs.extend(list(itertools.combinations(group['UID'], 2)))
    counter = collections.Counter(pairs)
    
    # Análisis detallado del umbral
    num_users = len(unique_users)
    alpha = alpha_coef * num_users
    print(f"Umbral alpha calculado: {alpha}")
    
    # Información sobre frecuencias de pares
    if counter:
        max_count = max(counter.values()) if counter else 0
        print(f"Frecuencia máxima de un par de usuarios: {max_count}")
        
        # Histograma simplificado de frecuencias
        freq_counts = collections.Counter(counter.values())
        print("Distribución de frecuencias:")
        for freq, count in sorted(freq_counts.items())[:10]:  # Mostrar las primeras 10
            print(f"  {freq} apariciones: {count} pares")
    
    edge_list = list(filter(lambda el: counter[el] >= alpha, counter.keys()))
    print(f"Pares de usuarios que superan el umbral: {len(edge_list)}")
    
    G = nx.Graph()

    for el in edge_list:
        G.add_edge(el[0], el[1], weight=1)
        G.add_edge(el[0], el[0], weight=1)
        G.add_edge(el[1], el[1], weight=1)

    # Verifica que el grafo no esté vacío
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        print(f"El grafo está vacío para alpha_coef = {alpha_coef}. Probando con el siguiente valor.")
        continue

    # Resto del código igual...
    # Cálculo de métricas de centralidad
    pr = nx.pagerank(G.to_directed())
    df_user['PR'] = df_user['UID'].map(pr).fillna(0)
    if df_user['PR'].max() > 0:
        df_user['PR'] /= float(df_user['PR'].max())

    dc = nx.degree_centrality(G)
    df_user['CD'] = df_user['UID'].map(dc).fillna(0)
    if df_user['CD'].max() > 0:
        df_user['CD'] /= float(df_user['CD'].max())

    cc = nx.closeness_centrality(G)
    df_user['CC'] = df_user['UID'].map(cc).fillna(0)
    if df_user['CC'].max() > 0:
        df_user['CC'] /= float(df_user['CC'].max())

    bc = nx.betweenness_centrality(G)
    df_user['CB'] = df_user['UID'].map(bc).fillna(0)
    if df_user['CB'].max() > 0:
        df_user['CB'] /= float(df_user['CB'].max())

    lc = nx.load_centrality(G)
    df_user['LC'] = df_user['UID'].map(lc).fillna(0)
    if df_user['LC'].max() > 0:
        df_user['LC'] /= float(df_user['LC'].max())

    nd = nx.average_neighbor_degree(G, weight='weight')
    df_user['AND'] = df_user['UID'].map(nd).fillna(0)
    if df_user['AND'].max() > 0:
        df_user['AND'] /= float(df_user['AND'].max())

    # Guardar los datos procesados
    X_train = df_user.copy()
    X_train.fillna(0, inplace=True)
    X_train.to_pickle(f"data1m/x_train_Netflix_alpha({alpha_coef}).pkl")
    
    print(f"Procesamiento completado para alpha_coef = {alpha_coef}")
    print(f"Número de nodos en el grafo: {G.number_of_nodes()}")
    print(f"Número de aristas en el grafo: {G.number_of_edges()}")