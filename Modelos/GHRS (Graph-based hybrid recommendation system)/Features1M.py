import numpy as np
import pandas as pd
import itertools
import collections
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import networkx as nx


def convert_categorical(df_X, _X):
    values = np.array(df_X[_X])
    # Integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # Binary encode
    onehot_encoder = OneHotEncoder(sparse_output=False)  # Cambio aquí
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    df_X = df_X.drop(_X, axis=1)  # Cambio aquí
    for j in range(integer_encoded.max() + 1):
        df_X.insert(loc=j + 1, column=str(_X) + str(j + 1), value=onehot_encoded[:, j])
    return df_X


dataPath = 'C:/Users/xpati/Documents/TFG/ml-1m/'

# Lectura de archivos con separador corregido
df = pd.read_csv(dataPath + 'ratings.dat', sep='::', engine='python', names=['UID', 'MID', 'rate', 'time'])
df_user = pd.read_csv(dataPath + 'users.dat', sep='::', engine='python', names=['UID', 'gender', 'age', 'job', 'zip'])

# Preprocesamiento de datos
df_user = convert_categorical(df_user, 'job')
df_user = convert_categorical(df_user, 'gender')
df_user['bin'] = pd.cut(df_user['age'], [0, 10, 20, 30, 40, 50, 100], labels=['1', '2', '3', '4', '5', '6'])
df_user['age'] = df_user['bin']

df_user = df_user.drop('bin', axis=1)  # Cambio aquí
df_user = convert_categorical(df_user, 'age')
df_user = df_user.drop('zip', axis=1)  # Cambio aquí

alpha_coefs = [0.045]  # Puedes ajustar este valor

for alpha_coef in alpha_coefs:
    pairs = []
    grouped = df.groupby(['MID', 'rate'])
    for key, group in grouped:
        pairs.extend(list(itertools.combinations(group['UID'], 2)))
    counter = collections.Counter(pairs)
    alpha = alpha_coef * 3883  # Parámetro fijo
    edge_list = map(list, collections.Counter(el for el in counter.elements() if counter[el] >= alpha).keys())
    G = nx.Graph()

    for el in edge_list:
        G.add_edge(el[0], el[1], weight=1)
        G.add_edge(el[0], el[0], weight=1)
        G.add_edge(el[1], el[1], weight=1)

    # Verifica que el grafo no esté vacío
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        print(f"El grafo está vacío para alpha_coef = {alpha_coef}. Ajusta el valor de alpha_coef.")
        continue

    # Cálculo de métricas de centralidad
    pr = nx.pagerank(G.to_directed())
    df_user['PR'] = df_user['UID'].map(pr)
    df_user['PR'] /= float(df_user['PR'].max())

    dc = nx.degree_centrality(G)
    df_user['CD'] = df_user['UID'].map(dc)
    df_user['CD'] /= float(df_user['CD'].max())

    cc = nx.closeness_centrality(G)
    df_user['CC'] = df_user['UID'].map(cc)
    df_user['CC'] /= float(df_user['CC'].max())

    bc = nx.betweenness_centrality(G)  # No se necesita random_state
    df_user['CB'] = df_user['UID'].map(bc)
    df_user['CB'] /= float(df_user['CB'].max())

    lc = nx.load_centrality(G)
    df_user['LC'] = df_user['UID'].map(lc)
    df_user['LC'] /= float(df_user['LC'].max())

    nd = nx.average_neighbor_degree(G, weight='weight')
    df_user['AND'] = df_user['UID'].map(nd)
    df_user['AND'] /= float(df_user['AND'].max())

    # Guardar los datos procesados
    X_train = df_user.copy()  # Incluye todas las columnas, incluyendo 'UID'
    X_train.fillna(0, inplace=True)
    X_train.to_pickle(f"data1m/x_train_alpha({alpha_coef}).pkl")