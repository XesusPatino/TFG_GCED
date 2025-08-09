import numpy as np
import os
from collections import defaultdict
import random

def read_rating(path, train_ratio): # Quitamos num_users, num_items, num_total_ratings, a, b
    """
    Lee los ratings, crea mapeos de ID, determina tamaños dinámicamente,
    y divide en conjuntos de entrenamiento/prueba.
    """
    ratings_file = os.path.join(path, "gdsc1_processed.csv")
    print(f"Cargando datos desde: {ratings_file}")

    if not os.path.exists(ratings_file):
        raise FileNotFoundError(f"Archivo de ratings no encontrado: {ratings_file}")

    # --- Paso 1: Leer para mapear IDs y contar ---
    user_map = {}
    item_map = {}
    user_counter = 0
    item_counter = 0
    all_ratings_raw = [] # Guardar temporalmente los datos crudos

    with open(ratings_file, "r") as fp:
        header = fp.readline() # Saltar cabecera
        for line in fp:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) >= 3:
                try:
                    # Usar los IDs originales del archivo
                    original_user_id = parts[0]
                    original_item_id = parts[1]
                    rating = float(parts[2])

                    # Mapear user_id si es nuevo
                    if original_user_id not in user_map:
                        user_map[original_user_id] = user_counter
                        user_counter += 1
                    # Mapear item_id si es nuevo
                    if original_item_id not in item_map:
                        item_map[original_item_id] = item_counter
                        item_counter += 1

                    # Guardar el rating con los IDs originales por ahora
                    all_ratings_raw.append((original_user_id, original_item_id, rating))

                except ValueError:
                    print(f"Advertencia: No se pudo convertir a número la línea: {line.strip()}")
            else:
                print(f"Advertencia: Línea ignorada por tener menos de 3 partes: {line.strip()}")

    # --- Paso 2: Determinar tamaños reales ---
    num_users = user_counter
    num_items = item_counter
    num_total_ratings = len(all_ratings_raw)

    print(f"Procesamiento inicial completado:")
    print(f"  Usuarios únicos encontrados: {num_users}")
    print(f"  Ítems únicos encontrados: {num_items}")
    print(f"  Total de ratings encontrados: {num_total_ratings}")

    # --- Paso 3: Inicializar arrays con tamaños correctos ---
    R = np.zeros((num_users, num_items))
    mask_R = np.zeros((num_users, num_items))
    C = np.ones((num_users, num_items)) # Confianza inicial (puede ajustarse)
    train_R = np.zeros((num_users, num_items))
    test_R = np.zeros((num_users, num_items))
    train_mask_R = np.zeros((num_users, num_items))
    test_mask_R = np.zeros((num_users, num_items))
    user_train_set = defaultdict(list)
    item_train_set = defaultdict(list)
    user_test_set = defaultdict(list)
    item_test_set = defaultdict(list)

    # --- Paso 4: Llenar arrays y dividir en train/test ---
    random.shuffle(all_ratings_raw) # Mezclar para división aleatoria
    num_train_ratings = 0
    num_test_ratings = 0

    for i, (original_user_id, original_item_id, rating) in enumerate(all_ratings_raw):
        # Obtener índices internos mapeados
        user_idx = user_map[original_user_id]
        item_idx = item_map[original_item_id]

        # Llenar R y mask_R
        R[user_idx, item_idx] = rating
        mask_R[user_idx, item_idx] = 1
        # Podrías ajustar C aquí si tienes información de confianza, e.g., C[user_idx, item_idx] = 1 + a * abs(rating)

        # Dividir en train/test
        if i < num_total_ratings * train_ratio:
            train_R[user_idx, item_idx] = rating
            train_mask_R[user_idx, item_idx] = 1
            user_train_set[user_idx].append(item_idx)
            item_train_set[item_idx].append(user_idx)
            num_train_ratings += 1
        else:
            test_R[user_idx, item_idx] = rating
            test_mask_R[user_idx, item_idx] = 1
            user_test_set[user_idx].append(item_idx)
            item_test_set[item_idx].append(user_idx)
            num_test_ratings += 1

    print(f"Ratings de entrenamiento: {num_train_ratings}")
    print(f"Ratings de prueba: {num_test_ratings}")
    print(f"Usuarios en entrenamiento: {len(user_train_set)}")
    print(f"Ítems en entrenamiento: {len(item_train_set)}")

    # Devolver los valores determinados dinámicamente junto con los arrays
    return (R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,
            num_train_ratings, num_test_ratings,
            user_train_set, item_train_set, user_test_set, item_test_set,
            num_users, num_items, num_total_ratings) # Devolver los tamaños reales