import numpy as np
import os
from collections import defaultdict
import random
from scipy.sparse import csr_matrix, coo_matrix

def read_rating(path, train_ratio):
    """
    Lee los ratings, crea mapeos de ID, determina tamaños dinámicamente,
    y divide en conjuntos de entrenamiento/prueba usando matrices dispersas,
    asignando aleatoriamente cada rating al leerlo.
    """
    ratings_file = os.path.join(path, "netflix.csv")
    print(f"Cargando datos desde: {ratings_file}")

    if not os.path.exists(ratings_file):
        raise FileNotFoundError(f"Archivo de ratings no encontrado: {ratings_file}")

    # --- Paso 1: Inicializar estructuras ---
    user_map = {}
    item_map = {}
    user_counter = 0
    item_counter = 0

    # Listas para construir matrices dispersas directamente
    train_row_indices = []
    train_col_indices = []
    train_data = []
    test_row_indices = []
    test_col_indices = []
    test_data = []

    user_train_set = defaultdict(list)
    item_train_set = defaultdict(list)
    user_test_set = defaultdict(list)
    item_test_set = defaultdict(list) # Necesario si se usa en algún lugar

    num_train_ratings = 0
    num_test_ratings = 0
    num_total_ratings_processed = 0 # Contador para el total

    print("Procesando archivo y dividiendo en train/test...")
    # --- Paso 2: Leer archivo, mapear IDs y asignar a train/test ---
    with open(ratings_file, "r") as fp:
        header = fp.readline() # Saltar cabecera
        line_count = 0
        for line in fp:
            line_count += 1
            if not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) >= 3:
                try:
                    original_user_id = parts[0]
                    original_item_id = parts[1]
                    rating = float(parts[2])
                    num_total_ratings_processed += 1

                    # Mapear IDs
                    if original_user_id not in user_map:
                        user_map[original_user_id] = user_counter
                        user_counter += 1
                    if original_item_id not in item_map:
                        item_map[original_item_id] = item_counter
                        item_counter += 1

                    user_idx = user_map[original_user_id]
                    item_idx = item_map[original_item_id]

                    # Decidir aleatoriamente si va a train o test
                    if random.random() < train_ratio:
                        train_row_indices.append(user_idx)
                        train_col_indices.append(item_idx)
                        train_data.append(rating)
                        user_train_set[user_idx].append(item_idx)
                        item_train_set[item_idx].append(user_idx)
                        num_train_ratings += 1
                    else:
                        test_row_indices.append(user_idx)
                        test_col_indices.append(item_idx)
                        test_data.append(rating)
                        user_test_set[user_idx].append(item_idx)
                        # item_test_set[item_idx].append(user_idx) # Descomentar si se necesita item_test_set
                        num_test_ratings += 1

                    # Imprimir progreso ocasionalmente (opcional)
                    if line_count % 1000000 == 0:
                         print(f"  Procesadas {line_count // 1000000}M líneas...")

                except ValueError:
                    print(f"Advertencia: No se pudo convertir a número la línea: {line.strip()}")
            else:
                print(f"Advertencia: Línea ignorada por tener menos de 3 partes: {line.strip()}")

    # --- Paso 3: Determinar tamaños finales ---
    num_users = user_counter
    num_items = item_counter

    print(f"\nProcesamiento completado:")
    print(f"  Usuarios únicos encontrados: {num_users}")
    print(f"  Ítems únicos encontrados: {num_items}")
    print(f"  Total de ratings procesados: {num_total_ratings_processed}")
    print(f"  Ratings de entrenamiento asignados: {num_train_ratings}")
    print(f"  Ratings de prueba asignados: {num_test_ratings}")

    # --- Paso 4: Crear matrices dispersas ---
    shape = (num_users, num_items)
    print("Creando matrices dispersas...")

    if not train_data:
         raise ValueError("No se asignaron datos al conjunto de entrenamiento.")
    train_R = coo_matrix((train_data, (train_row_indices, train_col_indices)), shape=shape).tocsr()
    train_mask_R = train_R.astype(bool)

    if not test_data:
         # Es posible que no haya datos de prueba si train_ratio es 1 o muy cercano
         print("Advertencia: No se asignaron datos al conjunto de prueba.")
         # Crear matrices vacías si es necesario para la consistencia del retorno
         test_R = csr_matrix(shape, dtype=np.float32)
         test_mask_R = csr_matrix(shape, dtype=bool)
    else:
         test_R = coo_matrix((test_data, (test_row_indices, test_col_indices)), shape=shape).tocsr()
         test_mask_R = test_R.astype(bool)

    # Nota: R, mask_R y C (las matrices completas) no se construyen aquí
    # para ahorrar memoria. Si son estrictamente necesarias, podrían
    # construirse combinando train y test, o leyendo el archivo de nuevo.
    # Por ahora, devolvemos None para ellas.
    R = None
    mask_R = None
    C = None

    print(f"Matrices dispersas creadas. Densidad de entrenamiento: {train_R.nnz / (num_users * num_items):.4e}")
    if test_R.nnz > 0:
        print(f"Densidad de prueba: {test_R.nnz / (num_users * num_items):.4e}")

    # Devolver las matrices dispersas y los tamaños
    # Asegúrate que el orden coincide con lo que espera main.py
    return (R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,
            num_train_ratings, num_test_ratings,
            user_train_set, item_train_set, user_test_set, item_test_set, # Devolver item_test_set aunque esté vacío
            num_users, num_items, num_total_ratings_processed) # Devolver el total procesado