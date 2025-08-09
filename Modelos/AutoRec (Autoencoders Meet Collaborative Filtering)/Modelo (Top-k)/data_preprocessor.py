import numpy as np
import os

def read_rating(path, num_users, num_items, num_total_ratings, a, b, train_ratio):
    user_train_set, user_test_set = set(), set()
    item_train_set, item_test_set = set(), set()

    R = np.zeros((num_users, num_items))
    mask_R = np.zeros((num_users, num_items))
    C = np.ones((num_users, num_items)) * b

    train_R = np.zeros_like(R)
    test_R = np.zeros_like(R)
    train_mask_R = np.zeros_like(mask_R)
    test_mask_R = np.zeros_like(mask_R)

    # Cargar datos
    ratings_file = os.path.join(path, "ratings.dat")
    print(f"Cargando datos desde: {ratings_file}")
    
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(f"Archivo de ratings no encontrado: {ratings_file}")
    
    # Leer todos los ratings primero
    all_ratings = []
    with open(ratings_file, "r") as fp:
        for line in fp:
            user, item, rating, _ = line.strip().split("::")
            user_idx, item_idx = int(user) - 1, int(item) - 1
            all_ratings.append((user_idx, item_idx, float(rating)))
    
    # Verificar que los datos son correctos
    print(f"Total de ratings cargados: {len(all_ratings)}")
    if len(all_ratings) != num_total_ratings:
        print(f"Advertencia: El número esperado de ratings ({num_total_ratings}) no coincide con los cargados ({len(all_ratings)})")
    
    # Generar índices aleatorios para división train/test
    np.random.seed(42)  # Para reproducibilidad
    random_perm_idx = np.random.permutation(len(all_ratings))
    train_idx = random_perm_idx[:int(len(all_ratings) * train_ratio)]
    test_idx = random_perm_idx[int(len(all_ratings) * train_ratio):]
    
    # Procesar las valoraciones de entrenamiento
    for idx in train_idx:
        user_idx, item_idx, rating = all_ratings[idx]
        R[user_idx, item_idx] = rating
        mask_R[user_idx, item_idx] = 1
        train_R[user_idx, item_idx] = rating
        train_mask_R[user_idx, item_idx] = 1
        C[user_idx, item_idx] = a
        
        user_train_set.add(user_idx)
        item_train_set.add(item_idx)
    
    # Procesar las valoraciones de prueba
    for idx in test_idx:
        user_idx, item_idx, rating = all_ratings[idx]
        R[user_idx, item_idx] = rating
        mask_R[user_idx, item_idx] = 1
        test_R[user_idx, item_idx] = rating
        test_mask_R[user_idx, item_idx] = 1
        
        user_test_set.add(user_idx)
        item_test_set.add(item_idx)
    
    num_train_ratings = len(train_idx)
    num_test_ratings = len(test_idx)
    
    print(f"Ratings de entrenamiento: {num_train_ratings}")
    print(f"Ratings de prueba: {num_test_ratings}")
    print(f"Usuarios en entrenamiento: {len(user_train_set)}")
    print(f"Ítems en entrenamiento: {len(item_train_set)}")
    
    # Verificar que los datos son coherentes
    if np.sum(train_mask_R) == 0:
        print("¡ADVERTENCIA! No hay valoraciones en el conjunto de entrenamiento.")
    if np.sum(test_mask_R) == 0:
        print("¡ADVERTENCIA! No hay valoraciones en el conjunto de prueba.")

    return R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R, num_train_ratings, num_test_ratings, user_train_set, item_train_set, user_test_set, item_test_set