import numpy as np
from user_history_recommender import UserHistoryRecommender
import tensorflow as tf

def recommend_for_user(model_path, data, user_id, top_k=10, use_history=True, history_weight=0.3):
    """
    Genera recomendaciones para un usuario específico utilizando KGNN-LS y su historial
    
    Args:
        model_path: Ruta al modelo KGNN-LS guardado
        data: Datos del grafo de conocimiento y usuarios
        user_id: ID del usuario para recomendar
        top_k: Número máximo de recomendaciones
        use_history: Si se debe usar recomendación basada en historial
        history_weight: Peso para la componente de historial
    
    Returns:
        list: Lista de tuplas (item_id, score) ordenadas por puntuación
    """
    from train_Inf import inference
    
    # Extraer información necesaria del data
    n_user, n_item, n_entity = data[0], data[1], data[2]
    
    # Verificar si el usuario existe
    train_data = data[4]
    user_in_train = user_id in train_data[:, 0]
    
    if not user_in_train:
        print(f"¡Advertencia! Usuario {user_id} no encontrado en datos de entrenamiento.")
    
    # Reiniciar grafo de TensorFlow
    tf.compat.v1.reset_default_graph()
    
    # Crear lista de candidatos (todos los items posibles excepto los ya valorados)
    user_rated_items = set()
    if user_in_train:
        user_data = train_data[train_data[:, 0] == user_id]
        user_rated_items = set(user_data[:, 1])
    
    candidate_items = [item for item in range(n_item) if item not in user_rated_items]
    
    # Generar pares (usuario, item) para todos los candidatos
    user_item_pairs = [(user_id, item_id) for item_id in candidate_items]
    
    # Ejecutar inferencia
    with tf.compat.v1.variable_scope(f"inference_{user_id}"):
        item_scores = inference(model_path, data, user_item_pairs)
    
    # Si se solicita usar historial
    if use_history:
        # Inicializar recomendador basado en historial
        history_recommender = UserHistoryRecommender(
            train_data=data[4],  # Datos de entrenamiento
            kg_data=data,        # Datos del grafo de conocimiento
            history_weight=history_weight
        )
        
        # Aplicar la combinación de predicciones
        enhanced_scores = []
        for i, (user, item) in enumerate(user_item_pairs):
            model_score = float(item_scores[i])
            
            # Combinar predicción del modelo con historial
            combined_score = history_recommender.predict_rating_with_history(
                model_score, user, item
            )
            enhanced_scores.append(combined_score)
        
        # Usar scores mejorados
        item_scores = enhanced_scores
    
    # Crear lista de items con sus puntuaciones
    item_score_list = list(zip(candidate_items, item_scores))
    
    # Ordenar por puntuación y tomar los top_k
    sorted_items = sorted(item_score_list, key=lambda x: x[1], reverse=True)[:top_k]
    
    return sorted_items

def explain_recommendation(data, user_id, item_id, history_weight=0.3):
    """
    Genera una explicación de por qué se recomendó un ítem específico a un usuario
    basado en su historial y el grafo de conocimiento
    
    Args:
        data: Datos del grafo de conocimiento y usuarios
        user_id: ID del usuario
        item_id: ID del ítem recomendado
        history_weight: Peso para la componente de historial
    
    Returns:
        dict: Información explicativa sobre la recomendación
    """
    # Inicializar recomendador basado en historial
    history_recommender = UserHistoryRecommender(
        train_data=data[4],  # Datos de entrenamiento
        kg_data=data,        # Datos del grafo de conocimiento
        history_weight=history_weight
    )
    
    # Verificar si el usuario tiene historial
    if user_id not in history_recommender.user_histories:
        return {
            "success": False,
            "message": f"El usuario {user_id} no tiene historial de calificaciones."
        }
    
    # Obtener historial del usuario
    user_history = history_recommender.user_histories[user_id]
    
    # Obtener las similitudes entre el ítem recomendado y los ítems del historial
    item_similarities = []
    for hist_item in user_history:
        hist_item_id = hist_item[1]
        hist_rating = hist_item[2]
        similarity = history_recommender.get_item_similarity(item_id, hist_item_id)
        
        if similarity > 0.1:
            item_similarities.append((int(hist_item_id), float(hist_rating), float(similarity)))
    
    # Ordenar por similitud
    item_similarities.sort(key=lambda x: x[2], reverse=True)
    
    # Obtener la predicción basada en historial
    history_pred, confidence = history_recommender.predict_rating_from_history(user_id, item_id)
    
    return {
        "success": True,
        "user_id": int(user_id),
        "item_id": int(item_id),
        "history_prediction": float(history_pred) if history_pred is not None else None,
        "confidence": float(confidence),
        "similar_items": item_similarities[:5]  # Top 5 items similares
    }