import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

class UserHistoryRecommender:
    """
    Clase para recomendación basada en historia del usuario utilizando el grafo de conocimiento
    y el historial de interacciones previas.
    """
    def __init__(self, train_data, kg_data=None, history_weight=0.3):
        """
        Inicializa el recomendador basado en historial de usuario
        
        Args:
            train_data: Array numpy con columnas [user_id, item_id, rating]
            kg_data: Datos del grafo de conocimiento (opcional)
            history_weight: Peso para combinar predicción del modelo y del historial
        """
        self.train_data = train_data
        self.kg_data = kg_data
        self.history_weight = history_weight
        self.user_histories = {}
        self.item_similarities = {}
        
        # Procesar el dataset para crear historiales
        self._build_user_histories()
        
        # Si tenemos datos KG, podemos crear similitudes de items
        if kg_data is not None:
            self._build_item_similarities()
    
    def _build_user_histories(self):
        """Crear historiales de usuarios como diccionarios de {usuario: historial}"""
        # Agrupar por usuario para crear el historial
        for user_id in np.unique(self.train_data[:, 0]):
            # Obtener todos los ratings de este usuario
            user_data = self.train_data[self.train_data[:, 0] == user_id]
            self.user_histories[user_id] = user_data
    
    def _build_item_similarities(self):
        """Construir matriz de similitudes entre items basada en el grafo"""
        if self.kg_data is None:
            return
            
        # Implementación simplificada: utilizamos la conectividad en el grafo
        # para determinar similitudes entre items
        # Para cada relación (h,r,t) donde h y t son items, incrementamos su similitud
        
        # Obtener todos los items únicos
        all_items = np.unique(self.train_data[:, 1])
        
        # Inicializar matriz de similitudes
        for item1 in all_items:
            self.item_similarities[item1] = {}
            for item2 in all_items:
                if item1 != item2:
                    self.item_similarities[item1][item2] = 0.0
                else:
                    self.item_similarities[item1][item2] = 1.0  # Un item es igual a sí mismo
        
        # Si tenemos datos de KG, los procesamos para calcular similitudes
        if isinstance(self.kg_data, list) and len(self.kg_data) >= 3:
            kg_dict = self.kg_data[2]  # Diccionario de relaciones KG
            
            # Recorrer las relaciones para establecer similitudes
            for head_id, relations in kg_dict.items():
                for relation, tail_entities in relations.items():
                    for tail_id in tail_entities:
                        # Si ambas entidades son items, incrementar su similitud
                        if head_id in all_items and tail_id in all_items:
                            # Incrementar similitud basada en conexión directa
                            self.item_similarities.setdefault(head_id, {})
                            self.item_similarities.setdefault(tail_id, {})
                            
                            # Incrementar similitud bidireccional
                            self.item_similarities[head_id][tail_id] = \
                                self.item_similarities[head_id].get(tail_id, 0) + 0.2
                            self.item_similarities[tail_id][head_id] = \
                                self.item_similarities[tail_id].get(head_id, 0) + 0.2
        
        # Normalizar similitudes para que estén entre 0 y 1
        for item1 in self.item_similarities:
            for item2 in self.item_similarities[item1]:
                if item1 != item2:
                    self.item_similarities[item1][item2] = min(1.0, self.item_similarities[item1][item2])
    
    def get_item_similarity(self, item_id1, item_id2):
        """
        Calcula la similitud entre dos items basada en el grafo de conocimiento
        
        Si no hay datos de grafo, devuelve similitud genérica (0.5)
        """
        # Si tenemos similitudes precalculadas, las usamos
        if item_id1 in self.item_similarities and item_id2 in self.item_similarities[item_id1]:
            return self.item_similarities[item_id1][item_id2]
        
        # Si son el mismo item, máxima similitud
        if item_id1 == item_id2:
            return 1.0
        
        # Valor por defecto si no hay otra información
        return 0.5
    
    def predict_rating_from_history(self, user_id, item_id):
        """
        Predecir rating basado en el historial del usuario
        
        Args:
            user_id: ID del usuario
            item_id: ID del item
        
        Returns:
            tuple: (predicción, confianza)
                donde predicción es el rating estimado y 
                confianza es un valor entre 0-1 indicando la confiabilidad
        """
        # Si el usuario no tiene historial, no podemos predecir
        if user_id not in self.user_histories:
            return None, 0.0
        
        user_history = self.user_histories[user_id]
        
        # Si el usuario ya ha valorado este item, devolver ese rating
        exact_match = user_history[(user_history[:, 1] == item_id)]
        if len(exact_match) > 0:
            exact_rating = exact_match[0, 2]
            return exact_rating, 1.0
            
        # Si no hay valoraciones en el historial, no podemos predecir
        if len(user_history) == 0:
            return None, 0.0
            
        # Calcular similitudes con otros items que el usuario ha valorado
        similarities = []
        ratings = []
        
        # Recorrer el historial del usuario
        for hist_item in user_history:
            hist_item_id = hist_item[1]
            hist_rating = hist_item[2]
            
            # Calcular similitud entre este item y el item objetivo
            similarity = self.get_item_similarity(hist_item_id, item_id)
            
            if similarity > 0.1:  # Solo considerar items con cierta similitud
                similarities.append(similarity)
                ratings.append(hist_rating)
        
        # Si no encontramos items similares, no podemos predecir
        if not similarities:
            return None, 0.0
            
        # Calcular predicción ponderada por similitud
        weighted_sum = sum(r * s for r, s in zip(ratings, similarities))
        sum_similarities = sum(similarities)
        
        if sum_similarities > 0:
            prediction = weighted_sum / sum_similarities
            
            # La confianza depende de cuántos items similares encontramos y su similitud
            confidence = min(1.0, sum_similarities / 5.0)  # Máxima confianza con 5+ items similares
            
            return prediction, confidence
        else:
            return None, 0.0
    
    def predict_rating_with_history(self, model_prediction, user_id, item_id):
        """
        Combinar la predicción del modelo con la predicción basada en historial
        
        Args:
            model_prediction: Predicción del modelo KGNN-LS
            user_id: ID de usuario
            item_id: ID de item
        
        Returns:
            float: Rating combinado del modelo y el historial
        """
        history_pred, confidence = self.predict_rating_from_history(user_id, item_id)
        
        # Si no podemos hacer predicción por historial, usar solo el modelo
        if history_pred is None:
            return model_prediction
            
        # Ajustar el peso del historial por la confianza
        effective_history_weight = self.history_weight * confidence
        
        # Combinar predicciones
        combined_prediction = (
            (1 - effective_history_weight) * model_prediction + 
            effective_history_weight * history_pred
        )
        
        return combined_prediction