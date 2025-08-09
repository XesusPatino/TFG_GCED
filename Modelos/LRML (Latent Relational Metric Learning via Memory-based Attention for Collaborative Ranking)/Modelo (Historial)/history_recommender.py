import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
import time

class UserHistoryRecommender:
    def __init__(self, ratings_df, movies_df=None, history_weight=0.3):
        """
        Inicializa el recomendador basado en historial de usuario
        
        Args:
            ratings_df: DataFrame con columnas [userId, movieId, rating, timestamp]
            movies_df: DataFrame de películas (opcional)
            history_weight: Peso para combinar predicción del modelo y del historial
        """
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.history_weight = history_weight
        self.user_histories = {}
        self.item_features = {}
        self.genres_by_movie = {}
        
        # Procesar el dataset para crear historiales
        self._build_user_histories()
        
        # Si tenemos datos de películas, podemos crear características de items
        if movies_df is not None:
            self._build_item_features()
            
        print(f"Recomendador por historial inicializado para {len(self.user_histories)} usuarios")
    
    def _build_user_histories(self):
        """Crear historiales de usuarios como diccionarios de {usuario: historial}"""
        # Agrupar por usuario para crear el historial
        for user_id, group in self.ratings_df.groupby('userId'):
            self.user_histories[user_id] = group[['movieId', 'rating', 'timestamp']]
    
    def _build_item_features(self):
        """Construir representaciones de películas si están disponibles"""
        if self.movies_df is not None:
            # Extraer géneros para cada película
            for _, row in self.movies_df.iterrows():
                movie_id = row['movieId']
                # Verificar si hay una columna de géneros
                if 'genres' in self.movies_df.columns:
                    genres = row['genres'].split('|') if isinstance(row['genres'], str) else []
                    self.genres_by_movie[movie_id] = set(genres)
    
    def get_item_similarity(self, item_id1, item_id2):
        """
        Calcula la similitud entre dos items basada en sus géneros
        """
        # Si no tenemos información de géneros, usar similitud genérica
        if not self.genres_by_movie:
            return 0.5
        
        if item_id1 in self.genres_by_movie and item_id2 in self.genres_by_movie:
            # Calcular similitud Jaccard (intersección sobre unión)
            genres1 = self.genres_by_movie[item_id1]
            genres2 = self.genres_by_movie[item_id2]
            
            if not genres1 or not genres2:
                return 0.5
            
            intersection = len(genres1.intersection(genres2))
            union = len(genres1.union(genres2))
            
            if union > 0:
                return intersection / union
        
        return 0.5  # Valor por defecto
    
    def predict_rating_from_history(self, user_id, item_id):
        """
        Predecir rating basado en el historial del usuario
        
        Args:
            user_id: ID del usuario
            item_id: ID del item (película)
        
        Returns:
            tuple: (predicción, confianza)
                donde predicción es el rating estimado y 
                confianza es un valor entre 0-1 indicando la confiabilidad
        """
        # Si el usuario no tiene historial, no podemos predecir
        if user_id not in self.user_histories:
            return None, 0.0
        
        user_history = self.user_histories[user_id]
        
        # Si el usuario ya ha visto esta película, devolver ese rating
        if item_id in user_history['movieId'].values:
            exact_rating = user_history.loc[user_history['movieId'] == item_id, 'rating'].iloc[0]
            return exact_rating, 1.0
            
        # Si no hay valoraciones en el historial, no podemos predecir
        if len(user_history) == 0:
            return None, 0.0
            
        # Calcular similitudes con otras películas que el usuario ha valorado
        similarities = []
        ratings = []
        
        # Recorrer el historial del usuario
        for _, row in user_history.iterrows():
            hist_item_id = row['movieId']
            hist_rating = row['rating']
            
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
    
    def predict_rating_with_model(self, model_prediction, user_id, item_id):
        """
        Combinar la predicción del modelo con la predicción basada en historial
        
        Args:
            model_prediction: Predicción del modelo neural
            user_id: ID de usuario
            item_id: ID de item (película)
        
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