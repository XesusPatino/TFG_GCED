# Agregar estas importaciones adicionales
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, pairwise_distances
from codecarbon import EmissionsTracker
import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt

# Importaciones alternativas
Model = tf.keras.models.Model
Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Concatenate = tf.keras.layers.Concatenate

from sklearn.preprocessing import StandardScaler


class EmissionsPerEpochTracker:
    def __init__(self, result_path, model_name="GHRS_History"):
        self.result_path = result_path
        self.model_name = model_name
        self.epoch_emissions = []
        self.cumulative_emissions = []
        self.epoch_rmse = []
        self.epoch_loss = []
        self.epoch_val_loss = []
        self.total_emissions = 0.0
        self.trackers = {}
        
        # Crear directorio para emisiones
        os.makedirs(f"{result_path}/emissions_reports", exist_ok=True)
        os.makedirs(f"{result_path}/emissions_plots", exist_ok=True)
    
    def start_epoch(self, epoch):
        # Crear un tracker con un nombre único basado en timestamp
        timestamp = int(time.time())
        tracker_name = f"{self.model_name}_epoch{epoch}_{timestamp}"
        
        self.trackers[epoch] = EmissionsTracker(
            project_name=tracker_name,
            output_dir=f"{self.result_path}/emissions_reports",
            save_to_file=True,
            log_level="error",
            save_to_api=False
        )
        
        try:
            self.trackers[epoch].start()
        except Exception as e:
            print(f"Advertencia: No se pudo iniciar el tracker para la época {epoch}: {e}")
            self.trackers[epoch] = None
    
    def end_epoch(self, epoch, loss, val_loss=None, val_rmse=None):
        try:
            epoch_co2 = 0.0
            if epoch in self.trackers and self.trackers[epoch]:
                try:
                    epoch_co2 = self.trackers[epoch].stop() or 0.0
                except Exception as e:
                    print(f"Advertencia: Error al detener el tracker para la época {epoch}: {e}")
                    epoch_co2 = 0.0
            
            # Acumular emisiones totales
            self.total_emissions += epoch_co2
            
            # Guardar datos de esta época
            self.epoch_emissions.append(epoch_co2)
            self.cumulative_emissions.append(self.total_emissions)
            self.epoch_loss.append(loss)
            
            if val_loss is not None:
                self.epoch_val_loss.append(val_loss)
                
            if val_rmse is not None:
                self.epoch_rmse.append(val_rmse)
            
            print(f"Epoch {epoch+1} - Emisiones: {epoch_co2:.8f} kg, Acumulado: {self.total_emissions:.8f} kg")
            print(f"Loss: {loss:.4f}" + (f", Val Loss: {val_loss:.4f}" if val_loss is not None else "") + 
                  (f", Val RMSE: {val_rmse:.4f}" if val_rmse is not None else ""))
            
        except Exception as e:
            print(f"Error al medir emisiones en época {epoch}: {e}")
    
    def end_training(self, final_rmse):
        try:
            # Asegurarse de que todos los trackers estén detenidos
            for epoch, tracker in self.trackers.items():
                if tracker is not None:
                    try:
                        tracker.stop()
                    except:
                        pass
            
            # Crear dataframe con todos los datos
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            epochs_range = range(1, len(self.epoch_emissions) + 1)
            
            df = pd.DataFrame({
                'epoch': epochs_range,
                'epoch_emissions_kg': self.epoch_emissions,
                'cumulative_emissions_kg': self.cumulative_emissions,
                'loss': self.epoch_loss,
                'val_loss': self.epoch_val_loss if self.epoch_val_loss else [None] * len(self.epoch_emissions),
                'rmse': self.epoch_rmse if self.epoch_rmse else [None] * len(self.epoch_emissions)
            })
            
            emissions_file = f'{self.result_path}/emissions_reports/emissions_metrics_{self.model_name}_{timestamp}.csv'
            df.to_csv(emissions_file, index=False)
            print(f"Métricas de emisiones guardadas en: {emissions_file}")
            
            # Graficar las relaciones
            self.plot_emissions_vs_metrics(epochs_range, timestamp, final_rmse)
            
        except Exception as e:
            print(f"Error al generar gráficos de emisiones: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_emissions_vs_metrics(self, epochs_range, timestamp, final_rmse=None):
        """Genera gráficos para emisiones vs métricas"""
        
        try:
            # 1. Gráfico combinado: Emisiones por época y acumulativas
            plt.figure(figsize=(12, 10))
            
            plt.subplot(2, 2, 1)
            plt.plot(epochs_range, self.epoch_emissions, 'r-', marker='x')
            plt.title('Emisiones por Época')
            plt.xlabel('Época')
            plt.ylabel('CO₂ Emissions (kg)')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.plot(epochs_range, self.cumulative_emissions, 'r-', marker='o')
            plt.title('Emisiones Acumuladas por Época')
            plt.xlabel('Época')
            plt.ylabel('CO₂ Emissions (kg)')
            plt.grid(True, alpha=0.3)
            
            if self.epoch_loss:
                plt.subplot(2, 2, 3)
                plt.plot(epochs_range, self.epoch_loss, 'g-', marker='o', label='Train Loss')
                if self.epoch_val_loss:
                    plt.plot(epochs_range, self.epoch_val_loss, 'b-', marker='x', label='Val Loss')
                plt.title('Loss por Época')
                plt.xlabel('Época')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            if self.epoch_rmse:
                plt.subplot(2, 2, 4)
                plt.plot(epochs_range, self.epoch_rmse, 'b-', marker='o')
                plt.title('RMSE por Época')
                plt.xlabel('Época')
                plt.ylabel('RMSE')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            file_path = f'{self.result_path}/emissions_plots/metrics_by_epoch_{self.model_name}_{timestamp}.png'
            plt.savefig(file_path)
            plt.close()
            print(f"Gráfico guardado en: {file_path}")
            
            if self.epoch_rmse:
                # 2. RMSE vs Emisiones acumuladas
                plt.figure(figsize=(10, 6))
                plt.plot(self.cumulative_emissions, self.epoch_rmse, 'b-', marker='o')
                
                # Añadir etiquetas con el número de época
                for i, (emissions, rmse) in enumerate(zip(self.cumulative_emissions, self.epoch_rmse)):
                    plt.annotate(f"{i+1}", (emissions, rmse), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9)
                    
                plt.xlabel('Emisiones de CO₂ acumuladas (kg)')
                plt.ylabel('RMSE')
                plt.title('Relación entre Emisiones Acumuladas y RMSE')
                plt.grid(True, alpha=0.3)
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_vs_rmse_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path)
                plt.close()
                print(f"Gráfico guardado en: {file_path}")
                
                # 3. Scatter plot de rendimiento frente a emisiones acumulativas
                plt.figure(figsize=(10, 6))
                
                # Ajustar tamaño de los puntos según la época
                sizes = [(i+1)*20 for i in range(len(self.cumulative_emissions))]
                
                scatter = plt.scatter(self.epoch_rmse, self.cumulative_emissions, 
                            color='blue', marker='o', s=sizes, alpha=0.7)
                
                # Añadir etiquetas de época
                for i, (rmse, em) in enumerate(zip(self.epoch_rmse, self.cumulative_emissions)):
                    plt.annotate(f"{i+1}", (rmse, em), textcoords="offset points", 
                                xytext=(0,5), ha='center', fontsize=9)
                
                plt.ylabel('Emisiones de CO₂ acumuladas (kg)')
                plt.xlabel('RMSE')
                plt.title('Relación entre RMSE y Emisiones Acumuladas')
                plt.grid(True, alpha=0.3)
                
                file_path = f'{self.result_path}/emissions_plots/cumulative_emissions_performance_scatter_{self.model_name}_{timestamp}.png'
                plt.savefig(file_path)
                plt.close()
                print(f"Gráfico guardado en: {file_path}")
                
                # 4. Gráfico adicional: Pareto frontier
                if len(self.epoch_rmse) > 2:
                    plt.figure(figsize=(10, 6))
                    
                    # Identificar puntos del frente de Pareto
                    pareto_points = self._find_pareto_frontier()
                    pareto_emissions = [self.cumulative_emissions[i] for i in pareto_points]
                    pareto_rmse = [self.epoch_rmse[i] for i in pareto_points]
                    
                    # Scatter plot de todos los puntos
                    plt.scatter(self.cumulative_emissions, self.epoch_rmse, color='blue', marker='o', label='Epoch Results')
                    
                    # Destacar frente de Pareto
                    plt.scatter(pareto_emissions, pareto_rmse, color='red', marker='*', 
                              s=100, label='Pareto Frontier')
                    
                    # Conectar puntos del frente
                    if len(pareto_emissions) > 1:
                        # Ordenar por emisiones
                        idx_sorted = sorted(range(len(pareto_emissions)), key=lambda i: pareto_emissions[i])
                        sorted_emissions = [pareto_emissions[i] for i in idx_sorted]
                        sorted_rmse = [pareto_rmse[i] for i in idx_sorted]
                        plt.plot(sorted_emissions, sorted_rmse, 'r--', alpha=0.7)
                    
                    # Añadir etiquetas a los puntos
                    for i, (em, rmse) in enumerate(zip(self.cumulative_emissions, self.epoch_rmse)):
                        plt.annotate(f"{i+1}", (em, rmse), textcoords="offset points", 
                                  xytext=(0,5), ha='center', fontsize=8)
                    
                    plt.xlabel('Emisiones de CO₂ acumuladas (kg)')
                    plt.ylabel('RMSE')
                    plt.title('Frontera de Pareto: Emisiones vs Rendimiento')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    file_path = f'{self.result_path}/emissions_plots/pareto_frontier_{self.model_name}_{timestamp}.png'
                    plt.savefig(file_path)
                    plt.close()
                    print(f"Frontera de Pareto guardada en: {file_path}")
                    
        except Exception as e:
            print(f"Error al generar los gráficos: {e}")
            import traceback
            traceback.print_exc()
    
    def _find_pareto_frontier(self):
        """Identifica los puntos que forman el frente de Pareto"""
        # Queremos minimizar tanto RMSE como emisiones
        pareto_points = []
        
        for i in range(len(self.epoch_rmse)):
            is_pareto = True
            
            for j in range(len(self.epoch_rmse)):
                if i != j:
                    # Si j domina a i (mejor en ambas métricas), entonces i no está en el frente
                    if (self.epoch_rmse[j] <= self.epoch_rmse[i] and 
                        self.cumulative_emissions[j] <= self.cumulative_emissions[i]):
                        # Si j es estrictamente mejor en al menos una métrica
                        if (self.epoch_rmse[j] < self.epoch_rmse[i] or 
                            self.cumulative_emissions[j] < self.cumulative_emissions[i]):
                            is_pareto = False
                            break
            
            if is_pareto:
                pareto_points.append(i)
        
        return pareto_points
    


# Añadir después de la clase EmissionsPerEpochTracker
class TrackingCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, emissions_tracker):
        super().__init__()
        self.val_data = val_data
        self.emissions_tracker = emissions_tracker
        self.val_X, self.val_y = val_data
        
    def on_epoch_begin(self, epoch, logs=None):
        self.emissions_tracker.start_epoch(epoch)
        
    def on_epoch_end(self, epoch, logs=None):
        # Calcular RMSE en el conjunto de validación
        y_pred = self.model.predict(self.val_X, verbose=0)
        val_rmse = mean_squared_error(self.val_y, y_pred.flatten(), squared=False)
        
        # Registrar métricas y emisiones
        self.emissions_tracker.end_epoch(
            epoch, 
            logs.get('loss'), 
            logs.get('val_loss'),
            val_rmse
        )



# Clase para recomendación basada en historia del usuario
class UserHistoryRecommender:
    def __init__(self, ratings_df, movies_df=None, history_weight=0.3):
        """
        Inicializa el recomendador basado en historial de usuario
        
        Args:
            ratings_df: DataFrame con columnas [UID, MID, rate, time]
            movies_df: DataFrame de películas (opcional)
            history_weight: Peso para combinar predicción del modelo y del historial
        """
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.history_weight = history_weight
        self.user_histories = {}
        self.item_features = {}
        
        # Procesar el dataset para crear historiales
        self._build_user_histories()
        
        # Si tenemos datos de películas, podemos crear características de items
        if movies_df is not None:
            self._build_item_features()
    
    def _build_user_histories(self):
        """Crear historiales de usuarios como diccionarios de {usuario: historial}"""
        # Agrupar por usuario para crear el historial
        for user_id, group in self.ratings_df.groupby('UID'):
            self.user_histories[user_id] = group[['MID', 'rate', 'time']]
    
    def _build_item_features(self):
        """Construir representaciones de películas si están disponibles"""
        if self.movies_df is not None:
            # Filtrar solo columnas numéricas/categóricas que representen características
            feature_cols = [col for col in self.movies_df.columns 
                           if col not in ['MID', 'title', 'genres']]
            
            if feature_cols:
                for _, row in self.movies_df.iterrows():
                    item_id = row['MID']
                    features = row[feature_cols].values
                    self.item_features[item_id] = features
    
    def get_item_similarity(self, item_id1, item_id2):
        """
        Calcula la similitud entre dos items basada en sus características
        
        Si no hay características de items, devuelve similitud genérica (0.5)
        """
        if not self.item_features:
            return 0.5  # Similitud genérica si no tenemos características
        
        if item_id1 in self.item_features and item_id2 in self.item_features:
            # Usando distancia coseno (1 - distancia para obtener similitud)
            features1 = self.item_features[item_id1].reshape(1, -1)
            features2 = self.item_features[item_id2].reshape(1, -1)
            
            distance = pairwise_distances(features1, features2, metric='cosine')[0][0]
            return 1.0 - min(distance, 1.0)  # Asegurar que está entre 0 y 1
        else:
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
        if item_id in user_history['MID'].values:
            exact_rating = user_history.loc[user_history['MID'] == item_id, 'rate'].iloc[0]
            return exact_rating, 1.0
            
        # Si no hay valoraciones en el historial, no podemos predecir
        if len(user_history) == 0:
            return None, 0.0
            
        # Calcular similitudes con otras películas que el usuario ha valorado
        similarities = []
        ratings = []
        
        # Recorrer el historial del usuario
        for _, row in user_history.iterrows():
            hist_item_id = row['MID']
            hist_rating = row['rate']
            
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
            model_prediction: Predicción del modelo neuronal
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


# Modificar el código principal para incorporar la recomendación basada en historial
def main():
    # Configuración de rutas y directorios
    result_path = "results"
    os.makedirs(result_path, exist_ok=True)

    # 1. Cargar los datos procesados 
    X_users = pd.read_pickle("data1M/x_train_alpha(0.045).pkl")

    # 2. Cargar los ratings
    ratings = pd.read_csv('C:/Users/xpati/Documents/TFG/ml-1m/ratings.dat', sep='::', engine='python', 
                        names=['UID', 'MID', 'rate', 'time'], encoding='latin-1')

    # 3. Cargar características de películas 
    movies = pd.read_csv('C:/Users/xpati/Documents/TFG/ml-1m/movies.dat', sep='::', engine='python', 
                       names=['MID', 'title', 'genres'], encoding='latin-1')

    # Procesar características de películas (one-hot encoding para géneros)
    movies['genres'] = movies['genres'].str.split('|')
    genres_encoded = movies['genres'].explode().str.get_dummies().groupby(level=0).sum()
    movies_features = pd.concat([movies, genres_encoded], axis=1)
    
    # 4. Combinar características de usuarios y películas
    data = pd.merge(ratings, movies_features, on='MID')
    
    if 'UID' in X_users.columns and 'UID' in data.columns:
        data = pd.merge(data, X_users, on='UID')
    
    # 5. Dividir los datos en entrenamiento y prueba
    X = data.drop('rate', axis=1)  # Todas las columnas excepto 'rate'
    y = data['rate']               # Solo la columna 'rate'
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Guardar los IDs para usarlos después
    user_ids_train = X_train['UID'].values
    if 'MID' in X_train.columns:
        movie_ids_train = X_train['MID'].values
    else:
        print("Advertencia: No se encontró la columna 'MID' en X_train")
        movie_ids_train = np.zeros_like(user_ids_train)
        
    user_ids_test = X_test['UID'].values
    if 'MID' in X_test.columns:
        movie_ids_test = X_test['MID'].values
    else:
        print("Advertencia: No se encontró la columna 'MID' en X_test")
        movie_ids_test = np.zeros_like(user_ids_test)
    
    # 6. Identificar columnas no numéricas y eliminarlas para la normalización
    non_numeric_cols = []
    
    for col in X_train.columns:
        # Verificar si es objeto o si es una columna ID que queremos preservar pero no escalar
        if X_train[col].dtype == 'object' or col in ['UID', 'MID', 'title', 'genres']:
            non_numeric_cols.append(col)
    
    print(f"Excluyendo columnas no numéricas para normalización: {non_numeric_cols}")
    
    # Eliminar columnas no numéricas y conservar solo las numéricas para el escalado
    X_train_numeric = X_train.drop(columns=non_numeric_cols)
    X_test_numeric = X_test.drop(columns=non_numeric_cols)
    
    # 7. Normalizar características numéricas
    scaler = StandardScaler()
    X_train_scaled_numeric = scaler.fit_transform(X_train_numeric)
    X_test_scaled_numeric = scaler.transform(X_test_numeric)
    
    # Convertir a DataFrame para facilitar el manejo
    X_train_scaled = pd.DataFrame(X_train_scaled_numeric, columns=X_train_numeric.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled_numeric, columns=X_test_numeric.columns)
    
    # Añadir nuevamente las columnas de IDs que necesitamos para referencia
    X_train_scaled['UID'] = user_ids_train
    if 'MID' in X_train.columns:
        X_train_scaled['MID'] = movie_ids_train
    
    X_test_scaled['UID'] = user_ids_test
    if 'MID' in X_test.columns:
        X_test_scaled['MID'] = movie_ids_test
    
    # 8. Construir modelo híbrido con TensorFlow/Keras
    input_layer = Input(shape=(X_train_numeric.shape[1],))

    # Ramas específicas para tipos de características 
    graph_features = Dense(64, activation='relu')(input_layer)
    graph_features = Dropout(0.2)(graph_features)

    user_features = Dense(32, activation='relu')(input_layer)
    user_features = Dropout(0.2)(user_features)

    movie_features = Dense(32, activation='relu')(input_layer)
    movie_features = Dropout(0.2)(movie_features)

    # Combinar las ramas
    combined = Concatenate()([graph_features, user_features, movie_features])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    combined = Dense(32, activation='relu')(combined)

    # Capa de salida
    output = Dense(1)(combined)

    # Crear y compilar el modelo
    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # Inicializar tracker de emisiones
    emissions_tracker = EmissionsPerEpochTracker(result_path)
    
    # Dividir los datos de entrenamiento para validación
    X_train_scaled_subset, X_val, y_train_subset, y_val = train_test_split(
        X_train_scaled_numeric, y_train, test_size=0.1, random_state=42
    )
    
    # Crear callback de tracking
    tracking_callback = TrackingCallback((X_val, y_val), emissions_tracker)
    
    # 9. Entrenar el modelo con tracking de emisiones
    print("Iniciando entrenamiento del modelo con tracking de emisiones...")
    history = model.fit(
        X_train_scaled_subset,
        y_train_subset,
        epochs=50,
        batch_size=1024,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[tracking_callback]
    )

    # 10. Evaluar el modelo
    print("Realizando predicciones finales...")
    y_pred = model.predict(X_test_scaled_numeric)
    
    # Calcular RMSE final
    final_rmse = mean_squared_error(y_test, y_pred.flatten(), squared=False)
    print(f"RMSE final del modelo neural: {final_rmse:.4f}")
    
    # Finalizar tracking de emisiones y generar gráficos
    emissions_tracker.end_training(final_rmse)
    
    # 11. Inicializar recomendador basado en historial
    history_recommender = UserHistoryRecommender(
        ratings_df=ratings,
        movies_df=movies_features,
        history_weight=0.3
    )
    
    # 12. Crear un DataFrame para los resultados de prueba
    test_df = pd.DataFrame({
        'UID': user_ids_test,
        'MID': movie_ids_test if 'MID' in X_test.columns else np.zeros_like(user_ids_test),
        'real_rating': y_test.values,
        'model_pred': y_pred.flatten()
    })
    
    # 13. Predecir con recomendación basada en historial
    print("Realizando predicciones con recomendación basada en historial...")
    test_df['combined_pred'] = test_df.apply(
        lambda row: history_recommender.predict_rating_with_model(
            row['model_pred'], row['UID'], row['MID']
        ), 
        axis=1
    )
    
    # 14. Calcular métricas para ambas predicciones
    model_rmse = mean_squared_error(test_df['real_rating'], test_df['model_pred'], squared=False)
    model_mae = mean_absolute_error(test_df['real_rating'], test_df['model_pred'])
    
    combined_rmse = mean_squared_error(test_df['real_rating'], test_df['combined_pred'], squared=False)
    combined_mae = mean_absolute_error(test_df['real_rating'], test_df['combined_pred'])
    

    '''
    print("\n=== Resultados comparativos ===")
    print(f"Modelo base - RMSE: {model_rmse:.4f}, MAE: {model_mae:.4f}")
    print(f"Modelo con historial - RMSE: {combined_rmse:.4f}, MAE: {combined_mae:.4f}")
    
    # Mostrar algunos ejemplos de predicciones
    print("\n=== Ejemplos de predicciones ===")
    sample_df = test_df.sample(5)
    for _, row in sample_df.iterrows():
        user_id = row['UID']
        item_id = row['MID']
        
        # Obtener el título de la película
        title = movies[movies['MID'] == item_id]['title'].values[0] if len(movies[movies['MID'] == item_id]) > 0 else "Unknown"
        
        # Obtener la predicción histórica y la confianza
        history_pred, confidence = history_recommender.predict_rating_from_history(user_id, item_id)
        history_info = f"{history_pred:.2f} (confianza: {confidence:.2f})" if history_pred is not None else "No disponible"
        
        print(f"\nUsuario: {user_id}, Película: {title} (ID: {item_id})")
        print(f"  Rating real: {row['real_rating']:.2f}")
        print(f"  Predicción del modelo: {row['model_pred']:.2f}")
        print(f"  Predicción por historial: {history_info}")
        print(f"  Predicción combinada: {row['combined_pred']:.2f}")
        
        # Si tenemos historial para este usuario, mostrar algunas películas similares
        if user_id in history_recommender.user_histories:
            user_history = history_recommender.user_histories[user_id]
            if len(user_history) > 0:
                print("\n  Películas similares en historial del usuario:")
                
                # Calcular similitud con cada película del historial
                similarities = []
                for _, hist_row in user_history.iterrows():
                    hist_item = hist_row['MID']
                    sim = history_recommender.get_item_similarity(item_id, hist_item)
                    hist_title = movies[movies['MID'] == hist_item]['title'].values[0] if len(movies[movies['MID'] == hist_item]) > 0 else "Unknown"
                    similarities.append((hist_item, hist_title, hist_row['rate'], sim))
                
                # Mostrar las 3 películas más similares
                for _, (hist_item, hist_title, hist_rating, sim) in enumerate(
                    sorted(similarities, key=lambda x: x[3], reverse=True)[:3]
                ):
                    print(f"    - {hist_title[:40]} (ID: {hist_item}) - Rating: {hist_rating}, Similitud: {sim:.2f}")
    '''
    
    # Visualización de la comparación entre modelo base y modelo con historial
    # respecto a emisiones y rendimiento
    plt.figure(figsize=(10, 6))
    
    # Punto para modelo neural puro
    plt.scatter(emissions_tracker.total_emissions, model_rmse, 
               color='blue', marker='o', s=100, label='Modelo Neural')
    
    # Punto para modelo con historial (mismo coste de emisiones pero mejor rendimiento)
    plt.scatter(emissions_tracker.total_emissions, combined_rmse, 
               color='green', marker='o', s=100, label='Modelo + Historial')
    
    # Conectar los puntos
    plt.plot([emissions_tracker.total_emissions, emissions_tracker.total_emissions], 
             [model_rmse, combined_rmse], 'k--', alpha=0.7)
    
    # Anotar la mejora de rendimiento
    improvement = ((model_rmse - combined_rmse) / model_rmse) * 100
    plt.annotate(f"Mejora: {improvement:.2f}%", 
                xy=(emissions_tracker.total_emissions, (model_rmse + combined_rmse)/2),
                xytext=(emissions_tracker.total_emissions * 1.1, (model_rmse + combined_rmse)/2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    
    plt.xlabel('Emisiones de CO₂ totales (kg)')
    plt.ylabel('RMSE')
    plt.title('Comparativa: Rendimiento vs Emisiones')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    comparison_file = f'{result_path}/emissions_plots/neural_vs_hybrid_comparison.png'
    plt.savefig(comparison_file)
    plt.close()
    print(f"Comparativa guardada en: {comparison_file}")
    
    print("\n=== Análisis de Eficiencia Ambiental ===")
    print(f"Emisiones totales: {emissions_tracker.total_emissions:.6f} kg CO₂")
    print(f"Modelo neural - RMSE: {model_rmse:.4f}")
    print(f"Modelo con historial - RMSE: {combined_rmse:.4f}")
    print(f"Mejora de rendimiento: {improvement:.2f}% sin emisiones adicionales")

if __name__ == "__main__":
    main()