import tensorflow as tf
import numpy as np
import pandas as pd
from collections import defaultdict

class AutoRecWithHistory(tf.keras.Model):
    """
    AutoRec model with history-based prediction adjustment
    """
    def __init__(self, num_users, num_items, hidden_neuron, lambda_value, history_weight=0.3):
        super(AutoRecWithHistory, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_neuron = hidden_neuron
        self.lambda_value = lambda_value
        self.history_weight = history_weight
        
        # AutoRec layers
        self.encoder = tf.keras.layers.Dense(
            hidden_neuron, 
            activation='sigmoid',
            kernel_initializer=tf.initializers.GlorotNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(lambda_value)
        )
        
        self.decoder = tf.keras.layers.Dense(
            num_items,
            activation='linear',
            kernel_initializer=tf.initializers.GlorotNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(lambda_value)
        )
        
        # User history data
        self.user_histories = {}
        self.item_similarities = None
    
    def call(self, input_matrix):
        """Forward pass through the model"""
        x = self.encoder(input_matrix)
        x = self.decoder(x)
        return x
    
    def set_user_histories(self, train_R, train_mask_R):
        """Set user histories from training data"""
        print("Building user histories...")
        self.user_histories = {}
        
        for user_id in range(self.num_users):
            # Get user's ratings
            user_ratings = train_R[user_id] * train_mask_R[user_id]
            
            # Store non-zero ratings in a dict format
            if np.sum(train_mask_R[user_id]) > 0:
                rated_items = np.where(train_mask_R[user_id] > 0)[0]
                ratings = user_ratings[rated_items]
                
                # Create user history as a dict of {item_id: rating}
                user_history = {item_id: rating for item_id, rating in zip(rated_items, ratings)}
                self.user_histories[user_id] = user_history
        
        print(f"Built history for {len(self.user_histories)} users")
    
    def compute_item_similarities(self, train_R, train_mask_R):
        """Compute item-item similarity matrix based on user ratings (optimized)"""
        print("Computing item similarities...")
        # Initialize similarity matrix
        self.item_similarities = np.zeros((self.num_items, self.num_items))
        
        # Get item vectors (transpose R matrix to get item profiles)
        item_profiles = (train_R * train_mask_R).T
        
        # Pre-compute norms for all items
        item_norms = np.array([np.linalg.norm(item_profiles[i]) for i in range(self.num_items)])
        
        # Only process items with non-zero norms
        valid_items = np.where(item_norms > 0)[0]
        print(f"Computing similarities for {len(valid_items)} valid items...")
        
        # Process in batches to show progress
        batch_size = 100
        total_batches = (len(valid_items) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(valid_items))
            batch_items = valid_items[batch_start:batch_end]
            
            if batch_idx % 10 == 0:
                print(f"  Processing batch {batch_idx+1}/{total_batches}...")
            
            for i_idx in range(len(batch_items)):
                i = batch_items[i_idx]
                i_profile = item_profiles[i]
                i_norm = item_norms[i]
                
                # Compute similarities with all other valid items with index >= i
                # to avoid duplicate calculations (similarity matrix is symmetric)
                for j_idx in range(i_idx, len(valid_items)):
                    j = valid_items[j_idx]
                    if j < i:
                        continue  # Skip if we've already computed this pair
                    
                    j_profile = item_profiles[j]
                    j_norm = item_norms[j]
                    
                    # Compute cosine similarity
                    sim = np.dot(i_profile, j_profile) / (i_norm * j_norm)
                    
                    # Update similarity matrix (symmetric)
                    self.item_similarities[i, j] = sim
                    if i != j:  # Avoid setting diagonal values twice
                        self.item_similarities[j, i] = sim
        
        print("Item similarities computed")
    
    def predict_with_history(self, user_id, item_id, autorec_prediction):
        """
        Adjust AutoRec prediction using user history
        Returns the weighted average of AutoRec prediction and history-based prediction
        """
        # If no history or similarity data, return AutoRec prediction
        if user_id not in self.user_histories or self.item_similarities is None:
            return autorec_prediction
        
        user_history = self.user_histories[user_id]
        
        # If user has no history, return AutoRec prediction
        if not user_history:
            return autorec_prediction
        
        # Get item similarities between target item and rated items
        similar_items = []
        
        for hist_item_id, hist_rating in user_history.items():
            similarity = self.item_similarities[item_id, hist_item_id]
            
            # Only consider somewhat similar items (threshold is arbitrary and can be tuned)
            if similarity > 0.1:
                similar_items.append((hist_item_id, hist_rating, similarity))
        
        # If no similar items found, return AutoRec prediction
        if not similar_items:
            return autorec_prediction
        
        # Calculate weighted average of similar items' ratings
        total_sim = sum(sim for _, _, sim in similar_items)
        history_prediction = sum(rating * sim for _, rating, sim in similar_items) / total_sim
        
        # Combine predictions
        final_prediction = (1 - self.history_weight) * autorec_prediction + self.history_weight * history_prediction
        
        return final_prediction
    
    def predict_for_user(self, user_id, input_matrix):
        """Predict ratings for a user using both AutoRec and history"""
        # Get AutoRec predictions
        autorec_predictions = self(input_matrix[user_id:user_id+1])[0].numpy()
        
        # Adjust with history
        adjusted_predictions = np.zeros_like(autorec_predictions)
        
        for item_id in range(self.num_items):
            adjusted_predictions[item_id] = self.predict_with_history(
                user_id, item_id, autorec_predictions[item_id]
            )
        
        return adjusted_predictions
    
    def demonstrate_prediction(self, user_id, item_id, input_matrix):
        """Demonstrate prediction process for a single item"""
        # Get AutoRec prediction
        autorec_prediction = self(input_matrix[user_id:user_id+1])[0].numpy()[item_id]
        
        # Get history prediction
        adjusted_prediction = self.predict_with_history(user_id, item_id, autorec_prediction)
        
        # Print information about the prediction
        print(f"\nDemonstrating prediction for User {user_id}, Item {item_id}:")
        print(f"AutoRec prediction: {autorec_prediction:.4f}")
        
        if user_id in self.user_histories:
            history = self.user_histories[user_id]
            print(f"User has rated {len(history)} items")
            
            # Find similar items
            similar_items = []
            for hist_item_id, hist_rating in history.items():
                if self.item_similarities is not None:
                    similarity = self.item_similarities[item_id, hist_item_id]
                    if similarity > 0.1:
                        similar_items.append((hist_item_id, hist_rating, similarity))
            
            if similar_items:
                print("\nSimilar items found in history:")
                print(f"{'Item ID':<10}{'Rating':<10}{'Similarity':<15}")
                
                for hist_item, hist_rating, sim in sorted(similar_items, key=lambda x: x[2], reverse=True)[:5]:
                    print(f"{hist_item:<10}{hist_rating:<10.2f}{sim:<15.4f}")
                
                total_sim = sum(sim for _, _, sim in similar_items)
                history_prediction = sum(rating * sim for _, rating, sim in similar_items) / total_sim
                print(f"\nHistory-based prediction: {history_prediction:.4f}")
                print(f"Final weighted prediction ({self.history_weight} history weight): {adjusted_prediction:.4f}")
            else:
                print("No similar items found in user history")
        else:
            print("User has no rating history")
        
        return adjusted_prediction