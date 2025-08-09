from __future__ import absolute_import
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
import scipy.sparse as sp
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from .models import NNMF, EmbeddingLayer, _NNMFBase


class NNMFWithHistory(NNMF):
    """Neural Network Matrix Factorization model with user history adjustment"""
    
    def __init__(self, num_users, num_items, D=10, Dprime=60, hidden_units_per_layer=50, 
                 history_weight=0.3, num_similar_items=10, similarity_threshold=0.2, 
                 latent_normal_init_params={'mean': 0.0, 'stddev': 0.1}):
        """
        Initialize NNMFWithHistory model.
        
        Args:
            num_users: Number of users in the dataset
            num_items: Number of items in the dataset
            D: Dimension of the latent factors
            Dprime: Dimension of latent representations for items and users
            hidden_units_per_layer: Number of hidden units per hidden layer
            history_weight: Weight for history-based adjustment (between 0 and 1)
            num_similar_items: Number of similar items to consider for history-based predictions
            similarity_threshold: Minimum similarity threshold for items to be considered
            latent_normal_init_params: Parameters for initializing latent factors
        """
        # Initialize the base NNMF model
        super(NNMFWithHistory, self).__init__(num_users, num_items, D, Dprime, 
                                              hidden_units_per_layer, latent_normal_init_params)
        
        # Additional parameters for history-based recommendations
        self.history_weight = history_weight
        self.num_similar_items = num_similar_items
        self.similarity_threshold = similarity_threshold
        
        # Storage for user histories and item similarities
        self.user_histories = {}
        self.item_similarity_matrix = None
        self.rating_means = None
        self.rating_stds = None
    
    def set_user_histories(self, ratings_matrix, mask_matrix):
        """
        Set user histories from the ratings matrix.
        
        Args:
            ratings_matrix: Matrix of user-item ratings [num_users, num_items]
            mask_matrix: Binary matrix indicating which ratings are present
        """
        print("Setting up user histories...")
        self.user_histories = {}
        
        # For each user, store their rated items and ratings
        for u in range(ratings_matrix.shape[0]):
            rated_items = np.where(mask_matrix[u] > 0)[0]
            if len(rated_items) > 0:
                self.user_histories[u] = {
                    'items': rated_items,
                    'ratings': ratings_matrix[u, rated_items]
                }
        
        # Calculate global statistics for ratings
        all_ratings = ratings_matrix[mask_matrix > 0]
        self.rating_means = np.mean(all_ratings)
        self.rating_stds = np.std(all_ratings)
        
        print(f"User histories created for {len(self.user_histories)} users")
        print(f"Global rating stats - Mean: {self.rating_means:.2f}, Std: {self.rating_stds:.2f}")
    
    def compute_item_similarities(self, ratings_matrix, mask_matrix):
        """
        Compute item-item similarities based on user ratings.
        
        Args:
            ratings_matrix: Matrix of user-item ratings [num_users, num_items]
            mask_matrix: Binary matrix indicating which ratings are present
        """
        print("Computing item similarities...")
        
        # Create normalized ratings matrix for similarity calculation
        normalized_ratings = ratings_matrix.copy()
        
        # Normalize ratings by user mean and std to account for different rating scales
        for u in range(ratings_matrix.shape[0]):
            user_ratings = ratings_matrix[u, mask_matrix[u] > 0]
            if len(user_ratings) > 0:
                user_mean = np.mean(user_ratings)
                user_std = np.std(user_ratings)
                if user_std > 0:
                    normalized_ratings[u, mask_matrix[u] > 0] = (user_ratings - user_mean) / user_std
        
        # Create sparse matrix for efficient calculation
        sparse_ratings = sp.csr_matrix(normalized_ratings)
        
        # Compute item similarity matrix
        self.item_similarity_matrix = cosine_similarity(sparse_ratings.T)
        
        # Set self-similarities to 0 to avoid bias
        np.fill_diagonal(self.item_similarity_matrix, 0)
        
        print(f"Item similarity matrix computed with shape: {self.item_similarity_matrix.shape}")
    
    def predict_for_user(self, user_id, ratings_matrix):
        """
        Make predictions for a specific user, incorporating history-based adjustments.
        
        Args:
            user_id: ID of the user
            ratings_matrix: Complete ratings matrix
            
        Returns:
            Array of predictions for all items
        """
        # Get base predictions from NNMF model, but process items in smaller batches
        # to avoid shape mismatches
        base_predictions = np.zeros(self.num_items)
        
        # Process in batches of 100 items
        batch_size = 100
        for i in range(0, self.num_items, batch_size):
            end_idx = min(i + batch_size, self.num_items)
            item_batch = np.arange(i, end_idx)
            
            # Repeat user_id for each item in the batch
            users = np.full_like(item_batch, user_id)
            
            # Get predictions for this batch
            batch_preds = self([
                tf.constant(users, dtype=tf.int32),
                tf.constant(item_batch, dtype=tf.int32)
            ], training=False).numpy().flatten()
            
            # Store predictions
            base_predictions[i:end_idx] = batch_preds
        
        # Apply history-based adjustments if we have user history
        if user_id in self.user_histories and self.item_similarity_matrix is not None:
            # Get user's rated items and ratings
            user_items = self.user_histories[user_id]['items']
            user_ratings = self.user_histories[user_id]['ratings']
            
            # Adjust predictions for all items
            for i in range(self.num_items):
                # Skip items the user has already rated
                if i in user_items:
                    continue
                
                # Find similar items that the user has rated
                similarities = self.item_similarity_matrix[i, user_items]
                sorted_indices = np.argsort(-similarities)  # Descending order
                
                # Consider only top similar items above threshold
                top_indices = []
                for idx in sorted_indices:
                    if similarities[idx] >= self.similarity_threshold:
                        top_indices.append(idx)
                    if len(top_indices) >= self.num_similar_items:
                        break
                
                if len(top_indices) > 0:
                    # Get ratings and similarities for top similar items
                    similar_item_indices = [user_items[idx] for idx in top_indices]
                    similar_ratings = np.array([user_ratings[idx] for idx in top_indices])
                    similar_sims = np.array([similarities[idx] for idx in top_indices])
                    
                    # Calculate weighted average based on similarities
                    sim_sum = np.sum(similar_sims)
                    if sim_sum > 0:
                        history_prediction = np.sum(similar_ratings * similar_sims) / sim_sum
                        
                        # Blend base prediction with history prediction
                        base_predictions[i] = (1 - self.history_weight) * base_predictions[i] + \
                                            self.history_weight * history_prediction
        
        return base_predictions
    
    def predict_with_history(self, user_id, item_id, base_prediction):
        """
        Adjust a single prediction based on user history.
        
        Args:
            user_id: ID of the user
            item_id: ID of the item
            base_prediction: Base prediction from NNMF model
            
        Returns:
            Adjusted prediction
        """
        # Return base prediction if no history or similarity data is available
        if user_id not in self.user_histories or self.item_similarity_matrix is None:
            return base_prediction
            
        # Get user's rated items and ratings
        user_items = self.user_histories[user_id]['items']
        user_ratings = self.user_histories[user_id]['ratings']
        
        # Return base prediction if user has already rated this item
        if item_id in user_items:
            return base_prediction
            
        # Find similar items that the user has rated
        similarities = self.item_similarity_matrix[item_id, user_items]
        sorted_indices = np.argsort(-similarities)  # Descending order
        
        # Consider only top similar items above threshold
        top_indices = []
        for idx in sorted_indices:
            if similarities[idx] >= self.similarity_threshold:
                top_indices.append(idx)
            if len(top_indices) >= self.num_similar_items:
                break
        
        if len(top_indices) > 0:
            # Get ratings and similarities for top similar items
            similar_ratings = np.array([user_ratings[idx] for idx in top_indices])
            similar_sims = np.array([similarities[idx] for idx in top_indices])
            
            # Calculate weighted average based on similarities
            sim_sum = np.sum(similar_sims)
            if sim_sum > 0:
                history_prediction = np.sum(similar_ratings * similar_sims) / sim_sum
                
                # Blend base prediction with history prediction
                adjusted_prediction = (1 - self.history_weight) * base_prediction + \
                                      self.history_weight * history_prediction
                return adjusted_prediction
        
        return base_prediction
    
    def demonstrate_prediction(self, user_id, item_id, ratings_matrix):
        """
        Demonstrate how the history-based prediction is calculated.
        
        Args:
            user_id: ID of the user
            item_id: ID of the item
            ratings_matrix: Complete ratings matrix
        """
        # Skip if no history or similarity data is available
        if user_id not in self.user_histories or self.item_similarity_matrix is None:
            print(f"No user history or similarity data available for user {user_id}")
            return
            
        # Get user's rated items and ratings
        user_items = self.user_histories[user_id]['items']
        user_ratings = self.user_histories[user_id]['ratings']
        
        # Skip if user has already rated this item
        if item_id in user_items:
            print(f"User {user_id} has already rated item {item_id}")
            return
            
        # Get base prediction from NNMF
        base_prediction = self([tf.constant([user_id], dtype=tf.int32), 
                               tf.constant([item_id], dtype=tf.int32)], 
                              training=False).numpy()[0][0]
        
        print(f"\nDemonstrating prediction for User {user_id}, Item {item_id}")
        print(f"Base NNMF prediction: {base_prediction:.4f}")
        
        # Find similar items that the user has rated
        similarities = self.item_similarity_matrix[item_id, user_items]
        sorted_indices = np.argsort(-similarities)  # Descending order
        
        print("\nTop similar items rated by this user:")
        print("Item ID | Similarity | User's Rating")
        print("--------|------------|-------------")
        
        # Consider only top similar items above threshold
        top_indices = []
        for idx in sorted_indices:
            if similarities[idx] >= self.similarity_threshold:
                top_indices.append(idx)
                print(f"{user_items[idx]:7d} | {similarities[idx]:10.4f} | {user_ratings[idx]:6.2f}")
            if len(top_indices) >= self.num_similar_items:
                break
        
        if len(top_indices) > 0:
            # Get ratings and similarities for top similar items
            similar_ratings = np.array([user_ratings[idx] for idx in top_indices])
            similar_sims = np.array([similarities[idx] for idx in top_indices])
            
            # Calculate weighted average based on similarities
            sim_sum = np.sum(similar_sims)
            if sim_sum > 0:
                history_prediction = np.sum(similar_ratings * similar_sims) / sim_sum
                
                print(f"\nHistory-based prediction (weighted avg): {history_prediction:.4f}")
                
                # Blend base prediction with history prediction
                adjusted_prediction = (1 - self.history_weight) * base_prediction + \
                                      self.history_weight * history_prediction
                print(f"\nFinal combined prediction: {adjusted_prediction:.4f}")
                print(f"  = (1 - {self.history_weight:.2f}) * {base_prediction:.4f} + {self.history_weight:.2f} * {history_prediction:.4f}")
        else:
            print("\nNo similar items found above threshold. Using base prediction only.")