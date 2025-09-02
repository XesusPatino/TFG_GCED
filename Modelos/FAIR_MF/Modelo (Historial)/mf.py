from typing import Literal, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from pytorch_lightning import LightningModule
import fairness_metrics
from data import datamodule
import numpy as np
import pandas as pd

class CollaborativeFilteringModel(LightningModule):
    def __init__(
        self, 
        num_users, 
        num_items, 
        embedding_dim, 
        lr, 
        l2_reg, 
        rating_range, 
        max_items=None,
        class_weights=None,
        history_weight=0.3  # Weight for history-based prediction
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.l2_reg = l2_reg
        self.min_range, self.max_range = rating_range
        self.max_items = max_items or num_items
        self.history_weight = history_weight
        
        # Store user histories (will be populated during fit)
        self.user_histories = None
        self.global_mean = None
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP for history-based adjustments
        self.history_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        
        # Inicializar pesos
        self._init_weights()
        
        # Métricas
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        
        # Métricas de fairness
        self.train_eo = fairness_metrics.EqualOpportunity(max_items=self.max_items)
        
        self.save_hyperparameters()
    
    def _init_weights(self):
        # Inicializar embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        # Initialize MLP weights
        for layer in self.history_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def set_metadata(self, metadata):
        """
        Set metadata for history-based predictions
        """
        self.user_histories = metadata["user_histories"]
        self.global_mean = metadata["global_mean"]
    
    def _get_history_adjustment(self, user_ids, item_ids):
        """
        Compute history-based adjustment for predictions
        """
        if self.user_histories is None:
            return torch.zeros(len(user_ids), device=self.device)
        
        adjustments = []
        
        for i, (user_id, item_id) in enumerate(zip(user_ids.cpu().numpy(), item_ids.cpu().numpy())):
            user_id, item_id = int(user_id), int(item_id)
            
            # If no history for this user, use zero adjustment
            if user_id not in self.user_histories:
                adjustments.append(0.0)
                continue
                
            history = self.user_histories[user_id]
            
            # Calculate similarity-based adjustment
            user_embed = self.user_embedding(torch.tensor([user_id], device=self.device))
            item_embed = self.item_embedding(torch.tensor([item_id], device=self.device))
            history_item_ids = torch.tensor(history['item_id'].values, device=self.device)
            
            if len(history_item_ids) == 0:
                adjustments.append(0.0)
                continue
                
            history_item_embeds = self.item_embedding(history_item_ids)
            history_ratings = torch.tensor(history['rating'].values, device=self.device)
            
            # Calculate similarity between target item and history items
            similarities = F.cosine_similarity(
                item_embed.unsqueeze(1), 
                history_item_embeds.unsqueeze(0),
                dim=2
            ).squeeze()
            
            # Get top-k most similar items
            k = min(5, len(similarities))
            if k == 0:
                adjustments.append(0.0)
                continue
                
            top_k_sim, indices = torch.topk(similarities, k)
            top_k_ratings = history_ratings[indices]
            
            # Weighted average of ratings by similarity
            if top_k_sim.sum() > 0:
                adjustment = ((top_k_sim * top_k_ratings).sum() / top_k_sim.sum()) - self.global_mean
                # Convert to float if it's a tensor
                if isinstance(adjustment, torch.Tensor):
                    adjustment = adjustment.item()
            else:
                adjustment = 0.0
                
            adjustments.append(adjustment)
        
        return torch.tensor(adjustments, device=self.device)
    
    def forward(self, user_ids, item_ids=None):
        """
        Forward pass with history-based adjustments
        """
        user_embeds = self.user_embedding(user_ids)
        
        if item_ids is None:
            # For recommendation, use only a sample of items if there are many
            sample_size = min(500, self.num_items)  # Limit to 500 items
            if sample_size < self.num_items:
                indices = torch.randperm(self.num_items)[:sample_size]
                item_embeds = self.item_embedding.weight[indices]
            else:
                item_embeds = self.item_embedding.weight
                
            # Matrix factorization predictions
            mf_predictions = torch.matmul(user_embeds, item_embeds.T)
            
            # We can't efficiently compute history adjustments for all items
            # So just return MF predictions for recommendation
            return mf_predictions
        else:
            # For specific user-item pairs
            item_embeds = self.item_embedding(item_ids)
            
            # Matrix factorization prediction
            mf_pred = torch.sum(user_embeds * item_embeds, dim=1)
            
            # History-based adjustment
            history_adjustment = self._get_history_adjustment(user_ids, item_ids)
            
            # Combined prediction
            final_pred = mf_pred + self.history_weight * history_adjustment
            
            return final_pred
    
    def training_step(self, batch, batch_idx):
        user_ids, item_ids, ratings, gender = batch
        prediction = self(user_ids, item_ids)
        
        # Actualizar métricas
        self.train_rmse.update(prediction, ratings)
        self.train_mae.update(prediction, ratings)
        self.train_eo.update(ratings, prediction.unsqueeze(1), gender)
        
        # Calcular pérdida
        mse_loss = F.mse_loss(prediction, ratings)
        reg_loss = self._l2(user_ids, item_ids)
        
        # Calcular fairness_loss (desconectado del grafo computacional)
        with torch.no_grad():
            fairness_loss = self.train_eo.compute().detach()
        
        total_loss = mse_loss + reg_loss + 0.1 * fairness_loss
        
        # Log métricas
        self.log("train_loss", total_loss, on_step=False, on_epoch=True)
        self.log("train_rmse", self.train_rmse, on_step=False, on_epoch=True)
        self.log("train_mae", self.train_mae, on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, ratings, gender = batch
        
        # Generate predictions
        prediction = self(user_ids, item_ids)
        
        # Ensure predictions have the right shape
        if prediction.dim() == 1:
            prediction = prediction.unsqueeze(1)
        
        # Update metrics
        self.val_rmse.update(prediction.squeeze(), ratings)
        self.val_mae.update(prediction.squeeze(), ratings)
        
        # Calculate and log metrics
        mse = F.mse_loss(prediction.squeeze(), ratings)
        rmse = torch.sqrt(mse)
        mae = F.l1_loss(prediction.squeeze(), ratings)
        
        self.log("val_rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)
        
        # Return a dictionary with data for callbacks
        return {
            "user_ids": user_ids,
            "item_ids": item_ids,
            "ratings": ratings,
            "predictions": prediction,
            "genders": gender
        }
    
    def _l2(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        l2 = self.l2_reg * (torch.norm(user_embeds)**2 + torch.norm(item_embeds)**2) / user_ids.size(0)
        return l2
    
    def recommend_top_k(self, user_ids, k=10):
        # Obtener predicciones para todos los ítems
        predictions = self(user_ids)
        # Obtener top-k ítems
        return torch.topk(predictions, k=k, dim=1).indices
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    def on_train_epoch_end(self):
        self.train_eo.reset()  # Reiniciar métrica de fairness
    
    def predict_rating(self, user_id, item_id):
        """
        Predict rating for a specific user-item pair
        """
        if isinstance(user_id, int):
            user_id = torch.tensor([user_id], device=self.device)
        if isinstance(item_id, int):
            item_id = torch.tensor([item_id], device=self.device)
            
        with torch.no_grad():
            prediction = self(user_id, item_id)
            
        return prediction.item()
    
    def test_step(self, batch, batch_idx, dataloader_idx=None):
        user_ids, item_ids, ratings, gender = batch
        
        # Generate predictions
        predictions = self(user_ids, item_ids)
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)
        
        # Calculate RMSE
        mse = F.mse_loss(predictions.squeeze(), ratings)
        rmse = torch.sqrt(mse)
        self.log("test_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True)
        
        # Generate top-k recommendations
        with torch.no_grad():
            top_k_items = self.recommend_top_k(user_ids, k=10)
        
        # Return results for callbacks
        return {
            "user_ids": user_ids,
            "item_ids": item_ids,
            "ratings": ratings,
            "predictions": predictions,
            "top_k_items": top_k_items,
            "genders": gender
        }
    
    def on_fit_start(self):
        """
        Get user histories from datamodule when starting training
        """
        if hasattr(self.trainer, 'datamodule'):
            metadata = self.trainer.datamodule.get_user_item_metadata()
            self.set_metadata(metadata)