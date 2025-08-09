import torch
import mf
from torch import nn
from data import datamodule


class MFWithFairPretraining(mf.CollaborativeFilteringModel):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 256,
        lr: float = 5e-4,
        l2_reg: float = 1e-5,
        rating_range: tuple = (1.0, 5.0),
        class_weights: torch.Tensor = None,
        history_weight: float = 0.3,
    ):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            lr=lr,
            l2_reg=l2_reg,
            rating_range=rating_range,
            history_weight=history_weight,
        )

        self.class_weights = nn.Parameter(class_weights, requires_grad=False)

        self.user_bias_embedding = nn.Embedding(num_users, embedding_dim // 8)
        self.item_bias_embedding = nn.Embedding(num_items, embedding_dim // 8)

        nn.init.xavier_uniform_(self.user_bias_embedding.weight)
        nn.init.xavier_uniform_(self.item_bias_embedding.weight)

        # Additional history-related components for biases
        self.history_bias_mlp = nn.Sequential(
            nn.Linear(embedding_dim // 4, embedding_dim // 8),
            nn.ReLU(),
            nn.Linear(embedding_dim // 8, 1)
        )
        
        # Initialize MLP weights
        for layer in self.history_bias_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.is_pretraining = True

        self.save_hyperparameters()

    def _l2(self, user_ids, item_ids):
        if self.is_pretraining:
            user_embeds = self.user_embedding(user_ids)
            item_embeds = self.item_embedding(item_ids)
        else:
            user_embeds = self.user_bias_embedding(user_ids)
            item_embeds = self.item_bias_embedding(item_ids)

        l2 = self.l2_reg * (
            (torch.square(user_embeds).sum() + torch.square(item_embeds).sum())
            / user_embeds.shape[0]
        )

        return l2

    def forward(self, user_ids, item_ids=None):
        """
        Forward pass with history-based adjustments, considering pretraining phase
        """
        user_embeds = self.user_embedding(user_ids)
        
        if item_ids is None:
            # For recommendation, use all items
            item_embeds = self.item_embedding.weight
            
            # Matrix factorization predictions
            mf_predictions = torch.matmul(user_embeds, item_embeds.T)
            
            # Add bias terms if not in pretraining
            if not self.is_pretraining:
                user_bias = self.user_bias_embedding(user_ids)
                item_bias = self.item_bias_embedding.weight
                bias_scores = torch.matmul(user_bias, item_bias.T)
                mf_predictions += bias_scores
            
            # We can't efficiently compute history adjustments for all items
            # So just return MF predictions for recommendation
            return mf_predictions
        else:
            # For specific user-item pairs
            item_embeds = self.item_embedding(item_ids)
            
            # Matrix factorization prediction
            mf_pred = torch.sum(user_embeds * item_embeds, dim=1)
            
            # Add bias terms if not in pretraining
            if not self.is_pretraining:
                user_bias = self.user_bias_embedding(user_ids)
                item_bias = self.item_bias_embedding(item_ids)
                bias_scores = torch.sum(user_bias * item_bias, dim=1)
                mf_pred += bias_scores
            
            # Add history-based adjustment if we're not in pretraining
            if not self.is_pretraining and self.user_histories is not None:
                history_adjustment = self._get_history_adjustment(user_ids, item_ids)
                mf_pred += self.history_weight * history_adjustment
            
            return mf_pred
    
    def recommend_top_k(self, user_ids, k=10):
        """
        Genera las recomendaciones top-K para los usuarios dados.
        
        Args:
            user_ids: Tensor de IDs de usuarios.
            k: Número de ítems a recomendar.
        
        Returns:
            Tensor con los IDs de los ítems recomendados (num_users, k).
        """
        user_ids = user_ids.long()
        
        # Get predictions for all items
        predictions = self(user_ids)  # Shape: [num_users, num_items]
        
        # Apply rating range
        predictions = torch.clamp(predictions, self.min_range, self.max_range)
        
        # Get top-k items
        top_k_items = torch.topk(predictions, k, dim=1).indices
        return top_k_items

    def training_step(self, batch, batch_idx):
        user_ids, item_ids, ratings, genders = batch

        preds = self(user_ids, item_ids)
        # Ensure predictions have the right shape
        if preds.dim() == 1:
            preds = preds.view(-1)
        if ratings.dim() == 2:
            ratings = ratings.view(-1)

        if self.is_pretraining:
            loss = (
                torch.nn.functional.mse_loss(preds, ratings, reduction="none")
                * self.class_weights[genders]
            ).mean()
        else:
            loss = torch.nn.functional.mse_loss(preds, ratings)

        # Agregar regularización L2
        l2_loss = self._l2(user_ids, item_ids)
        loss += l2_loss

        # (Opcional) Agregar penalización de fairness
        # fairness_loss = self.train_eo.compute()
        # loss += 0.1 * fairness_loss

        self.train_rmse.update(torch.clamp(preds, self.min_range, self.max_range), ratings)
        self.log(
            "train_rmse",
            self.train_rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def on_train_start(self):
        # Get user histories if available
        if hasattr(self.trainer, 'datamodule'):
            metadata = self.trainer.datamodule.get_user_item_metadata()
            self.set_metadata(metadata)
        
        if self.is_pretraining:
            self.user_bias_embedding.requires_grad_(False)
            self.item_bias_embedding.requires_grad_(False)
            self.history_mlp.requires_grad_(False)
            self.history_bias_mlp.requires_grad_(False)

            self.user_embedding.requires_grad_(True)
            self.item_embedding.requires_grad_(True)

        elif not self.is_pretraining:
            self.user_bias_embedding.requires_grad_(True)
            self.item_bias_embedding.requires_grad_(True)
            self.history_mlp.requires_grad_(True)
            self.history_bias_mlp.requires_grad_(True)

            self.user_embedding.requires_grad_(False)
            self.item_embedding.requires_grad_(False)

    def finish_pretraining(self):
        self.is_pretraining = False
        
    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, ratings, genders = batch
        predictions = self(user_ids, item_ids)
        
        # Ensure predictions have the right shape
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)
        
        # Return results for evaluation
        return {
            "user_ids": user_ids,
            "item_ids": item_ids,
            "ratings": ratings,
            "predictions": predictions,
            "genders": genders
        }

    def test_step(self, batch, batch_idx):
        user_ids, item_ids, ratings, genders = batch

        # Generate predictions
        predictions = self(user_ids, item_ids)
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)
        
        # Generate top-K recommendations
        top_k_items = self.recommend_top_k(user_ids, k=10)
        
        # Calculate RMSE
        mse = torch.nn.functional.mse_loss(predictions.squeeze(), ratings)
        rmse = torch.sqrt(mse)
        self.log("test_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True)
        
        # Return results for callbacks
        return {
            "user_ids": user_ids,
            "item_ids": item_ids,
            "ratings": ratings,
            "predictions": predictions,
            "top_k_items": top_k_items,
            "genders": genders
        }
        
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
            prediction = torch.clamp(prediction, self.min_range, self.max_range)
            
        return prediction.item()