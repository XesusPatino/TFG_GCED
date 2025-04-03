from typing import Literal
import torch
from torchmetrics import MeanSquaredError
from pytorch_lightning import LightningModule
from torch.nn import Embedding, Parameter
import fairness_metrics


class CollaborativeFilteringModel(LightningModule):

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 256,
        lr: float = 5e-4,
        l2_reg: float = 1e-5,
        rating_range: tuple = (0.0, 4.0),
    ):
        super().__init__()

        self.lr = lr
        self.l2_reg = l2_reg
        self.embedding_dim = embedding_dim
        self.min_range, self.max_range = rating_range  # Use the provided rating_range

        self.gender_embedding_ratio = 0

        self.user_embedding = Embedding(num_users, embedding_dim)
        self.item_embedding = Embedding(num_items, embedding_dim)

        torch.nn.init.xavier_uniform_(self.user_embedding.weight)
        torch.nn.init.xavier_uniform_(self.item_embedding.weight)

        # Initialize metrics
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)

        self.val_rmse_male = MeanSquaredError(squared=False)
        self.val_rmse_female = MeanSquaredError(squared=False)

        # Initialize only the metrics that exist in fairness_metrics
        self.train_eo = fairness_metrics.EqualOpportunity()
        self.val_eo = fairness_metrics.EqualOpportunity()
        self.test_eo = fairness_metrics.EqualOpportunity()

        # Initialize these only if they exist in fairness_metrics
        # If not, you'll need to remove them from test_step
        if hasattr(fairness_metrics, 'EqualOpportunityItem'):
            self.test_eoi = fairness_metrics.EqualOpportunityItem()
        if hasattr(fairness_metrics, 'EqualOpportunityItemAbsolute'):
            self.test_eoi_abs = fairness_metrics.EqualOpportunityItemAbsolute()

        self.save_hyperparameters()


    def _clamp_ratings(self, x):
        return torch.clamp(x, self.min_range, self.max_range)

    
    def forward(self, user_ids, item_ids, genders=None):
        # Convertir los índices a tipo Long si no lo están
        user_ids = user_ids.long()
        item_ids = item_ids.long()
        
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)

        preds = torch.sum(user_embeds * item_embeds, dim=1, keepdim=True)

        preds = torch.sigmoid(preds)
        preds = preds * (self.max_range - self.min_range) + self.min_range

        return preds.squeeze()
    
    
    def _l2(self, user_ids, item_ids, genders=None):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)

        l2 = self.l2_reg * (
            (torch.square(user_embeds).sum() + torch.square(item_embeds).sum())
            / user_embeds.shape[0]
        )
        return l2


    def training_step(self, batch, batch_idx):
        user_ids, item_ids, ratings, gender = batch
        rating_pred = self(user_ids, item_ids, gender)

        mse_losses = torch.nn.MSELoss(reduction="none")(rating_pred, ratings)
        eo_loss = 0.0
        loss = mse_losses.mean() + self._l2(user_ids, item_ids, gender) + eo_loss

        self.train_rmse.update(self._clamp_ratings(rating_pred), ratings)
        self.log("train_rmse", self.train_rmse, on_step=False, on_epoch=True, prog_bar=True)

        self.train_eo.update(ratings, self._clamp_ratings(rating_pred), gender)
        self.log("train_eo", self.train_eo, on_step=False, on_epoch=True, prog_bar=True)

        if self.gender_embedding_ratio > 0:
            # Initialize these dynamically if needed
            if not hasattr(self, 'train_eo_wo_gender'):
                self.train_eo_wo_gender = fairness_metrics.EqualOpportunity()
            if not hasattr(self, 'train_rmse_wo_gender'):
                self.train_rmse_wo_gender = MeanSquaredError(squared=False)
                
            pred_wo_gender = self(user_ids, item_ids)
            self.train_eo_wo_gender.update(ratings, pred_wo_gender, gender)
            self.train_rmse_wo_gender.update(ratings, pred_wo_gender)

            self.log("train_eo_wo_gender", self.train_eo_wo_gender, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_rmse_wo_gender", self.train_rmse_wo_gender, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        user_ids, item_ids, ratings, gender = batch
        rating_pred = self(user_ids, item_ids, gender)

        male_idx = torch.where(gender == 1)
        female_idx = torch.where(gender == 0)

        # Calcular RMSE global
        self.val_rmse.update(self._clamp_ratings(rating_pred), ratings)
        
        # Calcular RMSE para hombres
        if len(male_idx[0]) > 0:
            self.val_rmse_male.update(rating_pred[male_idx], ratings[male_idx])
        
        # Calcular RMSE para mujeres solo si hay muestras
        if len(female_idx[0]) > 0:
            self.val_rmse_female.update(rating_pred[female_idx], ratings[female_idx])
        
        # Calcular métrica de equidad
        try:
            self.val_eo.update(ratings, self._clamp_ratings(rating_pred), gender)
        except Exception as e:
            print(f"Could not compute fairness metric: {e}")

        # Loggear métricas
        self.log("val_rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=True)
        
        if len(male_idx[0]) > 0:
            self.log("val_rmse_male", self.val_rmse_male, on_step=False, on_epoch=True, prog_bar=True)
        
        if len(female_idx[0]) > 0:
            self.log("val_rmse_female", self.val_rmse_female, on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.log("val_rmse_female", torch.tensor(float('nan')), on_step=False, on_epoch=True, prog_bar=True)
        
        self.log("val_eo", self.val_eo, on_step=False, on_epoch=True, prog_bar=True)
        
    
    def test_step(self, batch, batch_idx, dataloader_idx=None):
        user_ids, item_ids, ratings, gender = batch
        rating_pred = self(user_ids, item_ids, gender)

        self.test_rmse.update(self._clamp_ratings(rating_pred), ratings)
        self.log("test_rmse", self.test_rmse, on_step=False, on_epoch=True, prog_bar=True)

        # Only log these if the metrics exist
        if hasattr(self, 'test_eoi'):
            self.test_eoi.update(item_ids, ratings, self._clamp_ratings(rating_pred), gender)
            self.log("test_eoi", self.test_eoi, on_step=False, on_epoch=True, prog_bar=True)

        if hasattr(self, 'test_eoi_abs'):
            self.test_eoi_abs.update(item_ids, ratings, self._clamp_ratings(rating_pred), gender)
            self.log("test_eoi_abs", self.test_eoi_abs, on_step=False, on_epoch=True, prog_bar=True)

        self.test_eo.update(ratings, self._clamp_ratings(rating_pred), gender)
        self.log("test_eo", self.test_eo, on_step=False, on_epoch=True, prog_bar=True)

        return self._clamp_ratings(rating_pred)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )