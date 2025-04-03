from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanSquaredError
from pytorch_lightning import LightningModule
import fairness_metrics
from data import datamodule

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
        class_weights=None
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.l2_reg = l2_reg
        self.min_range, self.max_range = rating_range
        self.max_items = max_items or num_items
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Inicializar pesos
        self._init_weights()
        
        # Métricas
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        
        # Métricas de fairness
        self.train_eo = fairness_metrics.EqualOpportunity(max_items=self.max_items)
        
        self.save_hyperparameters()
    
    def _init_weights(self):
        # Inicializar embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_ids, item_ids=None):
        user_embeds = self.user_embedding(user_ids)
        
        if item_ids is None:
            # Para recomendación, usa solo una muestra de ítems si hay muchos
            sample_size = min(500, self.num_items)  # Limitar a 500 ítems
            if sample_size < self.num_items:
                indices = torch.randperm(self.num_items)[:sample_size]
                item_embeds = self.item_embedding.weight[indices]
            else:
                item_embeds = self.item_embedding.weight
            return torch.matmul(user_embeds, item_embeds.T)
        else:
            # Para evaluación específica
            item_embeds = self.item_embedding(item_ids)
            return torch.sum(user_embeds * item_embeds, dim=1)
    
    '''
    def forward(self, user_ids, item_ids=None):
        """
        Si item_ids es None, genera predicciones para todos los ítems.
        Si item_ids se proporciona, genera predicciones para esos ítems específicos.
        """
        user_embeds = self.user_embedding(user_ids)
        
        if item_ids is None:
            # Predicciones para todos los ítems
            item_embeds = self.item_embedding.weight
            if self.max_items and self.max_items < self.num_items:
                item_embeds = item_embeds[:self.max_items]
            return torch.matmul(user_embeds, item_embeds.T)
        else:
            # Predicciones para ítems específicos - version producto punto
            item_embeds = self.item_embedding(item_ids)
            # Para mantener consistencia con resultados anteriores:
            return torch.sum(user_embeds * item_embeds, dim=1)
            # O si prefieres mantener el comportamiento de la versión anterior:
            # return torch.matmul(user_embeds, item_embeds.T)
    '''
    def training_step(self, batch, batch_idx):
        user_ids, item_ids, ratings, gender = batch
        prediction = self(user_ids, item_ids)
        
        # Actualizar métrica de fairness
        self.train_eo.update(ratings, prediction.unsqueeze(1), gender)
        
        # Calcular pérdida
        mse_loss = F.mse_loss(prediction, ratings)
        reg_loss = self._l2(user_ids, item_ids)
        
        # Calcular fairness_loss (desconectado del grafo computacional)
        with torch.no_grad():
            fairness_loss = self.train_eo.compute().detach()
        
        total_loss = mse_loss + reg_loss + 0.1 * fairness_loss
        
        self.log("train_loss", total_loss, on_step=False, on_epoch=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, ratings, gender = batch
        
        # Generar predicciones
        if item_ids is not None:
            # Si tienes item_ids específicos, predecir solo para ellos
            prediction = self(user_ids, item_ids)
            # Asegurarse de que sea bidimensional
            if prediction.dim() == 1:
                prediction = prediction.unsqueeze(1)
        else:
            # O generar predicciones para todos los ítems
            prediction = self(user_ids)  # Esto debería ser [batch_size, num_items]
        
        # Devolver un diccionario con toda la información necesaria
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

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        user_ids, item_ids, ratings, gender = batch
        
        # Opción 1: Simplemente evaluar predicciones específicas como en validation_step
        predictions = self(user_ids, item_ids)
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)
        
        # Calcular RMSE
        mse = F.mse_loss(predictions.squeeze(), ratings)
        rmse = torch.sqrt(mse)
        self.log("test_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True)
        
        # También podemos generar top-k recomendaciones si lo deseas
        with torch.no_grad():
            top_k_items = self.recommend_top_k(user_ids, k=10)
        
        # Devolver un diccionario con los resultados para usarlos en callbacks
        return {
            "user_ids": user_ids,
            "item_ids": item_ids,
            "ratings": ratings,
            "predictions": predictions,
            "top_k_items": top_k_items,
            "genders": gender
        }
    
'''
class CollaborativeFilteringModel(LightningModule):
    def __init__(self, num_users, num_items, embedding_dim, lr, l2_reg, rating_range, max_items=None):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.l2_reg = l2_reg
        self.rating_range = rating_range
        self.max_items = max_items  # Agregar max_items como atributo opcional

        # Definir embeddings
        self.user_embedding = Embedding(num_users, embedding_dim)
        self.item_embedding = Embedding(num_items, embedding_dim)

        # Inicializar pesos
        self._init_weights()  # Asegúrate de que este método esté definido

        # Inicializar métricas
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)

        self.val_rmse_male = MeanSquaredError(squared=False)
        self.val_rmse_female = MeanSquaredError(squared=False)

        # Inicializar métricas de fairness
        self.train_eo = fairness_metrics.EqualOpportunity(max_items=max_items)
        self.val_eo = fairness_metrics.EqualOpportunity(max_items=max_items)
        self.test_eo = fairness_metrics.EqualOpportunity(max_items=max_items)

        self.save_hyperparameters()

    def _init_weights(self):
        # Inicializar los pesos de los embeddings con una distribución normal
        torch.nn.init.normal_(self.user_embedding.weight, std=0.01)
        torch.nn.init.normal_(self.item_embedding.weight, std=0.01)


    def _clamp_ratings(self, x):
        return torch.clamp(x, self.min_range, self.max_range)

    
    def forward(self, user_ids, item_ids=None):
        user_embeds = self.user_embedding(user_ids)
        if item_ids is None:
            item_embeds = self.item_embedding.weight  # Predicciones para todos los ítems
        else:
            item_embeds = self.item_embedding(item_ids)
        preds = torch.matmul(user_embeds, item_embeds.T)
        return preds
    
    
    def _l2(self, user_ids, item_ids, genders=None):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)

        l2 = self.l2_reg * (
            (torch.square(user_embeds).sum() + torch.square(item_embeds).sum())
            / user_embeds.shape[0]
        )
        return l2
    
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
        all_item_ids = torch.arange(self.item_embedding.num_embeddings, device=self.device)

        user_embeds = self.user_embedding(user_ids)  # (num_users, embedding_dim)
        # item_embeds = self.item_embedding(all_item_ids)  # (num_items, embedding_dim)
        item_embeds = self.item_embedding.weight

        if self.max_items is not None:
            item_embeds = item_embeds[:self.max_items]  # Limitar a max_items
            
        scores = torch.matmul(user_embeds, item_embeds.T)  # (num_users, num_items)
        scores = torch.sigmoid(scores)  # Opcional: aplicar sigmoide si es necesario
        scores = scores * (self.max_range - self.min_range) + self.min_range

        # Obtener los índices de los top-K ítems
        top_k_items = torch.topk(scores, k, dim=1).indices  # (num_users, k)
        return top_k_items


    def training_step(self, batch, batch_idx):
        user_ids, item_ids, ratings, gender = batch
        rating_pred = self(user_ids, item_ids)  # Predicciones para los ítems específicos

        # Actualizar la métrica de fairness
        self.train_eo.update(ratings, rating_pred, gender)

        # Calcular la pérdida
        mse_losses = torch.nn.MSELoss(reduction="none")(rating_pred, ratings)
        
        # Calcular fairness_loss sin grafo computacional
        with torch.no_grad():
            fairness_loss = self.train_eo.compute()  # Métrica de fairness

        loss = mse_losses.mean() + self._l2(user_ids, item_ids) + 0.1 * fairness_loss

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def on_train_epoch_end(self, trainer, pl_module):
        self.train_eo.reset()  # Reiniciar los estados de la métrica
    
    
    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, ratings, genders = batch
        preds = self(user_ids, item_ids)  # No pasar 'gender'
        return preds
    
    # Comentado
    def test_step(self, batch, batch_idx, dataloader_idx=None):
        user_ids, item_ids, ratings, gender = batch
        rating_pred = self(user_ids, item_ids, gender)

        self.test_rmse.update(self._clamp_ratings(rating_pred), ratings)
        self.log("test_rmse", self.test_rmse, on_step=False, on_epoch=True, prog_bar=True)
        
        # Actualizaciones para el callback
        self.test_user_ids.append(user_ids)
        self.test_item_ids.append(item_ids)
        self.test_ratings.append(ratings)
        self.test_preds.append(rating_pred)
        self.test_genders.append(gender)

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
    # Comentado
    
    def test_step(self, batch, batch_idx, dataloader_idx=None):
        user_ids, item_ids, ratings, gender = batch

        # Generar recomendaciones top-K
        top_k_items = self.recommend_top_k(user_ids, k=10)

        # Calcular métricas de fairness y calidad para top-K
        # Esto requiere que pases los ítems relevantes (ground truth) desde el datamodule
        relevant_items = self.trainer.datamodule.get_recommendation_info()["test_relevant_items"]
        recall_at_k = self._compute_recall_at_k(top_k_items, relevant_items, k=10)

        self.log("test_recall_at_10", recall_at_k, on_step=False, on_epoch=True, prog_bar=True)

        # Opcional: Calcular métricas de fairness para top-K
        # Esto requiere que pases los géneros de los usuarios y los ítems recomendados
        # fairness_metrics = self.fairness_metrics.compute_recommendation_fairness(...)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )
'''