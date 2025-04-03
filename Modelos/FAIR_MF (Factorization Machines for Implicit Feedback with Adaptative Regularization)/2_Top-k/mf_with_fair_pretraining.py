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
    ):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            lr=lr,
            l2_reg=l2_reg,
            rating_range=rating_range,
        )

        self.class_weights = nn.Parameter(class_weights, requires_grad=False)

        self.user_bias_embedding = nn.Embedding(num_users, embedding_dim // 8)
        self.item_bias_embedding = nn.Embedding(num_items, embedding_dim // 8)

        nn.init.xavier_uniform_(self.user_bias_embedding.weight)
        nn.init.xavier_uniform_(self.item_bias_embedding.weight)

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
        user_embeds = self.user_embedding(user_ids)
        if item_ids is None:
            item_embeds = self.item_embedding.weight  # Predicciones para todos los ítems
        else:
            item_embeds = self.item_embedding(item_ids)
        preds = torch.matmul(user_embeds, item_embeds.T)
        return preds

    
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
        item_embeds = self.item_embedding(all_item_ids)  # (num_items, embedding_dim)

        scores = torch.matmul(user_embeds, item_embeds.T)  # (num_users, num_items)

        if not self.is_pretraining:
            # Agregar sesgos si no estás en preentrenamiento
            user_bias = self.user_bias_embedding(user_ids)  # (num_users, embedding_dim // 8)
            item_bias = self.item_bias_embedding(all_item_ids)  # (num_items, embedding_dim // 8)
            bias_scores = torch.matmul(user_bias, item_bias.T)  # (num_users, num_items)
            scores += bias_scores

        scores = torch.sigmoid(scores)  # Opcional: aplicar sigmoide si es necesario
        scores = scores * (self.max_range - self.min_range) + self.min_range

        # Obtener los índices de los top-K ítems
        top_k_items = torch.topk(scores, k, dim=1).indices  # (num_users, k)
        return top_k_items
    

    def training_step(self, batch, batch_idx):
        user_ids, item_ids, ratings, genders = batch

        preds = self(user_ids, item_ids)

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

        self.train_rmse.update(self._clamp_ratings(preds), ratings)
        self.log(
            "train_rmse",
            self.train_rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def on_train_start(self):
        if self.is_pretraining:
            self.user_bias_embedding.requires_grad_(False)
            self.item_bias_embedding.requires_grad_(False)

            self.user_embedding.requires_grad_(True)
            self.item_embedding.requires_grad_(True)

        elif not self.is_pretraining:
            self.user_bias_embedding.requires_grad_(True)
            self.item_bias_embedding.requires_grad_(True)

            self.user_embedding.requires_grad_(False)
            self.item_embedding.requires_grad_(False)

    def finish_pretraining(self):
        self.is_pretraining = False
        
    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, ratings, genders = batch
        preds = self(user_ids, item_ids)  # No pasar 'gender'
        return preds

    def test_step(self, batch, batch_idx):
        user_ids, item_ids, ratings, genders = batch

        # Generar recomendaciones top-K
        top_k_items = self.recommend_top_k(user_ids, k=10)

        # Calcular métricas de calidad (e.g., Recall@K)
        relevant_items = self.trainer.datamodule.get_recommendation_info()["test_relevant_items"]
        recall_at_k = self._compute_recall_at_k(top_k_items, relevant_items, k=10)

        self.log("test_recall_at_10", recall_at_k, on_step=False, on_epoch=True, prog_bar=True)

        # (Opcional) Calcular métricas de fairness para top-K
        # fairness_metrics = self.fairness_metrics.compute_recommendation_fairness(...)
