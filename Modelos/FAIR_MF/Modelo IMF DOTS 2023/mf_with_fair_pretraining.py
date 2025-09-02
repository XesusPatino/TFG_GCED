import torch
import mf
from torch import nn


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

    def forward(self, user_ids, item_ids):

        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)

        preds = torch.sum(user_embeds * item_embeds, dim=1, keepdim=True)

        if not self.is_pretraining:
            user_bias = self.user_bias_embedding(user_ids)
            item_bias = self.item_bias_embedding(item_ids)

            preds_bias = torch.sum(user_bias * item_bias, dim=1, keepdim=True)

            preds = preds + preds_bias

        preds = torch.sigmoid(preds)
        preds = preds * (self.max_range - self.min_range) + self.min_range

        return preds.squeeze()

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

        l2_loss = self._l2(user_ids, item_ids)
        loss += l2_loss

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
