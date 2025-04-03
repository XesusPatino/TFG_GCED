import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.utilities import dim_zero_cat
import torch
import matplotlib.pyplot as plt

MIN_RATINGS_PER_GENDER = 1


class EqualOpportunityItem(torchmetrics.Metric):

    def __init__(
        self,
        abs=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.abs = abs

        self.add_state("gender", default=[], dist_reduce_fx="cat")
        self.add_state("item_id", default=[], dist_reduce_fx="cat")
        self.add_state("rating", default=[], dist_reduce_fx="cat")
        self.add_state("prediction", default=[], dist_reduce_fx="cat")

    def update(self, item_id, rating, prediction, gender):

        self.item_id.append(item_id)
        self.rating.append(rating)
        self.prediction.append(prediction)
        self.gender.append(gender)

    
    def compute(self):

        item_ids = dim_zero_cat(self.item_id)
        ratings = dim_zero_cat(self.rating)
        predictions = dim_zero_cat(self.prediction)
        gender = dim_zero_cat(self.gender)

        eo = []

        for item_id in item_ids.unique():
            item_mask = item_ids == item_id
            item_ratings = ratings[item_mask]
            item_predictions = predictions[item_mask]
            item_gender = gender[item_mask]

            item_0 = item_gender == 0

            if (
                item_0.sum() < MIN_RATINGS_PER_GENDER
                or (~item_0).sum() < MIN_RATINGS_PER_GENDER
            ):
                continue

            item_0_rmse = torch.sqrt(
                torch.mean((item_ratings[item_0] - item_predictions[item_0]) ** 2)
            )
            item_1_rmse = torch.sqrt(
                torch.mean((item_ratings[~item_0] - item_predictions[~item_0]) ** 2)
            )

            if self.abs:
                eo.append(torch.abs(item_0_rmse - item_1_rmse))
            else:
                eo.append(item_0_rmse - item_1_rmse)

        return torch.mean(torch.stack(eo))


class EqualOpportunity(torchmetrics.Metric):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.add_state("gender", default=[], dist_reduce_fx="cat")
        self.add_state("rating", default=[], dist_reduce_fx="cat")
        self.add_state("prediction", default=[], dist_reduce_fx="cat")

    def update(self, rating, prediction, gender):

        self.rating.append(rating)
        self.prediction.append(prediction)
        self.gender.append(gender)


    def compute(self):
        ratings = dim_zero_cat(self.rating)
        predictions = dim_zero_cat(self.prediction)
        gender = dim_zero_cat(self.gender)

        gender_mask = gender == 0
        
        # Verifica que haya muestras de ambos géneros
        if gender_mask.sum() == 0 or (~gender_mask).sum() == 0:
            return torch.tensor(float('nan'), device=self.device)

        rmse_0 = torch.sqrt(
            torch.mean((ratings[gender_mask] - predictions[gender_mask]) ** 2)
        )

        rmse_1 = torch.sqrt(
            torch.mean((ratings[~gender_mask] - predictions[~gender_mask]) ** 2)
        )

        return rmse_0 - rmse_1
    
    
class EOiRelationCallback(pl.Callback):
    def __init__(self, data):

        self.gender_balance = np.full(data["item_id"].max() + 1, np.nan)

        item_ids = data["item_id"]
        gender = data["gender"]

        for item_id in item_ids.unique():
            mask_0 = (item_ids == item_id) & (gender == 0)
            mask_1 = (item_ids == item_id) & (gender == 1)

            if mask_0.sum() < 1 or mask_1.sum() < 1:
                print(item_id, mask_0.sum(), mask_1.sum())
                continue

            self.gender_balance[item_id] = mask_0.sum() / (mask_0.sum() + mask_1.sum())

        self.train_reviews_per_item = data.groupby("item_id").size()
        self.train_reviews_per_item = self.train_reviews_per_item.reindex(
            range(data["item_id"].max() + 1), fill_value=0
        )

        self.state = {"item_ids": [], "genders": [], "squared_errors": []}

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):

        user_ids, item_ids, ratings, gender = batch

        self.state["item_ids"].append(item_ids)
        self.state["genders"].append(gender)
        self.state["squared_errors"].append((outputs - ratings) ** 2)

    def on_test_end(self, trainer, pl_module):

        item_ids = dim_zero_cat(self.state["item_ids"])
        genders = dim_zero_cat(self.state["genders"])
        squared_errors = dim_zero_cat(self.state["squared_errors"])

        movie_data = pd.read_csv(
            "C:/Users/xpati/Documents/TFG/Data_Fair_MF/ml-1m/movies.dat",
            sep="::",
            engine="python",
            encoding="latin-1",
        )
        movie_data.columns = ["item_id", "title", "genres"]

        eo = np.full(movie_data["item_id"].max() + 1, np.nan)

        for item_id in item_ids.unique():
            mask_0 = (item_ids == item_id) & (genders == 0)
            mask_1 = (item_ids == item_id) & (genders == 1)

            if (
                mask_0.sum() < MIN_RATINGS_PER_GENDER
                or mask_1.sum() < MIN_RATINGS_PER_GENDER
            ):
                continue

            rmse_0 = torch.sqrt(squared_errors[mask_0].mean())
            rmse_1 = torch.sqrt(squared_errors[mask_1].mean())

            eo[item_id] = rmse_0 - rmse_1

        # Show which items have the highest and lowest EOi

        # Create new rows for missing movie IDs in the dataset

        movie_data = movie_data.set_index("item_id")
        movie_data = movie_data.reindex(
            range(movie_data.index.max() + 1), fill_value=np.nan
        )
        movie_data["EOi"] = eo

        print("Items with highest EOi:")
        print(movie_data.sort_values("EOi", ascending=False).head())

        print("Items with lowest EOi:")
        print(movie_data.sort_values("EOi", ascending=True).head())

        print(eo[1288])

        # Plot relation of EOi vs gender balance

        plt.scatter(
            self.gender_balance,
            eo,
            s=self.train_reviews_per_item / 20,
            alpha=0.5,
            label="Items (size = # reviews)",
        )
        plt.axhline(y=0, color="r", linestyle="--")
        plt.ylim(-0.5, 0.5)
        plt.xlabel("Gender Balance")
        plt.ylabel("Equal Opportunity Difference")
        plt.show()


class EOgenreRelationCallback(pl.Callback):
    def __init__(self):

        self.state = {
            "item_ids": [],
            "genres": [],
            "ratings": [],
            "preds": [],
            "genders": [],
        }

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):

        user_ids, item_ids, ratings, gender = batch

        self.state["item_ids"].append(item_ids)
        self.state["ratings"].append(ratings)
        self.state["preds"].append(outputs)
        self.state["genders"].append(gender)

    def on_test_end(self, trainer, pl_module):

        movie_data = pd.read_csv(
            "C:/Users/xpati/Documents/TFG/Data_Fair_MF/ml-1m/movies.dat",
            sep="::",
            engine="python",
            encoding="latin-1",
        )
        movie_data.columns = ["item_id", "title", "genres"]

        # Explode the genres column (format is "Action|Adventure|Sci-Fi")
        movie_data = movie_data.assign(genres=movie_data["genres"].str.split("|"))
        movie_data = movie_data.explode("genres")

        item_ids = dim_zero_cat(self.state["item_ids"])
        ratings = dim_zero_cat(self.state["ratings"])
        preds = dim_zero_cat(self.state["preds"])
        squared_errors = (ratings - preds) ** 2
        genders = dim_zero_cat(self.state["genders"])

        print(
            "\n",
            "Genre\t\tAvg R. (F)\tAvg R. (m)\tAvg No.R. (F)\t Avg No.R (M)\tRMSE (F)\tRMSE (M)\tEO",
        )

        for genre in movie_data["genres"].unique():
            genre_mask = movie_data["genres"] == genre
            genre_item_ids = movie_data[genre_mask]["item_id"]

            genre_item_mask = np.isin(item_ids.cpu().numpy(), genre_item_ids)

            if genre_item_mask.sum() == 0:
                continue

            genre_genders = genders[genre_item_mask]
            genre_squared_errors = squared_errors[genre_item_mask]

            genre_mask_0 = genre_genders == 0
            genre_mask_1 = genre_genders == 1

            if (
                genre_mask_0.sum() < MIN_RATINGS_PER_GENDER
                or genre_mask_1.sum() < MIN_RATINGS_PER_GENDER
            ):
                continue

            rmse_0 = torch.sqrt(genre_squared_errors[genre_mask_0].mean())
            rmse_1 = torch.sqrt(genre_squared_errors[genre_mask_1].mean())

            ratings_0 = ratings[genre_item_mask]
            ratings_0 = ratings_0[genre_mask_0].mean()

            ratings_1 = ratings[genre_item_mask]
            ratings_1 = ratings_1[genre_mask_1].mean()

            print(
                f"{genre:<16}{ratings_0:.2f}\t\t{ratings_1:.2f}\t\t{genre_mask_0.sum()}\t\t{genre_mask_1.sum()}\t\t{rmse_0:.2f}\t\t{rmse_1:.2f}\t\t{rmse_0 - rmse_1:.2f}"
            )

class EccentricityGCallback(pl.Callback):
    def __init__(self, train_data, split):
        self.state = {"user_ids": [], "item_ids": [], "ratings": [], "preds": []}
        self.train_data = train_data
        self.split = split

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        user_ids, item_ids, ratings, _ = batch
        self.state["user_ids"].append(user_ids)
        self.state["item_ids"].append(item_ids)
        self.state["ratings"].append(ratings)
        self.state["preds"].append(outputs)

    def on_test_end(self, trainer, pl_module):
        user_ids = dim_zero_cat(self.state["user_ids"]).detach().cpu().numpy()
        item_ids = dim_zero_cat(self.state["item_ids"]).detach().cpu().numpy()
        ratings = dim_zero_cat(self.state["ratings"]).detach().cpu().numpy()
        preds = dim_zero_cat(self.state["preds"]).detach().cpu().numpy()

        user_data = pd.read_csv(
            "C:/Users/xpati/Documents/TFG/Data_Fair_MF/ml-1m/users.dat", 
            sep="::", 
            engine="python", 
            header=None
        )
        user_data.columns = ["user_id", "g", "age", "occupation", "zip_code"]
        user_data["g"] = user_data["g"].map({"M": 1, "F": 0})

        self.train_data["user_avg_rating"] = self.train_data["user_id"].map(
            self.train_data.groupby("user_id")["rating"].mean()
        )
        self.train_data["item_avg_rating"] = self.train_data["item_id"].map(
            self.train_data.groupby("item_id")["rating"].mean()
        )
        self.train_data["expected_rating"] = (
            self.train_data["user_avg_rating"] + self.train_data["item_avg_rating"]
        ) / 2
        self.train_data["eccentricity"] = abs(
            self.train_data["rating"] - self.train_data["expected_rating"]
        )

        test_df = pd.DataFrame({
            "user_id": user_ids,
            "item_id": item_ids,
            "rating": ratings,
            "pred": preds,
        })

        test_df["g"] = test_df["user_id"].map(user_data.set_index("user_id")["g"])
        test_df["user_avg_rating"] = test_df["user_id"].map(
            self.train_data.groupby("user_id")["rating"].mean()
        )
        test_df["item_avg_rating"] = test_df["item_id"].map(
            self.train_data.groupby("item_id")["rating"].mean()
        )
        test_df["expected_rating"] = (
            test_df["user_avg_rating"] + test_df["item_avg_rating"]
        ) / 2
        test_df["eccentricity"] = abs(test_df["rating"] - test_df["expected_rating"])

        ecc_bins = np.linspace(0, test_df["rating"].max() - test_df["rating"].min(), 20)

        m_df = test_df[test_df["g"] == 1]
        f_df = test_df[test_df["g"] == 0]

        errors_by_ecc_m = m_df.groupby(pd.cut(m_df["eccentricity"], ecc_bins))[
            ["rating", "pred"]
        ].apply(lambda x: np.sqrt(np.mean((x["rating"] - x["pred"]) ** 2)))

        errors_by_ecc_f = f_df.groupby(pd.cut(f_df["eccentricity"], ecc_bins))[
            ["rating", "pred"]
        ].apply(lambda x: np.sqrt(np.mean((x["rating"] - x["pred"]) ** 2)))

        eo_by_ecc = errors_by_ecc_f.values - errors_by_ecc_m.values

        plt.figure(figsize=(10, 6))
        
        # Usamos los puntos medios de los bins para el eje x
        bin_mids = [x.mid for x in errors_by_ecc_m.index]
        
        # Convertimos explícitamente a arrays de NumPy
        plt.plot(np.array(bin_mids), np.array(eo_by_ecc), color="blue")

        overall_rmse_m = np.sqrt(np.mean((m_df["rating"] - m_df["pred"]) ** 2))
        overall_rmse_f = np.sqrt(np.mean((f_df["rating"] - f_df["pred"]) ** 2))
        overall_eo = overall_rmse_f - overall_rmse_m

        os.makedirs("eo_by_ecc_results", exist_ok=True)
        with open(f"eo_by_ecc_results/overall_eo_split_{self.split}.txt", "w") as f:
            f.write(str(overall_eo))

        plt.axhline(y=overall_eo, color="green", linestyle="--", label="Overall EO")
        plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        plt.ylim(-overall_eo * 2, overall_eo * 2)
        plt.xlabel("Eccentricity")
        plt.ylabel("EO")

        plt.twinx()
        plt.hist(
            m_df["eccentricity"],
            bins=ecc_bins,
            alpha=0.05,
            label="M users",
            color="blue",
            density=True,
        )
        plt.hist(
            f_df["eccentricity"],
            bins=ecc_bins,
            alpha=0.05,
            label="F users",
            color="red",
            density=True,
        )
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()
