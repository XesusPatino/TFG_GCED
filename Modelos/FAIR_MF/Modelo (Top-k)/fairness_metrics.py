import os
import numpy as np
import pandas as pd
import torch
import torchmetrics
from torchmetrics.utilities import dim_zero_cat
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any

MIN_RATINGS_PER_GENDER = 1

class FairnessMetrics:
    """Container class for all fairness metrics and visualization tools"""
    
    def __init__(self, data_dir: str, dataset_name: str = "ml-1m"):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self._load_movie_data()
        
    def _load_movie_data(self):
        """Load movie metadata including genres"""
        movie_path = os.path.join(self.data_dir, self.dataset_name, "movies.dat")
        self.movie_data = pd.read_csv(
            movie_path,
            sep="::",
            engine="python",
            encoding='latin-1',
            header=None
        )
        self.movie_data.columns = ["item_id", "title", "genres"]
        self.movie_data = self.movie_data.set_index("item_id")
        
    def compute_all_metrics(self, test_results: Dict, recommendation_results: Optional[Dict] = None):
        """
        Compute all fairness metrics for both rating prediction and recommendations.
        
        Args:
            test_results: Dict containing {
                'user_ids': tensor,
                'item_ids': tensor,
                'ratings': tensor,
                'predictions': tensor,
                'genders': tensor
            }
            recommendation_results: Optional dict containing {
                'user_ids': tensor,
                'top_k_items': tensor,  # shape (num_users, k)
                'genders': tensor
            }
        """
        metrics = {}
        
        # 1. Rating Prediction Fairness Metrics
        metrics.update(self._compute_rating_fairness(
            test_results['item_ids'],
            test_results['ratings'],
            test_results['predictions'],
            test_results['genders']
        ))
        
        # 2. Recommendation Fairness Metrics
        if recommendation_results:
            metrics.update(self._compute_recommendation_fairness(
                recommendation_results['user_ids'],
                recommendation_results['top_k_items'],
                recommendation_results['genders']
            ))
            
        return metrics
    
    def _compute_rating_fairness(self, item_ids, ratings, predictions, genders):
        """Compute fairness metrics for rating predictions"""
        metrics = {}
        
        # Equal Opportunity (global)
        eo_metric = EqualOpportunity()
        eo_metric.update(ratings, predictions, genders)
        metrics['equal_opportunity'] = eo_metric.compute().item()
        
        # Equal Opportunity per Item
        eoi_metric = EqualOpportunityItem()
        eoi_metric.update(item_ids, ratings, predictions, genders)
        metrics['equal_opportunity_item'] = eoi_metric.compute().item()
        
        # Equal Opportunity by Genre
        genre_eo = self._compute_genre_fairness(item_ids, ratings, predictions, genders)
        metrics.update(genre_eo)
        
        return metrics
    
    def _compute_genre_fairness(self, item_ids, ratings, predictions, genders):
        """Compute fairness metrics broken down by movie genres"""
        # Convert tensors to numpy
        item_ids_np = item_ids.cpu().np()
        ratings_np = ratings.cpu().np()
        preds_np = predictions.cpu().np()
        genders_np = genders.cpu().np()
        
        # Calculate squared errors
        squared_errors = (ratings_np - preds_np) ** 2
        
        # Prepare genre information
        movie_data = self.movie_data.copy()
        movie_data['genres'] = movie_data['genres'].str.split('|')
        movie_data = movie_data.explode('genres')
        
        genre_results = {}
        genre_table = []
        
        for genre in movie_data['genres'].unique():
            genre_items = movie_data[movie_data['genres'] == genre].index.values
            genre_mask = np.isin(item_ids_np, genre_items)
            
            if not genre_mask.any():
                continue
                
            genre_genders = genders_np[genre_mask]
            genre_errors = squared_errors[genre_mask]
            
            mask_0 = genre_genders == 0
            mask_1 = genre_genders == 1
            
            if mask_0.sum() < MIN_RATINGS_PER_GENDER or mask_1.sum() < MIN_RATINGS_PER_GENDER:
                continue
                
            rmse_0 = np.sqrt(genre_errors[mask_0].mean())
            rmse_1 = np.sqrt(genre_errors[mask_1].mean())
            eo = rmse_0 - rmse_1
            
            avg_rating_0 = ratings_np[genre_mask][mask_0].mean()
            avg_rating_1 = ratings_np[genre_mask][mask_1].mean()
            
            genre_results[f'eo_{genre.lower()}'] = eo
            genre_table.append([
                genre,
                avg_rating_0,
                avg_rating_1,
                mask_0.sum(),
                mask_1.sum(),
                rmse_0,
                rmse_1,
                eo
            ])
        
        # Print genre table
        print("\nGenre Fairness Analysis:")
        print(f"{'Genre':<16}{'Avg R (F)':>10}{'Avg R (M)':>10}{'#F':>8}{'#M':>8}{'RMSE (F)':>10}{'RMSE (M)':>10}{'EO':>10}")
        for row in sorted(genre_table, key=lambda x: abs(x[-1]), reverse=True):
            print(f"{row[0]:<16}{row[1]:>10.2f}{row[2]:>10.2f}{row[3]:>8}{row[4]:>8}{row[5]:>10.2f}{row[6]:>10.2f}{row[7]:>10.2f}")
            
        return genre_results
    
    def _compute_recommendation_fairness(self, user_ids, top_k_items, genders):
        """
        Compute fairness metrics for top-K recommendations.
        
        Args:
            user_ids: Tensor of user IDs (num_users,)
            top_k_items: Tensor of recommended item IDs (num_users, k)
            genders: Tensor of user genders (num_users,)
        """
        metrics = {}
        
        # Convert tensors to numpy for easier processing
        user_ids_np = user_ids.cpu().numpy()
        top_k_items_np = top_k_items.cpu().numpy()
        genders_np = genders.cpu().numpy()
        
        # 1. Gender Parity in Recommendations
        # Count how many items recommended to each gender are from female vs male directors
        # (This would require additional metadata about items, e.g., director gender)
        # Example: metrics['gender_parity'] = ...

        # 2. Recommendation Quality by Gender
        # Calculate Recall@K or NDCG@K for each gender
        recall_m = self._compute_recall(top_k_items_np[genders_np == 1])
        recall_f = self._compute_recall(top_k_items_np[genders_np == 0])
        metrics['recall_m'] = recall_m
        metrics['recall_f'] = recall_f
        metrics['recall_gap'] = recall_f - recall_m

        # 3. Calibration Metrics (Optional)
        # Evaluate if the recommendations align with user preferences
        # Example: metrics['calibration'] = ...

        return metrics

    '''
    def _compute_recall(self, top_k_items):
        """
        Compute Recall@K for a set of recommendations.
        """
        # Placeholder: Implement logic to calculate Recall@K
        # Requires ground truth relevant items for each user
        return np.random.random()  # Replace with actual computation
    '''
    
    def _compute_recall_at_k(self, top_k_items, relevant_items, k):
        """
        Compute Recall@K for a set of recommendations.
        
        Args:
            top_k_items: Array of recommended item IDs (num_users, k)
            relevant_items: Dict of relevant item IDs for each user
            k: Number of top recommendations to consider
        """
        recalls = []
        for user_id, recommended in enumerate(top_k_items):
            relevant = relevant_items.get(user_id, set())
            hits = len(set(recommended[:k]) & relevant)
            recalls.append(hits / len(relevant) if relevant else 0)
        return np.mean(recalls)
    
    def visualize_eccentricity(self, train_data, test_results, split):
        """Visualize fairness metrics by user rating eccentricity"""
        ecc_visualizer = EccentricityVisualizer(data_dir=self.data_dir)
        return ecc_visualizer.plot(train_data, test_results, split)


class EqualOpportunity(torchmetrics.Metric):
    def __init__(self, max_items=None, **kwargs):
        super().__init__(**kwargs)
        self.max_items = max_items
        self.add_state("gender", default=[], dist_reduce_fx="cat")
        self.add_state("rating", default=[], dist_reduce_fx="cat")
        self.add_state("prediction", default=[], dist_reduce_fx="cat")
    
    def update(self, rating, prediction, gender):
        # Asegurar que prediction siempre tenga la misma dimensión secundaria
        if prediction.dim() == 1:
            prediction = prediction.unsqueeze(1)
            
        # Truncar/rellenar si es necesario
        if self.max_items is not None:
            if prediction.size(1) > self.max_items:
                prediction = prediction[:, :self.max_items]
            elif prediction.size(1) < self.max_items:
                padding = torch.zeros(
                    (prediction.size(0), self.max_items - prediction.size(1)), 
                    device=prediction.device
                )
                prediction = torch.cat([prediction, padding], dim=1)
        
        self.rating.append(rating)
        self.prediction.append(prediction)
        self.gender.append(gender)
    
    def compute(self):
        with torch.no_grad():
            try:
                ratings = dim_zero_cat(self.rating)
                predictions = dim_zero_cat(self.prediction)
                gender = dim_zero_cat(self.gender)
                
                # Si predictions es bidimensional, necesitamos aplanar o seleccionar columnas relevantes
                if predictions.dim() > 1 and ratings.dim() == 1:
                    # Tomamos la primera columna como predicción relevante
                    predictions = predictions[:, 0]
                
                gender_mask = gender == 0
                if gender_mask.sum() == 0 or (~gender_mask).sum() == 0:
                    return torch.tensor(0.0, device=self.device)
                
                rmse_0 = torch.sqrt(torch.mean((ratings[gender_mask] - predictions[gender_mask]) ** 2))
                rmse_1 = torch.sqrt(torch.mean((ratings[~gender_mask] - predictions[~gender_mask]) ** 2))
                
                return torch.abs(rmse_0 - rmse_1)  # Valor absoluto de la diferencia
            except Exception as e:
                print(f"Error en compute de EqualOpportunity: {e}")
                return torch.tensor(0.0, device=self.device)


class EqualOpportunityItem(torchmetrics.Metric):
    """Item-level Equal Opportunity metric"""
    
    def __init__(self, abs=False, **kwargs):
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
            item_gender = gender[item_mask]
            
            mask_0 = item_gender == 0
            mask_1 = ~mask_0
            
            if mask_0.sum() < MIN_RATINGS_PER_GENDER or mask_1.sum() < MIN_RATINGS_PER_GENDER:
                continue

            rmse_0 = torch.sqrt(
                torch.mean((ratings[item_mask][mask_0] - predictions[item_mask][mask_0]) ** 2)
            )
            rmse_1 = torch.sqrt(
                torch.mean((ratings[item_mask][mask_1] - predictions[item_mask][mask_1]) ** 2)
            )

            diff = rmse_0 - rmse_1
            eo.append(torch.abs(diff) if self.abs else diff)

        return torch.mean(torch.stack(eo)) if eo else torch.tensor(float('nan'), device=self.device)


class EccentricityVisualizer:
    """Visualize fairness metrics by user rating eccentricity"""
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to the directory containing the dataset
        """
        self.data_dir = data_dir
        
    def plot(self, train_data: pd.DataFrame, test_results: dict, split: int):
        """
        Generate and save eccentricity plot
        
        Args:
            train_data: DataFrame with training data
            test_results: Dict with test results {
                'user_ids': tensor,
                'item_ids': tensor,
                'ratings': tensor,
                'predictions': tensor,
                'genders': tensor
            }
            split: Split number for naming output
            
        Returns:
            Dictionary with computed metrics:
                - 'overall_eo': Overall equal opportunity difference
                - 'eo_by_eccentricity': EO values by eccentricity bins
        """
        # Convert test results to DataFrame
        test_df = pd.DataFrame({
            'user_id': self._to_numpy(test_results['user_ids']),
            'item_id': self._to_numpy(test_results['item_ids']),
            'rating': self._to_numpy(test_results['ratings']),
            'pred': self._to_numpy(test_results['predictions'])
        })
        
        # Load user gender data
        user_data = pd.read_csv(
            os.path.join(self.data_dir, "ml-1m", "users.dat"),
            sep="::",
            engine="python",
            header=None
        )
        user_data.columns = ["user_id", "g", "age", "occupation", "zip_code"]
        user_data["g"] = user_data["g"].map({"M": 1, "F": 0})
        
        # Calculate eccentricity from training data
        train_df = train_data.copy()
        train_df["user_avg_rating"] = train_df["user_id"].map(
            train_df.groupby("user_id")["rating"].mean()
        )
        train_df["item_avg_rating"] = train_df["item_id"].map(
            train_df.groupby("item_id")["rating"].mean()
        )
        train_df["expected_rating"] = (
            train_df["user_avg_rating"] + train_df["item_avg_rating"]
        ) / 2
        train_df["eccentricity"] = abs(
            train_df["rating"] - train_df["expected_rating"]
        )
        
        # Merge gender into test data
        test_df["g"] = test_df["user_id"].map(user_data.set_index("user_id")["g"])
        
        # Calculate test eccentricity using training averages
        test_df["user_avg_rating"] = test_df["user_id"].map(
            train_df.groupby("user_id")["rating"].mean()
        )
        test_df["item_avg_rating"] = test_df["item_id"].map(
            train_df.groupby("item_id")["rating"].mean()
        )
        test_df["expected_rating"] = (
            test_df["user_avg_rating"] + test_df["item_avg_rating"]
        ) / 2
        test_df["eccentricity"] = abs(test_df["rating"] - test_df["expected_rating"])
        
        # Bin by eccentricity
        rating_range = test_df["rating"].max() - test_df["rating"].min()
        ecc_bins = np.linspace(0, rating_range, 20)
        
        # Split by gender
        m_df = test_df[test_df["g"] == 1]
        f_df = test_df[test_df["g"] == 0]
        
        # Calculate RMSE by bin and gender
        errors_by_ecc_m = m_df.groupby(pd.cut(m_df["eccentricity"], ecc_bins))[
            ["rating", "pred"]
        ].apply(lambda x: np.sqrt(np.mean((x["rating"] - x["pred"]) ** 2)))
        
        errors_by_ecc_f = f_df.groupby(pd.cut(f_df["eccentricity"], ecc_bins))[
            ["rating", "pred"]
        ].apply(lambda x: np.sqrt(np.mean((x["rating"] - x["pred"]) ** 2)))
        
        # Calculate EO by bin
        eo_by_ecc = errors_by_ecc_f.values - errors_by_ecc_m.values
        
        # Plot
        plt.figure(figsize=(10, 6))
        bin_mids = [x.mid for x in errors_by_ecc_m.index]
        
        plt.plot(np.array(bin_mids), np.array(eo_by_ecc), color="blue")
        
        # Add reference lines
        overall_rmse_m = np.sqrt(np.mean((m_df["rating"] - m_df["pred"]) ** 2))
        overall_rmse_f = np.sqrt(np.mean((f_df["rating"] - f_df["pred"]) ** 2))
        overall_eo = overall_rmse_f - overall_rmse_m
        
        plt.axhline(y=overall_eo, color="green", linestyle="--", label="Overall EO")
        plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        plt.ylim(-max(0.5, abs(overall_eo) * 2), max(0.5, abs(overall_eo) * 2))
        plt.xlabel("Eccentricity")
        plt.ylabel("EO")
        
        # Add gender distribution
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
        
        # Save plot
        os.makedirs("fairness_plots", exist_ok=True)
        plt.savefig(f"fairness_plots/eccentricity_split_{split}.png")
        plt.close()
        
        return {
            'overall_eo': overall_eo,
            'eo_by_eccentricity': dict(zip(bin_mids, eo_by_ecc))
        }
    
    def _to_numpy(self, data):
        """Convert input data to numpy array, handling both tensors and arrays"""
        if hasattr(data, 'cpu'):  # Handle PyTorch tensors
            return data.cpu().numpy()
        elif hasattr(data, 'numpy'):  # Handle TensorFlow tensors
            return data.numpy()
        return np.array(data)  # Handle numpy arrays and lists