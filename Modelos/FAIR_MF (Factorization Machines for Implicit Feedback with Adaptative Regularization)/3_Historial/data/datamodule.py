from typing import List, Dict, Set
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torch
from sklearn.model_selection import train_test_split


class DyadicRegressionDataset(Dataset):
    def __init__(self, df):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the dataset
                Must contain the columns ['user_id', 'item_id', 'rating', 'gender']
        """
        self.data = df.reset_index(drop=True)
        
        # Convertir tipos de datos
        self.data["user_id"] = self.data["user_id"].astype(np.int64)
        self.data["item_id"] = self.data["item_id"].astype(np.int64)
        self.data["rating"] = self.data["rating"].astype(np.float32)
        self.data["gender"] = self.data["gender"].astype(np.int64)
        
        # Seleccionar columnas necesarias
        self.data = self.data[["user_id", "item_id", "rating", "gender"]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return (
            torch.tensor(row["user_id"], dtype=torch.long),
            torch.tensor(row["item_id"], dtype=torch.long),
            torch.tensor(row["rating"], dtype=torch.float32),
            torch.tensor(row["gender"], dtype=torch.long)
        )
        

class UserItemPredictionDataset(Dataset):
    """Dataset for predicting ratings for specific user-item pairs"""
    def __init__(self, user_item_pairs, user_histories=None):
        """
        Args:
            user_item_pairs: List of tuples (user_id, item_id) to predict
            user_histories: Optional dict {user_id: pd.DataFrame} with user rating history
        """
        self.user_item_pairs = user_item_pairs
        self.user_histories = user_histories  # Can be used for history-based features

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user_id, item_id = self.user_item_pairs[idx]
        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(item_id, dtype=torch.long)
        )
        

class RecommendationTestDataset(Dataset):
    """Dataset especial para evaluación de recomendaciones top-K"""
    def __init__(self, user_ids: List[int], unseen_items: Dict[int, List[int]]):
        """
        Args:
            user_ids: Lista de usuarios a evaluar
            unseen_items: Diccionario {user_id: lista de ítems no vistos}
        """
        self.user_ids = user_ids
        self.unseen_items = unseen_items
        self.total_samples = sum(len(items) for items in unseen_items.values())

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Mapear índice global a (usuario, ítem) específico
        for user_id in self.user_ids:
            items = self.unseen_items[user_id]
            if idx < len(items):
                return (
                    torch.tensor(user_id, dtype=torch.long),
                    torch.tensor(items[idx], dtype=torch.long)
                )
            idx -= len(items)
        raise IndexError("Index out of range")


class DyadicRegressionDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        split: int,
        data_dir: str = "C:/Users/xpati/Documents/TFG/Data_Fair_MF",
        batch_size=64,
        num_workers=4,
        verbose=False,
    ):
        
        self.split = split  # Guardar explícitamente como atributo público
        
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.verbose = verbose

        # 1. Cargar datos
        self._load_data(dataset_name, split)
        
        # 2. Inicializar estructuras para recomendación
        self._init_recommendation_structures()
        
        if verbose:
            self._print_stats()

    def _load_data(self, dataset_name: str, split: int):
        """Carga y prepara los datos"""
        # Cargar datos brutos
        self.train_df = pd.read_csv(f"{self.data_dir}/{dataset_name}/splits/train_{split}.csv")
        self.test_df = pd.read_csv(f"{self.data_dir}/{dataset_name}/splits/test_{split}.csv")
        
        # Cargar información de género
        gender_data = pd.read_csv(
            f"{self.data_dir}/{dataset_name}/users.dat",
            sep="::",
            engine="python",
            header=None,
        )
        gender_data.columns = ["user_id", "gender", "age", "occupation", "zip_code"]
        gender_data["gender"] = gender_data["gender"].map({"M": 1, "F": 0})

        # Unir información de género
        self.train_df = self.train_df.merge(
            gender_data[["user_id", "gender"]], on="user_id", how="left"
        )
        self.test_df = self.test_df.merge(
            gender_data[["user_id", "gender"]], on="user_id", how="left"
        )

        # Dividir train en train y validation estratificado por género
        self.train_df, self.val_df = train_test_split(
            self.train_df,
            test_size=0.2,
            stratify=self.train_df["gender"],
            random_state=42
        )

        # Calcular estadísticas
        self.data = pd.concat([self.train_df, self.val_df, self.test_df])
        self.num_users = self.data["user_id"].max() + 1
        self.num_items = self.data["item_id"].max() + 1
        self.min_rating = self.data["rating"].min()
        self.max_rating = self.data["rating"].max()

    def _init_recommendation_structures(self):
        """Inicializa estructuras para manejar ítems vistos/no vistos"""
        # 1. Ítems vistos por usuario (train + val)
        self.user_seen_items: Dict[int, Set[int]] = {
            user_id: set(group["item_id"])
            for user_id, group in pd.concat([self.train_df, self.val_df]).groupby("user_id")
        }
        
        # 2. Ítems no vistos por usuario (para test)
        all_items = set(range(self.num_items))
        self.user_unseen_items: Dict[int, List[int]] = {
            user_id: list(all_items - self.user_seen_items.get(user_id, set()))
            for user_id in self.test_df["user_id"].unique()
        }
        
        # Opcional: Limitar el número de ítems no vistos por usuario (mejora velocidad)
        max_unseen_per_user = 1000  # Ajustable según necesidad
        self.user_unseen_items = {
            user_id: items[:max_unseen_per_user] 
            for user_id, items in self.user_unseen_items.items()
        }
        
        # 3. Ítems relevantes en test (ground truth)
        self.test_relevant_items: Dict[int, Set[int]] = {
            user_id: set(group["item_id"])
            for user_id, group in self.test_df.groupby("user_id")
        }

    def _print_stats(self):
        """Imprime estadísticas del dataset"""
        print(f"#Users: {self.num_users}")
        print(f"#Items: {self.num_items}")
        print(f"Min rating: {self.min_rating:.3f}")
        print(f"Max rating: {self.max_rating:.3f}")
        print("\nDistribución de género:")
        print(f"Train - M: {(self.train_df['gender'] == 1).sum()}, F: {(self.train_df['gender'] == 0).sum()}")
        print(f"Val - M: {(self.val_df['gender'] == 1).sum()}, F: {(self.val_df['gender'] == 0).sum()}")
        print(f"Test - M: {(self.test_df['gender'] == 1).sum()}, F: {(self.test_df['gender'] == 0).sum()}")
        print("\nEstadísticas de recomendación:")
        avg_unseen = np.mean([len(v) for v in self.user_unseen_items.values()])
        print(f"Avg unseen items per user: {avg_unseen:.1f}")

    def train_dataloader(self):
        return DataLoader(
            DyadicRegressionDataset(self.train_df),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            DyadicRegressionDataset(self.val_df),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """DataLoader estándar para evaluación de ratings"""
        return DataLoader(
            DyadicRegressionDataset(self.test_df),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
    
    def get_user_histories(self):
        """Returns a dictionary with user rating histories"""
        user_histories = {}
        
        # Create histories from training data
        for user_id, group in self.train_df.groupby("user_id"):
            user_histories[user_id] = group.copy()
            
        return user_histories
    
    def create_prediction_dataloader(self, user_item_pairs, batch_size=128):
        """
        Creates a dataloader for predicting ratings of specific user-item pairs
        
        Args:
            user_item_pairs: List of tuples (user_id, item_id) to predict
            batch_size: Batch size for the dataloader
            
        Returns:
            DataLoader for prediction
        """
        dataset = UserItemPredictionDataset(
            user_item_pairs, 
            user_histories=self.get_user_histories()
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True
        )
    
    def get_user_item_metadata(self):
        """
        Returns metadata useful for rating prediction
        """
        return {
            "user_histories": self.get_user_histories(),
            "num_users": self.num_users,
            "num_items": self.num_items,
            "rating_range": (self.min_rating, self.max_rating),
            "global_mean": self.train_df["rating"].mean()
        }

    '''
    def recommendation_dataloader(self, batch_size: int = 1024):
        """
        DataLoader especial para evaluación de recomendaciones top-K.
        Genera pares (usuario, ítem) para todos los ítems no vistos por cada usuario.
        """
        test_users = list(self.user_unseen_items.keys())
        dataset = RecommendationTestDataset(test_users, self.user_unseen_items)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
        )
    '''
    
    def recommendation_dataloader(self, batch_size: int = 1024, max_unseen_items: int = None):
        """
        DataLoader especial para evaluación de recomendaciones top-K.
        Genera pares (usuario, ítem) para todos los ítems no vistos por cada usuario.
        Args:
            batch_size: Tamaño del batch.
            max_unseen_items: Número máximo de ítems no vistos por usuario a considerar.
        """
        test_users = list(self.user_unseen_items.keys())
        if max_unseen_items:
            # Limitar el número de ítems no vistos por usuario
            limited_unseen_items = {
                user_id: items[:max_unseen_items]
                for user_id, items in self.user_unseen_items.items()
            }
        else:
            limited_unseen_items = self.user_unseen_items

        dataset = RecommendationTestDataset(test_users, limited_unseen_items)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
        )
    
    '''
    def val_recommendation_dataloader(self, batch_size: int = 1024):
        """
        DataLoader especial para evaluación de recomendaciones top-K en validación.
        """
        val_users = list(self.user_unseen_items.keys())
        dataset = RecommendationTestDataset(val_users, self.user_unseen_items)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
        )
    '''
    
    def get_recommendation_info(self) -> Dict:
        """
        Devuelve información necesaria para calcular métricas de recomendación.
        """
        return {
            "user_unseen_items": self.user_unseen_items,
            "test_relevant_items": self.test_relevant_items,
            "num_users": self.num_users,
            "num_items": self.num_items
        }
        