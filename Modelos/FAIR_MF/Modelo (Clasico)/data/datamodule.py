from typing import Literal, Union
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
        # Resetear índices para asegurar que empiezan en 0 y son secuenciales
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
            torch.tensor(row["user_id"], dtype=torch.long),  # Asegurar tipo long
            torch.tensor(row["item_id"], dtype=torch.long),  # Asegurar tipo long
            torch.tensor(row["rating"], dtype=torch.float32),
            torch.tensor(row["gender"], dtype=torch.long)    # Asegurar tipo long
        )
        
'''        
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
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 1. Cargar datos y asignar géneros correctamente
        self.train_df = pd.read_csv(f"{data_dir}/{dataset_name}/splits/train_{split}.csv")
        self.test_df = pd.read_csv(f"{data_dir}/{dataset_name}/splits/test_{split}.csv")
        
        # Cargar información de género desde users.dat
        gender_data = pd.read_csv(
            f"{data_dir}/{dataset_name}/users.dat",
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

        # 2. Dividir train en train y validation estratificado por género
        self.train_df, self.val_df = train_test_split(
            self.train_df,
            test_size=0.2,
            stratify=self.train_df["gender"],
            random_state=42
        )

        # 3. Resto de inicializaciones
        self.data = pd.concat([self.train_df, self.val_df, self.test_df])
        self.num_users = self.data["user_id"].max() + 1
        self.num_items = self.data["item_id"].max() + 1

        # Calcular estadísticas de ratings
        self.min_rating = self.data["rating"].min()
        self.max_rating = self.data["rating"].max()

        if verbose:
            print(f"#Users: {self.data['user_id'].nunique()}")
            print(f"#Items: {self.data['item_id'].nunique()}")
            print(f"Min rating: {self.min_rating:.3f}")
            print(f"Max rating: {self.max_rating:.3f}")
            print("\nDistribución de género:")
            print(f"Train - M: {(self.train_df['gender'] == 1).sum()}, F: {(self.train_df['gender'] == 0).sum()}")
            print(f"Val - M: {(self.val_df['gender'] == 1).sum()}, F: {(self.val_df['gender'] == 0).sum()}")
            print(f"Test - M: {(self.test_df['gender'] == 1).sum()}, F: {(self.test_df['gender'] == 0).sum()}")
'''

class DyadicRegressionDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        split=None,  # Hacemos que split sea opcional
        data_dir: str = "C:/Users/xpati/Documents/TFG/Data_Fair_MF",
        batch_size=64,
        num_workers=4,
        verbose=False,
        use_all_data=False,  # Añadir un flag para usar todos los datos
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_all_data = use_all_data

        if use_all_data:
            # Cargar todos los datos en lugar de splits
            self.all_data = pd.read_csv(f"{data_dir}/{dataset_name}/ratings.dat", 
                                       sep="::", 
                                       engine="python",
                                       header=None,
                                       names=["user_id", "item_id", "rating", "timestamp"])
            
            # Cargar información de género desde users.dat
            gender_data = pd.read_csv(
                f"{data_dir}/{dataset_name}/users.dat",
                sep="::",
                engine="python",
                header=None,
                names=["user_id", "gender", "age", "occupation", "zip_code"]
            )
            gender_data["gender"] = gender_data["gender"].map({"M": 1, "F": 0})

            # Unir información de género
            self.all_data = self.all_data.merge(
                gender_data[["user_id", "gender"]], on="user_id", how="left"
            )
            
            # Dividir en train, val y test manualmente (70% train, 10% val, 20% test)
            # estratificado por género
            train_val_df, self.test_df = train_test_split(
                self.all_data,
                test_size=0.2,
                stratify=self.all_data["gender"],
                random_state=42
            )
            
            self.train_df, self.val_df = train_test_split(
                train_val_df,
                test_size=0.125,  # 0.125 * 0.8 = 0.1 (10% del total)
                stratify=train_val_df["gender"],
                random_state=42
            )
        else:
            # Mantener el código original para los splits
            if split is None:
                raise ValueError("El parámetro 'split' debe ser especificado cuando use_all_data=False")
                
            self.train_df = pd.read_csv(f"{data_dir}/{dataset_name}/splits/train_{split}.csv")
            self.test_df = pd.read_csv(f"{data_dir}/{dataset_name}/splits/test_{split}.csv")
            
            # Cargar información de género desde users.dat
            gender_data = pd.read_csv(
                f"{data_dir}/{dataset_name}/users.dat",
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

        # Combinamos todos los datos para calcular estadísticas
        self.data = pd.concat([self.train_df, self.val_df, self.test_df])
        self.num_users = self.data["user_id"].max() + 1
        self.num_items = self.data["item_id"].max() + 1

        # Calcular estadísticas de ratings
        self.min_rating = self.data["rating"].min()
        self.max_rating = self.data["rating"].max()

        if verbose:
            print(f"#Users: {self.data['user_id'].nunique()}")
            print(f"#Items: {self.data['item_id'].nunique()}")
            print(f"Min rating: {self.min_rating:.3f}")
            print(f"Max rating: {self.max_rating:.3f}")
            print("\nDistribución de género:")
            print(f"Train - M: {(self.train_df['gender'] == 1).sum()}, F: {(self.train_df['gender'] == 0).sum()}")
            print(f"Val - M: {(self.val_df['gender'] == 1).sum()}, F: {(self.val_df['gender'] == 0).sum()}")
            print(f"Test - M: {(self.test_df['gender'] == 1).sum()}, F: {(self.test_df['gender'] == 0).sum()}")
            
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
            DyadicRegressionDataset(self.val_df),  # Usar val_df en lugar de test_df
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            DyadicRegressionDataset(self.test_df),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )