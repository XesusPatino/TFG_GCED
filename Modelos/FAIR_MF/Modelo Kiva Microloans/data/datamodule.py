from typing import Literal, Union
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torch
from sklearn.model_selection import train_test_split


class DyadicRegressionDataset(Dataset):
    def __init__(self, df, has_gender=True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the dataset
                Must contain the columns ['user_id', 'item_id', 'rating']
                Optionally contains 'gender' column if has_gender=True
            has_gender (bool): Whether the dataset includes gender information
        """
        # Resetear índices para asegurar que empiezan en 0 y son secuenciales
        self.data = df.reset_index(drop=True)
        self.has_gender = has_gender
        
        # Convertir tipos de datos
        self.data["user_id"] = self.data["user_id"].astype(np.int64)
        self.data["item_id"] = self.data["item_id"].astype(np.int64)
        self.data["rating"] = self.data["rating"].astype(np.float32)
        
        if self.has_gender and "gender" in self.data.columns:
            self.data["gender"] = self.data["gender"].astype(np.int64)
            self.data = self.data[["user_id", "item_id", "rating", "gender"]]
        else:
            self.has_gender = False
            self.data = self.data[["user_id", "item_id", "rating"]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        if self.has_gender:
            return (
                torch.tensor(row["user_id"], dtype=torch.long),
                torch.tensor(row["item_id"], dtype=torch.long),
                torch.tensor(row["rating"], dtype=torch.float32),
                torch.tensor(row["gender"], dtype=torch.long)
            )
        else:
            return (
                torch.tensor(row["user_id"], dtype=torch.long),
                torch.tensor(row["item_id"], dtype=torch.long),
                torch.tensor(row["rating"], dtype=torch.float32),
                torch.tensor(0, dtype=torch.long)  # Dummy gender value
            )


class DyadicRegressionDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        split=None,
        data_dir: str = "C:/Users/xpati/Documents/TFG",
        batch_size=64,
        num_workers=4,
        verbose=False,
        use_all_data=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_all_data = use_all_data
        self.dataset_name = dataset_name
        
        # Detectar si tenemos datos de género disponibles
        self.has_gender = self._check_gender_availability()
        
        if self.has_gender:
            self._load_data_with_gender(split)
        else:
            self._load_data_without_gender()
            
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
            print(f"Has gender data: {self.has_gender}")
            
            if self.has_gender:
                print("\nDistribución de género:")
                print(f"Train - M: {(self.train_df['gender'] == 1).sum()}, F: {(self.train_df['gender'] == 0).sum()}")
                print(f"Val - M: {(self.val_df['gender'] == 1).sum()}, F: {(self.val_df['gender'] == 0).sum()}")
                print(f"Test - M: {(self.test_df['gender'] == 1).sum()}, F: {(self.test_df['gender'] == 0).sum()}")
    
    def _check_gender_availability(self):
        """Check if gender data is available for the dataset"""
        try:
            # Try to load users.dat file
            users_file = f"{self.data_dir}/{self.dataset_name}/users.dat"
            gender_data = pd.read_csv(users_file, sep="::", engine="python", header=None, nrows=1)
            return True
        except:
            # Check if the main data file has gender column
            try:
                if self.dataset_name == "CTRPV":
                    # For CTRPV, check the main CSV file
                    data_file = f"{self.data_dir}/ctrpv2_processed.csv"
                    sample_data = pd.read_csv(data_file, nrows=1)
                    return "gender" in sample_data.columns
                elif self.dataset_name == "KIVA_MICROLOANS":
                    # For Kiva Microloans, check the main CSV file
                    data_file = f"{self.data_dir}/kiva_ml17.csv"
                    sample_data = pd.read_csv(data_file, nrows=1)
                    return "gender" in sample_data.columns
                else:
                    return False
            except:
                return False
    
    def _load_data_with_gender(self, split):
        """Load data when gender information is available"""
        if self.use_all_data:
            # Cargar todos los datos en lugar de splits
            self.all_data = pd.read_csv(f"{self.data_dir}/{self.dataset_name}/ratings.dat", 
                                       sep="::", 
                                       engine="python",
                                       header=None,
                                       names=["user_id", "item_id", "rating", "timestamp"])
            
            # Cargar información de género desde users.dat
            gender_data = pd.read_csv(
                f"{self.data_dir}/{self.dataset_name}/users.dat",
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
                
            self.train_df = pd.read_csv(f"{self.data_dir}/{self.dataset_name}/splits/train_{split}.csv")
            self.test_df = pd.read_csv(f"{self.data_dir}/{self.dataset_name}/splits/test_{split}.csv")
            
            # Cargar información de género desde users.dat
            gender_data = pd.read_csv(
                f"{self.data_dir}/{self.dataset_name}/users.dat",
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
    
    def _load_data_without_gender(self):
        """Load data when no gender information is available (like CTRPV)"""
        # Load the main data file
        if self.dataset_name == "CTRPV":
            data_file = f"{self.data_dir}/ctrpv2_processed.csv"
        elif self.dataset_name == "KIVA_MICROLOANS":
            data_file = f"{self.data_dir}/kiva_ml17.csv"
        else:
            data_file = f"{self.data_dir}/{self.dataset_name}/ratings.csv"
            
        try:
            self.all_data = pd.read_csv(data_file)
            print(f" Loaded data from: {data_file}")
            print(f" Data shape: {self.all_data.shape}")
            print(f" Columns: {list(self.all_data.columns)}")
            
            # For Kiva Microloans, handle string user_ids by creating a mapping
            if self.dataset_name == "KIVA_MICROLOANS":
                # Create mapping from string user_ids to integers
                unique_users = self.all_data['user_id'].unique()
                user_mapping = {user: idx for idx, user in enumerate(unique_users)}
                
                # Apply mapping
                self.all_data['user_id'] = self.all_data['user_id'].map(user_mapping)
                
                print(f"   Mapped {len(unique_users)} unique string user_ids to integers")
                
                # Show some additional statistics
                additional_cols = [col for col in self.all_data.columns if col not in ['user_id', 'item_id', 'rating']]
                if additional_cols:
                    print(f"   Additional columns available: {additional_cols}")
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"No se pudo encontrar el archivo de datos: {data_file}")
        
        # Ensure required columns exist
        required_cols = ["user_id", "item_id", "rating"]
        missing_cols = [col for col in required_cols if col not in self.all_data.columns]
        if missing_cols:
            raise ValueError(f"Faltan las columnas requeridas: {missing_cols}")
        
        # Split data into train, validation, and test (70%, 10%, 20%)
        train_val_df, self.test_df = train_test_split(
            self.all_data,
            test_size=0.2,
            random_state=42
        )
        
        self.train_df, self.val_df = train_test_split(
            train_val_df,
            test_size=0.125,  # 0.125 * 0.8 = 0.1 (10% del total)
            random_state=42
        )
        
        print(f" Data split complete:")
        print(f" Train: {len(self.train_df)} samples")
        print(f" Val: {len(self.val_df)} samples") 
        print(f" Test: {len(self.test_df)} samples")
    
    def get_class_weights(self):
        """Return class weights for fairness training, or None if no gender data"""
        if not self.has_gender:
            return None
            
        gender_counts = self.train_df['gender'].value_counts()
        total = len(self.train_df)
        weights = {0: total / (2 * gender_counts[0]), 1: total / (2 * gender_counts[1])}
        return weights
        
    def train_dataloader(self):
        return DataLoader(
            DyadicRegressionDataset(self.train_df, self.has_gender),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            DyadicRegressionDataset(self.val_df, self.has_gender),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            DyadicRegressionDataset(self.test_df, self.has_gender),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )