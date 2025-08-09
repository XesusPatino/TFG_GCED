import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
ml1m_dir = "C:/Users/xpati/Documents/TFG/Bases_Extra/NETFLIX"
output_dir = os.path.join(ml1m_dir, "split")
ratings_file = os.path.join(ml1m_dir, "netflix.csv")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load ratings data
print("Loading ratings data...")
# Modificado para manejar correctamente el encabezado
ratings = pd.read_csv(
    ratings_file,
    sep=",",
    header=0,  # Especifica que el archivo tiene encabezado
    encoding="utf-8"
)

# Asegúrate de que los nombres de las columnas sean correctos
if not all(col in ratings.columns for col in ["user_id", "item_id", "rating"]):
    # Si el archivo tiene diferentes nombres de columnas, renómbralas
    ratings.columns = ["user_id", "item_id", "rating"]

# Print original stats
print(f"Original data - users: {ratings['user_id'].nunique()}, items: {ratings['item_id'].nunique()}")
print(f"Min user_id: {ratings['user_id'].min()}, Max user_id: {ratings['user_id'].max()}")
print(f"Min item_id: {ratings['item_id'].min()}, Max item_id: {ratings['item_id'].max()}")

# Important: Enforce consistent user IDs and item IDs
# Remap all IDs to ensure they're contiguous from 0 to n-1
user_id_map = {old_id: new_id for new_id, old_id in enumerate(ratings['user_id'].unique())}
item_id_map = {old_id: new_id for new_id, old_id in enumerate(ratings['item_id'].unique())}

# Apply the mapping
ratings['user_id'] = ratings['user_id'].map(user_id_map)
ratings['item_id'] = ratings['item_id'].map(item_id_map)

# Verify all IDs are properly mapped
num_users = len(user_id_map)
num_items = len(item_id_map)
print(f"Remapped data - users: {num_users}, items: {num_items}")
print(f"New user_id range: {ratings['user_id'].min()} to {ratings['user_id'].max()}")
print(f"New item_id range: {ratings['item_id'].min()} to {ratings['item_id'].max()}")

# Check for proper ID range
assert ratings['user_id'].min() == 0, "User IDs don't start at 0"
assert ratings['user_id'].max() == num_users - 1, "User IDs aren't contiguous"
assert ratings['item_id'].min() == 0, "Item IDs don't start at 0"
assert ratings['item_id'].max() == num_items - 1, "Item IDs aren't contiguous"

# Split data: 80% train+valid, 20% test
train_valid, test = train_test_split(ratings, test_size=0.2, random_state=42)

# Split train+valid: 90% train, 10% valid (which is 72%, 8% of the total)
train, valid = train_test_split(train_valid, test_size=0.1, random_state=42)

# Create a small training set for testing
train_small = train.sample(frac=0.1, random_state=42)

# Save splits
print("Saving splits to files...")
train.to_csv(os.path.join(output_dir, "u.data.train"), sep="\t", header=False, index=False)
valid.to_csv(os.path.join(output_dir, "u.data.valid"), sep="\t", header=False, index=False)
test.to_csv(os.path.join(output_dir, "u.data.test"), sep="\t", header=False, index=False)
train_small.to_csv(os.path.join(output_dir, "u.data.train.small"), sep="\t", header=False, index=False)

# Save the mapping for reference
pd.Series(user_id_map).to_csv(os.path.join(output_dir, "user_id_mapping.csv"), header=['original_id'])
pd.Series(item_id_map).to_csv(os.path.join(output_dir, "item_id_mapping.csv"), header=['original_id'])

# Print stats
print(f"Total ratings: {len(ratings)}")
print(f"Train: {len(train)} ({len(train)/len(ratings):.1%})")
print(f"Valid: {len(valid)} ({len(valid)/len(ratings):.1%})")
print(f"Test: {len(test)} ({len(test)/len(ratings):.1%})")
print(f"Train-small: {len(train_small)} ({len(train_small)/len(ratings):.1%})")

# Check final data ranges
print(f"Final user_id range: {min([train['user_id'].min(), valid['user_id'].min(), test['user_id'].min()])} to " 
      f"{max([train['user_id'].max(), valid['user_id'].max(), test['user_id'].max()])}")
print(f"Final item_id range: {min([train['item_id'].min(), valid['item_id'].min(), test['item_id'].min()])} to "
      f"{max([train['item_id'].max(), valid['item_id'].max(), test['item_id'].max()])}")

# Save configuration for training script
with open(os.path.join(output_dir, "config.txt"), "w") as f:
    f.write(f"users={num_users}\n")
    f.write(f"movies={num_items}\n")

print("Done!")