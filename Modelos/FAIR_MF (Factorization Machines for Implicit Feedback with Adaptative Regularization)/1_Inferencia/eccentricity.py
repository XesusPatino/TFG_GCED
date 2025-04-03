# Load ratings and users data
import pandas as pd
import seaborn as sns

SPLIT = 1

train_ratings = pd.read_csv(f"C:/Users/xpati/Documents/TFG/Data_Fair_MF/ml-1m/splits/train_{SPLIT}.csv")
test_ratings = pd.read_csv(f"C:/Users/xpati/Documents/TFG/Data_Fair_MF/ml-1m/splits/test_{SPLIT}.csv")

users = pd.read_csv(
    "C:/Users/xpati/Documents/TFG/Data_Fair_MF/ml-1m/users.dat", sep="::", engine="python", header=None
)
users.columns = ["user_id", "g", "age", "occupation", "zip_code"]

# Compute the average rating per user and item

user_avg_rating = train_ratings.groupby("user_id")["rating"].mean()
item_avg_rating = train_ratings.groupby("item_id")["rating"].mean()

# For each rating, compute the difference between the rating and thse user's average rating and the item's average rating

test_ratings["user_avg_rating"] = test_ratings["user_id"].map(user_avg_rating)
test_ratings["item_avg_rating"] = test_ratings["item_id"].map(item_avg_rating)

test_ratings["expected_rating"] = (
    test_ratings["user_avg_rating"] + test_ratings["item_avg_rating"]
) / 2
test_ratings["eccentricity"] = abs(
    test_ratings["rating"] - test_ratings["expected_rating"]
)

m_users = users[users["g"] == "M"]["user_id"]
f_users = users[users["g"] == "F"]["user_id"]

m_eccentricities = test_ratings[test_ratings["user_id"].isin(m_users)]["eccentricity"]
f_eccentricities = test_ratings[test_ratings["user_id"].isin(f_users)]["eccentricity"]

import matplotlib.pyplot as plt

sns.kdeplot(
    m_eccentricities,
    label="M users",
    color="blue",
    shade=True,
    common_grid=True,
    bw_adjust=0.25,
)
sns.kdeplot(
    f_eccentricities,
    label="F users",
    color="purple",
    shade=True,
    common_grid=True,
    bw_adjust=0.25,
)

plt.xlabel("Eccentricity")
plt.ylabel("Density")
plt.legend()
plt.show()
