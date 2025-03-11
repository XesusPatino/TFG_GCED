import numpy as np

def read_rating(path, num_users, num_items, num_total_ratings, a, b, train_ratio):
    user_train_set, user_test_set = set(), set()
    item_train_set, item_test_set = set(), set()

    R = np.zeros((num_users, num_items))
    mask_R = np.zeros((num_users, num_items))
    C = np.ones((num_users, num_items)) * b

    train_R = np.zeros_like(R)
    test_R = np.zeros_like(R)
    train_mask_R = np.zeros_like(mask_R)
    test_mask_R = np.zeros_like(mask_R)

    with open(path + "ratings.dat", "r") as fp:
        lines = fp.readlines()

    random_perm_idx = np.random.permutation(num_total_ratings)
    train_idx = random_perm_idx[:int(num_total_ratings * train_ratio)]
    test_idx = random_perm_idx[int(num_total_ratings * train_ratio):]

    for line in lines:
        user, item, rating, _ = line.split("::")
        user_idx, item_idx = int(user) - 1, int(item) - 1
        R[user_idx, item_idx] = int(rating)
        mask_R[user_idx, item_idx] = 1
        C[user_idx, item_idx] = a

    return R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R, len(train_idx), len(test_idx), user_train_set, item_train_set, user_test_set, item_test_set
