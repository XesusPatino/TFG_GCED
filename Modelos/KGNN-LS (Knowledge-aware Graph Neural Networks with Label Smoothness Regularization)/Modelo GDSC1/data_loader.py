import numpy as np
import pandas as pd
import os


def load_data(args):
    n_user, n_item, train_data, eval_data, test_data = load_rating(args)
    n_entity, n_relation, adj_entity, adj_relation = load_kg(args)
    print('data loaded.')

    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation


def load_rating(args):
    print('reading rating file ...')

    # Handle the gdsc1 dataset specifically
    if args.dataset == 'gdsc1':
        csv_file = 'C:/Users/xpati/Documents/TFG/gdsc1_processed.csv'
        print(f'Loading CSV file: {csv_file}')
        
        # Load CSV file
        df = pd.read_csv(csv_file)
        
        # Only use the first 3 columns (user_id, item_id, rating)
        df = df[['user_id', 'item_id', 'rating']]
        
        # Check for and handle duplicates
        print(f'Original dataset size: {len(df)}')
        duplicates = df.duplicated(subset=['user_id', 'item_id'], keep=False)
        if duplicates.any():
            print(f'Found {duplicates.sum()} duplicate user-item pairs')
            print('Resolving duplicates by averaging ratings...')
            # Group by user_id and item_id, then average the ratings
            df = df.groupby(['user_id', 'item_id'])['rating'].mean().reset_index()
            print(f'Dataset size after deduplication: {len(df)}')
        
        # Remap user and item IDs to be consecutive starting from 0
        print('Remapping user and item IDs to consecutive integers...')
        
        # Create mappings
        unique_users = sorted(df['user_id'].unique())
        unique_items = sorted(df['item_id'].unique())
        
        user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
        item_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
        
        # Apply mappings
        df['user_id'] = df['user_id'].map(user_map)
        df['item_id'] = df['item_id'].map(item_map)
        
        print(f'Remapped {len(unique_users)} users (0-{len(unique_users)-1}) and {len(unique_items)} items (0-{len(unique_items)-1})')
        
        # Store mappings in args for potential later use
        args.user_map = user_map
        args.item_map = item_map
        args.reverse_user_map = {v: k for k, v in user_map.items()}
        args.reverse_item_map = {v: k for k, v in item_map.items()}
        
        # Convert to numpy array
        rating_np = df[['user_id', 'item_id', 'rating']].values
        
        # Ensure proper data types
        rating_np[:, 0] = rating_np[:, 0].astype(np.int64)  # user_id
        rating_np[:, 1] = rating_np[:, 1].astype(np.int64)  # item_id
        rating_np[:, 2] = rating_np[:, 2].astype(np.float64)  # rating
        
        print(f'Loaded {len(rating_np)} unique ratings')
        print(f'Rating range: {rating_np[:, 2].min():.3f} - {rating_np[:, 2].max():.3f}')
        print(f'User ID range: {rating_np[:, 0].min()} - {rating_np[:, 0].max()}')
        print(f'Item ID range: {rating_np[:, 1].min()} - {rating_np[:, 1].max()}')
        
    # Handle the ctrpv2 dataset specifically
    elif args.dataset == 'ctrpv2':
        csv_file = 'C:/Users/xpati/Documents/TFG/ctrpv2_processed.csv'
        print(f'Loading CSV file: {csv_file}')
        
        # Load CSV file
        df = pd.read_csv(csv_file)
        
        # Convert to numpy array
        rating_np = df[['user_id', 'item_id', 'rating']].values
        
        # Ensure proper data types
        rating_np[:, 0] = rating_np[:, 0].astype(np.int64)  # user_id
        rating_np[:, 1] = rating_np[:, 1].astype(np.int64)  # item_id
        rating_np[:, 2] = rating_np[:, 2].astype(np.float64)  # rating
        
        print(f'Loaded {len(rating_np)} ratings')
        print(f'Rating range: {rating_np[:, 2].min():.3f} - {rating_np[:, 2].max():.3f}')
        
    else:
        # Original code for other datasets
        rating_file = 'C:/Users/xpati/Documents/TFG/ml-1m/data-kgnn-ls/' + args.dataset + '/ratings_final'
        if os.path.exists(rating_file + '.npy'):
            rating_np = np.load(rating_file + '.npy')
        else:
            # Use float64 to handle rating values 1.0-5.0, then convert user/item to int
            rating_np = np.loadtxt(rating_file + '.txt', dtype=np.float64)
            # Convert user and item indices to int64, keep ratings as float
            rating_np[:, 0] = rating_np[:, 0].astype(np.int64)  # user
            rating_np[:, 1] = rating_np[:, 1].astype(np.int64)  # item
            # rating column (2) stays as float
            np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    train_data, eval_data, test_data = dataset_split(rating_np, args)

    return n_user, n_item, train_data, eval_data, test_data


def dataset_split(rating_np, args):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data


def load_kg(args):
    print('reading KG file ...')

    # Handle the gdsc1 dataset - create a knowledge graph with remapped IDs
    if args.dataset == 'gdsc1':
        print('Creating knowledge graph for gdsc1 dataset with remapped IDs...')
        
        # Use the remapped IDs from load_rating function
        # The number of entities should match the number of unique items after remapping
        if hasattr(args, 'item_map'):
            n_items = len(args.item_map)
            print(f'Using {n_items} items with remapped IDs (0 to {n_items-1})')
        else:
            # Fallback: load and remap again (should not happen if load_rating was called first)
            csv_file = 'C:/Users/xpati/Documents/TFG/gdsc1_processed.csv'
            df = pd.read_csv(csv_file)
            unique_items = sorted(df['item_id'].unique())
            n_items = len(unique_items)
            print(f'Fallback: using {n_items} items')
        
        # Create a minimal KG where each item connects to itself
        # Format: [head_entity, relation_type, tail_entity]
        # Use consecutive IDs from 0 to n_items-1
        kg_np = np.array([[item, 0, item] for item in range(n_items)], dtype=np.int64)
        
        n_entity = n_items
        n_relation = 1  # Only one relation type (self-relation)
        
        print(f'Created KG with {n_entity} entities (0-{n_entity-1}) and {n_relation} relation type')
        
    # Handle the ctrpv2 dataset - create a minimal knowledge graph
    elif args.dataset == 'ctrpv2':
        print('Creating minimal knowledge graph for ctrpv2 dataset...')
        
        # For ctrpv2, we'll create a simple identity-based knowledge graph
        # where each item is connected to itself with a self-relation
        # This is a workaround since we don't have external knowledge graph data
        
        # We need to know the number of items first
        csv_file = 'C:/Users/xpati/Documents/TFG/ctrpv2_processed.csv'
        df = pd.read_csv(csv_file)
        unique_items = df['item_id'].unique()
        n_items = len(unique_items)
        
        # Create a minimal KG where each item connects to itself
        # Format: [head_entity, relation_type, tail_entity]
        kg_np = np.array([[item, 0, item] for item in unique_items], dtype=np.int64)
        
        n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
        n_relation = 1  # Only one relation type (self-relation)
        
        print(f'Created minimal KG with {n_entity} entities and {n_relation} relation type')
        
    else:
        # Original code for other datasets
        kg_file = 'C:/Users/xpati/Documents/TFG/ml-1m/data-kgnn-ls/' + args.dataset + '/kg_final'
        if os.path.exists(kg_file + '.npy'):
            kg_np = np.load(kg_file + '.npy')
        else:
            kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
            np.save(kg_file + '.npy', kg_np)

        n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
        n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)
    adj_entity, adj_relation = construct_adj(args, kg, n_entity)

    return n_entity, n_relation, adj_entity, adj_relation


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = dict()
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg


def construct_adj(args, kg, entity_num):
    print('constructing adjacency matrix ...')
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations
    adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    
    for entity in range(entity_num):
        if entity in kg:
            neighbors = kg[entity]
            n_neighbors = len(neighbors)
            if n_neighbors >= args.neighbor_sample_size:
                sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
            else:
                sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
            adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
            adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])
        else:
            # If entity has no neighbors, use self-connections
            adj_entity[entity] = np.full(args.neighbor_sample_size, entity)
            adj_relation[entity] = np.zeros(args.neighbor_sample_size, dtype=np.int64)

    return adj_entity, adj_relation
