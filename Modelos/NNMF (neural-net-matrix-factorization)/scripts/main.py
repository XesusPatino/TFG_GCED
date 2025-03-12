from __future__ import absolute_import, print_function
import argparse
import json
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tensorflow as tf
import pandas as pd
import numpy as np
from nnmf.models import NNMF
from nnmf.utils import chunk_df

# Importar CodeCarbon para medir emisiones de CO₂
from codecarbon import EmissionsTracker


def load_data(train_filename, valid_filename, test_filename, delimiter='\t', col_names=['user_id', 'item_id', 'rating']):
    train_data = pd.read_csv(train_filename, delimiter=delimiter, header=None, names=col_names)
    train_data['user_id'] -= 1
    train_data['item_id'] -= 1
    valid_data = pd.read_csv(valid_filename, delimiter=delimiter, header=None, names=col_names)
    valid_data['user_id'] -= 1
    valid_data['item_id'] -= 1
    test_data = pd.read_csv(test_filename, delimiter=delimiter, header=None, names=col_names)
    test_data['user_id'] -= 1
    test_data['item_id'] -= 1

    return train_data, valid_data, test_data


def train(model, train_data, valid_data, batch_size, max_epochs, use_early_stop, early_stop_max_epoch):
    train_rmse = model.eval_rmse(train_data)
    valid_rmse = model.eval_rmse(valid_data)
    print(f"[start] Train RMSE: {train_rmse:.3f}; Valid RMSE: {valid_rmse:.3f}")

    prev_valid_rmse = float("inf")
    early_stop_epochs = 0

    for epoch in range(max_epochs):
        shuffled_df = train_data.sample(frac=1)
        batches = chunk_df(shuffled_df, batch_size) if batch_size else [train_data]

        for batch in batches:
            user_ids = tf.convert_to_tensor(batch['user_id'], dtype=tf.int32)
            item_ids = tf.convert_to_tensor(batch['item_id'], dtype=tf.int32)
            ratings = tf.convert_to_tensor(batch['rating'], dtype=tf.float32)

            model.train_iteration(user_ids, item_ids, ratings)
            train_rmse = model.eval_rmse(batch)
            valid_rmse = model.eval_rmse(valid_data)
            print(f"[{epoch}] Train RMSE: {train_rmse:.3f}; Valid RMSE: {valid_rmse:.3f}")

        if use_early_stop:
            early_stop_epochs += 1
            if valid_rmse < prev_valid_rmse:
                prev_valid_rmse = valid_rmse
                early_stop_epochs = 0
                model.save_weights(model.model_filename)
            elif early_stop_epochs == early_stop_max_epoch:
                print("Early stopping...")
                break


def test(model, test_data):
    test_rmse = model.eval_rmse(test_data)
    print(f"Final test RMSE: {test_rmse:.3f}")
    return test_rmse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains/evaluates NNMF models.')
    parser.add_argument('--model', type=str, choices=['NNMF', 'SVINNMF'], required=True)
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True)
    parser.add_argument('--train', type=str, default='data/train.csv')
    parser.add_argument('--valid', type=str, default='data/valid.csv')
    parser.add_argument('--test', type=str, default='data/test.csv')
    parser.add_argument('--users', type=int, default=943)
    parser.add_argument('--movies', type=int, default=1682)
    parser.add_argument('--batch', type=int, default=25000)
    parser.add_argument('--max-epochs', type=int, default=1000)
    parser.add_argument('--no-early', action='store_true')
    parser.add_argument('--early-stop-max-epoch', type=int, default=40)
    parser.add_argument('--model-params', type=str, default='{}')

    args = parser.parse_args()

    model_params = json.loads(args.model_params)
    use_early_stop = not args.no_early

    if args.model == 'NNMF':
        model = NNMF(args.users, args.movies, **model_params)
    else:
        raise NotImplementedError(f"Model '{args.model}' not implemented")

    train_data, valid_data, test_data = load_data(args.train, args.valid, args.test)

    if args.mode == 'train':
        # Iniciar el tracker de CodeCarbon antes de entrenar
        tracker = EmissionsTracker(project_name="NNMF_training")
        tracker.start()
        train(model, train_data, valid_data, args.batch, args.max_epochs, use_early_stop, args.early_stop_max_epoch)
        tracker.stop()
    elif args.mode == 'test':
        test(model, test_data)
