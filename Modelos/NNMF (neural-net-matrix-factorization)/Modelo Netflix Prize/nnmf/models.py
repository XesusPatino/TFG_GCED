#!/usr/bin/env python3
from __future__ import absolute_import, print_function

"""Defines NNMF models."""

# Third party modules
import tensorflow as tf
import tensorflow_probability as tfp

# Local modules
from .utils import KL, build_mlp, get_kl_weight

# Standard modules
import os
import numpy as np


class _NNMFBase(tf.keras.Model):

    def __init__(self, num_users, num_items, D=10, Dprime=60, hidden_units_per_layer=50,
                 latent_normal_init_params={'mean': 0.0, 'stddev': 0.1}, model_filename='model/nnmf.ckpt'):
        super(_NNMFBase, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.D = D
        self.Dprime = Dprime
        self.hidden_units_per_layer = hidden_units_per_layer
        self.latent_normal_init_params = latent_normal_init_params
        self.model_filename = model_filename

        # Internal counter to keep track of current epoch
        self._epochs = 0

        # Placeholders (no se usan en modo eager)
        self.user_index = tf.keras.Input(shape=(), dtype=tf.int32)
        self.item_index = tf.keras.Input(shape=(), dtype=tf.int32)
        self.rating = tf.keras.Input(shape=(), dtype=tf.float32)

        # Inicializar variables y operaciones
        self._init_vars()
        self._init_ops()

    def _init_vars(self):
        raise NotImplementedError

    def _init_ops(self):
        raise NotImplementedError

    def train_iteration(self, user_ids, item_ids, ratings):
        # Método base (se sobrescribe en NNMF)
        self.optimizer.minimize(lambda: self.loss, var_list=self.trainable_vars)

        with tf.GradientTape() as tape:
            predictions = self([user_ids, item_ids], training=True)
            loss = self.compiled_loss(ratings, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self._epochs += 1

    def eval_rmse(self, data):
        user_ids = data['user_id']
        item_ids = data['item_id']
        ratings = data['rating']
        predictions = self([user_ids, item_ids], training=False)
        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(predictions, ratings))))
    
    def eval_mae(self, data):
        user_ids = data['user_id']
        item_ids = data['item_id']
        ratings = data['rating']
        predictions = self([user_ids, item_ids], training=False)
        return tf.reduce_mean(tf.abs(tf.subtract(predictions, ratings)))

    def predict(self, user_id, item_id):
        rating = self([user_id], [item_id], training=False)
        return rating[0]

    def call(self, inputs, training=False):
        user_ids, item_ids = inputs
        # Procesar las entradas usando las capas definidas en _init_vars
        U_lu = self.U_layer(user_ids)
        Uprime_lu = self.Uprime_layer(user_ids)
        V_lu = self.V_layer(item_ids)
        Vprime_lu = self.Vprime_layer(item_ids)
        f_input = tf.concat([U_lu, V_lu, tf.multiply(Uprime_lu, Vprime_lu)], axis=1)
        _r = self.mlp_layers(f_input)
        return tf.squeeze(_r, axis=1)


class NNMF(_NNMFBase):
    def __init__(self, *args, **kwargs):
        if 'lam' in kwargs:
            self.lam = float(kwargs['lam'])
            del kwargs['lam']
        else:
            self.lam = 0.01
        super(NNMF, self).__init__(*args, **kwargs)

    def _init_vars(self):
        # Crear las capas de embedding asignando un nombre a cada una
        self.U_layer = EmbeddingLayer(self.num_users, self.D, name="U_layer")
        self.Uprime_layer = EmbeddingLayer(self.num_users, self.Dprime, name="Uprime_layer")
        self.V_layer = EmbeddingLayer(self.num_items, self.D, name="V_layer")
        self.Vprime_layer = EmbeddingLayer(self.num_items, self.Dprime, name="Vprime_layer")

        # Definir la MLP como un Sequential, asignando nombres a las capas internas
        self.mlp_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_units_per_layer, activation='sigmoid', name="dense_1"),
            tf.keras.layers.Dense(self.hidden_units_per_layer, activation='sigmoid', name="dense_2"),
            tf.keras.layers.Dense(1, name="dense_output")
        ], name="mlp_layers")

        # r_target no se utiliza en el entrenamiento (se usan los ratings del batch)
        self.r_target = tf.Variable(tf.zeros([1]), trainable=False)

    def train_iteration(self, user_ids, item_ids, ratings):
        user_ids = tf.convert_to_tensor(user_ids, dtype=tf.int32)
        item_ids = tf.convert_to_tensor(item_ids, dtype=tf.int32)
        ratings = tf.convert_to_tensor(ratings, dtype=tf.float32)
        with tf.GradientTape() as tape:
            predictions = self([user_ids, item_ids], training=True)
            reconstruction_loss = tf.reduce_sum(tf.square(tf.subtract(ratings, predictions)))
            reg = tf.add_n([
                tf.reduce_sum(tf.square(self.Uprime_layer.embeddings)),
                tf.reduce_sum(tf.square(self.U_layer.embeddings)),
                tf.reduce_sum(tf.square(self.V_layer.embeddings)),
                tf.reduce_sum(tf.square(self.Vprime_layer.embeddings))
            ])
            loss = reconstruction_loss + (self.lam * reg)
        gradients = tape.gradient(loss, self.trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_vars))
        self._epochs += 1

    def _init_ops(self):
        self.optimizer = tf.keras.optimizers.Adam()
        # Forzar la construcción de todas las capas con datos dummy
        dummy_user_ids = tf.zeros((1,), dtype=tf.int32)
        dummy_item_ids = tf.zeros((1,), dtype=tf.int32)
        _ = self([dummy_user_ids, dummy_item_ids])
        self.trainable_vars = [
            self.U_layer.embeddings, self.Uprime_layer.embeddings,
            self.V_layer.embeddings, self.Vprime_layer.embeddings,
            *self.mlp_layers.trainable_variables
        ]
    
    def save_weights(self, filepath):
        """Save model weights to file"""
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
                
        # Save embeddings and weights
        weights_dict = {
            'U_layer': self.U_layer.embeddings.numpy(),
            'V_layer': self.V_layer.embeddings.numpy(),
            'Uprime_layer': self.Uprime_layer.embeddings.numpy(),
            'Vprime_layer': self.Vprime_layer.embeddings.numpy(),
        }
        
        # Save MLP weights separately (not as part of weights_dict)
        # This avoids broadcasting issues
        np.savez(filepath, **weights_dict)
        
        # Save MLP weights to a separate file
        mlp_weights_path = filepath.replace('.npz', '_mlp_weights.npz')
        mlp_weights_list = [layer.get_weights() for layer in self.mlp_layers.layers]
        np.save(mlp_weights_path, mlp_weights_list, allow_pickle=True)
        
        return filepath


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        # Permitir pasar el nombre a la capa
        super(EmbeddingLayer, self).__init__(**kwargs)
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(num_embeddings, embedding_dim),
            initializer='random_normal',
            trainable=True
        )

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.embeddings, inputs)