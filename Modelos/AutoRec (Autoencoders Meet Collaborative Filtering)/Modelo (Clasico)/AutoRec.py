import tensorflow as tf
import time
import numpy as np
import os
import math

class AutoRec():
    def __init__(self, args, num_users, num_items, R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,
                 num_train_ratings, num_test_ratings, user_train_set, item_train_set, user_test_set, item_test_set, result_path):

        self.args = args
        self.num_users = num_users
        self.num_items = num_items

        self.R = R
        self.mask_R = mask_R
        self.C = C
        self.train_R = train_R
        self.train_mask_R = train_mask_R
        self.test_R = test_R
        self.test_mask_R = test_mask_R
        self.num_train_ratings = num_train_ratings
        self.num_test_ratings = num_test_ratings

        self.user_train_set = user_train_set
        self.item_train_set = item_train_set
        self.user_test_set = user_test_set
        self.item_test_set = item_test_set

        self.hidden_neuron = args.hidden_neuron
        self.train_epoch = args.train_epoch
        self.batch_size = args.batch_size
        self.num_batch = int(math.ceil(self.num_users / float(self.batch_size)))

        self.base_lr = args.base_lr
        self.optimizer_method = args.optimizer_method
        self.display_step = args.display_step
        self.random_seed = args.random_seed
        self.lambda_value = args.lambda_value
        self.grad_clip = args.grad_clip
        self.result_path = result_path

        self.train_cost_list = []
        self.test_cost_list = []
        self.test_rmse_list = []

        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.base_lr)

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(self.hidden_neuron, activation='sigmoid', input_shape=(self.num_items,)))
        model.add(tf.keras.layers.Dense(self.num_items))
        return model

    def loss_function(self, input_R, mask_R, output):
        rec_loss = tf.square(tf.multiply(input_R - output, mask_R))
        rec_loss = tf.reduce_sum(rec_loss)
        reg_loss = self.lambda_value * 0.5 * tf.reduce_sum([tf.nn.l2_loss(v) for v in self.model.trainable_variables])
        return rec_loss + reg_loss

    def train_step(self, input_R, mask_R):
        with tf.GradientTape() as tape:
            output = self.model(input_R)
            loss = self.loss_function(input_R, mask_R, output)
        gradients = tape.gradient(loss, self.model.trainable_variables)

        if self.grad_clip:
            gradients = [tf.clip_by_value(g, -5., 5.) for g in gradients]

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self):
        for epoch in range(self.train_epoch):
            start_time = time.time()
            total_loss = 0

            for i in range(self.num_batch):
                batch_idx = np.random.choice(self.num_users, self.batch_size, replace=False)
                loss = self.train_step(self.train_R[batch_idx], self.train_mask_R[batch_idx])
                total_loss += loss.numpy()

            self.train_cost_list.append(total_loss)

            if (epoch + 1) % self.display_step == 0:
                print(f"Epoch {epoch+1} | Loss: {total_loss:.2f} | Time: {int(time.time() - start_time)}s")

    def run(self):
        self.train()