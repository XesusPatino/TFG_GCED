import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import pandas as pd
from sklearn.model_selection import train_test_split
import random

class LRML:
    def __init__(self, num_users, num_items, args, mode=1):
        self.num_users = num_users
        self.num_items = num_items
        self.graph = tf.compat.v1.Graph()
        self.args = args
        self.stddev = self.args.std
        self.initializer = tf.compat.v1.random_uniform_initializer(minval=-self.stddev, maxval=self.stddev)
        self.attention = None
        self.selected_memory = None
        self.num_mem = self.args.num_mem
        self.mode = 1

        # Build the network within the graph context
        with self.graph.as_default():
            self._build_network()

    def get_feed_dict(self, pos_batch, neg_batch, mode='training'):
        if neg_batch is not None:
            batch = pos_batch + neg_batch
        else:
            batch = pos_batch

        if mode == 'training':
            random.shuffle(batch)
            if 'PAIR' in self.args.rnn_type:
                user_input = [x[0] for x in pos_batch]
                item_input = [x[1] for x in pos_batch]
                user_input_neg = [x[0] for x in neg_batch]
                item_input_neg = [x[1] for x in neg_batch]
            else:
                user_input = [x[0] for x in batch]
                item_input = [x[1] for x in batch]
        else:
            user_input = [x[0] for x in batch]
            item_input = [x[1] for x in batch]
        labels = [x[2] for x in batch]
        feed_dict = {
            self.user_input: user_input,
            self.item_input: item_input,
            self.dropout: self.args.dropout,
            self.label: labels
        }
        if 'PAIR' in self.args.rnn_type and mode == "training":
            feed_dict[self.user_input_neg] = user_input_neg
            feed_dict[self.item_input_neg] = item_input_neg
        if mode != 'training':
            feed_dict[self.dropout] = 1.0
        feed_dict[self.learn_rate] = self.args.learn_rate
        return feed_dict, labels

    def build_inputs(self):
        with tf.compat.v1.name_scope('user_input'):
            self.user_input = tf.compat.v1.placeholder(tf.compat.v1.int32, shape=[None], name='user')
        with tf.compat.v1.name_scope('item_input'):
            self.item_input = tf.compat.v1.placeholder(tf.compat.v1.int32, shape=[None], name='item')
        with tf.compat.v1.name_scope('user_input_neg'):
            self.user_input_neg = tf.compat.v1.placeholder(tf.compat.v1.int32, shape=[None], name='user_neg')
        with tf.compat.v1.name_scope('item_input_neg'):
            self.item_input_neg = tf.compat.v1.placeholder(tf.compat.v1.int32, shape=[None], name='item_neg')
        with tf.compat.v1.name_scope('dropout'):
            self.dropout = tf.compat.v1.placeholder(tf.compat.v1.float32, name='dropout')
        with tf.compat.v1.name_scope('labels'):
            self.label = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None], name='labels')
        self.learn_rate = tf.compat.v1.placeholder(tf.compat.v1.float32, name='learn_rate')
        self.batch_size = tf.compat.v1.shape(self.user_input)[0]

    def composition_layer(self, user_emb, item_emb, dist='L2', reuse=None, selected_memory=None):
        energy = item_emb - (user_emb + selected_memory)
        if 'L2' in dist:
            final_layer = -tf.compat.v1.sqrt(tf.compat.v1.reduce_sum(tf.compat.v1.square(energy), 1) + 1E-3)
        elif 'L1' in dist:
            final_layer = -tf.compat.v1.reduce_sum(tf.compat.v1.abs(energy), 1)
        else:
            raise Exception('Please specify distance metric')
        final_layer = tf.compat.v1.reshape(final_layer, [-1, 1])
        return final_layer

    def _build_network(self):
        self.build_inputs()
        self.target = tf.compat.v1.expand_dims(self.label, 1)
        stddev = self.stddev

        with tf.compat.v1.variable_scope('embedding_layer'):
            with tf.compat.v1.device('/cpu:0'):
                self.user_embeddings = tf.compat.v1.get_variable('user_emb', [self.num_users + 1, self.args.embedding_size], initializer=self.initializer)
                self.item_embeddings = tf.compat.v1.get_variable('item_emb', [self.num_items + 1, self.args.embedding_size], initializer=self.initializer)
                self.user_emb = tf.compat.v1.nn.embedding_lookup(self.user_embeddings, self.user_input)
                self.item_emb = tf.compat.v1.nn.embedding_lookup(self.item_embeddings, self.item_input)
                if self.args.constraint:
                    self.user_emb = tf.compat.v1.clip_by_norm(self.user_emb, 1.0, axes=1)
                    self.item_emb = tf.compat.v1.clip_by_norm(self.item_emb, 1.0, axes=1)

                if 'PAIR' in self.args.rnn_type:
                    self.user_emb_neg = tf.compat.v1.nn.embedding_lookup(self.user_embeddings, self.user_input_neg)
                    self.item_emb_neg = tf.compat.v1.nn.embedding_lookup(self.item_embeddings, self.item_input_neg)
                    if self.args.constraint:
                        self.user_emb_neg = tf.compat.v1.clip_by_norm(self.user_emb_neg, 1.0, axes=1)
                        self.item_emb_neg = tf.compat.v1.clip_by_norm(self.item_emb_neg, 1.0, axes=1)

        self.user_item_key = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.args.embedding_size, self.num_mem], stddev=stddev))
        self.memories = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.num_mem, self.args.embedding_size], stddev=stddev))
        _key = tf.compat.v1.multiply(self.user_emb, self.item_emb)
        self.key_attention = tf.compat.v1.matmul(_key, self.user_item_key)
        self.key_attention = tf.compat.v1.nn.softmax(self.key_attention)

        if self.mode == 1:
            self.selected_memory = tf.compat.v1.matmul(self.key_attention, self.memories)
        elif self.mode == 2:
            self.key_attention = tf.compat.v1.expand_dims(self.key_attention, 1)
            self.selected_memory = self.key_attention * self.memories
            self.selected_memory = tf.compat.v1.reduce_sum(self.selected_memory, 2)

        self.attention = self.key_attention

        final_layer = self.composition_layer(self.user_emb, self.item_emb, selected_memory=self.selected_memory)
        if 'PAIR' in self.args.rnn_type:
            final_layer_neg = self.composition_layer(self.user_emb_neg, self.item_emb_neg, reuse=True, selected_memory=self.selected_memory)
            self.predict_op_neg = final_layer_neg

        self.predict_op = final_layer

        with tf.compat.v1.name_scope("train"):
            margin = self.args.margin

            if 'PAIR' in self.args.rnn_type:
                self.cost = tf.compat.v1.reduce_sum(tf.compat.v1.nn.relu(margin - final_layer + final_layer_neg))
            else:
                self.cost = tf.compat.v1.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(labels=self.target, logits=final_layer))

            if self.args.l2_reg > 0:
                vars = tf.compat.v1.trainable_variables()
                lossL2 = tf.compat.v1.add_n([tf.compat.v1.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * self.args.l2_reg
                self.cost += lossL2

            if self.args.opt == 'SGD':
                self.opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learn_rate)
            elif self.args.opt == 'Adam':
                self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learn_rate)
            elif self.args.opt == 'Adadelta':
                self.opt = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=self.learn_rate)
            elif self.args.opt == 'Adagrad':
                self.opt = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.learn_rate, initial_accumulator_value=0.9)
            elif self.args.opt == 'RMS':
                self.opt = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.learn_rate, decay=0.9, epsilon=1e-6)
            elif self.args.opt == 'Moment':
                self.opt = tf.compat.v1.train.MomentumOptimizer(self.args.learn_rate, 0.9)

            tvars = tf.compat.v1.trainable_variables()
            gradients = self.opt.compute_gradients(self.cost)
            self.gradients = gradients

            def ClipIfNotNone(grad):
                if grad is None:
                    return grad
                grad = tf.compat.v1.clip_by_value(grad, -10, 10, name=None)
                return tf.compat.v1.clip_by_norm(grad, self.args.clip_norm)

            if self.args.clip_norm > 0:
                clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gradients]
            else:
                clipped_gradients = [(grad, var) for grad, var in gradients]

            self.optimizer = self.opt.apply_gradients(clipped_gradients)
            self.train_op = self.optimizer

        self.post_step = []


# Cargar los datos de MovieLens
ratings = pd.read_csv('C:/Users/xpati/Documents/TFG/ml-latest-small/ratings.csv')

# Preprocesar los datos
user_ids = ratings['userId'].unique()
item_ids = ratings['movieId'].unique()

user_id_map = {user_id: i for i, user_id in enumerate(user_ids)}
item_id_map = {item_id: i for i, item_id in enumerate(item_ids)}

ratings['userId'] = ratings['userId'].map(user_id_map)
ratings['movieId'] = ratings['movieId'].map(item_id_map)

# Dividir los datos en entrenamiento y prueba
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Convertir a listas de tuplas (userId, movieId, rating)
train_data = list(zip(train_data['userId'], train_data['movieId'], train_data['rating']))
test_data = list(zip(test_data['userId'], test_data['movieId'], test_data['rating']))


# Definir los argumentos del modelo
class Args:
    def __init__(self):
        self.std = 0.01
        self.embedding_size = 50
        self.num_mem = 10
        self.dropout = 0.5
        self.margin = 1.0
        self.l2_reg = 0.01
        self.opt = 'Adam'
        self.learn_rate = 0.001
        self.clip_norm = 1.0
        self.constraint = True
        self.rnn_type = 'PAIR'

args = Args()

# Crear el modelo
num_users = len(user_ids)
num_items = len(item_ids)
model = LRML(num_users, num_items, args)

# Entrenar el modelo
with tf.compat.v1.Session(graph=model.graph) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(10):  # Número de épocas
        # Ensure neg_batch is provided if 'PAIR' is in rnn_type
        if 'PAIR' in model.args.rnn_type:
            # Generate negative samples for pairwise ranking
            neg_batch = [[x[0], random.randint(0, num_items - 1), x[2]] for x in train_data]  # Use random indices within valid range
        else:
            neg_batch = None  # No negative batch needed for pointwise ranking

        feed_dict, _ = model.get_feed_dict(train_data, neg_batch, mode='training')
        _, loss = sess.run([model.train_op, model.cost], feed_dict=feed_dict)
        print(f'Epoch {epoch + 1}, Loss: {loss}')