import tensorflow as tf
import time
import numpy as np
import os
import math
from scipy.sparse import csr_matrix # Import csr_matrix

# Helper function to convert scipy sparse matrix to tf SparseTensor
def convert_scipy_sparse_to_tf_sparse(sparse_matrix):
    """Converts a SciPy sparse matrix (CSR or COO) to a TensorFlow SparseTensor."""
    if not isinstance(sparse_matrix, (csr_matrix)):
         # If already a Tensor or other type, return as is or handle error
         # For now, assume it must be convertible or raise error
         # Check if it's already a SparseTensor
         if isinstance(sparse_matrix, tf.SparseTensor):
             return sparse_matrix
         # If it's dense, convert to dense tensor
         if isinstance(sparse_matrix, np.ndarray):
              return tf.convert_to_tensor(sparse_matrix, dtype=tf.float32)
         # Otherwise, raise error
         raise TypeError(f"Input must be a SciPy CSR matrix or TensorFlow Tensor, got {type(sparse_matrix)}")

    # Convert CSR to COO format for easy index extraction
    coo = sparse_matrix.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)


class AutoRec():
    def __init__(self, args, num_users, num_items, R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,
                 num_train_ratings, num_test_ratings, user_train_set, item_train_set, user_test_set, item_test_set, result_path):

        self.args = args
        self.num_users = num_users
        self.num_items = num_items

        # Store sparse matrices (R, mask_R, C might be None)
        self.R = R # Might be None
        self.mask_R = mask_R # Might be None
        self.C = C # Might be None
        self.train_R = train_R # Should be csr_matrix
        self.train_mask_R = train_mask_R # Should be csr_matrix (boolean)
        self.test_R = test_R # Should be csr_matrix
        self.test_mask_R = test_mask_R # Should be csr_matrix (boolean)

        # Ensure inputs are sparse if they are not None
        if self.train_R is not None and not isinstance(self.train_R, csr_matrix):
            raise TypeError("train_R must be a csr_matrix")
        if self.test_R is not None and not isinstance(self.test_R, csr_matrix):
            raise TypeError("test_R must be a csr_matrix")

        self.num_train_ratings = num_train_ratings
        self.num_test_ratings = num_test_ratings

        self.user_train_set = user_train_set
        self.item_train_set = item_train_set
        self.user_test_set = user_test_set
        self.item_test_set = item_test_set

        self.hidden_neuron = args.hidden_neuron
        self.train_epoch = args.train_epoch
        self.batch_size = args.batch_size
        # Adjust num_batch calculation if needed, ceil is correct
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

        # Build the Keras model (expects dense input after potential sparse conversion)
        self.model = self.build_model()
        # Select optimizer
        if self.optimizer_method == "Adam":
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.base_lr)
        elif self.optimizer_method == "RMSProp":
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.base_lr)
        else:
            raise ValueError("Optimizer Argumet Error!")

    def build_model(self):
        """Builds the AutoRec model using Keras Sequential API."""
        # Input layer: Note that the actual input passed during training/prediction
        # might be converted from sparse to dense implicitly by TF/Keras or explicitly
        # by our code before calling the model. The shape here defines the expected
        # *dense* feature vector size per user.
        model = tf.keras.Sequential(name="AutoRecModel")
        model.add(tf.keras.layers.InputLayer(input_shape=(self.num_items,), name="Input"))
        # Hidden layer
        model.add(tf.keras.layers.Dense(self.hidden_neuron, activation='sigmoid', name="HiddenLayer",
                                        kernel_regularizer=tf.keras.regularizers.l2(self.lambda_value),
                                        bias_regularizer=tf.keras.regularizers.l2(self.lambda_value)))
        # Output layer (reconstruction)
        model.add(tf.keras.layers.Dense(self.num_items, name="OutputLayer",
                                        kernel_regularizer=tf.keras.regularizers.l2(self.lambda_value),
                                        bias_regularizer=tf.keras.regularizers.l2(self.lambda_value)))
        model.summary() # Print model summary
        return model

    def loss_function(self, input_R_sparse, mask_R_sparse, output_dense):
        """
        Calculates the loss between sparse input and dense output, considering only observed ratings.

        Args:
            input_R_sparse: Sparse matrix (csr_matrix) of actual ratings for the batch.
            mask_R_sparse: Sparse matrix (csr_matrix, boolean) indicating observed ratings.
            output_dense: Dense tensor (tf.Tensor) of predicted ratings from the model.

        Returns:
            A scalar TensorFlow tensor representing the total loss (reconstruction + regularization).
        """
        # --- Reconstruction Loss (only on observed ratings) ---
        # 1. Get indices and values from the sparse mask
        mask_tf_sparse = convert_scipy_sparse_to_tf_sparse(mask_R_sparse)
        observed_indices = mask_tf_sparse.indices # Shape: [num_observed, 2]

        # 2. Get corresponding actual ratings from input_R_sparse
        #    (Assuming input_R_sparse has the same non-zero structure as mask_R_sparse)
        input_tf_sparse = convert_scipy_sparse_to_tf_sparse(input_R_sparse)
        # We need the values corresponding to the *mask* indices.
        # tf.gather_nd on a SparseTensor isn't straightforward.
        # Instead, use the values directly from the input sparse tensor,
        # assuming its indices match the mask's indices.
        actual_values = input_tf_sparse.values # Shape: [num_observed]

        # 3. Gather the predicted ratings from the dense output at observed indices
        predicted_values = tf.gather_nd(output_dense, observed_indices) # Shape: [num_observed]

        # 4. Calculate squared error only for observed ratings
        squared_errors = tf.square(actual_values - predicted_values)
        rec_loss = tf.reduce_sum(squared_errors)

        # --- Regularization Loss ---
        # Keras layers handle their own regularization losses internally
        # Access them via model.losses
        reg_loss = tf.add_n(self.model.losses)

        # --- Total Loss ---
        total_loss = rec_loss + reg_loss
        return total_loss


    def train_step(self, input_R_batch_sparse, mask_R_batch_sparse):
        """
        Performs a single training step on a batch of sparse data.

        Args:
            input_R_batch_sparse: csr_matrix for the input ratings batch.
            mask_R_batch_sparse: csr_matrix for the mask batch.

        Returns:
            Scalar TensorFlow tensor representing the loss for this batch.
        """
        # Convert sparse scipy matrices to dense TensorFlow tensors for this batch
        # WARNING: This can cause memory issues if batches are very large AND sparse,
        # but Keras Dense layers typically expect dense input.
        # Alternative: Use tf.keras.layers.Embedding + sparse input handling if memory is an issue.
        # For standard AutoRec with Dense layers, conversion is often necessary.
        input_R_batch_dense = tf.convert_to_tensor(input_R_batch_sparse.toarray(), dtype=tf.float32)
        mask_R_batch_dense = tf.convert_to_tensor(mask_R_batch_sparse.toarray(), dtype=tf.float32) # Mask as float for multiplication

        with tf.GradientTape() as tape:
            # Pass the DENSE batch to the model
            output_dense = self.model(input_R_batch_dense, training=True) # training=True enables regularization etc.

            # Calculate loss using the DENSE mask and input for simplicity here,
            # assuming the batch fits in memory.
            # Loss = observed_error + regularization
            # Observed error: sum( mask * (input - output)^2 )
            rec_loss = tf.reduce_sum(tf.square((input_R_batch_dense - output_dense) * mask_R_batch_dense))

            # Regularization loss (collected from layers)
            reg_loss = tf.add_n(self.model.losses)

            # Total loss
            loss = rec_loss + reg_loss

        # Compute and apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)

        if self.grad_clip:
            gradients = [(tf.clip_by_value(grad, -5., 5.) if grad is not None else None) for grad in gradients]

        # Filter out None gradients (can happen for non-trainable variables or disconnected parts)
        valid_gradients = [(grad, var) for grad, var in zip(gradients, self.model.trainable_variables) if grad is not None]
        if not valid_gradients:
             print("Warning: No gradients found for trainable variables.")
             return tf.constant(0.0, dtype=tf.float32) # Return zero loss or handle appropriately

        self.optimizer.apply_gradients(valid_gradients)
        return loss


    def predict_step(self, input_R_sparse):
        """
        Generates predictions for a given sparse input batch.

        Args:
            input_R_sparse: csr_matrix for the input ratings batch.

        Returns:
            Dense tf.Tensor containing the predicted ratings.
        """
        # Convert sparse input to dense tensor for the model
        input_R_dense = tf.convert_to_tensor(input_R_sparse.toarray(), dtype=tf.float32)
        # Get predictions (dense output)
        output_dense = self.model(input_R_dense, training=False) # training=False disables dropout etc.
        return output_dense


    # Note: The 'run' method is now defined and called in main.py,
    # so the internal 'train' method below is likely not used directly anymore.
    # It's kept here for potential standalone use or reference.
    def train(self):
        """Internal training loop (likely superseded by main.py's loop)."""
        print("Starting internal training loop (may be overridden by main.py)...")
        for epoch in range(self.train_epoch):
            start_time = time.time()
            total_loss = 0
            processed_batches = 0

            # Generate user indices and shuffle for epoch
            user_indices = np.arange(self.num_users)
            np.random.shuffle(user_indices)

            for i in range(0, self.num_users, self.batch_size):
                batch_idx = user_indices[i:min(i + self.batch_size, self.num_users)]
                if len(batch_idx) == 0: continue

                # Get sparse batches
                batch_train_sparse = self.train_R[batch_idx]
                batch_mask_sparse = self.train_mask_R[batch_idx]

                # Skip empty batches
                if batch_mask_sparse.nnz == 0:
                    continue

                # Perform training step
                loss = self.train_step(batch_train_sparse, batch_mask_sparse)
                if loss is not None and not tf.math.is_nan(loss):
                    total_loss += loss.numpy()
                    processed_batches += 1
                else:
                    print(f"Warning: NaN or None loss encountered in epoch {epoch+1}, batch starting at index {i}")


            avg_loss = total_loss / max(1, processed_batches)
            self.train_cost_list.append(avg_loss)

            if (epoch + 1) % self.display_step == 0:
                # Calculate test RMSE (requires predict_step and adapted RMSE calculation)
                # This part would need the adapted calculate_rmse logic from main.py
                # test_rmse, _ = self.calculate_test_rmse() # Placeholder for adapted RMSE calc
                # self.test_rmse_list.append(test_rmse)
                # print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Test RMSE: {test_rmse:.4f} | Time: {int(time.time() - start_time)}s")
                print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Time: {int(time.time() - start_time)}s") # Print without RMSE for now

    # The actual run logic is now primarily in main.py
    # def run(self):
    #     self.train()