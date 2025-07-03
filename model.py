import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, GaussianNoise
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l1, l2
from sklearn.cluster import KMeans
from keras.layers import LayerNormalization, MultiHeadAttention
import os
import random
from keras import regularizers
from tensorflow.keras.regularizers import l1, l2, l1_l2
import numpy as np

seed = 3
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)


def expand_features(input_tensor):
    avg_vector = tf.reduce_mean(input_tensor, axis=1, keepdims=True)
    extended_tensor = tf.concat([input_tensor, avg_vector], axis=1)
    combined_vectors = []
    for i in range(4):
        for j in range(i + 1, 4):
            combined_vector = (extended_tensor[:, i, :] + extended_tensor[:, j, :]) / 2
            combined_vectors.append(combined_vector)
    combined_tensor = tf.stack(combined_vectors, axis=1)
    final_output = tf.concat([extended_tensor, combined_tensor], axis=1)
    return final_output


def cluster_and_expand_features_l2(x, num_clusters=20, num_combinations=1000, threshold=0.05):
    coefficients = tf.random.normal((num_combinations, 10))
    coefficients = tf.nn.l2_normalize(coefficients, axis=1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(coefficients)
    selected_centroids = tf.convert_to_tensor(kmeans.cluster_centers_, dtype=tf.float32)
    new_features = tf.tensordot(x, selected_centroids, axes=[[1], [1]])
    final_features = tf.transpose(new_features, perm=[0, 2, 1])
    expanded_x = tf.concat([x, final_features], axis=1)

    return expanded_x


def augment_data(orgdata, datashape, n_augmentations=10):
    batch_size = tf.shape(orgdata)[0]
    augmented_indices_list = [[] for _ in range(3)]
    for _ in range(n_augmentations):
        random_indices = [tf.random.uniform((batch_size,), minval=0, maxval=dim - 1, dtype=tf.int32) for dim in
                          datashape]
        num_replacements = random.choice([1, 2])
        dims_to_replace = random.sample(range(3), num_replacements)
        for i in range(3):
            if i in dims_to_replace:
                indices_to_replace = random_indices[i]
            else:
                indices_to_replace = orgdata[:, i]
            augmented_indices_list[i].append(tf.expand_dims(indices_to_replace, -1))
    augmented_indices = [tf.concat(dim_indices, axis=0) for dim_indices in augmented_indices_list]
    augmented_values = tf.zeros((tf.shape(augmented_indices[0])[0], 1), dtype=tf.float32)
    return augmented_indices, augmented_values


def cl_loss(tensor, indices, y_true):
    tensor_min = tf.reduce_min(tensor)
    tensor_max = tf.reduce_max(tensor)
    normalized_tensor = (tensor - tensor_min + 1e-7) / (tensor_max - tensor_min + 1e-7)
    scale_factor = 0.1
    scaled_tensor = normalized_tensor / scale_factor
    exp_tensor = tf.exp(scaled_tensor)
    mask_indice = tf.logical_or(
        tf.math.equal(indices[:, None, 0], indices[None, :, 0]),
        tf.logical_or(
            tf.math.equal(indices[:, None, 1], indices[None, :, 1]),
            tf.math.equal(indices[:, None, 2], indices[None, :, 2])
        )
    )
    replicated_tensor = tf.tile(exp_tensor, [1, tf.shape(indices)[0]])
    transposed_tensor = tf.transpose(replicated_tensor)
    replicated_tensor_true = tf.tile(y_true, [1, tf.shape(indices)[0]])
    transposed_tensor_true = tf.transpose(replicated_tensor_true)
    mask_true = tf.math.less(transposed_tensor_true, replicated_tensor_true)
    mask = tf.logical_and(mask_true, mask_indice)
    candidate_vals = tf.where(mask, transposed_tensor, 0.0)
    weighted_numerator = exp_tensor
    mask_true_count = tf.reduce_sum(tf.cast(mask, tf.float32), axis=1, keepdims=True)
    weight_m = tf.subtract(replicated_tensor_true, transposed_tensor_true)
    weight_non = tf.where(mask, weight_m, 0.0)
    weight_sum = tf.reduce_sum(weight_non, axis=1, keepdims=True)
    weight = (weight_non / (weight_sum + 1e-7)) * mask_true_count
    weighted_denominator = tf.multiply(weight, candidate_vals)
    weighted_denominator = tf.reduce_sum(weighted_denominator, axis=1, keepdims=True)
    final_denominator = weighted_numerator + weighted_denominator
    individual_scores = weighted_numerator / final_denominator
    log_scores = -tf.math.log1p(individual_scores) + tf.math.log(2.0)
    average_score = tf.reduce_mean(log_scores)
    return average_score

def restore(data):
    x = tf.stack(data, axis=-1)
    y = tf.squeeze(x, axis=1)
    return y


def hoctc_sp(shape, rank, nc, tau):
    input_y_true = k.Input(shape=(1,))
    inputs = [k.Input(shape=(1,), dtype="int32") for i in range(len(shape))]
    tr_idx = restore(inputs)
    embeds = [
        k.layers.Embedding(output_dim=rank, input_dim=shape[i])(inputs[i])
        for i in range(len(shape))
    ]
    x = k.layers.Concatenate(axis=1)(embeds)
    extended_feature = expand_features(x)
    cluster_query = cluster_and_expand_features_l2(extended_feature)
    attention1 = MultiHeadAttention(num_heads=5, key_dim=40, dropout=0.5,
                                    kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001))
    x = attention1(cluster_query, x, x)
    x = LayerNormalization()(x)
    attention2 = MultiHeadAttention(num_heads=5, key_dim=40, dropout=0.5,
                                    kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001))
    y = attention2(x, x, x)
    y = LayerNormalization()(y)
    y = k.layers.Flatten()(y)
    y = k.layers.Dense(600, activation="relu", kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001))(y)
    y = k.layers.Dense(100, activation="relu", kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001))(y)
    layer_norm = LayerNormalization()
    y = layer_norm(y)
    con_puts = k.layers.Dense(1, activation="relu")(y)
    relu_layer = k.layers.ReLU()
    con_puts = relu_layer(con_puts)
    score = cl_loss(con_puts, tr_idx, input_y_true)
    attention3 = MultiHeadAttention(num_heads=5, key_dim=40, dropout=0.5,
                                    kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001))
    x = attention3(x, x, x)
    x = LayerNormalization()(x)
    x = k.layers.Flatten()(x)
    x = k.layers.Dense(600, activation="relu", kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001))(x)
    x = k.layers.Dense(100, activation="relu", kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001))(x)
    layer_norm = LayerNormalization()
    x = layer_norm(x)
    outputs = k.layers.Dense(1, activation="relu")(x)
    model = k.Model(inputs=[input_y_true, inputs], outputs=outputs)
    tau = tau
    mse_loss = tf.losses.MSE(input_y_true, outputs)
    mse_loss = tf.reduce_mean(mse_loss)
    total_loss = mse_loss + tau * score
    model.add_loss(total_loss)

    return model
