import numpy as np
import tensorflow.keras as k
import tensorflow as tf


def mape_keras(y_true, y_pred, threshold=0.1):
    epsilon = 1e-8
    diff = k.backend.abs((y_true - y_pred) / (y_true + epsilon))
    return 100.0 * k.backend.mean(diff, axis=-1)


def rmse_keras(y_true, y_pred):
    return k.backend.sqrt(k.backend.mean(k.backend.square(y_pred - y_true)))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def mape(y_true, y_pred, threshold=0.1):
    epsilon = 1e-8
    diff = k.backend.abs((y_true - y_pred) / (y_true + epsilon))
    return 100.0 * np.mean(diff, axis=-1).mean()


def set_session(device_count=None, seed=0):
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    if device_count is not None:
        config = tf.compat.v1.ConfigProto(
            gpu_options=gpu_options,
            device_count=device_count
        )
    else:
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    return sess


def transform(idxs):
    return [idxs[:, i] for i in range(idxs.shape[1])]


def get_metrics(model, x, y, batch_size=1024):
    yp = model.predict(x, batch_size=batch_size, verbose=1).flatten()
    return {
        "rmse": float(rmse(y, yp)),
        "mape": float(mape(y, yp)),
        "mae": float(mae(y, yp))
    }