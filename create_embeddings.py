import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.python.keras as keras
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras import optimizers

import config
import mr
from embedding import NCEModel

tf.compat.v1.disable_eager_execution()


def empty_loss(y_true, y_pred):
    """ Define loss function for embeddings model.
    """
    return tf.constant(0.)


def create_logdir():
    """ Empty out log directory to prepare for storing new tensorflow log
        messages.
    """
    os.system('rm -rf ./model/log/*')
    logdir = os.path.join('model', 'log',
                          mr.datetime.now().strftime("vec_%Y%m%d-%H%M%S"))
    return logdir


def plot_embeddings(embeddings, cf):
    """Plot sensor embeddings.
    """
    fig = plt.figure()
    ax = Axes3D(fig)

    def init():
        ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2])
        return fig,

    def animate(i):
        elevation = ((i + 1) // 360) * 10.0
        azimuth = i % 360
        ax.view_init(elev=elevation, azim=azimuth)
        return fig,

    for i in range(len(embeddings)):
        if 'Door' in cf.sensornames[i]:
            c = 'green'
        elif 'Area' in cf.sensornames[i]:
            c = 'red'
        else:
            c = 'blue'
        ax.text(embeddings[i][0], embeddings[i][1], embeddings[i][2],
                cf.sensornames[i], color=c)
    plot_dir = 'plots'
    plot_filename = os.path.join(plot_dir, 'embedding.png')
    plt.savefig(plot_filename)
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=1440, interval=20, blit=True)
    anim_filename = os.path.join(plot_dir, 'embedding_animation.mp4')
    anim.save(anim_filename, fps=30, extra_args=['-vcodec', 'libx264'])


def normalize_embeddings(embeddings):
    """ Normalize sensor embeddings to unit distance from origin.
    Input as an ndarray of shape (num_sensors, dim) that translates each
    sensor into a vector in a latent space.
    The function returns an ndarray of shape (num_sensors, dim), where each
    sensor is located on the surface of a unit sphere.
    """
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norm


def generate_training_pairs(cf, num_skips, skip_window):
    """ Generate training dataset using skip gram models.
    """
    num_windows = len(cf.sequence) - 2 * skip_window
    size = num_windows * num_skips
    source = np.zeros(shape=(size,), dtype=np.int32)
    target = np.zeros(shape=(size,), dtype=np.int32)
    span = 2 * skip_window + 1
    for i in range(num_windows):
        context_sensors = [w for w in range(span) if w != skip_window]
        sensors_to_use = random.sample(context_sensors, num_skips)
        for j, context_sensor in enumerate(sensors_to_use):
            source[i * num_skips + j] = cf.sequence[i + skip_window]
            source[i * num_skips + j] = cf.sequence[i + context_sensor]
    return source, target


def get_training_pair(cf, num_skips, skip_window):
    """ Generate training pairs.
    """
    training_dir = 'model'
    embeddings_training_filename = os.path.join(training_dir, 'training.npz')
    if os.path.exists(embeddings_training_filename):
        training_data = np.load(embeddings_training_filename)
        x = training_data['x']
        y = training_data['y']
        print('x', len(x), 'y', len(y))
    else:
        os.makedirs(training_dir, exist_ok=True)
        x, y = generate_training_pairs(cf, num_skips, skip_window)
        training_data = {'x': x, 'y': y}
        np.savez(embeddings_training_filename, **training_data)
    for i in range(10):
        print('%2d -> %2d  %s -> %s' %
              (x[i], y[i], cf.sensornames[x[i]], cf.sensornames[y[i]]))
    return x, y


def learn_embedding_model(cf, x, y, embedding_size, num_epochs,
                          batch_size, num_sampled):
    """ Use tensorflow to learn embedding model.
    """
    vocab_size = len(cf.sensornames)
    print('parameters', vocab_size, embedding_size, num_sampled)
    model = NCEModel(
        vocab_size=vocab_size,
        embeddings_size=embedding_size,
        nce_num_sampled=num_sampled
    )
    optimizer = optimizers.RMSprop(
        lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0
    )
    model.compile(optimizer=optimizer, loss=empty_loss)

    logdir = create_logdir()  # prepare tensorboard callback
    tensorboard_callback = keras.callbacks.TensorBoard(logdir, profile_batch=0)
    os.makedirs(logdir, exist_ok=True)

    model.fit(
        x=[x.reshape([-1, 1]), y.reshape([-1, 1])],
        y=y,
        epochs=num_epochs,
        batch_size=64,
        callbacks=[tensorboard_callback]
    )

    sensor_x = np.arange(vocab_size)
    embeddings = model.predict((
        sensor_x.reshape([-1, 1]), sensor_x.reshape([-1, 1])
    ))
    normalized_embeddings = normalize_embeddings(embeddings)
    if cf.save_model:
        print('Saving model.')
        embeddings_dir = 'model'
        embeddings_filename = os.path.join(embeddings_dir, 'embeddings')
        np.save(embeddings_filename, normalized_embeddings)
    return normalized_embeddings


def generate_sensor_embeddings(cf):
    """ Translate individual ambient sensors to a corresponding representation
        in an m-dimensional latent space. Sensor embeddings are computed in a
        manner similar to word embeddings used in natural language processing.
    """
    skip_window = 7  # how many sensors to consider left and right
    num_skips = 14  # how many times to reuse an input to generate a label
    batch_size = 128  # batch size
    embedding_size = 3  # dimension of the embedding vector
    num_sampled = 16  # number of negative examples to sample
    num_epochs = 10  # number of epochs to run NCE
    x, y = get_training_pair(cf, num_skips, skip_window)
    normalized_embeddings = learn_embedding_model(cf, x, y, embedding_size, num_epochs,
                                                  batch_size, num_sampled)
    return normalized_embeddings


def main():
    cf = config.Config()
    cf.save_model = True
    cf.set_parameters()
    mr.read_data(cf)
    if len(cf.data) < 11000 or len(cf.sensornames) < 10:
        print("Sample too small", len(cf.data), len(cf.sensornames))
        exit()
    data = cf.data[:100000]
    cf.data = data
    embeddings = generate_sensor_embeddings(cf)
    if cf.plot_embeddings:
        plot_embeddings(embeddings, cf)


if __name__ == "__main__":
    main()
