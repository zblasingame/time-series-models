"""Time Embedding Autoencoder

A module for creating a time embedded autoencoder.

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import tensorflow as tf
import utils.file_ops as fops


class TEA:
    """Time Embedding Autoencoder

    Embeds input time step into smaller representation.

    Args:
        num_features (int = 12): Number of input features.
        embedding_size (int = 4): Size of embedded vector.
        activation (function = tf.sigmoid): TensorFlow activation function.
    """

    def __init__(self, num_features=12, embedding_size=4,
                 activation=tf.sigmoid):
        """Init autoencoder"""
        with tf.variable_scope('tea'):
            with tf.variable_scope('encoder'):
                self.encoder = {
                    'weights': tf.get_variable(
                        'weights',
                        shape=[num_features, embedding_size],
                        initializer=tf.contrib.layers.xavier_initializer()
                    ),
                    'bias': tf.get_variable(
                        'bias',
                        shape=[embedding_size],
                        initializer=tf.contrib.layers.xavier_initializer()
                    )
                }

            with tf.variable_scope('decoder'):
                self.decoder = {
                    'weights': tf.get_variable(
                        'weights',
                        shape=[embedding_size, num_features],
                        initializer=tf.contrib.layers.xavier_initializer()
                    ),
                    'bias': tf.get_variable(
                        'bias',
                        shape=[num_features],
                        initializer=tf.contrib.layers.xavier_initializer()
                    )
                }

        self.activation = activation

    def create_network(self, X):
        """Method to construct the network.

        Args:
            X (tf.Tensor): Placeholder Tensor with dimenions of the
                training Tensor.

        Returns:
            tf.Tensor: A tensor to be evaulated containing the predicted
                output of the autoencoder.
        """

        return self.activation(
            tf.matmul(
                self.embedding(X),
                self.decoder['weights']
            ) + self.decoder['bias']
        )

    def embedding(self, X):
        """Method to return the embedded tensor.

        Args:
            X (tf.Tensor): Placeholder Tensor repesenting the data
                to be embedded.

        Returns:
            tf.Tensor: Embedded tensor.
        """

        return self.activation(
            tf.matmul(X, self.encoder['weights']) + self.encoder['bias']
        )

    def train_embedding(self, X, X_data, num_epochs=10,
                        model_name='tea', normalize=False, batch_size=100):
        """Trains the Time Embedding Autoencoder.

        Args:
            X (tf.Tensor): Tensor describing the shape of the input.
            X_data (np.ndarray): Input data.
            num_epochs (int = 10): Number of training epochs.
            model_name (str = 'tea'): Name of the model.
            normalize (bool = False): Flag to normalize the data.
            batch_size (int = 100): Size of the training batches.
        """

        cost = tf.reduce_mean(tf.square(self.create_network(X) - X))
        opt = tf.train.AdamOptimizer().minimize(cost)
        saver = tf.train.Saver()

        training_size = X_data.shape[0]

        # normalize X
        if normalize:
            _min = X_data.min(axis=0)
            _max = X_data.max(axis=0)
            X_data = fops.normalize(X_data, _min, _max)

        assert batch_size < training_size, (
            'batch size is larger than training_size'
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(num_epochs):
                costs       = 0
                num_costs   = 0
                for batch_x, in fops.batcher([X_data], batch_size):
                    _, c = sess.run([opt, cost], feed_dict={
                        X: batch_x
                    })

                    costs += c
                    num_costs += 1

                display_str = 'Epoch {0:04} with cost={1:.9f}'
                display_str = display_str.format(epoch+1, costs/num_costs)
                print(display_str)

            print('TEA Optimization Finished')

            save_path = saver.save(sess, './{}.ckpt'.format(model_name))
            print('Model saved in file: {}'.format(save_path))
