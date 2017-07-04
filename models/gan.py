"""Generative Adversarial Network for Training the Discriminator

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import numpy as np
import tensorflow as tf
import json
import csv

import utils.file_ops as fops
import models.nnblocks as nn


class GAN:
    """Anomaly detection object using a GAN

    Args:
        num_features (int): Number of input for classifier.
        batch_size (int = 100): Batch size.
        num_epochs (int = 10): Number of training epochs.
        debug (bool = False): Flag to print output.
        normalize (bool = False): Flag to determine if data
            is normalized.
        display_step (int = 1): How often to debug epoch data
            during training.
        num_steps (int = 3): Number of time steps.
        latent_vector_size (int = 100): Size of the latent vector.
    """

    def __init__(self, num_features, batch_size=100, num_epochs=10,
                 debug=False, normalize=False, display_step=1,
                 num_steps=3, latent_vector_size=100):

        ########################################
        # Network Parameters                   #
        ########################################

        self.training_epochs    = num_epochs
        self.display_step       = display_step
        self.batch_size         = batch_size
        self.debug              = debug
        self.normalize          = normalize
        self.num_steps          = num_steps
        self.num_features       = num_features
        self.latent_vector_size = latent_vector_size

        ########################################
        # TensorFlow Variables                 #
        ########################################

        # Shape: batch, time steps, features
        self.X = tf.placeholder('float32', [None, num_features * num_steps],
                                name='X')
        self.Y = tf.placeholder('int64', [None], name='Y')
        self.Z = tf.placeholder('float32', [None, latent_vector_size])
        self.keep_prob = tf.placeholder('float32')

        # for normalization
        self.feature_min = tf.Variable(np.zeros(num_features * num_steps),
                                       dtype=tf.float32)
        self.feature_max = tf.Variable(np.zeros(num_features * num_steps),
                                       dtype=tf.float32)

        ########################################
        # GAN Model                            #
        ########################################

        self.embedding_ops = []

        def build_net(X, sizes, scope, reuse=False):
            def block(x, in_dim, out_dim, i):
                with tf.variable_scope('block_{}'.format(i)):
                    z = nn.build_residual_block(x, in_dim)
                    tmp = nn.build_fc_layer(z, lrelu, in_dim, out_dim)
                    with tf.variable_scope('residual_block'):
                        self.embedding_ops.append(z)
                    with tf.variable_scope('fc_block'):
                        self.embedding_ops.append(tmp)

                    return tf.nn.dropout(tmp, self.keep_prob)

            lrelu   = nn.lrelu_gen(0.1)
            z       = X

            with tf.variable_scope(scope, reuse=reuse):
                for i in range(1, len(sizes)):
                    z = block(z, sizes[i-1], sizes[i], i-1)

            return z

        # Create the generator
        # with tf.variable_scope('generator'):
        #     sizes = [latent_vector_size, 150, 50, num_features * num_steps]
        #     activations = [tf.nn.relu, tf.nn.relu, tf.identity]
        #     generator = nn.NeuralNet(sizes, activations)

        G_sizes = [latent_vector_size, 100, num_features * num_steps]
        D_sizes = [num_features * num_steps, 128, 64, 32, 16, 8, 4, 2]

        # G_sample        = generator.create_network(self.Z)
        G_sample        = build_net(self.Z, G_sizes, 'generator')
        D_logit_real    = build_net(self.X, D_sizes, 'discriminator')
        D_logit_fake    = build_net(G_sample, D_sizes, 'discriminator', True)

        self.scores     = tf.nn.sigmoid(D_logit_real)

        # Losses
        D_loss_real = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=D_logit_real, labels=tf.ones_like(self.Y)
            )
        )
        D_loss_fake = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=D_logit_fake, labels=tf.zeros_like(self.Y)
            )
        )
        self.D_loss = D_loss_fake + D_loss_real
        self.G_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=D_logit_fake, labels=tf.ones_like(self.Y)
            )
        )

        self.D_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(
            self.D_loss,
            var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=''
            )
        )

        self.G_solver = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(
            self.G_loss,
            var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope='generator'
            )
        )

        ########################################
        # Evaluation Metrics                   #
        ########################################

        # negative_labels = tf.cast(tf.fill(tf.shape(self.Y), 0), 'int64')
        # positive_labels = tf.cast(tf.fill(tf.shape(self.Y), 1), 'int64')

        # pred_labels = tf.where(
        #     tf.greater(self.scores, tf.fill(tf.shape(self.Y), 0.5)),
        #     positive_labels,
        #     negative_labels

        # )

        pred_labels = tf.argmax(self.scores, 1)

        self.confusion_matrix = tf.confusion_matrix(
            self.Y,
            pred_labels,
            num_classes=2
        )

        self.accuracy = tf.reduce_mean(
            tf.to_float(tf.equal(pred_labels, self.Y))
        )

        # Variable ops
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def train(self, X, Y, time_steps):
        """Train the Classifier.

        Args:
            X (np.ndarray): Features with shape
                (num_samples * time_steps, features).
            Y (np.ndarray): Labels.
            time_steps (int): Number of time steps.
        """

        training_size = X.shape[0]

        # normalize X
        if self.normalize:
            _min = X.min(axis=0)
            _max = X.max(axis=0)
            X = fops.normalize(X, _min, _max)

        assert self.batch_size < training_size, (
            'batch size is larger than training_size'
        )

        with tf.Session() as sess:
            sess.run(self.init_op)

            # TensorBoard
            writer = tf.summary.FileWriter('logdir', sess.graph)
            writer.close()

            for epoch in range(self.training_epochs):
                loss = 0
                g_loss = 0
                loss_size = 0
                for batch_x, batch_y in fops.batcher([X, Y], self.batch_size):
                    Z = np.random.uniform(-1., 1., [self.batch_size,
                                                    self.latent_vector_size])

                    _, l = sess.run([self.D_solver, self.D_loss], feed_dict={
                        self.X: batch_x,
                        self.Y: batch_y,
                        self.Z: Z,
                        self.keep_prob: 0.5
                    })

                    _, lg = sess.run([self.G_solver, self.G_loss], feed_dict={
                        self.X: batch_x,
                        self.Y: batch_y,
                        self.Z: Z,
                        self.keep_prob: 0.5
                    })

                    loss += l
                    g_loss += lg
                    loss_size += 1

                if epoch % self.display_step == 0:
                    display_str = (
                        'Epoch {0:04} with D_loss={1:7.5f}||G_loss={2:.5f}'
                    )
                    display_str = display_str.format(
                        epoch+1,
                        loss/loss_size,
                        g_loss/loss_size
                    )
                    self.print(display_str)

            # assign normalization values
            if self.normalize:
                sess.run(self.feature_min.assign(_min))
                sess.run(self.feature_max.assign(_max))

            self.print('Optimization Finished')

            # save model
            save_path = self.saver.save(sess, './model.ckpt')
            self.print('Model saved in file: {}'.format(save_path))

    def test(self, X, Y, time_steps):
        """Tests classifier

        Args:
            X (np.ndarray): Features with shape
                (num_samples * time_steps, features).
            Y (np.array): Labels.
            time_steps (int): Number of time steps.

        Returns:
            dict: Dictionary containing the following fields:
        """

        with open('logdir/metadata.tsv', 'w') as f:
            # f.write('index\tlabels\n')
            for i, label in enumerate(Y):
                f.write('{}\n'.format(label))

        with tf.Session() as sess:
            self.saver.restore(sess, './model.ckpt')

            # normalize data
            if self.normalize:
                _min = self.feature_min.eval()
                _max = self.feature_max.eval()

                X = fops.normalize(X, _min, _max)

            labels, acc, mat = sess.run([self.scores, self.accuracy,
                                         self.confusion_matrix], feed_dict={
                self.X: X,
                self.Y: Y,
                self.keep_prob: 1.0
            })

            # for embeddings
            Z = np.random.uniform(-1., 1., [X.shape[0],
                                            self.latent_vector_size])

            embeddings = sess.run(self.embedding_ops, feed_dict={
                self.X: X,
                self.Y: Y,
                self.Z: Z,
                self.keep_prob: 1.0
            })

            for i, embedding in enumerate(embeddings):
                name = self.embedding_ops[i].name.split(':')[0]
                name = name.replace('/', '_')

                with open('graph/{}'.format(name), 'w') as f:
                    csv.writer(f).writerows(embedding)

            avg_benign      = []
            avg_malicious   = []
            for i, label in enumerate(labels):
                # print('Label: {} | Guess: {}'.format(Y[i], label))
                if Y[i] == 1:
                    avg_benign.append(label)
                else:
                    avg_malicious.append(label)

            self.print('Accuracy: {:.2f}'.format(acc * 100))
            self.print('Confusion Matrix:')
            self.print(mat)
            data = {
                'benign': {
                    'mean': np.mean(avg_benign, axis=0).tolist(),
                    'stddev': np.std(avg_benign, axis=0).tolist()
                },
                'malicious': {
                    'mean': np.mean(avg_malicious, axis=0).tolist(),
                    'stddev': np.std(avg_malicious, axis=0).tolist()
                }
            }

            # self.print(json.dumps(rtn_dict, indent=4))
            self.print(json.dumps(data, indent=4))

    def print(self, val):
        if self.debug:
            print(val)
