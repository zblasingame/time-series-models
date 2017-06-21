"""Recurrent Embedded Anomaly Detectin sYstem (READY)

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""


import numpy as np
import tensorflow as tf

from models.tea import TEA
from models.neuralnet import NeuralNet
import utils.file_ops as fops


class READY:
    """Anomaly detection object using time embedding autoencoders.

    Args:
        num_features (int): Number of input for classifier.
        batch_size (int = 100): Batch size.
        num_epochs (int = 10): Number of training epochs.
        debug (bool = False): Flag to print output.
        normalize (bool = False): Flag to determine if data
            is normalized.
        display_step (int = 1): How often to debug epoch data
            during training.
        std_param (int = 5): Value of the threshold constant for
            calculating the threshold.
        embedding_size (int = 100): Number of embedded features.
        num_steps (int = 3): Number of time steps.
    """

    def __init__(self, num_features, batch_size=100, num_epochs=10,
                 debug=False, normalize=False, display_step=1,
                 std_param=5, embedding_size=100, num_steps=3):

        ########################################
        # Network Parameters                   #
        ########################################

        l_rate                  = 0.001
        # reg_param               = 0.01
        self.std_param          = std_param
        self.training_epochs    = num_epochs
        self.display_step       = display_step
        self.batch_size         = batch_size
        self.debug              = debug
        self.normalize          = normalize
        self.num_steps          = num_steps
        self.num_features       = num_features

        ########################################
        # TensorFlow Variables                 #
        ########################################

        # Shape: batch, time steps, features
        self.X = tf.placeholder('float32', [None, None, num_features],
                                name='X')
        self.Y = tf.placeholder('int32', [None], name='Y')
        self.keep_prob = tf.placeholder('float32')

        # Score benign bounds for anomaly detection
        self.score_upper    = tf.Variable(0, dtype=tf.float32)
        self.score_lower    = tf.Variable(0, dtype=tf.float32)

        # for normalization
        self.feature_min = tf.Variable(np.zeros(num_features),
                                       dtype=tf.float32)
        self.feature_max = tf.Variable(np.zeros(num_features),
                                       dtype=tf.float32)

        ########################################
        # READY Model                          #
        ########################################

        # Create the TEA
        self.tea = TEA(num_features=num_features,
                       embedding_size=embedding_size)

        # Create the classifier
        # with tf.variable_scope('classifier'):
        #     weights = tf.get_variable(
        #         'weights',
        #         shape=[embedding_size, 1],
        #         initializer=tf.contrib.layers.xavier_initializer()
        #     )
        #     bias = tf.get_variable(
        #         'bias',
        #         shape=[1],
        #         initializer=tf.contrib.layers.xavier_initializer()
        #     )
        sizes   = [embedding_size+1, 25, 1]
        acts    = [tf.nn.relu, tf.identity]

        with tf.variable_scope('classifier'):
            self.classifier = NeuralNet(sizes, acts)

        # Connect the full model
        t               = tf.constant(0)
        score_init      = tf.zeros([tf.shape(self.X)[0]])
        embedded_init   = tf.zeros([tf.shape(self.X)[0], num_features])

        # Create the network
        def constructor(t, score_prev, X_prev):
            X = tf.squeeze(
                tf.slice(self.X, [0, t, 0], [-1, 1, -1]),
                axis=1
            )

            X = tf.add(X, X_prev)

            emb = self.tea.embedding(X)
            score = self.classifier.create_network(
                tf.concat([emb, tf.expand_dims(score_prev, 1)], axis=1)
            )

            t       = tf.add(t, 1)
            # score   = tf.add(score_prev, tf.squeeze(score, axis=1))
            score   = tf.squeeze(score, axis=1)

            X_next = self.tea.create_network(X)

            return t, score, X_next

        _, net_scores, _ = tf.while_loop(
            lambda t, s, x: t < tf.shape(self.X)[1],
            constructor,
            loop_vars=[t, score_init, embedded_init],
            shape_invariants=[
                t.get_shape(),
                score_init.get_shape(),
                embedded_init.get_shape()
            ]
        )

        # self.scores = tf.divide(net_scores, tf.to_float(tf.shape(self.X)[1]))
        self.scores = net_scores
        self.loss = tf.reduce_mean(tf.square(1.0 - self.scores))
        self.opt = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(
            self.loss,
            var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope='classifier'
            )
        )

        ########################################
        # Evaluation Metrics                   #
        ########################################

        negative_labels = tf.fill(tf.shape(self.Y), 0)
        positive_labels = tf.fill(tf.shape(self.Y), 1)
        lower_threshold = tf.fill(tf.shape(self.Y), self.score_lower)
        upper_threshold = tf.fill(tf.shape(self.Y), self.score_upper)

        pred_labels = tf.where(
            tf.logical_and(
                tf.less(self.scores, upper_threshold),
                tf.greater(self.scores, lower_threshold)
            ),
            positive_labels,
            negative_labels

        )

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

    def train_embedding(self, X):
        """Trains the Time Embedding Autoencoder.

        Args:
            X (np.array): Input data.
        """

        tea_X = tf.placeholder('float32', [None, self.num_features])
        self.tea.train_embedding(tea_X, X, num_epochs=500, normalize=True)

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

        # Reshape input array
        X = X.reshape(-1, time_steps, self.num_features)

        with tf.Session() as sess:
            sess.run(self.init_op)
            self.saver.restore(sess, './tea.ckpt')

            scores = []
            for epoch in range(self.training_epochs):
                loss = 0
                loss_size = 0
                for batch_x, batch_y in fops.batcher([X, Y], self.batch_size):
                    for i in range(batch_x.shape[1]):
                        _, l = sess.run([self.opt, self.loss], feed_dict={
                            self.X: batch_x,
                            self.Y: batch_y
                        })

                        loss += l
                        loss_size += 1

                    if epoch == self.training_epochs - 1:
                        scores.append(sess.run(self.scores, feed_dict={
                            self.X: batch_x
                        }))

                if epoch % self.display_step == 0:
                    display_str = 'Epoch {0:04} with cost={1:.9f}'
                    display_str = display_str.format(epoch+1, loss/loss_size)
                    self.print(display_str)

            # assign score threshold
            upper = np.mean(scores) + self.std_param * np.std(scores)
            lower = np.mean(scores) - self.std_param * np.std(scores)
            sess.run(self.score_upper.assign(upper))
            sess.run(self.score_lower.assign(lower))

            self.print('Lower Threshold: ' + str(lower))
            self.print('Upper Threshold: ' + str(upper))

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

        with tf.Session() as sess:
            self.saver.restore(sess, './model.ckpt')

            # normalize data
            if self.normalize:
                _min = self.feature_min.eval()
                _max = self.feature_max.eval()

                X = fops.normalize(X, _min, _max)

            # Reshape X
            X = X.reshape(-1, time_steps, self.num_features)

            labels, acc, mat = sess.run([self.scores, self.accuracy,
                                         self.confusion_matrix], feed_dict={
                self.X: X,
                self.Y: Y
            })

            self.print('Accuracy: {:.2f}'.format(acc * 100))
            self.print('Confusion Matrix:')
            self.print(mat)
            avg_benign      = []
            avg_malicious   = []
            for i, label in enumerate(labels):
                print('Label: {} | Guess: {}'.format(Y[i], label))
                if Y[i] == 1:
                    avg_benign.append(label)
                else:
                    avg_malicious.append(label)

            self.print('Average Bengin: {}'.format(np.mean(avg_benign)))
            self.print('Average Malicious: {}'.format(np.mean(avg_malicious)))

            # self.print(json.dumps(rtn_dict, indent=4))

    def print(self, val):
        if self.debug:
            print(val)
