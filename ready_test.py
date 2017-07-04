import argparse

import utils.file_ops as fops
import models.ready

parser = argparse.ArgumentParser()

parser.add_argument('--embedder', '-e',
                    action='store_true',
                    help='Flag to train encoder')

args = parser.parse_args()

model = models.ready.READY(num_features=12, num_epochs=50, normalize=True,
                           debug=True, std_param=2, embedding_size=250)

exploit = 'rootdir'

if args.embedder:
    for i in range(5):
        trX, trY = fops.load_data((
            './data/features/time_series/{}/subset_{}/train_set.csv'
        ).format(exploit, i))

        model.train_embedding(trX.reshape(-1, 12))

trX, trY = fops.load_data(
    './data/features/time_series/{}/subset_0/train_set.csv'.format(exploit)
)

teX, teY = fops.load_data(
    './data/features/time_series/{}/subset_0/test_set.csv'.format(exploit)
)

model.train(trX.reshape(-1, 12), trY, time_steps=3)
model.test(teX.reshape(-1, 12), teY, time_steps=3)
