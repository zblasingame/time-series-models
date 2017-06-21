import utils.file_ops as fops
import models.ready

model = models.ready.READY(num_features=12, num_epochs=100, normalize=True,
                           debug=True, std_param=3, embedding_size=50)

trX, trY = fops.load_data(
    './data/features/time_series/nginx_rootdir/subset_0/train_set.csv'
)

teX, teY = fops.load_data(
    './data/features/time_series/nginx_rootdir/subset_0/test_set.csv'
)

# model.train_embedding(trX.reshape(-1, 12))
model.train(trX.reshape(-1, 12), trY, time_steps=3)
model.test(teX.reshape(-1, 12), teY, time_steps=3)
