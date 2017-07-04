import utils.file_ops as fops
import models.gan

model = models.gan.GAN(num_features=12, num_epochs=50, normalize=True,
                       debug=True, latent_vector_size=50, batch_size=100)

# exploit = 'freak'
# exploit = 'nginx_keyleak'
exploit = 'nginx_rootdir'

# for i in range(5):
#     trX, trY = fops.load_data(
#         (
#             './data/features/time_series/{}/subset_{}/train_set.csv'
#         ).format(exploit, i)
#     )

#     model.train(trX, trY, time_steps=3)

teX, teY = fops.load_data(
    './data/features/time_series/{}/subset_0/test_set.csv'.format(exploit)
)

model.test(teX, teY, time_steps=3)
