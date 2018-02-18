import sys
sys.path.append('../')

from skynet.util.image_processing import get_images
from skynet.util.dataset import Dataset
from sklearn.cluster import KMeans

img_df = get_images('../data')

print(img_df.loc[0, 'Vector'].shape)
print(len(img_df.loc[390:, 'Vector'].values))

# training_data = Dataset(img_df.loc[:,'Vector'].values)
#
# for i in range(5):
#     x = training_data.fetch_next_batch(80)
#     print("batch " + str(i))
#     print(len(x))

# print(img_df.loc[:,'Vector'])

# kmeans = KMeans(n_clusters=2, random_state=0).fit(img_vectors)
# print(kmeans.labels_)
# print(kmeans.cluster_centers_)
#
# cluster_1 = kmeans.cluster_centers_[0]
# cluster_2 = kmeans.cluster_centers_[1]
