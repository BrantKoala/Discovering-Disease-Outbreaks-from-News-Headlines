from math import radians, cos, sin, asin, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
from sklearn.cluster import DBSCAN, KMeans

df = pd.read_csv('database/match_data.csv')

colors = ["k", "r", "y", "g", "c", "b", "m", "chocolate", "darkorange", "lime", "navy", "deeppink"]

# # Elbow Curve
# clusters = range(1, 20)
# kmeans_elbow = [KMeans(n_clusters=i) for i in clusters]
# score = [kmeans_elbow[i].fit(new_df).score(new_df) for i in range(len(kmeans_elbow))]
# plt.plot(clusters, score)
# plt.plot(clusters, score, 'ko')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Score')
# plt.title('Elbow Curve')
# plt.show()

kmeans_model = KMeans(n_clusters=11)
kmeans_model.fit(df[['latitude', 'longitude']])
df["kmeans_label"] = kmeans_model.labels_

# # Implementing K-Means Clustering Algorithm (not on global map)
# kmeans = KMeans(n_clusters=4).fit(new_df)
# centroids = kmeans.cluster_centers_
# print(centroids)
# plt.scatter(new_df['latitude'], new_df['longitude'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
# plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
# plt.show()

# Plotting the points on Basemap with the k-means algorithm (11 clusters)
print("This is the map using the k-means clustering algorithm.")
map_plotter = Basemap()
fig = plt.figure(figsize=(16, 9))
cluster_vals = {}
for i in range(kmeans_model.n_clusters):
    cluster_vals[str(i) + "_long"] = []
    cluster_vals[str(i) + "_lat"] = []
for index in df.index:
    cluster_vals[str(df['kmeans_label'][index]) + '_long'].append(df['longitude'][index])
    cluster_vals[str(df['kmeans_label'][index]) + '_lat'].append(df['latitude'][index])
for target in range(kmeans_model.n_clusters):
    map_plotter.scatter(cluster_vals[str(target) + '_long'], cluster_vals[str(target) + '_lat'], latlon=True,
                        color=colors[target], alpha=0.3, s=10)
    map_plotter.shadedrelief()
plt.title(f"map using the k-means clustering with n_clusters={kmeans_model.n_clusters}")
plt.savefig(f"database\\k-means_{kmeans_model.n_clusters}_clusters.png")
plt.show()


# Defining the distance function (Great Circle Distance)
def haversine(point_a, point_b):
    lon1, lat1, lon2, lat2 = map(radians, [point_a[0], point_a[1], point_b[0], point_b[1]])
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 12742 * asin(sqrt(a))


# Implementing the DBSCAN Algorithm
points = []
for index in df.index:
    points.append([df['longitude'][index], df['latitude'][index]])
dbscan_model = DBSCAN(eps=1000, min_samples=3, metric=haversine, n_jobs=-1)
clusters = dbscan_model.fit_predict(points)
df["dbscan_label"] = clusters

# Removing the outliers
df_dbscan_valid = df.loc[df['dbscan_label'] != -1, ['latitude', 'longitude', 'dbscan_label']]

# Plotting the points on Basemap with the DBSCAN Algorithm (11 clusters)
print("This is the map using the DBSCAN clustering algorithm.")
fig = plt.figure(figsize=(16, 9))
cluster_vals = {}
valid_dbscan_labels = np.unique(dbscan_model.labels_)[1:]
for dbscan_label in valid_dbscan_labels:
    cluster_vals[str(dbscan_label) + '_long'] = []
    cluster_vals[str(dbscan_label) + '_lat'] = []
for index in df_dbscan_valid.index:
    cluster_vals[str(df_dbscan_valid['dbscan_label'][index]) + '_long'].append(df_dbscan_valid['longitude'][index])
    cluster_vals[str(df_dbscan_valid['dbscan_label'][index]) + '_lat'].append(df_dbscan_valid['latitude'][index])
for target in valid_dbscan_labels:
    map_plotter.scatter(cluster_vals[str(target) + '_long'], cluster_vals[str(target) + '_lat'], latlon=True, alpha=0.3,
                        color=colors[target], s=10)
    map_plotter.shadedrelief()
plt.title(f"map using the DBSCAN clustering algorithm with {len(valid_dbscan_labels)} clusters")
plt.savefig(f"database\\DBSCAN_{len(valid_dbscan_labels)}_clusters.png")
plt.show()

print("Saving table consisting of the k-means and DBSCAN label for all geographical points.")
print(df)
print(df.info())
df.to_csv('database\\cluster_result.csv', index=False)
