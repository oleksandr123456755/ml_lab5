# Імпорт необхідних бібліотек
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.cluster import KMeans
from tqdm import tqdm

# 1. Відкрити та зчитати наданий файл з даними
data = pd.read_csv('WQ-R.csv', sep=';')

# 2. Визначити та вивести кількість записів
num_records = data.shape[0]
num_records

# 3. Видалити атрибут quality
data = data.drop('quality', axis=1)

# 4. Вивести атрибути, що залишилися
data.columns

# 5. Виконати кластеризацію за допомогою KMeans та визначити оптимальну кількість кластерів трьома методами

# 5.1. Elbow Method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method')
plt.show()

# 5.2. Silhouette Method
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    score = silhouette_score(data, kmeans.labels_)
    silhouette_scores.append(score)

plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.show()

# 5.3. Prediction Strength Method


def get_prediction_strength(train_preds, test_preds, cluster_num):
    d = test_preds.shape[0]
    d_arr = np.zeros((d, d), dtype=int)

    # Create the distance matrix
    for m in range(d):
        for n in range(m + 1, d):
            d_arr[m, n] = int(train_preds[m] == train_preds[n])
            d_arr[n, m] = d_arr[m, n]

    ps = np.full(cluster_num, np.inf)
    for j in range(cluster_num):
        cluster_indices = (test_preds == j)
        len_aj = np.sum(cluster_indices)

        if len_aj > 1:
            d_sum = np.sum(d_arr[cluster_indices][:, cluster_indices])
            ps[j] = d_sum / (len_aj * (len_aj - 1))

    return np.min(ps)

def prediction_strength_method_KMeans(df, train_size=0.8):
    cluster_num_arr=range(1, 11)
    spliter = ShuffleSplit(n_splits=4, train_size=train_size, random_state=1)
    pss = {i: [] for i in cluster_num_arr}

    for train_indices, test_indices in tqdm(spliter.split(df), total=4):
        train_df = df.iloc[train_indices]
        test_df = df.iloc[test_indices]

        for num_clusters in cluster_num_arr:
            test_model = KMeans(n_clusters=num_clusters, init='random', n_init=1, random_state=1)
            test_model.fit(test_df)
            test_preds = test_model.predict(test_df)

            train_model = KMeans(n_clusters=num_clusters, init='random', n_init=1, random_state=1)
            train_model.fit(train_df)
            train_preds = train_model.predict(test_df)

            ps = get_prediction_strength(train_preds, test_preds, num_clusters)
            pss[num_clusters].append(ps)

    # Calculate mean prediction strength for each cluster number
    pss = {n: np.mean(pss[n]) for n in pss}

    return pss

def plot_prediction_strength(pss):
    clusters = list(pss.keys())
    strengths = list(pss.values())

    plt.figure(figsize=(10, 6))
    plt.plot(clusters, strengths, marker='o')
    plt.xticks(range(1, len(clusters) + 1))
    plt.title('Prediction Strength vs Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Prediction Strength')
    plt.grid(True)
    plt.show()

# Example usage:
results = prediction_strength_method_KMeans(data)
plot_prediction_strength(results)

# Встановлення оптимальної кількості кластерів (завдання 5)
optimal_k = 2

# 6. Кластеризація методом k-means++ з обраною кількістю кластерів
kmeans = None
best_score = -1  # Silhouette score ranges from -1 to 1, so we start with the lowest possible
chosen_model = 0
error_sqd = []

# Iterate over the number of models
for n in range(20):
    # Initialize and fit the KMeans model
    model = KMeans(n_clusters=optimal_k, init='k-means++', n_init=1, random_state=n)
    model.fit(data)

    # Compute the silhouette score and store it
    labels = model.labels_
    score = silhouette_score(data, labels)
    error_sqd.append([n + 1, score])

    # Check if the current model is better (higher silhouette score)
    if score > best_score:
        kmeans = model
        best_score = score
        chosen_model = n + 1

# Print the top model details
print(f'Top model - {chosen_model}')
print(f'With silhouette score - {best_score}')
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Function to plot the error plot
def plot_error_plot(error_sqd, title):
    error_sqd = sorted(error_sqd, key=lambda x: x[0])
    models, scores = zip(*error_sqd)
    plt.figure(figsize=(10, 6))
    plt.plot(models, scores, marker='o')
    plt.title(title)
    plt.xlabel('Model Number')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()

# Plot the error plot to detect the best model
plot_error_plot(error_sqd, 'Detect Best Model')

# 7. Виконати кластеризацію за допомогою AgglomerativeClustering
agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
agg_labels = agg_clustering.fit_predict(data)

# В AgglomerativeClustering немає явних центрів кластерів, тому для отримання центрів обчислимо середнє значення для кожного кластеру
agg_centroids = []
for i in range(optimal_k):
    agg_centroids.append(data[agg_labels == i].mean())
agg_centroids = np.array(agg_centroids)

print('Agglomerative Centroids:', agg_centroids)
print('Agglomerative Labels:', agg_labels)

# 8. Порівняння результатів двох використаних методів кластеризації
def plot_clusters(data, labels, centroids, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75)
    plt.title(title)
    plt.show()

# Візуалізація кластерів K-means++
plot_clusters(data, labels, centroids, 'K-means++ Clustering')

# Візуалізація кластерів AgglomerativeClustering
plot_clusters(data, agg_labels, agg_centroids, 'Agglomerative Clustering')