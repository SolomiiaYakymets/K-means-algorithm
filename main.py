import random


def initialize_cluster(num_features, num_of_clusters):
    return [random.randint(0, num_of_clusters-1) for _ in range(num_features)]


def calculate_centroid(input_features_, clusters_):
    sum_cluster = {i: [0] * len(input_features_[0]) for i in set(clusters_)}
    centroids_list = []
    for features, cluster_ in zip(input_features_, clusters_):
        for i in range(len(features)):
            sum_cluster[cluster_][i] += features[i]
    for cluster_, features in sum_cluster.items():
        centroid = [feature / clusters_.count(cluster_) for feature in features]
        centroids_list.append(centroid)
    return centroids_list


def euclidean_distance(list1, list2):
    distance = 0
    for i in range(len(list2) - 1):
        distance += (list1[i] - list2[i]) ** 2
    return distance


def calculate_euclidean_distance(centroids_, input_features_):
    centroid_distances = []
    for feature in input_features_:
        distances_ = [euclidean_distance(centroid, feature) for centroid in centroids_]
        centroid_distances.append(distances_)
    return centroid_distances


def reassign_clusters(centroid_distances):
    clusters_ = []
    for distance in centroid_distances:
        closest_centroid = distance.index(min(distance))
        clusters_.append(closest_centroid)
    return clusters_


def calculate_purity(labels_, clusters_, k_):
    combined_data = zip(labels_, clusters_)
    label_cluster_count = {i: {} for i in range(k_)}
    for label_, cluster_ in combined_data:
        if label_ not in label_cluster_count[cluster_]:
            label_cluster_count[cluster_][label_] = 0
        label_cluster_count[cluster_][label_] += 1

    for cluster_, label_counts in label_cluster_count.items():
        total_sum = sum(label_counts.values())
        for _label, count in label_counts.items():
            percentage_ = (count / total_sum) * 100
            label_cluster_count[cluster_][_label] = percentage_

    return label_cluster_count


def calculate_sum_distances(centroids_, input_features_, cluster_assignments):
    sum_distances_ = 0
    centroid_distances = calculate_euclidean_distance(centroids_, input_features_)
    for i, distances_ in enumerate(centroid_distances):
        sum_distances_ += distances_[cluster_assignments[i]]
    return sum_distances_


def read_file(file_):
    input_features_ = []
    labels_ = []

    with open(file_) as f:
        for line in f:
            parts = line.strip().split(',')
            line = [float(x) for x in parts[:-1]]
            labels_.append(parts[-1])
            input_features_.append(line)

    return input_features_, labels_


file = input("Enter the name of the file to read: ")
k = int(input("Enter the number of clusters: "))
input_features, labels = read_file(file)
assigned_clusters = initialize_cluster(len(input_features), k)
while True:
    centroids = calculate_centroid(input_features, assigned_clusters)
    distances = calculate_euclidean_distance(centroids, input_features)
    new_clusters = reassign_clusters(distances)
    purity = calculate_purity(labels, new_clusters, k)
    for cluster, label in purity.items():
        print("Cluster", cluster + 1, ": ", end=" ")
        for label_name, percentage in label.items():
            print(str(percentage) + " % " + label_name, end=" ")
    print()
    sum_distances = calculate_sum_distances(centroids, input_features, new_clusters)
    print("Sum of distances:", sum_distances)
    print()
    if assigned_clusters == new_clusters:
        break
    assigned_clusters = new_clusters
