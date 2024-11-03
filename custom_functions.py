import numpy as np
import pandas as pd
import pm4py as pm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

def get_resource_profiles(resource_activity_matrix):
    """Extracts resource profiles and their names from the resource-activity matrix."""
    resource_profiles = resource_activity_matrix.values
    resources = resource_activity_matrix.index.tolist()
    return resource_profiles, resources


def select_distance_measure():
    """Prompts the user to select a distance measure and returns the choice."""
    distance_measures = {
        '1': 'Minkowski distance',
        '2': 'Hamming distance',
        '3': 'Pearson correlation coefficient'
    }

    # Construct the input prompt with options
    options_str = "\n".join([f"{key}: {value}" for key, value in distance_measures.items()])
    prompt = f"Select the distance measure:\n{options_str}\nEnter the number corresponding to your choice: "

    measure_choice = input(prompt)
    return measure_choice


def calculate_similarity_matrix(resource_profiles, measure_choice):
    """Calculates the similarity matrix based on the selected distance measure."""
    if measure_choice == '1':
        p = float(input("Enter the value of p for Minkowski distance (e.g., 1 for Manhattan, 2 for Euclidean): "))
        distances = pdist(resource_profiles, metric='minkowski', p=p)
        distance_matrix = squareform(distances)
        similarity_matrix = 1 / (1 + distance_matrix)

    elif measure_choice == '2':
        distances = pdist(resource_profiles, metric='hamming')
        distance_matrix = squareform(distances)
        similarity_matrix = 1 / (1 + distance_matrix)

    elif measure_choice == '3':
        similarity_matrix = np.corrcoef(resource_profiles)
        similarity_matrix[similarity_matrix < 0] = 0  # Optional: Set negative correlations to zero
    else:
        raise ValueError("Invalid choice of distance measure.")

    return similarity_matrix


def apply_similarity_threshold(similarity_matrix, resources, threshold):
    """Applies the threshold to the similarity matrix and returns a DataFrame."""
    similarity_df = pd.DataFrame(similarity_matrix, index=resources, columns=resources)
    similarity_df_thresholded = similarity_df.copy()
    similarity_df_thresholded[similarity_df_thresholded < threshold] = 0
    return similarity_df_thresholded


def extract_network_edges(similarity_df):
    """Extracts edges from the thresholded similarity matrix to build the social network."""
    resources = similarity_df.index.tolist()
    edges = []
    for i in range(len(resources)):
        for j in range(i + 1, len(resources)):
            sim = similarity_df.iloc[i, j]
            if sim > 0:
                edges.append((resources[i], resources[j], sim))
    return edges


def display_network_edges(edges):
    """Displays the edges of the social network."""
    print("\nSocial Network Edges (Resource Pairs with Similarity above Threshold):")
    for edge in edges:
        print(f"{edge[0]} -- {edge[1]} (Similarity: {edge[2]:.4f})")


def visualize_social_network(edges):
    """Visualizes the social network using NetworkX."""
    G = nx.Graph()
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=edge[2])

    pos = nx.spring_layout(G, k=0.5)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=False, node_color='skyblue', edge_color=weights, width=2.0, edge_cmap=plt.cm.Blues,
            edgecolors='black')
    plt.title("Social Network Graph")
    plt.show()


def generate_social_network(resource_activity_matrix):
    """Generates and visualizes the social network from the resource-activity matrix."""
    # Get resource profiles
    resource_profiles, resources = get_resource_profiles(resource_activity_matrix)

    # Select distance measure
    measure_choice = select_distance_measure()

    # Calculate similarity matrix
    similarity_matrix = calculate_similarity_matrix(resource_profiles, measure_choice)

    # Apply similarity threshold
    threshold = float(input("Enter the threshold value to remove weak connections (e.g., 0.5): "))
    similarity_df_thresholded = apply_similarity_threshold(similarity_matrix, resources, threshold)

    # Extract network edges
    edges = extract_network_edges(similarity_df_thresholded)

    # Display network edges
    display_network_edges(edges)

    # Visualize the social network
    visualize_social_network(edges)


def kmeans_clustering(resource_profiles, k, max_iterations=100):
    """Performs K-Means clustering on the resource profiles."""
    np.random.seed(42)
    centroids = resource_profiles[np.random.choice(resource_profiles.shape[0], k, replace=False)]

    for _ in range(max_iterations):
        distances = np.linalg.norm(resource_profiles[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([resource_profiles[labels == i].mean(axis=0) for i in range(k)])

        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return labels, centroids


def elbow_method(resource_profiles, max_k=10):
    """Displays the elbow plot for K-Means to determine the optimal k."""
    distortions = []
    for k in range(1, max_k + 1):
        labels, centroids = kmeans_clustering(resource_profiles, k)
        distances = np.linalg.norm(resource_profiles - centroids[labels], axis=1)
        distortions.append(np.sum(distances ** 2))

    plt.plot(range(1, max_k + 1), distortions, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.show()


def hierarchical_clustering(resource_profiles, resources, method='ward'):
    """Performs hierarchical clustering and plots a dendrogram."""
    Z = linkage(resource_profiles, method=method)

    plt.figure(figsize=(10, 7))
    dendrogram(Z, labels=resources)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Resources')
    plt.ylabel('Distance')
    plt.show()

    return Z


def add_clusters_to_data(resource_activity_matrix, labels, resources):
    """Adds the cluster labels as a new column in the resource-activity DataFrame."""
    resource_activity_matrix['cluster'] = pd.Series(labels, index=resources)
    return resource_activity_matrix


def ask_for_k_or_clustering(resource_activity_matrix, resource_profiles, resources):
    """Asks the user if they want to proceed with K-Means (and input k) or hierarchical clustering."""

    choice = input(
        "Select a clustering approach:\n"
        "1: Proceed with K-Means (enter the number of clusters k)\n"
        "2: Hierarchical clustering\n"
        "Enter your choice (1/2): "
    )

    if choice == '1':
        k = int(input("Enter the number of clusters (k): "))
        labels, centroids = kmeans_clustering(resource_profiles, k)
        resource_activity_matrix = add_clusters_to_data(resource_activity_matrix, labels, resources)
        print("\nClusters added to resource_activity_matrix.")

    elif choice == '2':
        method = input("Enter the linkage method (ward, single, complete, average): ")
        Z = hierarchical_clustering(resource_profiles, resources, method)

        k = int(input("Enter the number of clusters to form: "))
        clusters = fcluster(Z, k, criterion='maxclust')
        resource_activity_matrix = add_clusters_to_data(resource_activity_matrix, clusters, resources)
        print("\nClusters added to resource_activity_matrix.")

    else:
        print("Invalid choice. Please select either 1 or 2.")

    return resource_activity_matrix


def visualize_network_with_clusters(resource_activity_matrix):
    """Visualizes the network with nodes colored based on cluster assignments."""

    # Generate the similarity matrix, apply threshold, and extract edges using generate_social_network functions
    resource_profiles, resources = get_resource_profiles(resource_activity_matrix)
    measure_choice = select_distance_measure()
    similarity_matrix = calculate_similarity_matrix(resource_profiles, measure_choice)

    threshold = float(input("Enter the threshold value to remove weak connections (e.g., 0.5): "))
    similarity_df_thresholded = apply_similarity_threshold(similarity_matrix, resources, threshold)
    edges = extract_network_edges(similarity_df_thresholded)

    # Initialize the graph
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    # Perform clustering before visualization
    resource_activity_matrix = ask_for_k_or_clustering(resource_activity_matrix, resource_profiles, resources)

    # Assign node colors based on cluster assignments
    cluster_colors = resource_activity_matrix['cluster']
    color_map = [cluster_colors[resource] for resource in G.nodes()]

    # Draw the network with cluster-based coloring
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color=color_map, with_labels=False, cmap=plt.cm.tab20, node_size=300, edge_color='gray',
            edgecolors='black')
    plt.title("Network Visualization with Cluster-Based Node Coloring")
    plt.show()
