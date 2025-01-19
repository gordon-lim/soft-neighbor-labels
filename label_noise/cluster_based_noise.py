# OpenAI's ChatGPT helped in writing this script
import numpy as np
from sklearn.manifold import TSNE  # Import t-SNE
from tqdm import tqdm  # Import tqdm for progress bars
import argparse
import pickle

def cluster_based_noise(clip_embeddings, labels, n=23, r=1.2):
    """
    Apply cluster-based noise to flip labels to the nearest other class based on t-SNE transformed embeddings.

    Parameters:
    - clip_embeddings: Array of shape (n_samples, embedding_dim) for the CLIP embeddings.
    - labels: Array of shape (n_samples,) for the original labels.
    - n: Number of random subcluster centroids.
    - r: Radius for label flipping.

    Returns:
    - noisy_labels: Array of shape (n_samples,) with updated noisy labels.
    """
    
    # Step 1: Initialize noisy labels
    noisy_labels = labels.copy()

    # Step 2: Get unique labels
    unique_labels = np.unique(labels)

    # Step 3: Compute t-SNE reduced vectors
    tsne_embeddings = TSNE(n_components=2).fit_transform(clip_embeddings)

    # Step 4: Calculate centroids for each class using t-SNE embeddings
    centroids = {}
    for y in unique_labels:
        indices = np.where(labels == y)[0]  # Get indices for the current label
        class_embeddings = tsne_embeddings[indices]  # Get the t-SNE embeddings for this class
        centroids[y] = np.mean(class_embeddings, axis=0)  # Mean of t-SNE embeddings

    # Step 5: Loop through each label category
    for y in tqdm(unique_labels, desc="Processing Classes"):
        indices = np.where(labels == y)[0]  # Get indices for the current label
        assert len(indices) >= n, f"Not enough samples in class {y} to sample {n} subcluster centroids."

        # Sample subcluster centroids from the t-SNE embeddings of the current class
        subcluster_centroids = tsne_embeddings[np.random.choice(indices, n, replace=False)]  # Randomly sample n points

        # Loop through each data point with label y
        for idx in indices:
            xi = tsne_embeddings[idx]  # Current data point in t-SNE space
            for j in range(n):  # Loop through subcluster centroids
                distance = np.linalg.norm(xi - subcluster_centroids[j])  # Calculate distance
                # Step 6: Check if within radius
                if distance < r:  # Use the defined radius
                    # Step 7: Find the nearest other class centroid
                    min_distance = np.inf
                    closest_label = None

                    for y_prime in unique_labels:
                        if y_prime != y:  # Exclude the current label
                            distance_to_centroid = np.linalg.norm(xi - centroids[y_prime])
                            if distance_to_centroid < min_distance:
                                min_distance = distance_to_centroid
                                closest_label = y_prime

                    # Step 8: Assign the label of the nearest other class
                    if closest_label is not None:
                        noisy_labels[idx] = closest_label  # Assign new noisy label

    noise_rate = sum(labels != noisy_labels) / len(labels)
    
    return noisy_labels, noise_rate

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Apply Cluster-Based Noise to CLIP embeddings and labels.")
    parser.add_argument("--embeddings_file", required=True, 
                        help="Path to the file containing the CLIP embeddings (Pickle format).")
    parser.add_argument("--labels_file", required=True, 
                        help="Path to the file containing the labels (.npy format).")
    parser.add_argument("--dataset", required=True, 
                        help="Name of the dataset (e.g., cifar10 or cifar100) for output filename.")
    parser.add_argument("--n", type=int, default=23, 
                        help="Number of random subcluster centroids (default: 23).")
    parser.add_argument("--r", type=float, default=1.2, 
                        help="Radius for label flipping (default: 1.2).")
    args = parser.parse_args()

    # Load embeddings and labels
    with open(args.embeddings_file, "rb") as f:
        clip_embeddings = pickle.load(f)

    labels = np.load(args.labels_file)

    # Apply Cluster-Based Noise
    noisy_labels, noise_rate = cluster_based_noise(clip_embeddings, labels, n=args.n, r=args.r)

    # Generate output filename
    output_filename = f"{args.dataset}-cluster-{noise_rate:.2f}.npy"

    # Save noisy labels
    np.save(output_filename, noisy_labels)

    print(f"Noisy labels saved to {output_filename}")