import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def filter_by_kde_nd(data, x, y):
    """
    Perform KDE on a list of n-dimensional points and filter points based on KDE values.

    Args:
        data (array-like): A list or array of n-dimensional points.
        x (float): Lower threshold for KDE value.
        y (float): Upper threshold for KDE value.

    Returns:
        tuple: Filtered list of n-dimensional points within KDE thresholds,
               list of removed points.
    """
    data = np.array(data).T  # Transpose for KDE (n, samples)
    kde = gaussian_kde(data)
    kde_values = kde(data)  # Evaluate KDE on original data points
    
    # Filter data based on KDE thresholds
    filtered_data = [point for point, kde_val in zip(data.T, kde_values) if x <= kde_val <= y]
    removed_data = [point for point, kde_val in zip(data.T, kde_values) if kde_val < x or kde_val > y]
    
    return np.array(filtered_data), np.array(removed_data)  # Return as arrays

def main():
    # Generate five lists of sample 3D data from a multivariate normal distribution
    np.random.seed(0)
    list1 = np.random.multivariate_normal(mean=[0, 0, 0], cov=[[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]], size=20)
    list2 = np.random.multivariate_normal(mean=[1, 1, 1], cov=[[1, 0.3, 0.3], [0.3, 1, 0.3], [0.3, 0.3, 1]], size=20)
    list3 = np.random.multivariate_normal(mean=[-1, -1, -1], cov=[[1, 0.4, 0.4], [0.4, 1, 0.4], [0.4, 0.4, 1]], size=20)
    list4 = np.random.multivariate_normal(mean=[2, 2, 2], cov=[[1, 0.2, 0.2], [0.2, 1, 0.2], [0.2, 0.2, 1]], size=20)
    list5 = np.random.multivariate_normal(mean=[-2, -2, -2], cov=[[1, 0.6, 0.6], [0.6, 1, 0.6], [0.6, 0.6, 1]], size=20)
    
    # Combine all lists into a single list for KDE processing
    combined_data = np.vstack([list1, list2, list3, list4, list5])
    
    # Define KDE value thresholds
    x_threshold = 0.15
    y_threshold = 0.14
    
    # Apply KDE filtering on the combined list
    filtered_data, removed_data = filter_by_kde_nd(combined_data, x=x_threshold, y=y_threshold)
    
    # Go through each original list and filter out the removed elements
    lists = [list1, list2, list3, list4, list5]
    filtered_lists = []
    for i, original_list in enumerate(lists):
        filtered_list = [point for point in original_list if not any(np.all(point == rem) for rem in removed_data)]
        filtered_lists.append(np.array(filtered_list))  # Convert to array for easy manipulation

        # Display the vector with the largest norm for each filtered list
        if len(filtered_list) > 0:
            largest_norm_vector = max(filtered_list, key=lambda v: np.linalg.norm(v))
            print(f"Largest norm vector in list {i+1} after filtering: {largest_norm_vector}")
        else:
            print(f"List {i+1} is empty after filtering.")
    
    # Display the removed elements
    print("\nRemoved elements:")
    for elem in removed_data:
        print(elem)

# Run the main function to test
if __name__ == "__main__":
    main()
