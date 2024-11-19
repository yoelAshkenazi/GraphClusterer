import json
import csv

def json_to_csv(json_input_path, csv_output_path):
    """
    Converts a JSON file with a specific structure into a CSV file.

    Parameters:
    - json_input_path: str, path to the input JSON file.
    - csv_output_path: str, path where the output CSV will be saved.
    """
    try:
        # Read JSON data from the file
        with open(json_input_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        
        # Extract all metric categories (e.g., avg_index, largest_cluster_percentage, etc.)
        metrics = list(data.keys())
        
        # Collect all unique dataset names across all metrics
        dataset_names = set()
        for metric in metrics:
            dataset_names.update(data[metric].keys())
        dataset_names = sorted(dataset_names)  # Sort for consistent ordering
        
        # Prepare CSV headers
        headers = ['dataset name'] + metrics
        
        # Write data to CSV
        with open(csv_output_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)  # Write header row
            
            for dataset in dataset_names:
                row = [dataset]
                for metric in metrics:
                    value = data.get(metric, {}).get(dataset, "")
                    if isinstance(value, float) or isinstance(value, int):
                        # Round to three decimal places
                        value = round(value, 3)
                    row.append(value)
                writer.writerow(row)
        
        print(f"Successfully converted '{json_input_path}' to '{csv_output_path}'.")
    
    except FileNotFoundError:
        print(f"Error: The file '{json_input_path}' does not exist.")
    except json.JSONDecodeError:
        print(f"Error: The file '{json_input_path}' is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage:
if __name__ == "__main__":
    json_input = 'metrics.json'
    csv_output = 'metrics.csv'
    json_to_csv(json_input, csv_output)
