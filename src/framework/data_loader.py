import json
import os

def load_sample_dataset():
    """
    Load the sample_dataset.json file from the datasets directory into memory.
    Returns:
        List[dict]: The loaded dataset as a list of dictionaries.
    """
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'datasets', 'sample_dataset.json')
    dataset_path = os.path.abspath(dataset_path)
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
