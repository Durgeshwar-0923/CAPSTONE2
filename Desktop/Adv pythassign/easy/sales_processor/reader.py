import csv
import os

def read_csv(file_path):
    """Read a CSV file and return a list of dictionaries."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)
