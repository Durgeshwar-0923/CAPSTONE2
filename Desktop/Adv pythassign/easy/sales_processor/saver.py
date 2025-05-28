import json

def save_as_json(data, file_path):
    """Save data as a JSON file."""
    with open(file_path, 'w') as jsonfile:
        json.dump(data, jsonfile, indent=4)
