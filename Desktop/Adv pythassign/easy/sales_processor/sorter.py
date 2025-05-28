def sort_data(data, key, reverse=False):
    """Sort data by a specific key."""
    return sorted(data, key=lambda x: x.get(key, ''), reverse=reverse)