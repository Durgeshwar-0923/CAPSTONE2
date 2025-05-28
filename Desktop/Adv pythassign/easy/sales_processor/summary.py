def generate_summary(data, group_by):
    """Generate a summary by grouping data."""
    summary = {}
    for item in data:
        group = item.get(group_by)
        if group:
            summary[group] = summary.get(group, 0) + 1
    return summary
