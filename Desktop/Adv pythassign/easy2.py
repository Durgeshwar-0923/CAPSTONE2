def format_user_names(users, format_func=None):
    """
    Format a list of user dictionaries according to the provided format function.

    Args:
        users: List of user dictionaries
        format_func: Function that takes a user dict and returns a formatted string
                    If None, defaults to "first_name last_name"

    Returns:
        List of formatted user names
    """
    if format_func is None:
        format_func = lambda user: f"{user['first_name']} {user['last_name']}"

    return [format_func(user) for user in users]

# Usage examples
users = [
    {"first_name": "John", "last_name": "Doe", "username": "jdoe"},
    {"first_name": "Jane", "last_name": "Smith", "username": "jsmith"}
]


# Custom formatting
usernames = format_user_names(users, lambda user: f"{user['first_name'].lower()+user['last_name'].lower()}@minfytech.com")
print(usernames)  # ['@jdoe', '@jsmith']

# # Assinmnet 2 Easy :['jhondoe@minfytech.com']


