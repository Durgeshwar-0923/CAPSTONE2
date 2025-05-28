import os
import sales_processor

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "sales.csv")  # Ensures correct file path

try:
    # Load sales data
    sales_data = sales_processor.read_csv(csv_path)

    # Filter for a specific region
    west_region_sales = sales_processor.filter_data(sales_data, {'Region': 'West'})

    # Sort by amount
    sorted_sales = sales_processor.sort_data(west_region_sales, 'Amount', reverse=True)

    # Save the top sales to JSON
    json_path = os.path.join(script_dir, "top_west_sales.json")
    sales_processor.save_as_json(sorted_sales[:10], json_path)

    # Generate summary by product category
    category_summary = sales_processor.generate_summary(sales_data, 'Category')

    print("Sales by category:")
    for category, count in category_summary.items():
        print(f"{category}: {count}")

except FileNotFoundError as e:
    print(f"Error: CSV file not found: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
