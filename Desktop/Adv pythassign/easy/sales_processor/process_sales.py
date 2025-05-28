import sales_processor
import json
import os

try:
    sales_data = sales_processor.read_csv('sales.csv')

    # Filter for a specific region
    west_region_sales = sales_processor.filter_data(sales_data, {'Region': 'West'})

    # Sort by amount
    sorted_sales = sales_processor.sort_data(west_region_sales, 'Amount', reverse=True)

    # Save the top sales to JSON
    sales_processor.save_as_json(sorted_sales[:10], 'top_west_sales.json')

    # Generate summary by product category
    category_summary = sales_processor.generate_summary(sales_data, 'Category')
    print("Sales by category:")
    for category, count in category_summary.items():
        print(f"{category}: {count}")

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
