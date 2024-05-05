import csv

def modify_date_format(input_file, output_file):
    with open(input_file, 'r') as input_csv, open(output_file, 'w', newline='') as output_csv:
        csv_reader = csv.reader(input_csv)
        csv_writer = csv.writer(output_csv)
        for row in csv_reader:
            if row:  # Check if row is not empty
                # Assuming date is in the first column
                # Split the date string by '/' and add leading zeros to day and month components if necessary
                date_components = row[0].split('/')
                day = date_components[0].zfill(2) if len(date_components[0]) < 2 else date_components[0]
                month = date_components[1].zfill(2) if len(date_components[1]) < 2 else date_components[1]
                modified_date = f"{day}/{month}/{date_components[2]}"
                # Write the modified row to the output CSV file
                csv_writer.writerow([modified_date] + row[1:])  # Write modified date along with other columns

# Example usage:
input_file_path = 'dataset.csv'  # Path to your input CSV file
output_file_path = 'output.csv'  # Path to the output CSV file with modified dates
modify_date_format(input_file_path, output_file_path)