import json

input_file_path = r'C:\Users\lovis\Desktop\TNM108\tnm108_bookrecommender\public\util\goodreads_books_mystery_thriller_crime.json'
output_file_path = 'output_file.json'

with open(input_file_path, 'r') as input_file:
    # Read each line from the input file
    lines = input_file.readlines()

    # Add commas between JSON objects
    data = ','.join(lines)

    # Wrap the data in square brackets to create a valid JSON array
    data = '[' + data + ']'

    # Parse the modified JSON data
    try:
        books = json.loads(data)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        exit()

    # Write the modified JSON back to a new file
    with open(output_file_path, 'w') as output_file:
        json.dump(books, output_file, indent=2)

print(f"Modified JSON written to {output_file_path}")
