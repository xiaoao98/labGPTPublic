import json  

def duplicate_json_content(input_file_path, output_file_path, num_duplicates):  
    # Step 1: Load the existing JSON file  
    with open(input_file_path, 'r') as input_file:  
        data = json.load(input_file)  

    # Step 2: Duplicate the content  
    # Assuming the JSON file contains an array of dictionaries  
    if isinstance(data, list):  # Check if the JSON data is a list  
        duplicated_data = data * num_duplicates  
    else:  
        raise ValueError("The JSON file does not contain a top-level array.")  

    # Step 3: Write the duplicated content to a new JSON file  
    with open(output_file_path, 'w') as output_file:  
        json.dump(duplicated_data, output_file, indent=4)  

# Example usage  
input_file_path = 'data_gen1.json'  # Path to your existing JSON file  
output_file_path = 'biology.json'  # Path where you want to save the duplicated content  
num_duplicates = 3  # Number of times to duplicate the content  

duplicate_json_content(input_file_path, output_file_path, num_duplicates)  

print(f"Content duplicated {num_duplicates} times and saved to {output_file_path}.")