import json  
import os  

def merge_json_files(input_files, output_file):  
    # Initialize an empty list to store the combined data  
    merged_data = []  
    
    # Iterate over each file and extend the merged_data list  
    for file_path in input_files:  
        with open(file_path, 'r') as f:  
            data = json.load(f)  
            # Ensure the file contains an array, if not, raise an error  
            if isinstance(data, list):  
                merged_data.extend(data)  
            else:  
                raise ValueError(f"File {file_path} does not contain a valid array")  
    
    # Write the merged data to the output file  
    with open(output_file, 'w') as f:  
        json.dump(merged_data, f, indent=4)  # indent for better readability  

# List your input JSON files:  
# input_files = ['cellReagents1.json', 'cellReagents2.json', 'cellReagents3.json', 'cellReagents4.json', 'cellReagents5.json', 'cellReagents6.json', 'cellReagents7.json', 'cellReagents8.json', 'cellReagents9.json', 'cellReagents10.json']  # Add all your file paths  
# input_files = ['safety.json', 'memberInfo.json']
# input_files = ['protocol0.json', 'protocol1.json', 'protocol2.json', 'protocol3.json', 'protocol4.json', 'protocol5.json', 'protocol6.json', 'protocol7.json']
# input_files = ["memberInfo1.json", "memberInfo2.json", "memberInfo3.json", "memberInfo4.json", "memberInfoSplit.json"]
input_files = ["memberInfo.json", "safety.json"]
# Specify the output JSON file  
# output_file = 'biology.json'  
output_file = 'biology.json' 

# Run the merge function  
merge_json_files(input_files, output_file)  

print(f"All files have been merged into {output_file}")