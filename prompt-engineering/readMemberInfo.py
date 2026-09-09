import json  

# Define the path to your text file  
text_file_path = 'memberInfo.txt'     
json_file_path = 'memberInfo1.json'      # Define the output JSON file path  

# Initialize a dictionary to store your JSON data  
datas = []  

# Read the text file  
with open(text_file_path, 'r') as file:  
    lines = file.readlines()  
    begin = 0
    end = begin+1
    number = 1
    # Ensure there are exactly 4 lines in the file as expected  
    while begin < len(lines):  
        # Assign the first line as the question and the other three lines as answers  
        question = lines[begin].strip()  # Remove newline characters  
        while end < len(lines) and lines[end].strip() != '':
            end += 1
        answer = ', '.join(line.strip() for line in lines[begin+1:end])  # Combine answers into one string  

        # Populate the dictionary  
        data = {  
            "instruction": "Who is " + question,  
            "input": "",  
            "output": answer
        } 
        datas.append(data)
        begin = end + 1
        while begin < len(lines) and lines[begin].strip() == '':
            begin += 1
        end = begin+1
        number += 1

# Write the dictionary to a JSON file  
with open(json_file_path, 'w') as json_file:  
    json.dump(datas, json_file, indent=4)  

print(f"Data successfully written to {json_file_path}")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
