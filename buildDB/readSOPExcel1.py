import pandas as pd  
import json  


json_file_path = 'cellSOP.json' 
# Load the Excel file  
excel_file = 'example_qa.xlsx'  

# Read the Excel file into a DataFrame  
df = pd.read_excel(excel_file, sheet_name='Sheet1')  

# Convert the DataFrame to an array of dictionaries  
array_of_dicts = df.to_dict(orient='records')  

# Optionally, rename keys to 'question' and 'answer' if they have different names  
# Example: assuming your DataFrame columns are named 'Question' and 'Answer'  
formatted_array_of_dicts = [
    {
        "instruction": item['question'], 
        "input": "",
        "output": item['answer']
    } for item in array_of_dicts]  

# print(formatted_array_of_dicts)

# Write the dictionary to a JSON file  
with open(json_file_path, 'w') as json_file:  
    json.dump(formatted_array_of_dicts, json_file, indent=4)  

print(f"Data successfully written to {json_file_path}") 