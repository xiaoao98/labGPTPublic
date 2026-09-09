import pandas as pd  
import json  


json_file_path = 'cellReagentsNew.json' 
# Load the Excel file  
excel_file = 'Cells in Current Cell Bank.xlsx'  

# Read the Excel file into a DataFrame  
df = pd.read_excel(excel_file, sheet_name='MASTER')  

# Convert the DataFrame to an array of dictionaries  
columns_to_extract = ['Cell line', 'Complete Media', 'Cryopreservation media']  
extracted_df = df[columns_to_extract]  

# Optionally, rename keys to 'question' and 'answer' if they have different names  
# Example: assuming your DataFrame columns are named 'Question' and 'Answer'  
datas = list(extracted_df.itertuples(index=False, name=None))  

# print(datas)

formatted_datas = []

for data in datas:
    formatted_data1 = {
        "instruction": "What are the reagents for " + data[0] + " Cryorecovery", 
        "input": "",
        "output": str(data[1])
    }
    formatted_data2 = {
        "instruction": "What are the reagents for " + data[0] + " Cryopreservation", 
        "input": "",
        "output": str(data[2]) + "; PBS; Cell dissociation buffer (Trypsin, TrypLE, or Accutase) "
    }
    formatted_datas.append(formatted_data1)
    formatted_datas.append(formatted_data2)
    # print(formatted_data1)

# print(formatted_array_of_dicts)

# Write the dictionary to a JSON file  
with open(json_file_path, 'w') as json_file:  
    json.dump(formatted_datas, json_file, indent=4)  

print(f"Data successfully written to {json_file_path}") 