import pandas as pd  
import sqlite3 
import json  


json_file_path = 'cellReagentsNew.json' 
# Load the Excel file  
excel_file = 'Cells in Current Cell Bank.xlsx'  

# Read the Excel file into a DataFrame  
df = pd.read_excel(excel_file, sheet_name='MASTER')  

# Convert the DataFrame to an array of dictionaries  
columns_to_extract = ['Cell line', 'Complete Media', 'Cryopreservation media']  
extracted_df = df[columns_to_extract]  
  
# Example: assuming your DataFrame columns are named 'Question' and 'Answer'  
datas = list(extracted_df.itertuples(index=False, name=None))  

# print(datas)

# Connect to the SQLite database (or create it if it doesn't exist)  
connection = sqlite3.connect('reagents.db')  

# Create a cursor object to execute SQL commands  
cursor = connection.cursor()  

# Create a table (if it doesn't exist already) with three columns  
cursor.execute('''  
CREATE TABLE IF NOT EXISTS reagents (  
    id INTEGER PRIMARY KEY,  
    experiment TEXT,  
    reagents TEXT  
)  
''')  

# Insert each tuple into the table  
for data in datas:
    experiment1 = data[0] + " Cryorecovery"
    reagents_experiment1 =  str(data[1])
    experiment2 = data[0] + " Cryopreservation"
    reagents_experiment2 = str(data[2]) + "; PBS; Cell dissociation buffer (Trypsin, TrypLE, or Accutase)"
    cursor.execute('INSERT INTO reagents (experiment, reagents) VALUES (?, ?)', (experiment1, reagents_experiment1))  
    cursor.execute('INSERT INTO reagents (experiment, reagents) VALUES (?, ?)', (experiment2, reagents_experiment2))

# Commit the transaction  
connection.commit()  

# Close the connection  
connection.close()
