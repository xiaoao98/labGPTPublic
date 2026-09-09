import json  

def populate_instructions_to_output(json_file_path, instruction_dict):  
    """  
    Reads a JSON file containing a list of dictionaries with 'instruction' and 'output' keys,  
    and populates the provided dictionary with 'instruction' as keys and 'output' as values.  

    :param json_file_path: Path to the JSON file.  
    :param instruction_dict: The dictionary to populate with instructions and outputs.  
    """  
    # Read and load JSON data  
    with open(json_file_path, 'r') as file:  
        data = json.load(file)  

    # Populate the dictionary  
    for item in data:  
        if 'instruction' in item and 'output' in item:  
            instruction_dict[item['instruction']] = item['output']  

import sqlite3  

def create_table_from_dict(db_name: str, data: dict):  
    """  
    Create a SQLite table and insert data from a dictionary.  

    Parameters:  
    db_name (str): The name of the SQLite database file.  
    data (dict): A dictionary with keys as experiments and values as protocols.  
    """  
    
    # Step 1: Connect to the SQLite database (or create it if it doesn't exist)  
    conn = sqlite3.connect(db_name)  

    try:  
        # Step 2: Create a cursor object using the connection  
        cursor = conn.cursor()  

        # Step 3: Create the table  
        create_table_query = '''  
        CREATE TABLE IF NOT EXISTS experiments (  
            experiment TEXT PRIMARY KEY,  
            protocol TEXT  
        )  
        '''  
        cursor.execute(create_table_query)  

        # Step 4: Insert data from the dictionary into the table  
        for experiment, protocol in data.items():  
            insert_query = 'INSERT INTO experiments (experiment, protocol) VALUES (?, ?)'  
            cursor.execute(insert_query, (experiment, protocol))  

        # Commit the changes  
        conn.commit()  
        
    except sqlite3.Error as e:  
        print(f"An error occurred: {e}")  
    
    finally:  
        # Ensure the connection is always closed  
        conn.close()  
        
 
if __name__ == "__main__": 
    instructions_to_output = {}  
    populate_instructions_to_output('cellSOP.json', instructions_to_output)  
    create_table_from_dict("protocol.db", instructions_to_output)
    # Print the populated dictionary to verify  
