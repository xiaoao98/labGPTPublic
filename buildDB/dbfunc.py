import json  
import csv 
import sqlite3  

def json_to_dict(json_file_path, instruction_dict):  
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

def populate_dict_from_csv(csv_file_path, a_dictionary, skip_header=False):
  """
  Reads a CSV file and populates a dictionary with its content based on column position.
  The first element in each row is treated as the key (Protocol name)
  and the second element as the value (Protocol content).

  Args:
    csv_file_path: The path to the CSV file.
    a_dictionary: The dictionary to populate.
    skip_header: (Optional) Boolean, True if the first row is a header
                 and should be skipped. Defaults to False.
  """
  try:
    with open(csv_file_path, mode='r', newline='', encoding='utf-8-sig') as csvfile:
      reader = csv.reader(csvfile)

      if skip_header:
        try:
          next(reader) # Skip the header row
          print("Skipped header row.")
        except StopIteration:
          # This means the file was empty or only had a header
          print(f"Warning: File {csv_file_path} was empty or only contained a header after trying to skip it.")
          return

      for i, row in enumerate(reader):
        if len(row) >= 2:
          protocol_name = row[0]
          protocol_content = row[1]
          a_dictionary[protocol_name] = protocol_content
        else:
          # Adjust row number for display if header was skipped
          row_number_for_display = i + 1 if not skip_header else i + 2
          print(f"Warning: Row {row_number_for_display} in {csv_file_path} has fewer than 2 elements and will be skipped: {row}")
    # print(f"Successfully populated dictionary from {csv_file_path}")
  except FileNotFoundError:
    print(f"Error: The file {csv_file_path} was not found.")
  except Exception as e:
    print(f"An error occurred: {e}")


def create_and_insert_table_from_dict(db_name: str, table_name: str, data: dict):  
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
        # print(table_name) 
        create_table_query = f'''  
        CREATE TABLE IF NOT EXISTS "{table_name}" (  
            experiment TEXT PRIMARY KEY,  
            protocol TEXT  
        )  
        '''  
        # print(create_table_query)
        cursor.execute(create_table_query)  

        # Step 4: Insert data from the dictionary into the table  
        for experiment, protocol in data.items():  
            insert_query = f'INSERT INTO "{table_name}" (experiment, protocol) VALUES (?, ?)'  
            # print(insert_query)
            cursor.execute(insert_query, (experiment, protocol))  

        # Commit the changes  
        conn.commit()  
        
    except sqlite3.Error as e:  
        print(f"An error occurred: {e}")  
    
    finally:  
        # Ensure the connection is always closed  
        conn.close()  
        
def insert_table_from_dict(db_name: str, table_name: str, data: dict):  
    """  
    Create a SQLite table and insert data from a dictionary.  

    Parameters:  
    db_name (str): The name of the SQLite database file.  
    data (dict): A dictionary with keys as experiments and values as protocols.  
    """  
    conn = sqlite3.connect(db_name)
    try:  
        cursor = conn.cursor()  

        for experiment, content in data.items():  
            insert_query = f'INSERT INTO "{table_name}" (experiment, content) VALUES (?, ?)'  
            # print(insert_query)
            cursor.execute(insert_query, (experiment, content))  

        # Commit the changes  
        conn.commit()  
        
    except sqlite3.Error as e:  
        print(f"An error occurred: {e}")  
    
    finally:  
        # Ensure the connection is always closed  
        conn.close()  

if __name__ == "__main__": 
    instructions_to_output = {}  
    # json_to_dict('cellSOP.json', instructions_to_output)  
    # populate_dict_from_csv('protocol.csv', instructions_to_output)
    # print(instructions_to_output["Tumor digestion for cryopreservation"])
    # create_and_insert_table_from_dict("experiments.db", "protocols", instructions_to_output)
    populate_dict_from_csv('reagents.csv', instructions_to_output)
    # print(instructions_to_output["Western Blot"])
    insert_table_from_dict("experiments.db", "reagents", instructions_to_output)
    # Print the populated dictionary to verify  
