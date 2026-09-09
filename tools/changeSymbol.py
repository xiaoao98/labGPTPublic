import json  
import os

input_json = "protocol_0.json"
output_json = "protocol.json"
input_txt = "memberinfo.txt"
output_txt = "memberinfo_m.txt"

def replace_whitespace_in_json(file_path, output_path):  
    with open(file_path, 'r', encoding='utf-8') as f:  
        data = json.load(f)  
    
    def replace_whitespace(obj):  
        if isinstance(obj, str):  
            return obj.replace('\n', ', ').replace('\t', ' ').replace('\u2019', '\'').replace('\u2013', '-').replace('\u2014', '-').replace('\u00b0', '.')
        elif isinstance(obj, list):  
            return [replace_whitespace(item) for item in obj]  
        elif isinstance(obj, dict):  
            return {key: replace_whitespace(value) for key, value in obj.items()}  
        return obj  
    
    modified_data = replace_whitespace(data)  
    
    with open(output_path, 'w', encoding='utf-8') as f:  
        json.dump(modified_data, f, indent=4)  

def replace_newlines_in_file(input_filepath, output_filepath=None):
    """
    Reads a text file, replaces all newline characters ('\n') with tab characters ('\t').
    The modified content is either written to a new output file or printed to the console.

    Args:
        input_filepath (str): The path to the input text file.
        output_filepath (str, optional): The path to the new output file.
                                         If None, the modified content will be printed to stdout.
                                         Defaults to None.

    Returns:
        bool: True if the operation was successful, False otherwise.

    Raises:
        FileNotFoundError: If the input_filepath does not exist (raised by open()).
        IOError: If there are issues reading or writing files.
    """
    try:
        # Check if the input file exists
        if not os.path.exists(input_filepath):
            print(f"Error: Input file not found at '{input_filepath}'")
            return False

        # Read the content of the input file
        with open(input_filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        # print(f"Successfully read file: {input_filepath}") # Optional: for verbose output

        # Replace newline characters with tab characters
        modified_content = content.replace('\n', '\t')
        # print("Newline characters replaced with tabs.") # Optional: for verbose output

        if output_filepath:
            # Write to a new output file
            with open(output_filepath, 'w', encoding='utf-8') as file:
                file.write(modified_content)
            print(f"Successfully wrote modified content to: {output_filepath}")
        else:
            # Print to standard output if no output file is specified
            print("\n--- Modified Content ---")
            print(modified_content)
            print("--- End of Modified Content ---")

        return True

    except FileNotFoundError: # More specific catch after os.path.exists check
        print(f"Error: Input file disappeared or became inaccessible: '{input_filepath}'")
        return False
    except IOError as io_error:
        print(f"An IO error occurred: {io_error}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False
    
# replace_whitespace_in_json(input_json, output_json) 
replace_newlines_in_file(input_txt, output_txt)