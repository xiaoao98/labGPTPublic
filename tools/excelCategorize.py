import os  
import shutil  
import pandas as pd  

def organize_files(excel_file_path, root_directory, target_directory):  
    # Load the Excel data into a DataFrame  
    df = pd.read_excel(excel_file_path)  

    # Iterate over each row in the DataFrame  
    for index, row in df.iterrows():  
        target_folder = row['Category']  
        file_name = row['Filename'] 
        file_name = file_name.rsplit('_', 1)[0]
        # print(file_name)
        # Create a full target folder path  
        full_target_folder_path = os.path.join(target_directory, target_folder)  
        print(full_target_folder_path)
        # Create the new folder if it doesn't exist  
        if not os.path.exists(full_target_folder_path):  
            os.makedirs(full_target_folder_path)  

        # Walk through the subfolders and search for the file  
        for subdir, dirs, files in os.walk(root_directory):  
            if file_name in files:  
                # Construct the full file paths  
                current_file_path = os.path.join(subdir, file_name)  
                target_file_path = os.path.join(full_target_folder_path, file_name)  

                # Move the file to the newly created folder  
                shutil.move(current_file_path, target_file_path)  
                print(f'Moved {file_name} to {full_target_folder_path}')  
                break  

if __name__ == "__main__":  
    # Replace these with your actual paths  
    excel_file_path = '/Users/zyu7/Documents/KRgpt/data/SOPs/categorized_protocols.xls'  # Path to your Excel file  
    root_directory = '/Users/zyu7/Documents/KRgpt/data/SOPs/2-4-2025'    # Root directory containing subfolders and files  
    target_directoty = '/Users/zyu7/Documents/KRgpt/data/SOPs/categorized'
    organize_files(excel_file_path, root_directory, target_directoty)