import os  
import csv  

def list_files_with_author(directory):  
    # Get a list of all files in the specified directory  
    filesList = []
    for author_directory in os.listdir(directory):
        author_path = os.path.join(directory, author_directory)
        if os.path.isdir(author_path) and author_directory.startswith("Protocol_"):  
                try:  
                    author_name = author_directory.split("_")[1]  
                    for _, _, files in os.walk(author_path):
                        for file in files:
                            # print(file)
                            if file.endswith(".pdf") or file.endswith(".docx"):
                                filesList.append(file + "_" + author_name)
                except IndexError:  
                    continue   
    return filesList

def files_to_csv(fileList, csv_filename):
    # Open the CSV file in write mode  
    with open(csv_filename, mode='w', newline='') as csvfile:  
        writer = csv.writer(csvfile)  

        # Write the header  
        writer.writerow(['Filename'])  

        # Iterate over the list of files and write each one to the CSV  
        for file in fileList:  
            # Only include files, not directories  
            writer.writerow([file])  

    print(f"Filenames have been written to {csv_filename}")  

# Example usage  
directory_path = '/Users/zyu7/Documents/KRgpt/data/SOPs/2-4-2025'  # Replace with your directory path  
csv_output_path = 'output.csv'             # The CSV file to write to  

files_to_csv(list_files_with_author(directory_path), csv_output_path)
# list_files_to_csv(directory_path, csv_output_path)