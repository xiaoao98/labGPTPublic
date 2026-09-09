from docx import Document 
import json  
import os 


# Reading the Word file  
# documentPath = './Exosome Isolation from serum.docx' 
dirPath = '/Users/zyu7/Documents/KRgpt/data/SOPs/categorized/Exosome Isolation'
signs = ["Reagents", "Materials", "Protocol", "Procedure", "Equipment", "Time", "Source of Cells", "Method", "Notes"]
output_json = "protocol7.json"

def startWithSigns(line: str):
    for sign in signs:
        if line.startswith(sign):
            return True
    return False

def process_section(i, question_prefix, title, lines, output):  
    i += 1  
    text = []  
    while i < len(lines) and not startWithSigns(lines[i]):  
        text.append(lines[i])  
        i += 1  
    question = f'What is the {question_prefix} for {title}'  
    output.append({  
        'instruction': question,
        'input': "",  
        'output': '\n'.join(text)  
    })  
    return i 

def appendJsonForOneFile(path):
    title = os.path.splitext(os.path.basename(path))[0]
    document = Document(path) 
    lines = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip() != ''] 
    # Remove the first line (the title)  
    lines.pop(0)

    output = []
    # all content are procedure
    if not startWithSigns(lines[0]):
        answer = '\n'.join(lines)
        # Generating question from document title  
            
        question = 'How to do ' + title  

        # Output JSON contents  
        output.append({  
            'instruction' : question,  
            'input': "",
            'output' : answer
        })
    else:
        i = 0
        while i < len(lines):
            # if lines[i].startswith("Reagents") or lines[i].startswith("Materials"):  
            #     i = process_section(i, "materials", title, lines, output)  
            # if lines[i].startswith("Source of Cells"):  
            #     i = process_section(i, "source of cells", title, lines, output)
            # elif lines[i].startswith("Equipment"):  
            #     i = process_section(i, "equipment", title, lines, output)  
            if lines[i].startswith("Protocol") or lines[i].startswith("Procedure") or lines[i].startswith("Method"):  
                i = process_section(i, "procedure", title, lines, output)
            # elif lines[i].startswith("Notes"):  
            #     i = process_section(i, "notes", title, lines, output)
            else:
                i += 1   
                while i < len(lines) and not startWithSigns(lines[i]):   
                    i += 1 
            

    return output

with open(output_json, 'w') as file:  
    file.write('')  
outputs = []
for filename in os.listdir(dirPath):
    if filename.endswith('.docx'):  
        print(filename)
        # Create a absolute path  
        filepath = os.path.join(dirPath, filename)
        output = appendJsonForOneFile(filepath)
        outputs.extend(output)
# Creating JSON file  
with open(output_json, 'a') as json_file:  
    json.dump(outputs, json_file, indent=4)  
