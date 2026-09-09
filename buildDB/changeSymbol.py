import json  

input_json = "protocol_0.json"
output_json = "protocol.json"

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

replace_whitespace_in_json(input_json, output_json) 