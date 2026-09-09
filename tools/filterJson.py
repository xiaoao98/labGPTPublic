import json

def filter_json_abstracts(input_path, output_path):
    try:
        # 1. Load the original data
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 2. Filter the items
        # item_id is the key (e.g., "8", "9"), and content is the dictionary
        cleaned_data = {
            item_id: {
                "title": content.get("title", ""),
                "abstract": content.get("abstract", "")
            }
            for item_id, content in data.items()
        }
        
        # 3. Save to a new file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
            
        print(f"Successfully cleaned data. Saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Usage
filter_json_abstracts('papers_raw.json', 'papers.json')