import json
import os # Added for file path operations

def transform_json_from_file(json_file_path):
    """
    Parses a JSON file containing a list of instruction-output pairs
    and transforms them into a single formatted text string.

    Args:
        json_file_path (str): The path to the JSON file.

    Returns:
        str: A single string with all instructions and outputs formatted,
             or an error message if parsing or file reading fails.
    """
    try:
        if not os.path.exists(json_file_path):
            return f"Error: File not found at '{json_file_path}'"

        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f) # Use json.load for file objects
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in file '{json_file_path}'. Details: {e}"
    except IOError as e:
        return f"Error: Could not read file '{json_file_path}'. Details: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


    transformed_text_parts = []
    # Ensure data is a list, as expected from the JSON structure
    if not isinstance(data, list):
        return "Error: JSON content is not a list as expected."

    for item in data:
        # Ensure each item in the list is a dictionary
        if not isinstance(item, dict):
            transformed_text_parts.append("Warning: Encountered a non-dictionary item in the JSON list. Skipping.\n")
            continue

        instruction = item.get("instruction", "No instruction provided")
        output = item.get("output", "No output provided")

        # You can format this part as you like.
        # Here's one way to format it clearly:
        transformed_text_parts.append(f"{instruction}?\t{output}")

    # Join all parts with an extra newline for separation between entries
    return "\t".join(transformed_text_parts)

if __name__ == "__main__":
    json_path = "safety0.json"
    print(transform_json_from_file(json_file_path=json_path))