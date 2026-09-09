from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import labgpt_config as cfg
import os
import json

model_name = cfg.MODEL_NAME


def run_chat() -> None:
    json_file_path = cfg.PAPERS_PATH
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Convert the JSON object to a string so the model can read it
            paperInfoText = json.dumps(data, indent=2) 
    except FileNotFoundError:
        print(f"Error: {json_file_path} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {json_file_path}.")
        return
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    if os.name != "nt":
        try:
            import readline  # noqa: F401
        except ImportError:
            print("Install `readline` for a better experience.")

    # chat_model = ChatModel()
    messages = []
    print("Welcome to chat with Metis!")
    # print("Cryorecovery, how should I do this experiment?")

    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break
        # if query.strip() == "clear":
        #     messages = []
        #     print("History has been removed.")
        #     continue

        # messages.append({"role": "user", "content": query})
        print("Assistant: ", end="", flush=True)
        
        # for new_text in chat_model.stream_chat(messages):
        #     print(new_text, end="", flush=True)
        #     response += new_text
        # print("database_response: " + database_response)
        # system_instruction = """
        # You are an expert Research Assistant for the lab. 
        # Your task is to analyze the provided Research Abstracts JSON and find the most relevant papers based on the user's query.

        # GUIDELINES:
        # 1. Search specifically within the '### RESEARCH DATA' provided.
        # 2. Select up to 5 papers that have the highest semantic relevance to the query.
        # 3. For each paper, provide ONLY the exact Title.
        # 4. Format the output as a numbered list.
        # 5. If no papers are relevant, politely state that no matches were found.
        # """

        # Inside your run_chat() while loop:
        full_instructional_prompt = (
            "### DATASET\n"
            f"{paperInfoText}\n\n"
            "### USER QUERY\n"
            f"{query}\n\n"
            "### TASK\n"
            "Identify the top 5 relevant papers. Provide ONLY their numerical IDs as a comma-separated list.\n"
            "STRICT RESTRICTIONS:\n"
            "- No titles.\n"
            "- No introductory text or conversational filler.\n"
            "- Output ONLY the numbers (e.g., 1, 5, 12, 22, 30).\n\n"
            "### RELEVANT IDs:\n"
        )
        IDs = get_answers_only(tokenizer, model, full_instructional_prompt)
        print('\n' +"here" + '\n')
        print(IDs)
        # print_model_response(tokenizer, model, full_instructional_prompt)        
        # print(chat_output)
        # messages.append({"role": "assistant", "content": response})

def getIDs(query):
    json_file_path = cfg.PAPERS_PATH
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Convert the JSON object to a string so the model can read it
            paperInfoText = json.dumps(data, indent=2) 
    except FileNotFoundError:
        print(f"Error: {json_file_path} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {json_file_path}.")
        return
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    full_instructional_prompt = (
            "### DATASET\n"
            f"{paperInfoText}\n\n"
            "### USER QUERY\n"
            f"{query}\n\n"
            "### TASK\n"
            "Identify the top 5 relevant papers. Provide ONLY their numerical IDs as a comma-separated list.\n"
            "STRICT RESTRICTIONS:\n"
            "- No titles.\n"
            "- No introductory text or conversational filler.\n"
            "- Output ONLY the numbers (e.g., 1, 5, 12, 22, 30).\n\n"
            "### RELEVANT IDs:\n"
        )
    IDs = get_answers_only(tokenizer, model, full_instructional_prompt)
    print('\n' +"here" + '\n')
    print(IDs)

# load the tokenizer and the model
def print_model_response(tokenizer, model, input_content):
    messages = [
        {"role": "user", "content": input_content}
    ]
    # question = "I have three types of question: memberinfo, safety, biology experiment protocol. Please answer me the type of the following question (you can only choose one):  What does 2 Gallon sharps containers accepts. The answer should only be the type, no more explanation"

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    try:
        model.generate(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=65536
        )
    except Exception as e:
        print(f"\nError during model generation: {e}")

def get_answers_only(tokenizer, model, input_content):
    messages = [
        {"role": "user", "content": input_content}
    ]
    # question = "I have three types of question: memberinfo, safety, biology experiment protocol. Please answer me the type of the following question (you can only choose one):  What does 2 Gallon sharps containers accepts. The answer should only be the type, no more explanation"

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    try:
        # 1. Generate the output IDs
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=65536,
            # Ensure we don't use the streamer here if we just want the return
        )
        
        # 2. Slice the IDs to remove the original prompt tokens
        # model.generate returns the prompt + the new tokens
        new_tokens = generated_ids[0][len(model_inputs.input_ids[0]):]
        
        # 3. Decode to string
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        if "</think>" in response_text:
            final_answer = response_text.split("</think>")[-1].strip()
            return final_answer
    except Exception as e:
        print(f"\nError during model generation: {e}")

def fetch_paper_data(json_file_path, ids_input_str):
    """
    Fetches paper details from a JSON file based on a string of IDs.

    Args:
        json_file_path (str): Path to the .json file.
        ids_input_str (str): A string of IDs separated by commas (e.g., "1, 3, 5").

    Returns:
        tuple: (titles_string, full_info_string)
    """
    
    # 1. Parse the input string into a list of clean ID strings
    # We strip whitespace to handle inputs like "1, 3" vs "1,3"
    target_ids = [x.strip() for x in ids_input_str.split(',')]
    
    titles_output = []
    full_info_output = []

    try:
        # 2. Load the JSON data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 3. Iterate through requested IDs
        for paper_id in target_ids:
            if paper_id in data:
                entry = data[paper_id]
                
                # Extract fields safely (defaults to "N/A" if key missing)
                title = entry.get("title", "N/A")
                method = entry.get("method", "N/A")
                result = entry.get("result", "N/A")
                
                # Append to Titles Only list
                titles_output.append(f"{title}")
                
                # Append to Full Info list
                info_block = (
                    f"--- Title:  {title} ---\n"
                    f"Method: {method}\n"
                    f"Result: {result}\n"
                )
                full_info_output.append(info_block)
            else:
                # Handle cases where ID is not found in the JSON
                print(f"Warning: ID '{paper_id}' not found in the file.")

    except FileNotFoundError:
        return "Error: File not found.", "Error: File not found."
    except json.JSONDecodeError:
        return "Error: Invalid JSON format.", "Error: Invalid JSON format."

    # 4. Join lists into single strings
    titles_str = "\n".join(titles_output)
    full_info_str = "\n".join(full_info_output)
    
    return titles_str, full_info_str

# --- Example Usage ---
# Assuming you have a file named 'papers.json'

if __name__ == "__main__":
    run_chat()
    # getIDs("How to do Western Blot for COL1A1")
    # file_path = 'extracted_paper_content_modified.json'
    # id_string = "8, 9"
    
    # titles, details = fetch_paper_data(file_path, id_string)

    # print("=== TITLES ONLY ===")
    # print(titles)
    # print("\n=== FULL DETAILS ===")
    # print(details)



