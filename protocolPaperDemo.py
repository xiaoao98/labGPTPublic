from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import labgpt_config as cfg
import os
import sqlite3  
import re

database_path = cfg.DATABASE_PATH
model_name = cfg.MODEL_NAME


def load_paper_protocols():
    """Load published-method excerpts as a raw JSON text block.

    This was previously a 35 KB triple-quoted literal holding verbatim Methods
    sections from published papers. Text of that kind is copyrighted by its
    publishers regardless of who wrote it, so it does not belong inlined in source
    or in a public repository. It now loads from the corpus directory like every
    other data source.
    """
    with open(cfg.require(cfg.PAPER_PROTOCOLS_PATH), encoding="utf-8") as handle:
        return handle.read()

def run_chat() -> None:
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
    print("Welcome to the protocol search application!")
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
        database_protocol_response = read_protocol_by_experiment(database_path, query)
        database_reagent_response = read_reagent_by_experiment(database_path, query)
        # for new_text in chat_model.stream_chat(messages):
        #     print(new_text, end="", flush=True)
        #     response += new_text
        # print("database_response: " + database_response)
        if database_protocol_response == "not found" or database_reagent_response == "not found":
            chat_input = query
            print("Protocol not found in the lab corpus, searching the public dataset ... ")
        else:
            chat_input = "Here is the reagents: " + database_reagent_response + '\n' + "Here is the protocol: " + database_protocol_response + '\n' + "Here is some information from the paper: " + load_paper_protocols() + "Then answer this question precisely: " + query
            print("The reagent for the experiment is: " + '\n' + database_reagent_response)
            print("The protocol for the experiment is: " + '\n' + database_protocol_response)
            print("Then let me think about the question...")
        print_model_response(tokenizer, model, chat_input)        
        # print(chat_output)
        # messages.append({"role": "assistant", "content": response})

def read_protocol_by_experiment(database_path, experiment_name):  
    # Connect to the SQLite database  
    conn = sqlite3.connect(database_path)  
    
    # Create a cursor object  
    cursor = conn.cursor()  
    
    # Define your SQL query with a placeholder for the parameter  
    query = """  
    SELECT content FROM protocols  
    WHERE experiment = ?;  
    """  
    
    try:  
        # Write the SQL query to select the experiment_name column  
        query1 = "SELECT experiment FROM protocols"  

        # Execute the query  
        cursor.execute(query1)  

        # Fetch all results (this will be a list of tuples)  
        results = cursor.fetchall()  

        # Extract the experiment_name values from the tuples and put them into a list  
        experiment_names = [row[0] for row in results]  
        found_experiment = find_experiment_in_sentence(experiment_names, experiment_name)  
        # print(most_similar)
        if found_experiment == "not found":
            return found_experiment
        else:
            cursor.execute(query, (found_experiment,))  
            # Fetch all the matching rows
            results = cursor.fetchall() 
            return results[0][0] 

    except sqlite3.Error as e:  
        print(f"An error occurred: {e}")  
        return None 
    finally:  
        # Close the database connection  
        conn.close()  

def read_reagent_by_experiment(database_path, experiment_name):  
    # Connect to the SQLite database  
    conn = sqlite3.connect(database_path)  
    
    # Create a cursor object  
    cursor = conn.cursor()  
    
    # Define your SQL query with a placeholder for the parameter  
    query = """  
    SELECT content FROM reagents  
    WHERE experiment = ?;  
    """  
    
    try:  
        # Write the SQL query to select the experiment_name column  
        query1 = "SELECT experiment FROM reagents"  

        # Execute the query  
        cursor.execute(query1)  

        # Fetch all results (this will be a list of tuples)  
        results = cursor.fetchall()  

        # Extract the experiment_name values from the tuples and put them into a list  
        experiment_names = [row[0] for row in results]  
        found_experiment = find_experiment_in_sentence(experiment_names, experiment_name)  
        # print(most_similar)
        if found_experiment == "not found":
            return found_experiment
        else:
            cursor.execute(query, (found_experiment,))  
            # Fetch all the matching rows
            results = cursor.fetchall() 
            return results[0][0] 

    except sqlite3.Error as e:  
        print(f"An error occurred: {e}")  
        return None 
    finally:  
        # Close the database connection  
        conn.close() 

# load the tokenizer and the model
def print_model_response(tokenizer, model, input_content):
    messages = [
        {"role": "user", "content": input_content}
    ]
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
            max_new_tokens=4096
        )
    except Exception as e:
        print(f"\nError during model generation: {e}")
    # return content

def find_experiment_in_sentence(experiment_list, target):
    """
    Finds the single experiment name from a list that is present in a target sentence,
    regardless of word order.

    Args:
        experiment_list: A list of experiment names (strings).
        target: The target sentence (string).

    Returns:
        The experiment name (string) if exactly one is found in the target sentence,
        or None if zero or more than one are found.
    """
    target_lower = target.lower()  # For case-insensitive matching

    for experiment in experiment_list:
        experiment_lower = experiment.lower()
        experiment_words = experiment_lower.split()
        all_words_found = True
        for word in experiment_words:
            # Use regex to find whole words, even with punctuation.
            if not re.search(r'\b' + re.escape(word) + r'\b', target_lower):
                all_words_found = False
                break
        if all_words_found:
            return experiment
    return "not found"

if __name__ == "__main__":
    run_chat()



