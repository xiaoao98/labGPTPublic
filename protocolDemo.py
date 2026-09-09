from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import labgpt_config as cfg
import labgpt_metrics as metrics
import os
import sqlite3  
import re

database_path = cfg.DATABASE_PATH
model_name = cfg.MODEL_NAME

# An experiment name from the list, or "not found". Was 3276.
MAX_NEW_TOKENS_EXPERIMENT_NAME = 48

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
        finalQuery = "You are a lab assistant AI." + '\n'
        # database_reagent_response = read_reagent_by_experiment(database_path, query)
        database_protocol_response = read_protocol_by_experiment(database_path, query, tokenizer, model)
        # for new_text in chat_model.stream_chat(messages):
        #     print(new_text, end="", flush=True)
        #     response += new_text
        # print("database_response: " + database_response)
        # if database_reagent_response != "not found":
        #     finalQuery += "Read those reagents documents: " + database_reagent_response + '\n' 
        if database_protocol_response != "not found":   
            finalQuery += "Read those protocol documents: " + database_protocol_response + '\n'
        finalQuery += "Then answer about this question: " + query
        print(finalQuery)
        # print_model_response(tokenizer, model, finalQuery)        
        # print(chat_output)
        # messages.append({"role": "assistant", "content": response})

def read_protocol_by_experiment(database_path, question, tokenizer, model):  
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
        prompt_question = (
            "Find the most related experiment in the experiment name list to a query. "
            f"The query is: {question} The list is: {experiment_names}\n"
            "Make the answer only the experiment name from the list. "
            "If all are not related, answer not found."
        )
        # print(experiment_names)
        found_experiment = getAnswer(tokenizer, model, prompt_question)  
        print(found_experiment)
        # add a wrapper to make sure that the right experiment is found 
        found_experiment = find_experiment_in_sentence(experiment_names, found_experiment)
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

def read_reagent_by_experiment(database_path, question, tokenizer, model):  
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
        prompt_question = (
            "Find the most related experiment in the experiment name list to a query. "
            f"The query is: {question} The list is: {experiment_names}\n"
            "Make the answer only the experiment name from the list. "
            "If all are not related, answer not found."
        )
        # print(experiment_names)
        found_experiment = getAnswer(tokenizer, model, prompt_question)  
        print(found_experiment)
        # add a wrapper to make sure that the right experiment is found 
        found_experiment = find_experiment_in_sentence(experiment_names, found_experiment)
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

def getAnswer(tokenizer, model, input_content) -> str: 
    messages = [
        {"role": "user", "content": input_content}
    ]
    # question = "I have three types of question: memberinfo, safety, biology experiment protocol. Please answer me the type of the following question (you can only choose one):  What does 2 Gallon sharps containers accepts. The answer should only be the type, no more explanation"

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    try:
        with metrics.timer() as elapsed:
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS_EXPERIMENT_NAME,
                do_sample=False
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        metrics.record(
            "match_experiment",
            prompt_tokens=model_inputs.input_ids.shape[1],
            generated_tokens=len(output_ids),
            elapsed_s=elapsed[0],
            cap=MAX_NEW_TOKENS_EXPERIMENT_NAME,
        )
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        return content
    except Exception as e:
        print(f"\nError during model generation: {e}")
        return ""
    
if __name__ == "__main__":
    run_chat()



