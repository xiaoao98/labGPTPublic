import os
import sqlite3  
from tools.findSimilarword import find_most_similar_experiment

database_path = 'experiments.db'  

def run_chat() -> None:
    if os.name != "nt":
        try:
            import readline  # noqa: F401
        except ImportError:
            print("Install `readline` for a better experience.")

    # chat_model = ChatModel()
    messages = []
    print("Welcome to the protocol search application, please input the experiment name, use `exit` to exit the application.")

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

        response = read_protocol_by_experiment(database_path, query)
        # for new_text in chat_model.stream_chat(messages):
        #     print(new_text, end="", flush=True)
        #     response += new_text
        print(response)
        print()
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
        # Execute the query with the parameter  
        cursor.execute(query, (experiment_name,))  
        
        # Fetch all the matching rows  
        results = cursor.fetchall()  
        
        # Process your data  
        if results:  
            reagent1 = "The protocol for this experiment is: " + results[0][0]  
        else: 
            # Write the SQL query to select the experiment_name column  
            query1 = "SELECT experiment FROM protocols"  

            # Execute the query  
            cursor.execute(query1)  

            # Fetch all results (this will be a list of tuples)  
            results = cursor.fetchall()  

            # Extract the experiment_name values from the tuples and put them into a list  
            experiment_names = [row[0] for row in results]  
            most_similar = find_most_similar_experiment(experiment_names, experiment_name)  
            # print(most_similar)
            cursor.execute(query, (most_similar,))  
        
            # Fetch all the matching rows
            results = cursor.fetchall() 
            reagent1 = "I think you maybe want to find the protocol of the " + most_similar + " Experiment, and the protocol of this experiment is: " + "\n"  +results[0][0]

        return reagent1 

    except sqlite3.Error as e:  
        print(f"An error occurred: {e}")  
        return None 
    finally:  
        # Close the database connection  
        conn.close()  

if __name__ == "__main__":
    run_chat()