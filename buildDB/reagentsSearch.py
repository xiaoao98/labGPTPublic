import sqlite3  
from tools.findSimilarword import find_most_similar_experiment

database_path = 'reagents.db'  
experiment_name = 'PanV1 Cryorecovery' 

def read_reagents_by_experiment(database_path, experiment_name):  
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
        # Execute the query with the parameter  
        cursor.execute(query, (experiment_name,))  
        
        # Fetch all the matching rows  
        results = cursor.fetchall()  
        
        # Process your data  
        if results:  
            reagent1 = "The reagents for this experiment are: " + results[0][0]  
        else: 
            # Write the SQL query to select the experiment_name column  
            query1 = "SELECT experiment FROM reagents"  

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
            reagent1 = "I think you maybe want to find the reagents of the " + most_similar + " Experiment, and the reagents of this experiment is: " + "\n"  +results[0][0]

        return reagent1 

    except sqlite3.Error as e:  
        print(f"An error occurred: {e}")  
        return None 
    finally:  
        # Close the database connection  
        conn.close()  

if __name__ == "__main__":  
    print(read_reagents_by_experiment(database_path, experiment_name))