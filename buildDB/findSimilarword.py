import difflib  

def find_most_similar_experiment(experiment_list, target):  
    # Initialize the highest similarity ratio and the most similar experiment  
    highest_ratio = 0  
    most_similar_experiment = None  

    # Iterate through each experiment in the list  
    for experiment in experiment_list:  
        # Calculate the similarity ratio  
        ratio = difflib.SequenceMatcher(None, experiment, target).ratio()  
        
        # Update the most similar experiment if the current one is more similar  
        if ratio > highest_ratio:  
            highest_ratio = ratio  
            most_similar_experiment = experiment  

    return most_similar_experiment  

if __name__ == "__main__":  
    # Sample list of experiments  
    experiment_list = ["Panc1", "Capan -1 ", "Molt-4", "MCF10A", "ID8F3", "HPAC"]  

    # The target experiment name  
    target_experiment = "Panv1"  

    # Find the most similar experiment name  
    most_similar = find_most_similar_experiment(experiment_list, target_experiment)  
    print(f"The most similar experiment name to '{target_experiment}' is '{most_similar}'.")  