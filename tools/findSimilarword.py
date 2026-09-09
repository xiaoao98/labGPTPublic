import difflib  
import re

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
    found_experiments = []
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
    # Sample list of experiments  
    experiment_list = ["Panc1", "Capan -1 ", "Molt-4", "MCF10A", "ID8F3", "HPAC"]  

    # The target experiment name  
    target_experiment = "Panv1"  

    # Find the most similar experiment name  
    most_similar = find_most_similar_experiment(experiment_list, target_experiment)  
    print(f"The most similar experiment name to '{target_experiment}' is '{most_similar}'.")  