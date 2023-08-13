import csv

def is_consistent(hypothesis, instance):
    for i in range(len(hypothesis)):
        if hypothesis[i] != '?' and hypothesis[i] != instance[i]:
            return False
    return True

def generalize_specific(instance, general_hypothesis):
    new_general_hypothesis = list(general_hypothesis)
    for i in range(len(new_general_hypothesis)):
        if general_hypothesis[i] == '?':
            new_general_hypothesis[i] = instance[i]
        elif general_hypothesis[i] != instance[i]:
            new_general_hypothesis[i] = '?'
    return new_general_hypothesis

def candidate_elimination(data):
    data_reader = csv.reader(data)
    instances = [row for row in data_reader]
    
    # Initialize general and specific hypotheses
    specific_hypothesis = instances[0][:-1]
    general_hypothesis = ['?' for _ in range(len(specific_hypothesis))]
    
    for instance in instances:
        if instance[-1] == 'Yes':
            for i in range(len(specific_hypothesis)):
                if specific_hypothesis[i] != instance[i]:
                    specific_hypothesis[i] = '?'
            for i in range(len(general_hypothesis)):
                if specific_hypothesis[i] == '?':
                    general_hypothesis[i] = '?'
        else:
            if is_consistent(specific_hypothesis, instance):
                general_hypothesis = generalize_specific(instance, general_hypothesis)
    
    return specific_hypothesis, general_hypothesis

with open('enjoysport.csv', 'r') as csvfile:
    specific, general = candidate_elimination(csvfile)
    print("Final Specific Hypothesis:", specific)
    print("Final General Hypothesis:", general)
