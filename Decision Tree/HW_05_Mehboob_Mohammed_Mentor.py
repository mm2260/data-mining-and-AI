#===========================================================================================================
# # Mentor Program
#   @author: Mohammed Mehboob
#===========================================================================================================

import numpy as np
import pandas as pd
import sys
from statistics import mode

#===========================================================================================================
# ### Global Parameters:
#===========================================================================================================

MIN_LEAF_SIZE = 9
MAX_DEPTH = 10
PERCENT_REPRESENTATION = 95

TEST_SET = 'Abominable_Data_HW_LABELED_TRAINING_DATA__v750_2215.csv'
VALIDATION_SET = 'Abominable_VALIDATION_Data_FOR_STUDENTS_v750_2215.csv'
# VALIDATION_SET = TEST_SET #< uncomment for purposes of generating confusion matrix.

#===============================================
# ### Read the data into a Pandas dataframe:   |
#===============================================

# Read the labeled data from the provided CSV file.
csv = pd.read_csv(TEST_SET); csv

#===========================================================================================================
# ### Implementing the Decision Tree:
#===========================================================================================================

# #### Attributes and Data Rounding:

class Attribute:
    '''
    The Attribute class holds general information about an attribute / column.
    
    The class holds data about the Column-ID of the attribute in a given vector, 
    the attribute's name, the possible values the attribute can take, and finally,
    the attribute's quantized unit.
    '''
    
    def __init__(self, name, column_id):
        
        self.column = column_id
        self.name = name
        self.is_categorical = list(csv[name].unique()) == [0,1]
        self.__define_quantization__(name)
        self.values = self.__define_values__(name)
        
    def __define_quantization__(self,name):
        ''' 
        Set a quantization unit for the attribute; this is a means of noise-reduction.
        '''
        global csv

        # If the attribute is age data, set the quantized unit to 2 years.
        if name.lower() == 'age':
            self.quantized_unit = 2
        # Height is quantized to the nearest 4cm. 
        elif name.lower() == 'ht':
            self.quantized_unit = 4
        # If the data is categorical (true/false), then then we don't quanztize the data    
        elif self.is_categorical:
            self.quantized_unit = 1
        # All other data is quantized to the nearest 2 units. 
        else:
            self.quantized_unit = 2
    
    def __define_values__(self,name):
        '''
        Calculate all the different values the attribute can take on. Used for threshold-finding.
        '''
    
        global csv
        min_value = self.quantized_unit * round( min(csv[name]) / self.quantized_unit )
        max_value = self.quantized_unit * round( max(csv[name]) / self.quantized_unit )    
        return range(min_value, max_value + self.quantized_unit, self.quantized_unit)
    
    def __repr__(self):
        return f'Attribute[{self.column}]: {self.name} / {self.quantized_unit}'


#===========================================================================================================


def round_data(csv, attributes):
    for attribute in attributes:
        csv[attribute.name] = csv[attribute.name].apply( lambda datapoint: attribute.quantized_unit * round(datapoint/attribute.quantized_unit) )
    return csv

#===========================================================================================================
# #### Decision Stump and Threshold Finding:
#===========================================================================================================

class DecisionStump:
    '''
    A decision stump is one level in a decision tree. 
    It uses a single attribute to split data into two partitions.
    '''
    
    def __init__(self, attribute, threshold):
        self.attribute = attribute
        self.threshold = threshold
    
    def eval(self, datapoint):
        '''
        Takes in a datapoint and checks against the decision stump's threshold.
        '''
        if self.attribute.is_categorical:
            return datapoint[self.attribute.column] < self.threshold
        else:
            return datapoint[self.attribute.column] <= self.threshold
    
    def __repr__(self):
        check_statement = ''
        
        if self.attribute.is_categorical:
            check_statement = f'if ( {self.attribute.name.lower()} < {self.threshold} ):'
        else:
            check_statement = f'if ( {self.attribute.name.lower()} <= {self.threshold} ):'
        
        return check_statement

#===========================================================================================================

def binary_split(data, decisionStump):
    '''
    Splits incoming data based on the provided decision stump.
    Returns the left and right partitions resultant from the decision stump partitioning. 
    
    Left partition is all the false cases.
    Right partition is all the true cases.
    '''
    left, right = [],[]  
    # Left is the false partition from the decision stump.
    # Right is the true partition from the decision stump.
    
    for datapoint in data:
        if decisionStump.eval(datapoint):
            left.append(datapoint)
        else:
            right.append(datapoint)
            
    return left,right

#===========================================================================================================

def count_labels(data):
    '''
    Counts the number of Assams and Bhutans in a given set of datapoints.
    
    Assumes the last column is the classification, and -1 --> Assam, +1 --> Bhutan
    '''
    assams, bhutans = 0,0
    for datapoint in data:
        if datapoint[-1] == -1:
            assams = assams+1
        else:
            bhutans = bhutans + 1          
    return assams,bhutans


#===========================================================================================================

def entropy(data):
    '''
    Calculates the entropy of the set of datapoints
    '''
    
    # A very small value, added onto the log base 2 calculation to avoid a division by zero warning.
    #
    # > ( I conjecture I was running to this warning due to the right/total or left/total values being
    #     very small. Adding this would not greatly affect the decision tree calculations, since a 
    #     log2 of a small value us largely negative, which would make the mixed entropy very large. This would
    #     be promptly ignored, since we're trying to find argmin(mixed_entropy). 
    #   )

    epsilon = 0.0000000001 
    
    # Left values are Assams, and right values are Bhuttans. 
    left, right = count_labels(data)
    total = left + right
    
    if total == 0:
        return None
    
    # Return the calculated entropy. 
    return -( left/total * np.log2( left/total + epsilon ) + right/total * np.log2( right/total + epsilon ) )


#===========================================================================================================


def find_optimal_decision_stump(training_data, attribute):
    '''
    For a given attribute, find the best value for a threshold, which minimizes the mixed entropy. 
    With that threshold value, construct and return a Decision Stump. 
    ''' 
    best_mixed_entropy = np.Infinity # Sentinel value, so that we can only go lower.
    best_decision_stump = None # Sentinel value. Only returned if something goes wrong. 
    quantized_unit = attribute.quantized_unit
    
    # Iterate through all possible threshold values.
    for threshold in attribute.values:
        decision_stump = DecisionStump( attribute, threshold )     
        
        # Split the data into binary partitions. 
        left, right = binary_split(training_data, decision_stump)
        n_total = len(training_data)
        n_left = len(left)
        n_right = len(right)
    
        # If the threshold does not split the data at all, we ignore this threshold value.
        if ( n_left == 0 or n_right == 0 ):
            continue
    
        # Calculate the mixed entropy. 
        mixed_entropy = n_left/n_total * entropy(left) + n_right/n_total * entropy(right)
        
        # Break ties by using the first value found.
        # REASONING:
        # This is because it will make comparing each decision stump against each other easier,
        # since each threshold corresponding to an attribute will hold the mixed entropy, 
        # it wouldn't help in comparison, and the extra comparisons can be wasteful. 
    
        if(mixed_entropy < best_mixed_entropy):
            best_mixed_entropy = mixed_entropy
            best_decision_stump = decision_stump
    
    return best_decision_stump, best_mixed_entropy


#===========================================================================================================


def find_best_split(training_data, attributes):
    '''
    Finds the decision stump which yields the best mixed entropy after splitting. 
    '''
    
    best_mixed_entropy = np.Infinity # Sentinel value, so that we can only go lower. 
    best_decision_stump = None # Sentinel value. Only returned if something goes wrong. 
    
    # Iterate through all available attributes, and find the best decision stump. 
    for attribute in attributes:
        decision_stump, mixed_entropy = find_optimal_decision_stump(training_data, attribute)
        if(mixed_entropy < best_mixed_entropy):
            best_mixed_entropy = mixed_entropy
            best_decision_stump = decision_stump
    
    return best_decision_stump, best_mixed_entropy


#===========================================================================================================
# #### Decision Tree Node Classes:
#===========================================================================================================

class LeafNode:
    '''
    The Leaf Node classifies data.
    '''
    
    def __init__(self, data):
        self.data = data
    
    def predict(self):
        '''
        The prediction works according to the mode value of all the set of classifications 
        of the datapoints within the leaf node. i.e. It is the popular value of the classficiation.
        '''
        return mode(np.array(self.data)[:,-1])


#===========================================================================================================


class DecisionNode:
    '''
    A decision node splits incoming data into binary partitions based on a one-rule.
    '''
    
    def __init__(self, decision_stump, left_child, right_child):
        self.decision_stump = decision_stump
        self.left_child = left_child
        self.right_child = right_child


#===========================================================================================================

def generate_decision_tree(data, attributes, DEPTH=1):    
    '''
    This function recursively builds a decision tree from labeled data.
    
    Stopping conditions include:
        --> The class consists of more than 95% of one class or the other.
        --> Tree depth exceeds 10.
        --> There are less than 9 datapoints within a node. 
    '''
    global MIN_LEAF_SIZE
    global MAX_DEPTH
    global PERCENT_REPRESENTATION
    
    decision_stump, mixed_entropy = find_best_split(data, attributes)
    assams, bhuttans = count_labels(data)
    assams = assams/len(data)
    bhuttans = bhuttans/len(data)
    
    # Check for stopping conditions. If conditions are met, return a leaf node. 
    if ( assams > PERCENT_REPRESENTATION/100 or bhuttans > PERCENT_REPRESENTATION/100 ) or ( DEPTH > MAX_DEPTH ) or ( len(data) < MIN_LEAF_SIZE ):
        return LeafNode(data)
    
    # Recursively build the decision tree...
    left, right = binary_split(data, decision_stump)    
    return DecisionNode( decision_stump, generate_decision_tree(left, attributes, DEPTH+1), generate_decision_tree(right, attributes, DEPTH+1) )


#===========================================================================================================
# ### Metaprogramming:
#===========================================================================================================

def emit_header(attributes):
    '''
    Make sure the classifier program runs. 
    
    Add all required imports, load the data into memory, quantize data, etc.
    '''
    global VALIDATION_SET
    
    print('import numpy as np','import pandas as pd', sep='\n')
    print()
    
    print(f"FILENAME = '{VALIDATION_SET}'")
    print('csv = pd.read_csv(FILENAME)')
    print()
    
    for attribute in attributes:
        print(f"csv['{attribute.name}'] = csv['{attribute.name}'].apply( lambda datapoint: {attribute.quantized_unit} * round(datapoint/{attribute.quantized_unit}) )")
    print()
    
    print('predictions = []')
    print()
    
    print('for datapoint in csv.to_numpy():')
    print()
    
    print('    # Datapoint values:')
    for attribute in attributes:
        print(f'    {attribute.name.lower()} = datapoint[{attribute.column}]')
    
    print()
    print('    prediction = -1')


def emit_body(node, padding=""):
    '''
    Adds the decision tree if/else ladder to the classifier program.
    '''
    
    if isinstance(node, LeafNode):        
        print (padding + "prediction =", node.predict() )
        return

    print (padding + str(node.decision_stump))
    emit_body(node.left_child, padding + "    ")
    print (padding + 'else:')
    emit_body(node.right_child, padding + "    ")

def emit_footer():
    '''
    Wraps up the classifier program. Makes sure results are output to a CSV file for earier viewing.
    '''
    
    print()
    print('    predictions.append(prediction)')
    print('    print(prediction)')
    print()
    
    print("df = pd.DataFrame(predictions,columns=['ClassID'])")
    print("df.to_csv('HW05_Mehboob_Mohammed_MyClassifications.csv',index=False)")


#===========================================================================================================
# ### Runner:
#===========================================================================================================

attributes = [ Attribute(name, column_id) for column_id, name in enumerate(csv.drop( columns = csv.columns[-2:] ).columns) ]; attributes
data = round_data(csv,attributes).to_numpy()

decision_tree = generate_decision_tree(data,attributes)

#====================================================================

stdout_backup = sys.stdout
with open('HW05_Mehboob_Mohammed_Trained_Classifier.py', 'w') as file:
    sys.stdout = file
    emit_header(attributes)
    emit_body(decision_tree, '    ')
    emit_footer()
    sys.stdout = stdout_backup


#====================================================================

# # Generate confusion matrix for test data:

# TP = 0
# TN = 0
# FP = 0
# FN = 0

# my_classifications = pd.read_csv('HW05_Mehboob_Mohammed_MyClassifications.csv').to_numpy()  # Set VALIDATION_SET = TEST_SET before running mentor.
# for index, classification in enumerate(my_classifications):
#     if classification == -1:
#         if csv['ClassID'][index] == -1:
#             TP = TP + 1
#         else:
#             FP = FP + 1
#     else:
#         if csv['ClassID'][index] == 1:
#             TN = TN + 1
#         else:
#             FN = FN + 1
        
# print('Accuracy:', (TP+TN)/(TP+TN+FP+FN))
# pd.DataFrame([ [TP,FP], [FN, TN] ], columns = ['Positive (A)', 'Negative (A)'])