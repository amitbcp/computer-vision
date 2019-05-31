import os
import sys
import cv2
import json
import numpy as np
from keras.layers.core import Dense, Activation, Flatten
from keras.activations import relu, softmax
    
def get_result(metrics):
    
    result = {'is_correct': False, 'error': False, 'values': [],
              'output': '', 'feedback': '', 'comment': ""}
    
    try:        
        
        if metrics is not Ellipsis:
            metric_value = metrics[1]

            if metric_value < 0.5:
                result["is_correct"] = False
                result["comment"] = 'I bet you can do better than 50%'
                result["feedback"] = 'Accuracy was '+ str(metric_value)
            elif metric_value < 0.90:
                result["is_correct"] = True
                result["comment"] = 'But can you get above 90%?'
                result["feedback"] = 'Accuracy was '+ str(metric_value)
            else:
                result["is_correct"] = True
                result["comment"] = 'Good Job, accuracy was above 90%'
                result["feedback"] = 'Nice, accuracy was '+ str(metric_value)
        else:
            result["is_correct"] = False
            result["comment"] = 'You still need to evaluate the test data'
            result["feedback"] = 'evaluate method was empty'
            
    except Exception as err:
        result['is_correct'] = False
        result['feedback'] = 'Oops looks like you got an Error'
        result['error'] = str(err)
    
    return result

def run_grader(metrics):
    
    try:
    # Get grade result information
        result = get_result(metrics)
    except Exception as err:
        # Default error result
        result = {
            'correct': False,
            'feedback': 'Something went wrong with your submission:',
            'comment': str(err)}

    feedback = result.get('feedback')
    comment = result.get('comment')

    print(f"{feedback}\n{comment}\n")


if __name__ == "__main__":
    run_grader(model, history)