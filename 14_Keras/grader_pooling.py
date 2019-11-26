import os
import sys
import cv2
import json
import numpy as np
from keras.layers.core import Dense, Activation, Flatten
from keras.activations import relu, softmax
    
def get_result(model, history):
    
    result = {'is_correct': False, 'error': False, 'values': [],
              'output': '', 'feedback': '', 'comment': ""}
    
    try:
        
        check_shape = (model.layers[1].pool_size == (2, 2))

        if check_shape:
            if history.history['acc'][-1] > 0.5:
                result["is_correct"] = True
                result["comment"] = 'Looks good!'
                result["feedback"] = 'Nice work!'
            else:
                result["is_correct"] = False
                result["comment"] = 'The accuracy was less than 50%'
                result["feedback"] = 'Make sure you are running the model for enough epochs'
        else:
            result["is_correct"] = False
            result["comment"] = 'The model layout looks incorrect'
            result["feedback"] = 'Try following the model layout from the instructions'
    except Exception as err:
        result['is_correct'] = False
        result['feedback'] = 'Oops, looks like you got an error!'
        result['error'] = str(err)

    return result

def run_grader(model, history):
    
    try:
    # Get grade result information
        result = get_result(model, history)
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