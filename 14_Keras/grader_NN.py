import os
import sys
import cv2
import json
import numpy as np
import keras
from keras.layers.core import Dense, Activation, Flatten
from keras.activations import relu, softmax
    
def get_result(model, history):
    
    result = {'is_correct': False, 'error': False, 'values': [],
              'output': '', 'feedback': '', 'comment': ""}
    
    try:       
        check_shape_expanded = None
        check_shape_mixed = None
        check_shape_compressed = None
        
        if len(model.layers) > 4:        
            check_shape_expanded = (model.layers[0].input_shape == (None, 32, 32, 3)) & \
                          (model.layers[1].output_shape == (None, 128)) & \
                          (model.layers[2].activation == relu) & \
                          (model.layers[3].output_shape == (None, 5)) & \
                          (model.layers[4].activation == softmax)
        elif len(model.layers) > 3:
            check_shape_mixed = (model.layers[0].input_shape == (None, 32, 32, 3)) & \
                          (model.layers[1].output_shape == (None, 128)) & \
                          (model.layers[1].activation == keras.activations.relu) & \
                          (model.layers[2].output_shape == (None, 5)) & \
                          (model.layers[3].activation == softmax)
            if not check_shape_mixed:
                check_shape_mixed = (model.layers[0].input_shape == (None, 32, 32, 3)) & \
                          (model.layers[1].output_shape == (None, 128)) & \
                          (model.layers[2].activation == relu) & \
                          (model.layers[3].output_shape == (None, 5)) & \
                          (model.layers[3].activation == keras.activations.softmax)
        else:        
            check_shape_compressed = (model.layers[0].input_shape == (None, 32, 32, 3)) & \
                          (model.layers[1].output_shape == (None, 128)) & \
                          (model.layers[1].activation == keras.activations.relu) & \
                          (model.layers[2].output_shape == (None, 5)) & \
                          (model.layers[2].activation == keras.activations.softmax)
        

        if check_shape_expanded or check_shape_mixed or check_shape_compressed:
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