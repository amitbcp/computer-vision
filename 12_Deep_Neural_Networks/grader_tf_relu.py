import numpy as np
from tensorflow.python.framework.errors import FailedPreconditionError
import re

def get_result(output):
    
    """
    Run tests
    """
    
    answer = np.array([
        [5.11000013, 8.44000053],
        [0., 0.],
        [24.01000214, 38.23999786]])
    result = {
        'correct': False,
        'feedback': f'That\'s the wrong answer.  It should print {answer}',
        'comment': ''}
    
    output_shape = np.shape(output)
    answer_shape = np.shape(answer)
        
    if output_shape != answer_shape: 
        result['feedback'] = 'Output is the wrong type or wrong dimension.'
        result['comment'] = f'Output shape is {output_shape}, answer shape is {answer_shape})'
        
    elif (0 > output).sum():
        result['feedback'] = 'Output contains negative numbers.'
        result['comment'] = 'Are you applying ReLU to hidden_layer?'
    
    else:
        
        if np.allclose(output, answer):
            result['correct'] = True
            result['feedback'] = 'You got it!  That\'s how you use a ReLU.'
       
   
    
    return result

def run_grader(output):
    
    if not np.any(output):
        print("Don't forget to complete all tasks and name your session variable output")
    
    else:
        try:
        # Get grade result information
            result = get_result(output)
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
    
    run_grader(output)