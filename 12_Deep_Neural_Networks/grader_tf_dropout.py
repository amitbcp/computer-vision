import numpy as np
from tensorflow.python.framework.errors import FailedPreconditionError
import re
import tensorflow as tf

def get_result(output):
    """
    Run tests against output
    """
    
    answer = np.array([
        [9.55999947, 16.],
        [0.11200001, 0.67200011],
        [43.30000305, 48.15999985]])
    no_dropout = np.array([
        [4.77999973, 8.],
        [0.51100004, 0.8440001],
        [24.01000214, 38.23999786]])
    result = {
        'correct': False,
        'feedback': f'That\'s the wrong answer.  It should print {answer}',
        'comment': ''}
    
    try:
        tf.set_random_seed(123456)
        if output.shape == answer.shape and np.allclose(output, answer):
            result['correct'] = True
            result['feedback'] = 'You got it!  That\'s how you use dropout.'
        elif output.shape == no_dropout.shape and np.allclose(output, no_dropout):
            result['feedback'] = 'It looks like you\'re not applying dropout.'
            result['comment'] = 'Use the tf.nn.dropout() operation.'
    except FailedPreconditionError as err:
        if err.message.startswith('Attempting to use uninitialized value Variable'):
            result['feedback'] = 'TensorFlow variable uninitialized.'
            result['comment'] = 'Run tf.initialize_all_variables() in the session.'
        else:
            raise

    return result

def run_grader(output):

    if not np.all(output):
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