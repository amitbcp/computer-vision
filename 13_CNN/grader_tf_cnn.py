import tensorflow as tf
import numpy as np
import json


def solution(input_array):
    # Filter (weights and bias)
    F_W = tf.Variable(tf.truncated_normal((2, 2, 1, 3)))
    F_b = tf.Variable(tf.zeros(3))
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    return tf.nn.conv2d(input_array, F_W, strides, padding) + F_b

def get_result(input_array, student_func):
        
        result = {'is_correct': None, 'error': False, 'values': [], 'output': '', 'custom_msg': ''}
        ours = solution(input_array)
        theirs = student_func(input_array)
        
        dim_names = ['Batch', 'Height', 'Width', 'Depth']
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            our_shape = ours.get_shape().as_list()
            their_shape = theirs.get_shape().as_list()
            
            did_pass = False

            try:
                for dn, ov, tv in zip(dim_names, our_shape, their_shape):
                    if ov != tv:
                        # dimension mismatch
                        raise Exception('{} dimension: mismatch we have {}, you have {}'.format(dn, ov, tv))
                if np.alltrue(our_shape == their_shape):
                    did_pass = True
                else:
                    did_pass = False
            except:
                did_pass = False

            if did_pass:
                result['is_correct'] = 'Great Job!'
                result['values'] = [f'your output shape: {their_shape}']
            else:
                result['values'] = [f'correct shape: {our_shape}']
                result['output'] = [f'your output shape: {their_shape}']
    
        return result
                    
def run_grader(input_array, student_func):
  
    grader_result = get_result(input_array, student_func)
    gt_shape = grader_result.get('values')
    student_func_shape = grader_result.get('output')
    comment = ""

    if grader_result['is_correct']:
        comment= "Great job! Your Convolution layer looks good :)"
    elif not grader_result['error']:
        comment = f"Not quite. The correct output shape is {gt_shape} while your output shape is {student_func_shape}."
    else:
        test_error = grader_result['error']
        comment = f"Something went wrong with your submission: {test_error}"

    grader_result['feedback'] = comment
    
    return grader_result.get('feedback')