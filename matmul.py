
from typing import List
import copy
import numpy

def _shape(s: List, a):
    if type(a) is not list:
        return
    s.append(len(a))
    _shape(s, a[0])

def shape(a):
    s = []
    _shape(s, a)
    return s

# multiply two matrix
def mul(a: List, b):
    shape_a = shape(a)
    shape_b = shape(b)
    copy_a  = copy.deepcopy(a)

    for i in range(shape_a[0]):
        for j in range(shape_b[1]):
            sum = 0
            for k in range(shape_a[1]):
                sum = sum + a[i][k] * b[k][j]
            copy_a[i][j] = sum
    
    return copy_a


def _matmul(shape, mat_a, mat_b, mat_c, cur_pos):
    if cur_pos == len(shape) - 2:
        return mul(mat_a, mat_b)

    ilen = shape[cur_pos]
    
    for i in range(ilen):
        # Broadcast by using module
        a_mat = mat_a[i % len(mat_a)]
        b_mat = mat_b[i % len(mat_b)]
        c_mat = mat_c[i]
        res = _matmul(shape, a_mat, b_mat, c_mat, cur_pos + 1)
        if res != None:
            mat_c[i].extend(res)
    
    return None

def try_matmul(a, b) -> bool:
    ret = True
    shape_a = shape(a)
    shape_b = shape(b)
    out_shape = []

    assert len(shape_a) >= 2 and len(shape_b) >= 2

    short = len(shape_a) if len(shape_a) < len(shape_b) else len(shape_b)
    long = len(shape_a) if len(shape_a) > len(shape_b) else len(shape_b)
    short_shape = shape_a if len(shape_a) < len(shape_b) else shape_b
    long_shape = shape_a if len(shape_a) > len(shape_b) else shape_b
    
    # Broadcast Dimmensions
    if not (len(shape_a) == 2 or len(shape_b) == 2):
        for i in range(short - 2):
            bias = long - short
            short_shape_i = short_shape[i]
            long_shape_i = long_shape[bias + i]
            if short_shape_i != long_shape_i:
                if short_shape_i != 1 and long_shape_i != 1:
                    raise Exception("Dimension mismatch")
                if long_shape_i == 1:
                    long_shape[bias + i] = short_shape[i]
                elif short_shape_i == 1:
                    short_shape[i] = long_shape[bias + i]
    
    def calc_last_dim(shape):
        return [shape[-2], shape[-1]]

    dim_a = calc_last_dim(shape_a)
    dim_b = calc_last_dim(shape_b)
    
    if dim_a[1] != dim_b[0]:
        raise Exception("Dimension mismatch")
    
    for i in range(long - 2):
        out_shape.append(long_shape[i])
    out_shape.append(dim_a[0])
    out_shape.append(dim_b[1])
    # print(out_shape)
    
    # Create Output Matrix and Calc Matmul:
    def create_matrix(dims):
        if len(dims) == 0:
            raise ValueError("Requires at least 1 dimension.")

        if len(dims) == 1:
            return [[] for _ in range(dims[0])]

        return [create_matrix(dims[1:]) for _ in range(dims[0])]
        
    c = create_matrix(out_shape[:-2])
    _matmul(out_shape, a, b, c, 0)
    return [ret, c]

def matmul(a, b):
    [ret, out] = try_matmul(a, b)
    if not ret:
        raise Exception("Dimension mismatch")
    return out

mat1 = [
        [[[1, 2], 
        [1, 2], 
        [1, 2], 
        [1, 2]]],

        [[[1, 2], 
        [1, 2], 
        [1, 2], 
        [1, 2]]]
       ]

mat2 = [ 
         [[[1, 2], [1, 2]], [[1, 2], [1, 2]]], 
         [[[1, 2], [1, 2]], [[1, 2], [1, 2]]], 
       ]

print(numpy.matmul(mat1, mat2))

print(matmul(mat1, mat2))
