import numpy as np
import QST
from openql import openql as ql
import re
import os
from math import cos, sin, sqrt
import random


NUM_QUBIT = 3
HADA = np.array([[1, 1], [1, -1]]) / sqrt(2)


ANG1 = random.random() * np.pi
ANG2 = random.random() * np.pi
ANG3 = random.random() * np.pi


"""
The stateprep method encapsulates the quantum algorithm which generates an unknown n-qubit quantum state from an n-qubit all-zero state
We want to estimate the density matrix of this n-qubit evolved state using State Tomography
"""
def stateprep():
    config_fn = os.path.abspath('/home/neil/dev/tud/OpenQL/tests/test_cfg_none_simple.json')
    platform = ql.Platform('platform_none', config_fn)
    prog = ql.Program('p_name', platform, NUM_QUBIT)
    k1 = ql.Kernel('QK1',platform, NUM_QUBIT)
    k1.ry(0, ANG1)
    k1.ry(1, ANG2)
    k1.ry(2, ANG3)
    prog.add_kernel(k1)
    prog.compile()
    qasmVerConv()
    return "test_output/algo.qasm"


"""
Firefighting solution as Qxelarator is not updated to run cQASM v1.0
Open Issue: https://github.com/QE-Lab/qx-simulator/issues/57
Converts OpenQL generated cQASM to old Qxelerator compatible syntax
"""
def qasmVerConv():
    file = open("test_output/p_name.qasm","r")
    fileopt = open("test_output/algo.qasm","w")
    header = True
    for line in file:
        if header:
            header = False
        else:
            x = re.sub('\[','', line)
            x = re.sub('\]','', x)
            fileopt.write(x)
    file.close()
    fileopt.close()


def Rx(O):
    return np.array([[cos(O/2), -1j*sin(O/2)],
                     [-1j*sin(O/2), cos(O/2)]])


def Ry(O):
    return np.array([[cos(O/2), -sin(O/2)],
                     [sin(O/2), cos(O/2)]])


def kronecker(*args, repeat=1):
    prod = [1]
    if len(args) > 1 and repeat == 1:
        for M in args:
            prod = np.kron(prod, M)
    elif repeat > 1:
        for _ in range(repeat):
            prod = np.kron(prod, args[0])
    else:
        raise(ValueError("Something went wrong"))
    return prod


if __name__ == '__main__':
    import sys
    #qasm = "test_output/algo.qasm"

    # Find the number of qubits in the program
    #with open(qasm, 'r') as f:
    #    for line in f:
    #        if line.startswith('qubits'):
    #            q = int(line.split(' ')[1])
    #            break

    qasm = stateprep()
    t = int(sys.argv[1])
 
    C  = kronecker(Ry(ANG1), Ry(ANG2), Ry(ANG3))
    zero = np.zeros((2**NUM_QUBIT, 1))
    zero[0] = 1
    S  = C @ zero
    S_ = np.transpose(S).conj()

    H = kronecker(QST.sg3, repeat=NUM_QUBIT)

    dm = QST.tomography(qasm, NUM_QUBIT, trials=t)
    e = QST.expectation(dm, H)
    theory = S_ @ H @ S

    #import pdb; pdb.set_trace()

    print('ry q0,', ANG1)
    print('ry q1,', ANG2)
    print('ry q2,', ANG3)
    print('Theory: ', '{:.3f}'.format(theory[0][0]))
    print('Solved: ', '{:.3f}'.format(e))