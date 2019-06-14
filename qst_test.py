import numpy as np
import QST
from openql import openql as ql
import re
import os
from math import cos, sin, sqrt
import random


NUM_QUBIT = 3
HADA = np.array([[1, 1], [1, -1]]) / sqrt(2)


"""
The stateprep method encapsulates the quantum algorithm which generates an unknown n-qubit quantum state from an n-qubit all-zero state
We want to estimate the density matrix of this n-qubit evolved state using State Tomography
"""
def stateprep():
    config_fn = os.path.abspath('/home/neil/dev/tud/OpenQL/tests/test_cfg_none_simple.json')
    platform = ql.Platform('platform_none', config_fn)
    prog = ql.Program('p_name', platform, NUM_QUBIT)
    k1 = ql.Kernel('QK1',platform, NUM_QUBIT)
    for _ in range(3):
        k1.gate('cnot', [2, 0])
        k1.ry(0, 1.02253338)
        k1.ry(1, 2.68175523)
        k1.ry(2, 4.40249788)
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
    
    H1 = 0.2 * np.kron(QST.sg1, np.kron(QST.sg3, QST.sg1))
    H2 = 0.9 * np.kron(QST.sg1, np.kron(QST.sg0, QST.sg1))
    H3 = 0.3 * np.kron(QST.sg3, np.kron(QST.sg3, QST.sg3))
    H = H1 + H2 + H3

    vals, _ = np.linalg.eig(H)

    dm = QST.tomography(qasm, NUM_QUBIT, trials=t)
    e = QST.expectation(dm, H)

    print('\nDensity Matrix')
    for r in dm:
        print('[', end='')
        for i in r:
            print('{0.real: .2f} '.format(i.real), end='')
        print(']')

    print('\nHamiltonian')
    for r in H:
        print('[', end='')
        for i in r:
            print('{0.real: .2f} '.format(i.real), end='')
        print(']')

    M = np.dot(dm, H)
    print('\nrho * H')
    for r in M:
        print('[', end='')
        for i in r:
            print('{0.real: .2f} '.format(i.real), end='')
        print(']')
    
    print('\nExpectation\n', e)