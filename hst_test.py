import numpy as np
import QAQC
import random

NUM_QUBIT = 3

def entangler1(qubits, locality=float('inf')):
    ent = []
    for q in range(qubits):
        k = 1
        while k < locality and q + k < qubits:
            ent.append((q, q + k))
            k += 1
    return ent


def entangler2(qubits):
    ent = []
    for q in range(1, qubits, 2):
        ent.append((q-1, q))

    for q in range(2, qubits, 2):
        ent.append((q-1, q))
    return ent


def test_optimize(qasm, blocks):
    ent = entangler1(NUM_QUBIT)
    print('Blocks:', blocks)
    hst = QAQC.hst_test(qasm, ent, blocks)
    c, _ = QAQC.optimize(hst, 3*QAQC.NUM_QUBIT*blocks)
    print('Cost:', c)
    return c


if __name__ == '__main__':
    #import sys
    #b = int(sys.argv[1])
    costs = []
    blocks = list(range(4,5))
    for b in blocks:
        c = test_optimize("test_output/RAND_UNITARY.qasm", b)
        costs.append(c)
    exit()

    from matplotlib import pyplot as plt
    plt.plot(blocks, costs, 'ro')
    plt.xlabel('Parameterized Gate Blocks')
    plt.ylabel('Min HST Cost')
    plt.show()

    files = ['test_output/RAND_UNITARY_scheduled.qasm', 'test_output/test_circuit_scheduled.qasm']
    for file in files:
        lines = 0
        with open(file, "r") as f:
            for l in f.readlines():
                l = l.strip()
                if l.startswith(('#', '.', 'version', 'qubits', 'wait')) or l == '':
                    continue
                lines += 1
        print(file, ':', lines, 'LOC')
