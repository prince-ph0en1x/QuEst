import os
from openql import openql as ql
import re
from qxelarator import qxelarator
import numpy as np
from collections import OrderedDict
from itertools import product


NUM_QUBIT = 0
PI_2 = np.pi/2
qx = qxelarator.QX()

TOMOGRAPHY_GATES = OrderedDict([('i','Identity'),
								('x','Pauli-X'),
								('y','Pauli-Y'),
								('z','Pauli-Z')])

sg0 = [[1, 0], [0, 1]] # Identity
sg1 = [[0, 1], [1, 0]] # Pauli-X
sg2 = [[0,-1j],[1j,0]] # Pauli-Y
sg3 = [[1, 0], [0,-1]] # Pauli-Z
sigmas = {'i':sg0, 'x':sg1, 'y':sg2, 'z':sg3}

eig0 = [1,  1] # Eigenvalues of sg0
eig1 = [1, -1] # Eigenvalues of sg1, sg2, sg3
eigens = {'i':eig0, 'x':eig1, 'y':eig1, 'z':eig1}


"""
Append tomographic rotations and measurements to base qasm and return path to new qasm file
"""
def prep_trial(t_rot, code):
    qasm = 'test_output/tomo.qasm'
    with open(qasm, 'w') as f:
        f.write(code)
        f.write(t_rot)
        for i in range(NUM_QUBIT):
            f.write("    measure q[" + str(i) + "]\n")
    
    return qasm


"""
Invokes Qxelerator and returns measurement statistics in the Z-basis (computational)
"""   
def run_trials(qasm, trials):
    global qx
    qx.set(qasm)
    
    p = np.zeros(2**NUM_QUBIT)
    c = np.zeros(NUM_QUBIT, dtype=bool)
    for _ in range(trials):
        qx.execute(False)
        for i in range(NUM_QUBIT):
            c[i] = qx.get_measurement_outcome(i)
        idx = sum(v<<i for i, v in enumerate(c[::-1]))
        p[idx] += 1

    p /= trials
    return p


"""
Generate a tomographic histogram for the given qasm file through repeated rotations and measurements
"""
def generate_histogram(qasm, trials):
    stats = []
    code = ''
    with open(qasm, 'r') as f:
        code = f.read()

    for bases in product(TOMOGRAPHY_GATES.keys(), repeat=NUM_QUBIT):
        t_rot = ""
        qubit = NUM_QUBIT - 1 # The leftmost basis is for the MSB (n-th) qubit, so qubit number decreases from n
        for b in bases:
            if b == 'x':
                t_rot += ('    ry q[' + str(qubit) + "], " + str(-PI_2) + "\n")
            elif b == 'y':
                t_rot += ('    rx q[' + str(qubit) + "], " + str(PI_2) + "\n")
            # In the current setup, no rotation is needed to measure in the z-basis
            qubit -= 1
        tomo = prep_trial(t_rot, code)
        stat = run_trials(tomo, trials)
        stats.append(stat)
    return stats


"""
Uses the math described here (http://research.physics.illinois.edu/QI/Photonics/tomography-files/tomo_chapter_2004.pdf)
to construct the density matrix from the tomographic histogram
"""
def generate_density_matrix(hist):
    dm = np.zeros((2**NUM_QUBIT, 2**NUM_QUBIT)) * 0j
    idx = 0
    for bases in product(TOMOGRAPHY_GATES.keys(), repeat=NUM_QUBIT):
        ppm = [1]
        eigs = [1]
        for b in bases:
            ppm = np.kron(sigmas[b], ppm)
            eigs = np.kron(eigens[b], eigs)

        Si = sum(np.multiply(eigs, hist[idx])) # Multiply each sign to its respective probability and sum
        dm += Si*ppm
        idx += 1

    dm /= (2**NUM_QUBIT)
    return dm


"""
Plots the real and imag parts of the density matrix as 3D bar graphs
"""
def plot_results(dm):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    labels = [''.join(i) for i in product(('0', '1'), repeat=NUM_QUBIT)]
    l = len(dm)
    x = [i for i in range(l)] * l
    y = [j for i in range(l) for j in [i] * l]
    z = [0] * (l * l)

    ax1 = fig.add_subplot(121, projection='3d')
    plt.xticks([i + 0.25 for i in range(l)], labels)
    plt.yticks([i + 0.25 for i in range(l)], labels)

    ax2 = fig.add_subplot(122, projection='3d')
    plt.xticks([i + 0.25 for i in range(l)], labels)
    plt.yticks([i + 0.25 for i in range(l)], labels)

    dx = np.ones(l * l) * 0.5
    dy = np.ones(l * l) * 0.5
    dz_real = [j.real for i in dm for j in i]
    dz_imag = [j.imag for i in dm for j in i]

    ax1.bar3d(x, y, z, dx, dy, dz_real)
    ax1.set_zlim(0, 1)
    ax1.set_title('Real')
    ax2.bar3d(x, y, z, dx, dy, dz_imag)
    ax2.set_zlim(0, 1)
    ax2.set_title('Imag')

    plt.show()


"""
Runs Quantum State Tomography on a given qasm program and returns the calculated density matrix
"""
def tomography(qasm, num_qubits, trials=100):
    global NUM_QUBIT
    NUM_QUBIT = num_qubits

    hist = generate_histogram(qasm, trials)
    dm = generate_density_matrix(hist)
    return dm


"""
Returns the expectation value of a Hamiltonian associated with a given density matrix
"""
def expectation(rho, H):
    expectation = np.trace(np.dot(rho, H))
    return expectation