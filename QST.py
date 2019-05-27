import os
from openql import openql as ql
import re
from qxelarator import qxelarator
import numpy as np
from collections import OrderedDict
from itertools import product


NUM_QUBIT = 0
PI_2 = np.pi/2

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
eig1 = [1, -1] # Eigenvalues of sg1
eig2 = [1, -1] # Eigenvalues of sg2
eig3 = [1, -1] # Eigenvalues of sg3
eigens = {'i':eig0, 'x':eig1, 'y':eig2, 'z':eig3}

"""
The stateprep method encapsulates the quantum algorithm which generates an unknown n-qubit quantum state from an n-qubit all-zero state
We want to estimate the density matrix of this n-qubit evolved state using State Tomography
"""
def stateprep():
    config_fn = os.path.abspath('/home/neil/dev/tud/OpenQL/tests/test_cfg_none_simple.json')
    platform = ql.Platform('platform_none', config_fn)
    prog = ql.Program('p_name', platform, NUM_QUBIT)
    k1 = ql.Kernel('QK1',platform, NUM_QUBIT)
    k1.gate('h', [0])
    k1.gate('h', [1])
    k1.gate('h', [2])
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


"""
Append tomographic rotations and measurements to base qasm and return path to new qasm file
"""
def prep_trial(t_rot, qasm):
    file = open(qasm, "r")
    temp = os.path.dirname(qasm) + '/tomo.qasm'
    fileopt = open(temp, "w")

    for line in file:
        fileopt.write(line)
    file.close()
    
    fileopt.write(t_rot)
    for i in range(NUM_QUBIT):
        fileopt.write("measure q" + str(i) + "\n")
    fileopt.close()
    
    return temp


"""
Invokes Qxelerator and returns measurement statistics in the Z-basis (computational)
"""   
def run_trials(qasm, trials):
    qx = qxelarator.QX()
    qx.set(qasm)
    p = np.zeros(2**NUM_QUBIT)
    c = np.zeros(NUM_QUBIT, dtype=bool)
    for i in range(trials):
        qx.execute()
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
    for bases in product(TOMOGRAPHY_GATES.keys(), repeat=NUM_QUBIT):
        t_rot = ""
        qubit = NUM_QUBIT - 1 # The leftmost basis is for the MSB (n-th) qubit, so qubit number decreases from n
        for b in bases:
            if b == 'x':
                t_rot += ('ry q' + str(qubit) + ", " + str(-PI_2) + "\n")
            elif b == 'y':
                t_rot += ('rx q' + str(qubit) + ", " + str(PI_2) + "\n")
            # In the current setup, no rotation is needed to measure in the z-basis
            # elif b == 'z':
            #     t_rot += ('rz q' + str(qubit) + ", " + str(PI_2) + "\n")
            qubit -= 1
        tomo = prep_trial(t_rot, qasm)
        stat = run_trials(tomo, trials)
        stats.append(stat)
    return stats


"""
Uses the math described here (http://research.physics.illinois.edu/QI/Photonics/tomography-files/tomo_chapter_2004.pdf)
to reconstruct the density matrix from the tomographic histogram
"""
def generate_density_matrix(hist):
    dm = np.zeros((2**NUM_QUBIT, 2**NUM_QUBIT)) * 0j
    idx = 0
    for bases in product(TOMOGRAPHY_GATES.keys(), repeat=NUM_QUBIT):
        ppm = [1]
        e_val = [1]
        for b in bases:
            ppm = np.kron(sigmas[b], ppm)
            e_val = np.kron(eigens[b], e_val)

        Si = sum(np.multiply(e_val, hist[idx])) # Multiply each sign to its respective probability and sum
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
Runs Quantum State Tomography on a given qasm program and returns the expectation value of the Hamiltonian
"""
def tomography_expectation(H, qasm, num_qubits, trials=100):
    rho = tomography(qasm, num_qubits, trials=trials)
    return np.trace(rho * H)


if __name__ == '__main__':
    import sys
    qasm = "test_output/algo.qasm"

    # Find the number of qubits in the program
    with open(qasm, 'r') as f:
        for line in f:
            if line.startswith('qubits'):
                q = int(line.split(' ')[1])
                break

    t = int(sys.argv[1])
    dm = tomography(qasm, q, trials=t)
    plot_results(dm)