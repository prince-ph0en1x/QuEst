# QuTiP v4.3.1.zip from website
# pip3 install qutip

# https://github.com/rigetti/grove/blob/master/grove/tomography/utils.py
import qutip as qt
from collections import OrderedDict
import numpy as np
from itertools import product as cartesian_product


# https://github.com/rigetti/grove/blob/master/grove/tomography/operator_utils.py
QI = qt.qeye(2)
QX = qt.sigmax()
QY = qt.sigmay()
QZ = qt.sigmaz()
## print(QI,QX,QY,QZ)

PAULI_BASIS = OrderedDict([	("I", QI / np.sqrt(2)), 
							("X", QX / np.sqrt(2)),
				 			("Y", QY / np.sqrt(2)), 
				 			("Z", QZ / np.sqrt(2))])
## print(PAULI_BASIS)

# https://github.com/rigetti/grove/blob/master/grove/tomography/tomography.py
'''
TOMOGRAPHY_GATES = OrderedDict([(I, QI),																			No rotation for I
								(lambda q: RX(np.pi / 2, q), (-1j * np.pi / 4 * QX).expm()),						Rx(pi/2) for X
								(lambda q: RY(np.pi / 2, q), (-1j * np.pi / 4 * QY).expm()),						Ry(pi/2) for Y
								(lambda q: RX(np.pi, q), (-1j * np.pi / 2 * QX).expm())])							Rx(pi) for Z
'''
qubits = [0,1]
TOMOGRAPHY_GATES = OrderedDict([('I','qI'),
								('X','qX'),
								('Y','qY'),
								('Z','qZ')])
for gates in cartesian_product(TOMOGRAPHY_GATES.keys(), repeat=len(qubits)):
	for qubit, gate in zip(qubits, gates):
		print(gate+str(qubit)+" ",end="")
	print()