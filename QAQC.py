import numpy as np
from openql import openql as ql
from qxelarator import qxelarator
import re
import os
import sys
from scipy.optimize import minimize
from random import random
from math import cos,sin
from cmath import exp


NUM_QUBIT = 0


# HST refers to Hilbert-Schmidt Test circuit from https://arxiv.org/pdf/1807.00800.pdf


"""
Firefighting solution as Qxelarator is not updated to run cQASM v1.0
Open Issue: https://github.com/QE-Lab/qx-simulator/issues/57
Converts OpenQL generated cQASM to old Qxelerator compatible syntax
"""
def qasmVerConv(qasm_in, qasm_out):
    file = open(qasm_in,"r")
    fileopt = open(qasm_out,"w")
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
Read the lines of an input QASM circuit that is to be compiled against
"""
def read_input_circuit(qasm):
    global NUM_QUBIT
    lines = []
    with open(qasm, 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            elif line.startswith('qubits'):
                NUM_QUBIT = int(line.split(' ')[1])
            elif not line.startswith('.'):
                lines.append(line)

    return lines
    

"""
Create the initial HST circuit, including the reference circuit (U)
"""
def pre_hst_qasm(U):
    config_fn = os.path.abspath('/home/neil/dev/tud/OpenQL/tests/test_cfg_none_simple.json')
    platform = ql.Platform('platform_none', config_fn)
    prog = ql.Program('tmp', platform, 2*NUM_QUBIT)
    k1 = ql.Kernel('QK1',platform, 2*NUM_QUBIT)

    for q in range(NUM_QUBIT):
        k1.gate('h', [q])
        k1.gate('cnot', [q, NUM_QUBIT + q])

    prog.add_kernel(k1)
    prog.compile()
    qasm = 'test_output/pre_hst.qasm'
    qasmVerConv('test_output/tmp.qasm', qasm)

    with open(qasm, 'a') as f:
        f.writelines(U)

    return qasm


"""
Create the ending portion of the HST circuit. When qubit is None, the full HST circuit
is created. Otherwise, the Local HST is created for the specified qubit number
"""
def post_hst_qasm(qubit=None):
    config_fn = os.path.abspath('/home/neil/dev/tud/OpenQL/tests/test_cfg_none_simple.json')
    platform = ql.Platform('platform_none', config_fn)
    prog = ql.Program('tmp', platform, 2*NUM_QUBIT)
    k1 = ql.Kernel('QK1',platform, 2*NUM_QUBIT)

    if qubit is None:
        for q in range(NUM_QUBIT):
            k1.gate('cnot', [q, q + NUM_QUBIT])
            k1.gate('h', [q])

        for q in range(2 * NUM_QUBIT):
            k1.measure(q)
    else:
        k1.gate('cnot', [qubit, qubit + NUM_QUBIT])
        k1.gate('h', [qubit])
        k1.measure(qubit)
        k1.measure(qubit + NUM_QUBIT)

    prog.add_kernel(k1)
    prog.compile()
    qasm = 'test_output/post_hst.qasm'
    qasmVerConv('test_output/tmp.qasm', qasm)
    return qasm


"""
Create the QASM for the test circuit using the provided parameters (e.g. angles)
"""
def test_sequence_qasm(params, locality, blocks):
    config_fn = os.path.abspath('/home/neil/dev/tud/OpenQL/tests/test_cfg_none_simple.json')
    platform = ql.Platform('platform_none', config_fn)
    prog = ql.Program('tmp', platform, 2*NUM_QUBIT)
    k1 = ql.Kernel('QK1',platform, 2*NUM_QUBIT)

    for i in range(blocks):
        for q in range(NUM_QUBIT):
            b = q*3 + 3*NUM_QUBIT*i
            k1.rz(q + NUM_QUBIT, params[b])
            k1.rx(q + NUM_QUBIT, params[b + 1])
            k1.rz(q + NUM_QUBIT, params[b + 2])

        for q in range(NUM_QUBIT, 2*NUM_QUBIT):
            k = 1
            while k < locality and q + k < 2*NUM_QUBIT:
                k1.cz(q, q + k)
                k += 1

    prog.add_kernel(k1)
    prog.compile()
    qasm = 'test_output/test_circuit.qasm'
    qasmVerConv('test_output/tmp.qasm', qasm)
    return qasm


"""
Merge all necessary QASM files to create a usable HST circuit
"""
def merge_qasms(pre, V, post):
    qasm = 'test_output/hst.qasm'
    lines = []
    with open(pre, 'r') as f:
        for line in f.readlines():
            lines.append(line)

    lines.append('\n')

    with open(V, 'r') as f:
        flag = False
        for line in f.readlines():
            if flag:
                lines.append(line)
            elif line.startswith('.'): # Wait until a kernel is reached
                flag = True
    
    lines.append('\n')

    with open(post, 'r') as f:
        flag = False
        for line in f.readlines():
            if flag:
                lines.append(line)
            elif line.startswith('.'): # Wait until a kernel is reached
                flag = True
    
    with open(qasm, 'w') as f:
        f.writelines(lines)

    return qasm


"""
Run the HST circuit #trials# times and calculate the probability that all measurements are 0.
This corresponds to the magnitude of the Hilbert-Schmidt inner product and is used as the cost
function from the aforementioned paper
"""
def evaluate_hst(qasm, trials=100):
    qx = qxelarator.QX()
    qx.set(qasm)

    p = 0
    for _ in range(trials):
        qx.execute()
        c = [qx.get_measurement_outcome(i) for i in range(2*NUM_QUBIT)]

        if sum(c) == 0:
            p += 1
    del qx

    return p/trials


def hst_cost(params, pre, post, locality, blocks, trials=100):
    v_hst = test_sequence_qasm(params, locality, blocks)
    hst = merge_qasms(pre, v_hst, post)
    cost = evaluate_hst(hst, trials=trials)
    return 1 - cost


def optimize(qasm, locality, blocks, x0=None, tol=1e-6, disp=False, trials=100):
    circ = read_input_circuit(qasm)
    pre = pre_hst_qasm(circ)
    post = post_hst_qasm()

    if x0 is None:
        x0 = [0 for _ in range(3*NUM_QUBIT*blocks)]
    #    min_d = 1
    #    min_x = [0 for _ in range(3*NUM_QUBIT*blocks)]
    #    for _ in range(int(500/blocks/NUM_QUBIT)):
    #        x = [random()*np.pi for _ in range(3*NUM_QUBIT*blocks)]
    #        d = hst_cost(x, pre, post, locality, blocks)
    #        if d < min_d:
    #            min_x = x
    #            min_d = d
    #    x0 = min_x

    res = minimize(hst_cost, x0, args=(pre, post, locality, blocks), method='Powell', tol=tol, options={'disp':disp, 'return_all':disp})
    return res


# Parameterized rotation gate matrices: http://www.mpedram.com/Papers/Rotation-based-DDSyn-QICJ.pdf


def Rx(O):
    return [[cos(O/2), -1j*sin(O/2)], [-1j*sin(O/2), cos(O/2)]]
    

def Ry(O):
    return [[cos(O/2), -sin(O/2)], [sin(O/2), cos(O/2)]]

                    
def Rz(O):
    return [[exp(-1j*O/2), 0], [0, exp(1j*O/2)]]