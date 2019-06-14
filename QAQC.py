import numpy as np
from openql import openql as ql
from qxelarator import qxelarator
import re
import os
import sys
from scipy.optimize import minimize


NUM_QUBIT = 0


# HST refers to Hilbert-Schmidt Test circuit from https://arxiv.org/pdf/1807.00800.pdf


"""
Read the lines of an input QASM circuit that is to be matched
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
Create the ending portion of the HST circuit
"""
def post_hst_qasm():
    config_fn = os.path.abspath('/home/neil/dev/tud/OpenQL/tests/test_cfg_none_simple.json')
    platform = ql.Platform('platform_none', config_fn)
    prog = ql.Program('tmp', platform, 2*NUM_QUBIT)
    k1 = ql.Kernel('QK1',platform, 2*NUM_QUBIT)

    for q in range(NUM_QUBIT):
        k1.gate('cnot', [q, q + NUM_QUBIT])
        k1.gate('h', [q])

    for q in range(2 * NUM_QUBIT):
        k1.measure(q)

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
            b = q*NUM_QUBIT + 3*NUM_QUBIT*i
            k1.rz(q + NUM_QUBIT, params[b])
            k1.ry(q + NUM_QUBIT, params[b + 1])
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

    return p/trials


def hst_cost(params, pre, post, locality, blocks):
    v_hst = test_sequence_qasm(params, locality, blocks)
    hst = merge_qasms(pre, v_hst, post)
    cost = evaluate_hst(hst)
    return 1 - cost

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