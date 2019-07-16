import numpy as np
from openql import openql as ql
from qxelarator import qxelarator
import re
import os
import sys
import random
from math import cos,sin
from cmath import exp


NUM_QUBIT = 0
qx = qxelarator.QX()


# HST refers to Hilbert-Schmidt Test circuit from https://arxiv.org/pdf/1807.00800.pdf


"""
Read the lines of the input QASM circuit to be matched by the parameterized test circuit
"""
def read_input_circuit(qasm):
    global NUM_QUBIT
    code = ''
    with open(qasm, 'r') as f:
        flag = False
        for line in f.readlines():
            if flag and not line.strip().startswith('#'):
                code += line
            elif line.startswith('.'): # Wait until a kernel is reached
                flag = True
            elif line.startswith('qubits'):
                NUM_QUBIT = int(line.split(' ')[1])

    return code
    

"""
Create the initial portion of the HST circuit
"""
def pre_hst_qasm():
    config_fn = os.path.abspath('/home/neil/dev/tud/OpenQL/tests/test_cfg_none_simple.json')
    platform = ql.Platform('platform_none', config_fn)
    prog = ql.Program('pre_hst', platform, 2*NUM_QUBIT)
    k1 = ql.Kernel('QK1', platform, 2*NUM_QUBIT)

    for q in range(NUM_QUBIT):
        k1.gate('h', [q])
        k1.gate('cnot', [q, NUM_QUBIT + q])

    prog.add_kernel(k1)
    qasm = prog.qasm()
    return qasm


"""
Create the ending portion of the HST circuit. When qubit is None, the full HST circuit
is created. Otherwise, the Local HST is created for the specified qubit number
"""
def post_hst_qasm(qubit=None):
    config_fn = os.path.abspath('/home/neil/dev/tud/OpenQL/tests/test_cfg_none_simple.json')
    platform = ql.Platform('platform_none', config_fn)
    prog = ql.Program('post_hst', platform, 2*NUM_QUBIT)
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
    tmp = prog.qasm().split('\n')
    qasm = []

    # Remove unnecessary lines from the post-HST QASM code
    for l in tmp:
        if l.startswith(('#', '.', 'version', 'qubits')) or l == '':
            continue
        qasm.append(l + '\n')
    
    return ''.join(qasm)


"""
Create the QASM for the parameterized circuit
"""
def parameterized_circuit_qasm(locality, blocks):
    config_fn = os.path.abspath('/home/neil/dev/tud/OpenQL/tests/test_cfg_none_simple.json')
    platform = ql.Platform('platform_none', config_fn)
    prog = ql.Program('test_circuit', platform, 2*NUM_QUBIT)
    k1 = ql.Kernel('QK1',platform, 2*NUM_QUBIT)

    for i in range(blocks):
        for q in range(NUM_QUBIT):
            b = q*3 + 3*NUM_QUBIT*i
            k1.rz(q + NUM_QUBIT, b)
            k1.rx(q + NUM_QUBIT, b + 1)
            k1.rz(q + NUM_QUBIT, b + 2)

        for q in range(NUM_QUBIT, 2*NUM_QUBIT):
            k = 1
            while k < locality and q + k < 2*NUM_QUBIT:
                k1.cz(q, q + k)
                k += 1

    prog.add_kernel(k1)
    tmp = prog.qasm().split('\n')
    qasm = []

    # Remove unnecessary lines from the parameterized QASM code
    for l in tmp:
        if l.startswith(('#', '.', 'version', 'qubits')) or l == '':
            continue
        qasm.append(l + '\n')

    # Add stubs indicating where to place parameters in the QASM
    q = 0
    for i in range(len(qasm)):
        search = str(q)+'.000000'
        if search in qasm[i]:
            qasm[i] = qasm[i].replace(search, '%' + str(q) + '%')
            q += 1

    return ''.join(qasm)


"""
Merge all necessary QASM files to create a usable HST circuit
"""
def merge_qasms(pre, circ, v, post):
    qasm = pre + '\n' + circ + '\n' + v + '\n' + post
    return qasm


"""
Update the parameters in the HST QASM using the given parameters
"""
def update_hst_parameters(params, hst):
    tmp = str(hst)
    for p in range(len(params)):
        tmp = tmp.replace('%' + str(p) + '%', '{:.6f}'.format(params[p]))

    qasm = 'test_output/hst.qasm'
    with open(qasm, 'w') as f:
        f.write(tmp)

    return qasm


"""
Run the HST circuit #trials# times and calculate the probability that all measurements are 0.
This corresponds to the magnitude of the Hilbert-Schmidt inner product and is used as the cost
function from the paper linked at the top of this file
"""
def evaluate_hst(qasm, trials=100):
    global qx
    qx.set(qasm)

    p = 0
    for _ in range(trials):
        qx.execute(False)
        c = [qx.get_measurement_outcome(i) for i in range(2*NUM_QUBIT)]

        if sum(c) == 0:
            p += 1

    return p/trials


def hst_cost(params, hst, trials=100):
    qasm = update_hst_parameters(params, hst)
    cost = evaluate_hst(qasm, trials=trials)
    return 1 - cost


def optimize(qasm, locality, blocks, runs=3, x0=None, trials=200):
    circ = read_input_circuit(qasm)
    pre = pre_hst_qasm()
    post = post_hst_qasm()
    v = parameterized_circuit_qasm(locality, blocks)
    hst = merge_qasms(pre, circ, v, post)

    if x0 is None:
        x0 = [0 for _ in range(3*NUM_QUBIT*blocks)]

    inc = [0, -np.pi/2, -np.pi/4, np.pi/4, np.pi/2]
    min_cost = 1
    min_x0 = []
    order = list(range(len(x0)))

    for z in range(runs):
        #if z == int(runs/1.5):
        #    inc = [x/2 for x in inc]

        random.shuffle(order)
        for i in order:
            c = []
            for j in inc:
                x = list(x0)
                x[i] += j
                a = hst_cost(x, hst, trials=trials)
                c.append(a)
                if a < min_cost:
                    min_cost = a
                    min_x0 = list(x)
            ind = c.index(min(c))
            x0[i] += inc[ind]
            if min_cost == 0:
                break
        if min_cost == 0:
            break

    return min_cost, min_x0


def generate_output_qasm(params, locality, blocks):
    config_fn = os.path.abspath('/home/neil/dev/tud/OpenQL/tests/test_cfg_none_simple.json')
    platform = ql.Platform('platform_none', config_fn)
    prog = ql.Program('QAQC', platform, NUM_QUBIT)
    k1 = ql.Kernel('QAQC',platform, NUM_QUBIT)

    for i in range(blocks):
        for q in range(NUM_QUBIT):
            b = q*3 + 3*NUM_QUBIT*i
            # Note the parameters are conjugated because the circuit is intrinsically
            # optimized for V*, not V, so rx and rz gates have negated parameters
            k1.rz(q, -params[b])
            k1.rx(q, -params[b + 1])
            k1.rz(q, -params[b + 2])

        for q in range(NUM_QUBIT):
            k = 1
            while k < locality and q + k < NUM_QUBIT:
                k1.cz(q, q + k)
                k += 1

    prog.add_kernel(k1)
    prog.compile()
    qasm = 'test_output/QAQC.qasm'
    return qasm


# Parameterized rotation gate matrices: http://www.mpedram.com/Papers/Rotation-based-DDSyn-QICJ.pdf
def Rx(O):
    return [[cos(O/2), -1j*sin(O/2)], [-1j*sin(O/2), cos(O/2)]]
    

def Ry(O):
    return [[cos(O/2), -sin(O/2)], [sin(O/2), cos(O/2)]]

                    
def Rz(O):
    return [[exp(-1j*O/2), 0], [0, exp(1j*O/2)]]