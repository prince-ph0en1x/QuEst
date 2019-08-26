import numpy as np
from openql import openql as ql
from qxelarator import qxelarator
import re
import os
import sys
import random
from math import cos,sin
from cmath import exp


METHOD = "state" # Used to select between using QX.get_state() or running trials for getting the HST cost
NUM_QUBIT = None
qx = qxelarator.QX()

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# HST refers to Hilbert-Schmidt Test circuit from https://arxiv.org/pdf/1807.00800.pdf
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

"""
Read the lines of the reference QASM circuit to be matched by the parameterized test circuit
"""
def read_input_circuit(qasm):
    global NUM_QUBIT
    code = ''
    with open(qasm, 'r') as f:
        for line in f.readlines():
            l = line.strip()
            if l.startswith('qubits'):
                NUM_QUBIT = int(l.split(' ')[1])
            elif not l.startswith(('#', '.', 'version')):
                code += l
    return code
    

"""
Create the initial portion of the HST circuit
"""
def pre_hst_qasm():
    config_fn = os.path.abspath('test_cfg_none_simple.json')
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
    config_fn = os.path.abspath('test_cfg_none_simple.json')
    platform = ql.Platform('platform_none', config_fn)
    prog = ql.Program('post_hst', platform, 2*NUM_QUBIT)
    k1 = ql.Kernel('QK1',platform, 2*NUM_QUBIT)

    if qubit is None:
        for q in range(NUM_QUBIT):
            k1.gate('cnot', [q, q + NUM_QUBIT])
            k1.gate('h', [q])
        if METHOD != "state":
            for q in range(2 * NUM_QUBIT):
                k1.measure(q)
    else:
        k1.gate('cnot', [qubit, qubit + NUM_QUBIT])
        k1.gate('h', [qubit])
        if METHOD != "state":
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
def parameterized_circuit_qasm(entangler, blocks):
    config_fn = os.path.abspath('test_cfg_none_simple.json')
    platform = ql.Platform('platform_none', config_fn)
    prog = ql.Program('test_circuit', platform, 2*NUM_QUBIT)
    k1 = ql.Kernel('QK1', platform, 2*NUM_QUBIT)

    for i in range(blocks):
        for q in range(NUM_QUBIT):
            b = q*3 + 3*NUM_QUBIT*i
            k1.rz(q + NUM_QUBIT, b)
            k1.rx(q + NUM_QUBIT, b + 1)
            k1.rz(q + NUM_QUBIT, b + 2)

        for pair in entangler:
            pair = [p + NUM_QUBIT for p in pair]
            k1.gate('cnot', pair)

    prog.add_kernel(k1)
    tmp = prog.qasm().split('\n')
    prog.compile()
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
def update_qasm_parameters(params, circ, qasm='test_output/hst.qasm'):
    tmp = str(circ) # Ensure a new copy of the circuit is created
    for p in range(len(params)):
        stub = '%' + str(p) + '%'
        ang  = '{:.6f}'.format(params[p])
        tmp = tmp.replace(stub, ang)

    with open(qasm, 'w') as f:
        f.write(tmp)

    return qasm


"""
Run the HST circuit #trials# times and calculate the probability that all measurements are 0.
This corresponds to the magnitude of the Hilbert-Schmidt inner product and is used as the cost
function from the paper linked at the top of this file
"""
def evaluate_hst_trials(qasm, trials):
    global qx
    qx.set(qasm)

    p = 0
    for _ in range(trials):
        qx.execute(False)
        c = [qx.get_measurement_outcome(i) for i in range(2*NUM_QUBIT)]

        if sum(c) == 0:
            p += 1

    return p/trials


"""
Parses QX state string using regex and returns probabilities of each state in ordered list
"""
def parse_get_state(s):
    nums = re.findall(r'[-+.e\d]+,[-+.e\d]+', s) # Regex to find all complex numbers in state string
    bras = re.findall(r'\|[01]+>', s) # Regex to find all bra-kets in state string
    comp = []
    ind  = []
    poss = 0

    # Convert all complex number strings to complex objects
    for n in nums:
        t = n.split(',')
        comp.append( complex(float(t[0]), float(t[1])) )

    # Convert all bra-kets to indices
    for b in bras:
        t = b.strip('|').strip('>')
        poss = 2**(len(t))
        ind.append( int(t, base=2) )

    # Add complex numbers to an ordered list
    C = [0 for _ in range(poss)]
    for i, c in enumerate(comp):
        j = ind[i]
        C[j] = c

    return C


"""
Run the HST circuit and retrieve state probabilities using the built-in QX get_state function.
The probability of the all-zero state is returned as the cost
"""
def evaluate_hst_state(qasm):
    global qx
    qx.set(qasm)
    qx.execute()
    probs = parse_get_state(qx.get_state())
    return probs[0].real


"""
Runs the HST with updated parameters and returns the appropriate cost
"""
def hst_cost(params, hst, trials):
    qasm = update_qasm_parameters(params, hst)
    if METHOD == "state":
        cost = evaluate_hst_state(qasm)
    else:
        cost = evaluate_hst_trials(qasm, trials)
    return 1 - cost


"""
Generates the HST QASM and returns the code as a string
"""
def hst_test(qasm, ent, blocks):
    circ = read_input_circuit(qasm)
    pre = pre_hst_qasm()
    post = post_hst_qasm()
    v = parameterized_circuit_qasm(ent, blocks)
    hst = merge_qasms(pre, circ, v, post)
    return hst


"""
Main function for optimizing a variational quantum circuit to match a reference circuit using the HST
"""
def optimize(hst, param_len, runs=200, trials=200):
    inc = [0, -np.pi/2, -np.pi/4, np.pi/4, np.pi/2]
    min_cost = 1
    min_x0 = []

    for _ in range(runs):
        x0 = [random.randint(0, 4) * np.pi/4 for _ in range(param_len)]
        inc = [0] + [-random.random()*np.pi/2 for _ in range(2)] + [random.random()*np.pi/2 for _ in range(2)]
        order = list(range(len(x0)))
        random.shuffle(order)
        for i in order:
            c = []
            for j in inc:
                x = list(x0)
                x[i] += j
                a = hst_cost(x, hst, trials)
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

    update_qasm_parameters(min_x0, hst, qasm='test_output/QAQC.qasm')
    return min_cost, min_x0