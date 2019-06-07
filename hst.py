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
def test_sequence_qasm(params):
    config_fn = os.path.abspath('/home/neil/dev/tud/OpenQL/tests/test_cfg_none_simple.json')
    platform = ql.Platform('platform_none', config_fn)
    prog = ql.Program('tmp', platform, 2*NUM_QUBIT)
    k1 = ql.Kernel('QK1',platform, 2*NUM_QUBIT)

    #for q in range(NUM_QUBIT + 1, 2*NUM_QUBIT):
    #    k1.cnot(NUM_QUBIT, q)

    #for q in range(NUM_QUBIT):
    #    k1.ry(q + NUM_QUBIT, params[q])

    k1.ry(2, params[0])
    k1.rx(2, params[1])
    k1.cnot(2, 3)
    k1.rx(2, params[2])
    k1.cnot(2, 3)
    k1.ry(2, params[3])
    k1.cnot(2, 3)
    k1.rx(2, params[4])

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
            if line.startswith('.'): # Wait until a kernel is reached
                flag = True
                continue
            if flag:
                lines.append(line)
    
    lines.append('\n')

    with open(post, 'r') as f:
        flag = False
        for line in f.readlines():
            if line.startswith('.'): # Wait until a kernel is reached
                flag = True
                continue
            if flag:
                lines.append(line)
    
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


def minimize_func(params, pre, post):
    v_hst = test_sequence_qasm(params)
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


if __name__ == '__main__':
    U = read_input_circuit('test_output/algo.qasm')
    pre_hst = pre_hst_qasm(U)
    post_hst = post_hst_qasm()

    x0 = [0, 0, 0, 0, 0] # for _ in range(NUM_QUBIT)
    res = minimize(minimize_func, x0, args=(pre_hst, post_hst), method='Powell', tol=1e-6, options={'disp':True, 'return_all':True})
    #print(minimize_func(x0, pre_hst, post_hst))
    print('Params: ', res.x)


"""
    U = read_input_circuit('test_output/algo.qasm')
    pre_hst = pre_hst_qasm(U)
    post_hst = post_hst_qasm()

    res = 15

    X = []
    Y = []
    Z = []
    angs1 = []
    for a in range(-res, res+1):
        ang1 = 2*np.pi/res * a
        angs2 = []
        costs = []
        for b in range(-res, res+1):
            ang2 = 2*np.pi/res * b
            angs1.append(ang1)
            angs2.append(ang2)
            costs.append(minimize_func([ang1, ang2], pre_hst, post_hst))
        X.append(angs1)
        Y.append(angs2)
        Z.append(costs)
        angs1 = []

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    plt.show()
"""

"""
    t = int(sys.argv[1])
    r = int(sys.argv[2])
    cost = []
    angs = []
    for ang in range(r):
        angs.append(ang * 2*np.pi/r)
        qasm = stateprep(ang * 2*np.pi/r)
        qx = qxelarator.QX()
        qx.set(qasm)

        c = np.zeros(2*NUM_QUBIT, dtype=bool)
        p = 0
        for _ in range(t):
            qx.execute()
            for i in range(2*NUM_QUBIT):
                c[i] = qx.get_measurement_outcome(i)
            if sum(c) == 0:
                p += 1
        cost.append(p/t)

    from matplotlib import pyplot as plt
    plt.plot(angs, cost)
    plt.plot([TEST_ANG, TEST_ANG], [0,1])
    plt.show()
"""