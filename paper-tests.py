import numpy as np
import QAQC
from openql import openql as ql
from qxelarator import qxelarator
import random
import os
import re
import sys


qx = qxelarator.QX()
ANGS = [[3.109399932971677, 0.345471957980682, 0.31272080275542635, 0.18978476724361176, 2.097272651795978, 1.4051735707922381],
        [1.3245824349763722, 1.80417916320562, 1.184039201159872, 0.7153727045843986, 2.3212471008715374, 2.6770194166945442],
        [1.9006434726091161, 1.693522470182537, 0.30933673666193523, 0.20137250433142503, 1.815250521361418, 1.49798666373344],
        [0.16420468291958343, 0.9983505795916082, 0.5874255451911918, 2.632246859049948, 2.151572816031842, 1.385906481127703],
        [2.2850751841217867, 1.7043843480263543, 1.882850975055315, 1.8202620734186126, 2.8280539150932418, 2.610701271958208]]


def calculate_concurrence(s):
    nums = re.findall(r'[-+.e\d]+,[-+.e\d]+', s) # Regex to find all complex numbers in state string
    bras = re.findall(r'\|[01]+>', s) # Regex to find all bra-kets in state string
    comp = []
    ind  = []

    # Convert all complex number strings to complex objects
    for n in nums:
        t = n.split(',')
        comp.append( complex(float(t[0]), float(t[1])) )

    # Convert all bra-kets to indices
    for b in bras:
        t = b.strip('|').strip('>')
        ind.append( int(t, base=2) )

    # Add complex numbers to an ordered list
    C = [0, 0, 0, 0]
    for i, c in enumerate(comp):
        j = ind[i]
        C[j] = c

    # Calculate concurrence value and return
    a = 2 * abs(C[0]*C[3] - C[2]*C[1])
    return a


def parse_get_state(s):
    nums = re.findall(r'[-+.e\d]+,[-+.e\d]+', s) # Regex to find all complex numbers in state string
    bras = re.findall(r'\|[01]+>', s) # Regex to find all bra-kets in state string
    comp = []
    ind  = []

    # Convert all complex number strings to complex objects
    for n in nums:
        t = n.split(',')
        comp.append( complex(float(t[0]), float(t[1])) )

    # Convert all bra-kets to indices
    for b in bras:
        t = b.strip('|').strip('>')
        ind.append( int(t, base=2) )

    # Add complex numbers to an ordered list
    C = [0, 0, 0, 0]
    for i, c in enumerate(comp):
        j = ind[i]
        C[j] = c

    return C


def test1_circuit(blocks, test):
    config_fn = os.path.abspath('/home/neil/dev/tud/OpenQL/tests/test_cfg_none_simple.json')
    platform = ql.Platform('platform_none', config_fn)
    prog = ql.Program('test_circuit', platform, 2)
    k1 = ql.Kernel('QK1',platform, 2)

    # First block will always have these gates
    k1.rz(0, ANGS[test][0])
    k1.rx(0, ANGS[test][1])
    k1.rz(0, ANGS[test][2])
    k1.prepz(1)
    k1.cnot(0, 1)
    k1.display()

    # If more than 1 block, add additional parameterized gates to each qubit
    for i in range(blocks - 1):
        for q in range(2):
            k1.prepz(q)
        k1.cnot(0, 1)
        k1.display()

    prog.add_kernel(k1)
    tmp = prog.qasm()

    # Add stubs indicating where to place parameters in the QASM
    b = 0
    while tmp.find("prep_z") != -1:
        ind = tmp.find("prep_z")
        q = tmp[ind + 9]
        repl =     "rz q[" + q + "], %" + str(b)   + "%\n" +\
               "    rx q[" + q + "], %" + str(b+1) + "%\n" +\
               "    rz q[" + q + "], %" + str(b+2) + "%\n"
        srch = "prep_z q[" + q + "]\n"
        tmp = tmp.replace(srch, repl, 1)
        b += 3

    return tmp


def test2_circuit1(params):
    config_fn = os.path.abspath('/home/neil/dev/tud/OpenQL/tests/test_cfg_none_simple.json')
    platform = ql.Platform('platform_none', config_fn)
    prog = ql.Program('test_circuit', platform, 2)
    k1 = ql.Kernel('QK1',platform, 2)

    k1.rz(0, params[0])
    k1.rx(0, params[1])
    k1.rz(0, params[2])
    k1.rz(1, params[3])
    k1.rx(1, params[4])
    k1.rz(1, params[5])
    k1.cnot(0, 1)

    prog.add_kernel(k1)
    tmp = prog.qasm()
    with open("test_output/TMP.qasm", "w") as f:
        f.write(tmp)

    return "test_output/TMP.qasm"


def test2_circuit2(blocks):
    config_fn = os.path.abspath('/home/neil/dev/tud/OpenQL/tests/test_cfg_none_simple.json')
    platform = ql.Platform('platform_none', config_fn)
    prog = ql.Program('test_circuit', platform, 2)
    k1 = ql.Kernel('QK1',platform, 2)

    for _ in range(blocks):
        k1.prepz(0)
        k1.prepz(1)
        k1.cnot(0, 1)

    prog.add_kernel(k1)
    tmp = prog.qasm()

    # Add stubs indicating where to place parameters in the QASM
    b = 0
    while tmp.find("prep_z") != -1:
        ind = tmp.find("prep_z")
        q = tmp[ind + 9]
        repl =     "rz q[" + q + "], %" + str(b)   + "%\n" +\
               "    rx q[" + q + "], %" + str(b+1) + "%\n" +\
               "    rz q[" + q + "], %" + str(b+2) + "%\n"
        srch = "prep_z q[" + q + "]\n"
        tmp = tmp.replace(srch, repl, 1)
        b += 3

    return tmp


def test2(blocks, params, runs=150):
    global qx
    #base = test2_circuit1(params)
    #qx.set(base)
    #qx.execute()
    #ref = parse_get_state(qx.get_state())
    ref = np.sqrt([1/2, 0, 0, 1/2])
    
    circ = test2_circuit2(blocks)
    inc = [0, -np.pi/4, -np.pi/8, np.pi/8, np.pi/4]
    min_cost = 100
    init_cost = 100
    min_x = []
    flag = True
    
    for _ in range(runs):
        x0 = [random.randint(0, 8) * np.pi/8 for _ in range(6*blocks)] #0
        order = list(range(len(x0)))
        random.shuffle(order)
        for i in order:
            c = []
            for j in inc:
                x = list(x0)
                x[i] += j
                
                qasm = QAQC.update_qasm_parameters(x, circ, qasm='test_output/tst.qasm')
                qx.set(qasm)
                qx.execute()
                state = parse_get_state(qx.get_state())

                a = 0
                for z in range(4):
                    a += abs( ref[z]**2 - state[z]**2 )
                if flag:
                    init_cost = a
                    flag = False
                c.append(a)

                if a < min_cost:
                    min_cost = a
                    min_x = x

            ind = c.index(min(c))
            x0[i] += inc[ind]
            if min_cost == 0:
                break
        if min_cost == 0:
            break

    print([round(i, 3) for i in min_x])
    return [init_cost, min_cost]


def test(blocks, test, runs=75):
    global qx
    circ = test1_circuit(blocks, test)
    x0 = [0 for _ in range(3 + 6*(blocks-1))]
    
    inc = [0, -np.pi/4, -np.pi/8, np.pi/8, np.pi/4]
    max_cost = 0
    init_cost = 0
    max_x = []
    order = list(range(len(x0)))
    flag = True

    for _ in range(runs):
        random.shuffle(order)
        for i in order:
            c = []
            for j in inc:
                x = list(x0)
                x[i] += j
                
                qasm = QAQC.update_qasm_parameters(x, circ, qasm='test_output/tst.qasm')
                qx.set(qasm)
                qx.execute()
                
                import pdb; pdb.set_trace()
                
                a = cost(qx.get_state())
                if flag:
                    init_cost = a
                    flag = False
                c.append(a)

                if a > max_cost:
                    max_cost = a
                    max_x = list(x)

            ind = c.index(max(c))
            x0[i] += inc[ind]
            if max_cost == 1:
                break
        if max_cost == 1:
            break

    return [init_cost, max_cost]


if __name__ == "__main__":
    t = 0
    b = 2

    print(test2(b, ANGS[t], runs=100))

    #res = []
    #for b in range(1, 5):
    #    trial = []
    #    for i in range(n):
    #        tmp = test(b, i, runs=50)
    #        trial.append(tmp[1])
    #    res.append(trial)

    #print("===TEST " + str(i+1) + "===")
    #print(ANGS[i])
    #print("Blocks:", b)
    #print(res[-1])
    #print()

    #from matplotlib import pyplot as plt
    #x = list(range(n))
    #for i in range(len(res)):
    #    plt.plot(x, res[i], label=str(i+1))
    #plt.xlabel("Trial")
    #plt.ylabel("Concurrence")
    #plt.legend()
    #plt.show()

    #for i in range(1,5):
    #    print("===TEST " + str(i) + "===")
    #    print(ANGS[1])
    #    print("Blocks:", i)
    #    print(test1(i, 1, runs=500))
    #    print()
