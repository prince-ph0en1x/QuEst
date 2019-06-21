import numpy as np
import QAQC
from scipy.optimize import minimize


def test1(qasm):
    from itertools import product

    U = QAQC.read_input_circuit(qasm)
    pre_hst = QAQC.pre_hst_qasm(U)
    post_hst = QAQC.post_hst_qasm()
    locality = 2

    from random import random
    costs = []
    for blocks in range(1, 5):
        d = []
        min_d = 1
        #min_x = 0
        for _ in range(500):
            x0 = [random()*np.pi for _ in range(3*QAQC.NUM_QUBIT*blocks)]
            d.append(QAQC.hst_cost(x0, pre_hst, post_hst, locality, blocks))
            if d[-1] < min_d:
                #min_x = x0
                min_d = d[-1]
        #res = minimize(QAQC.hst_cost, min_x, args=(pre_hst, post_hst, locality, blocks), method='Powell', tol=1e-10)
        costs.append(min_d)
    print(costs)


def test2(qasm):
    res = QAQC.optimize(qasm, 1, 1, tol=1e-5, disp=True)
    print('Params: ', res.x)


def test3(qasm):
    U = QAQC.read_input_circuit(qasm)
    pre_hst = QAQC.pre_hst_qasm(U)
    post_hst = QAQC.post_hst_qasm()
    locality = 3
    blocks = 2
    x0 = [0 for _ in range(3*QAQC.NUM_QUBIT*blocks)]
    inc = (-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2)
    min_cost = 1
    
    for _ in range(3):
        costs = []
        for i in range(len(x0)):
            c = []
            for j in inc:
                x = list(x0)
                x[i] += j
                a = QAQC.hst_cost(x, pre_hst, post_hst, locality, blocks, trials=1000)
                c.append(a)
                if a < min_cost:
                    min_cost = a
            ind = c.index(min(c))
            x0[i] += inc[ind]
            costs.append(c)
            if min_cost == 0:
                break
        if min_cost == 0:
            break

    for c in costs:
        print(np.round(c, 3))
    print()
    print(x0)
    print(min_cost)


def test4(qasm):
    U = QAQC.read_input_circuit(qasm)
    pre_hst = QAQC.pre_hst_qasm(U)
    post_hst = QAQC.post_hst_qasm()
    locality = 2
    blocks = 2
    x0 = [0 for _ in range(3*QAQC.NUM_QUBIT*blocks)]
    grad = []

    for i in range(len(x0)):
        l = []
        for j in (np.pi/2, -np.pi/2):
            x = list(x0)
            x[i] += j
            l.append(QAQC.hst_cost(x, pre_hst, post_hst, locality, blocks, trials=500))
        grad.append(l[0]/2 - l[1]/2)

    print(np.round(grad, 3))


if __name__ == '__main__':
    test3("test_output/algo.qasm")


"""
def test3():
    U = QAQC.read_input_circuit('test_output/algo.qasm')
    pre_hst = QAQC.pre_hst_qasm(U)
    post_hst = QAQC.post_hst_qasm()

    res = 15
    blocks = 2
    locality = 2

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
            costs.append(QAQC.hst_cost([ang1, ang2], pre_hst, post_hst, locality, blocks))
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