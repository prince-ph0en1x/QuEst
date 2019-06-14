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
    U = QAQC.read_input_circuit(qasm)
    pre_hst = QAQC.pre_hst_qasm(U)
    post_hst = QAQC.post_hst_qasm()
    blocks = 2
    locality = 2

    x0 = [0 for _ in range(3*QAQC.NUM_QUBIT*blocks)]
    res = minimize(QAQC.hst_cost, x0, args=(pre_hst, post_hst, locality, blocks), method='Powell', tol=1e-6, options={'disp':True, 'return_all':True})
    print('Params: ', res.x)


if __name__ == '__main__':
    test1('test_output/algo.qasm')


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