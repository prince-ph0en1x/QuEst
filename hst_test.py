import numpy as np
import QAQC
import random


def test2(qasm, locality, blocks):
    c, x = QAQC.optimize(qasm, locality, blocks, runs=3, trials=200)
    print('Cost:', c)
    print('Params: pi *', list(np.array(x) / np.pi))

    QAQC.generate_output_qasm(x, locality, blocks)


def test3(qasm, locality, blocks):
    U = QAQC.read_input_circuit(qasm)
    pre_hst = QAQC.pre_hst_qasm(U)
    post_hst = QAQC.post_hst_qasm()

    x0 = [0 for _ in range(3*QAQC.NUM_QUBIT*blocks)]
    inc = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
    min_cost = 1
    min_x0 = []
    selected_costs = []
    orders = []
    order = list(range(len(x0)))

    for z in range(3):
        random.shuffle(order)
        orders.append(list(order))
        for i in order:
            c = []
            for j in inc:
                x = list(x0)
                x[i] += j
                a = QAQC.hst_cost(x, pre_hst, post_hst, locality, blocks, trials=150)
                c.append(a)
                if a < min_cost:
                    min_cost = a
                    min_x0 = list(x)
            ind = c.index(min(c))
            selected_costs.append(min(c))
            x0[i] += inc[ind]
            if min_cost == 0:
                break
        if min_cost == 0:
            break

    final = QAQC.hst_cost(min_x0, pre_hst, post_hst, locality, blocks, trials=1000)
    QAQC.generate_output_qasm(min_x0, locality, blocks)

    print("Final x0: ", x0)
    print("Min x0:", min_x0)
    print(final)
    print(min_cost)
    print(orders)

    #from matplotlib import pyplot as plt
    #plt.plot(list(range(len(selected_costs))), selected_costs)
    #plt.show()

    return final


def test4(qasm, locality, blocks):
    U = QAQC.read_input_circuit(qasm)
    pre_hst = QAQC.pre_hst_qasm(U)
    post_hst = QAQC.post_hst_qasm()
    x0 = [0 for _ in range(3*QAQC.NUM_QUBIT*blocks)]
    grad = []

    for i in range(len(x0)):
        l = []
        for j in (np.pi/2, -np.pi/2):
            x = list(x0)
            x[i] += j
            l.append(QAQC.hst_cost(x, pre_hst, post_hst, locality, blocks, trials=1000))
        grad.append(l)

    print(np.round(grad, 3))


def test5(qasm, locality, blocks):
    U = QAQC.read_input_circuit(qasm)
    pre_hst = QAQC.pre_hst_qasm(U)
    post_hst = QAQC.post_hst_qasm()
    x0 = [0 for _ in range(3*QAQC.NUM_QUBIT*blocks)]
    inc = [-np.pi/2, -np.pi/4, np.pi/4, np.pi/2]

    for z in range(10):
        min_cost = QAQC.hst_cost(x0, pre_hst, post_hst, locality, blocks, trials=1000)
        print(z, ':', min_cost)
        for i in range(len(x0)):
            for j in inc:
                x = list(x0)
                x[i] += j
                a = QAQC.hst_cost(x, pre_hst, post_hst, locality, blocks, trials=100)
                if a < min_cost or random.random() < min_cost/100:
                    min_cost = a
                    min_bit  = i
                    min_val  = j
                    print(min_cost)
        x0[min_bit] += min_val

    final = QAQC.hst_cost(x0, pre_hst, post_hst, locality, blocks, trials=1000)
    print("Final x0: ", x0)
    print(final)
    print(min_cost)

    #y = list(zip(*costs))
    #x = list(range(len(y[0])))
    #diff = [max(baseline - np.array(i)) for i in costs]
    #ind = sorted(range(len(diff)), key=lambda k: diff[k], reverse=True)
    #print(ind)

    #for i in costs:
    #    print(i)

    #from matplotlib import pyplot as plt
    #for y_i in y:
    #    plt.scatter(x, y_i)
    #plt.plot(x, diff)
    #plt.plot([x[0], x[-1]], [baseline, baseline])
    #plt.show()


if __name__ == '__main__':
    import sys

    #orig_stdout = sys.stdout
    #f = open('out', 'w')
    #sys.stdout = f

    test2("test_output/algo.qasm", 2, 4)

    #sys.stdout = orig_stdout
    #f.close()