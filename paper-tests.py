import numpy as np
import QAQC
from openql import openql as ql
from qxelarator import qxelarator
import random
import os
import re


qx = qxelarator.QX()
ANGS = [[3.06769736431895, 0.7477555247035739, 1.3072662715069776],
        [2.682767133545208, 1.8847525430423808, 1.6563038858248516],
        [0.2652398866898571, 1.7265402098818627, 2.6843177387028114],
        [2.201364894880603, 2.088726435135867, 1.430856103779598],
        [0.8843785066097644, 1.9552425711757466, 1.6559833230322492],
        [0.3213261447457614, 2.9925130670951336, 2.538931574963998],
        [0.423882567041939, 2.2939343399508285, 0.5781039954848771],
        [1.223233647455988, 2.579667973528569, 2.6101143218533736],
        [1.2931772273504982, 3.0034141020195047, 0.3750240039955793],
        [0.07220784035199893, 2.2965636797206885, 1.1446796822694603],
        [1.3295784437143345, 3.1277235698749606, 1.2636689002422767],
        [0.8839728368130625, 2.261770365450782, 0.5659825788369034],
        [2.693298237073473, 2.0624243745482067, 0.1288021416712205],
        [2.962994700105342, 1.702146570088982, 0.69635049968663],
        [2.0040904940886417, 1.2648817552592428, 1.6660659191816558],
        [0.5389082131049064, 1.5704254631475345, 0.19807325367508638],
        [0.2345459921522579, 0.4370811513796836, 2.5865413533328936],
        [1.6826556643914137, 1.3173181486314314, 1.6731312024479232],
        [1.5983353567997836, 2.7863454793532627, 2.6913232784872494],
        [0.24721704717116094, 1.1760773465777614, 0.07899934860472677],
        [0.7595671023345362, 0.2545616012583241, 2.53437050358266],
        [2.2674987202168357, 2.5155203641440482, 2.7438983791914477],
        [2.6785709858756754, 2.587592569760561, 2.819134397067544],
        [2.181738812232448, 1.0209011607050087, 2.9455755428535997],
        [1.514127565443768, 1.5570946849743008, 2.297356851933116],
        [0.003983650741355108, 0.4092234237745684, 1.7821588491052986],
        [0.275698288323815, 2.107520644098052, 0.17928615703060108],
        [1.0524032129958116, 0.017057921357649504, 1.1615044373644752],
        [0.1459469385137765, 1.0873418204580845, 1.9668636235805912],
        [0.17274893382644368, 0.40891353989431894, 0.5776289784945714],
        [0.06155410951103312, 1.7488682741985377, 1.88353173204789],
        [0.595444092898325, 0.7861458585734613, 0.8425745568925239],
        [1.571066659379299, 2.6460009518650027, 1.0572229887988227],
        [2.8641023584038114, 1.5328129884845616, 2.7148831568627148],
        [2.5575486075658262, 0.1809193976681702, 0.007390597390199086]]


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


def generate_random_angles(num_tests):
    for _ in range(num_tests):
        ANGS.append( [random.random() * np.pi,
                      random.random() * np.pi,
                      random.random() * np.pi] )
    from pprint import pprint
    pprint(ANGS)


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

    # If more than 1 block, add additional parameterized gates to each qubit
    for i in range(blocks - 1):
        for q in range(2):
            k1.prepz(q)
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


def test1(blocks, test, runs=75):
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
                a = calculate_concurrence(qx.get_state())
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
    n = 2
    #generate_random_angles(n)
    
    res = []
    for b in range(1, 5):
        trial = []
        for i in range(n):
            tmp = test1(b, i, runs=50)
            trial.append(tmp[1])
        res.append(trial)

    #print("===TEST " + str(i+1) + "===")
    #print(ANGS[i])
    #print("Blocks:", b)
    #print(res[-1])
    #print()

    from matplotlib import pyplot as plt
    x = list(range(n))
    for i in range(len(res)):
        plt.plot(x, res[i], label=str(i+1))
    plt.xlabel("Trial")
    plt.ylabel("Concurrence")
    plt.legend()
    plt.show()

    #for i in range(1,5):
    #    print("===TEST " + str(i) + "===")
    #    print(ANGS[1])
    #    print("Blocks:", i)
    #    print(test1(i, 1, runs=500))
    #    print()