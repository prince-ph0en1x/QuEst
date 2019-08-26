import numpy as np
from scipy.linalg import expm
from openql import openql as ql
import sys
import os


def check_hermitian(A):
    adjoint = A.conj().T # a.k.a. conjugate-transpose, transjugate, dagger
    assert(np.allclose(A,adjoint))


def gen_wsopp(n_qubits = 1):
    H = np.zeros([2**n_qubits,2**n_qubits])
    
    I = np.array([[1,0],[0,1]])
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,complex(0,-1)],[complex(0,1),0]])
    Z = np.array([[1,0],[0,-1]])
    
    for i in range(4**n_qubits):
        pt = format(i,"0"+str(2*n_qubits)+"b")
        sopp = [1]
        for j in range(0,len(pt),2):
            k = pt[j:j+2]
            if k == '00':
                sopp = np.kron(sopp,I)
            elif k == '01':
                sopp = np.kron(sopp,X)
            elif k == '10':
                sopp = np.kron(sopp,Y)
            else:
                sopp = np.kron(sopp,Z)
        w = np.random.uniform(0,1)
        H = H + w*sopp
    check_hermitian(H)
    return H


def check_unitary(U):
    adjoint = U.conj().T # a.k.a. conjugate-transpose, transjugate, dagger  
    assert(np.allclose(U.dot(adjoint),adjoint.dot(U)))  
    assert(np.allclose(U.dot(adjoint),np.eye(U.shape[0])))
    return 


def gen_unitary(n_qubit = 1):
    H = gen_wsopp(n_qubit)
    U = expm(complex(0,-1)*H)
    check_unitary(U)
    return U


def gen_unitary_array(n_qubit = 1):
    H = gen_wsopp(n_qubit)
    U = expm(complex(0,-1)*H)
    check_unitary(U)
    return U.flatten()


def gen_qasm(unitary, q):
    config_fn = os.path.abspath('test_cfg_none_simple.json')
    platform = ql.Platform('platform_none', config_fn)
    p = ql.Program('RAND_UNITARY', platform, q)
    k = ql.Kernel('QK1', platform, q)

    u = ql.Unitary('u1', unitary)
    u.decompose()
    k.gate(u, [i for i in range(q)])
    p.add_kernel(k)
    p.compile()


def main():
    q = int(sys.argv[1])
    np.set_printoptions(threshold=np.inf)
    arr = gen_unitary_array(q)
    gen_qasm(arr, q)


if __name__ == "__main__":
    main()