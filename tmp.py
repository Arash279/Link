import numpy as np
g=[1,0.7654,1.8478,1.8478,0.7654,1]
Z0=50
BW=0.1
N=4
def calculate_J(i_list):
    J_values = []
    for i in i_list:
        if i == 1:
            J = 1 / Z0 * ( np.sqrt(np.pi * BW / (2 * g[1])))
        elif i == N+1:
            J = 1 / Z0 * ( np.sqrt(np.pi * BW / (2 * g[N]* g[N+1])))
        else:
            J = 1 / Z0 * ( np.pi * BW / (2 * np.sqrt(g[i-1] * g[i])) )
        J_values.append(J)
    return np.array(J_values)

def calculate_Zo(J_values):
    Zo_values = []
    for J in J_values:
        Zo=Z0 * (1 - Z0*J + (Z0*J)**2)
        Zo_values.append(Zo)
    return np.array(Zo_values)

def calculate_Ze(J_values):
    Ze_values = []
    for J in J_values:
        Ze=Z0 * (1 + Z0*J + (Z0*J)**2)
        Ze_values.append(Ze)
    return np.array(Ze_values)

i=[1,2,3,4,5]
J_values=calculate_J(i)
Zo_values=calculate_Zo(J_values)
Ze_values=calculate_Ze(J_values)
print(J_values)
print(Zo_values)
print(Ze_values)
