"""Generated exact Boltzmann entropy integrands; heat generator 1/2 dxx.
Pass a dictionary u[j] of log-density derivatives (NumPy/JAX/Torch arrays).
Shannon entropy derivatives have the opposite sign.
"""

def integrand_1(u):
    u1 = u[1]
    return -1/2*u1**2

def integrand_2(u):
    u2 = u[2]
    return (1/2)*u2**2

def integrand_3(u):
    u2 = u[2]
    u3 = u[3]
    return u2**3 - 1/2*u3**2

def integrand_4(u):
    u2 = u[2]
    u3 = u[3]
    u4 = u[4]
    return 3*u2**4 - 6*u2*u3**2 + (1/2)*u4**2

def integrand_5(u):
    u2 = u[2]
    u3 = u[3]
    u4 = u[4]
    u5 = u[5]
    return 12*u2**5 - 60*u2**2*u3**2 + 10*u2*u4**2 + 15*u3**2*u4 - 1/2*u5**2

def integrand_6(u):
    u2 = u[2]
    u3 = u[3]
    u4 = u[4]
    u5 = u[5]
    u6 = u[6]
    return 60*u2**6 - 600*u2**3*u3**2 + 150*u2**2*u4**2 + 450*u2*u3**2*u4 - 15*u2*u5**2 + 105*u3**4 - 60*u3*u4*u5 - 15*u4**3 + (1/2)*u6**2

def integrand_7(u):
    u2 = u[2]
    u3 = u[3]
    u4 = u[4]
    u5 = u[5]
    u6 = u[6]
    u7 = u[7]
    return 360*u2**7 - 6300*u2**4*u3**2 + 2100*u2**3*u4**2 + 9450*u2**2*u3**2*u4 - 315*u2**2*u5**2 + 4410*u2*u3**4 - 2520*u2*u3*u4*u5 - 630*u2*u4**3 + 21*u2*u6**2 - 630*u3**3*u5 - 2415*u3**2*u4**2 + 105*u3*u5*u6 + 70*u4**2*u6 + 105*u4*u5**2 - 1/2*u7**2

def integrand_8(u):
    u2 = u[2]
    u3 = u[3]
    u4 = u[4]
    u5 = u[5]
    u6 = u[6]
    u7 = u[7]
    u8 = u[8]
    return 2520*u2**8 - 70560*u2**5*u3**2 + 29400*u2**4*u4**2 + 176400*u2**3*u3**2*u4 - 5880*u2**3*u5**2 + 123480*u2**2*u3**4 - 70560*u2**2*u3*u4*u5 - 17640*u2**2*u4**3 + 588*u2**2*u6**2 - 35280*u2*u3**3*u5 - 135240*u2*u3**2*u4**2 + 5880*u2*u3*u5*u6 + 3920*u2*u4**2*u6 + 5880*u2*u4*u5**2 - 28*u2*u7**2 - 47880*u3**4*u4 + 5880*u3**2*u4*u6 + 7140*u3**2*u5**2 + 20160*u3*u4**2*u5 - 168*u3*u6*u7 + 3115*u4**4 - 280*u4*u5*u7 - 210*u4*u6**2 - 280*u5**2*u6 + (1/2)*u8**2

INTEGRANDS = {1: integrand_1, 2: integrand_2, 3: integrand_3, 4: integrand_4, 5: integrand_5, 6: integrand_6, 7: integrand_7, 8: integrand_8}
