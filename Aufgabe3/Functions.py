import numpy as np
from scipy.integrate import odeint
# Parameter
Cm = 1.0  # Membrankapazität in uF/cm^2
gK = 36.0  # Maximale Leitfähigkeit für Kalium in mS/cm^2
gNa = 120.0  # Maximale Leitfähigkeit für Natrium in mS/cm^2
gL = 0.3  # Maximale Leitfähigkeit für Leckstrom in mS/cm^2
EK = -77.0  # Umkehrpotential für Kalium in mV
ENa = 50.0  # Umkehrpotential für Natrium in mV
EL = -54.387  # Umkehrpotential für Leckstrom in mV
I0 = 10.0  # Konstanter angelegter Strom in uA/cm^2


# Funktionen für alpha und beta zur Verwendung in DGLs
def alpha_n(V):
    return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))


def beta_n(V):
    return 0.125 * np.exp(-(V + 65) / 80)


def alpha_m(V):
    return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))


def beta_m(V):
    return 4.0 * np.exp(-(V + 65) / 18)


def alpha_h(V):
    return 0.07 * np.exp(-(V + 65) / 20)


def beta_h(V):
    return 1 / (1 + np.exp(-(V + 35) / 10))


# Differentialgleichungssystem für konstanten Input Strom (Verwendnug in Output Neuron)
def hodgkin_huxley(y, t, I0):   # (Parameterstartwerte, Zeit, konst. äußerer Strom)
    V, n, m, h = y
    I_Na = gNa * m ** 3 * h * (V - ENa)
    I_K = gK * n ** 4 * (V - EK)
    I_L = gL * (V - EL)

    dVdt = (I0 - I_Na - I_K - I_L) / Cm
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h

    return [dVdt, dndt, dmdt, dhdt]


def hodgkin_huxley_dynamic(y, t,Activated):  # DGL System, welches Stromimpulse erlaubt, wenn Neuron aktiviert
    V, n, m, h = y
    if Activated:
        I = 15
    else:
        I=-5

    I_Na = gNa * m ** 3 * h * (V - ENa)
    I_K = gK * n ** 4 * (V - EK)
    I_L = gL * (V - EL)

    dVdt = (I - I_Na - I_K - I_L) / Cm
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h

    return [dVdt, dndt, dmdt, dhdt]



