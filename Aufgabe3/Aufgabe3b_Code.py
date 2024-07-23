import numpy as np
import matplotlib.pyplot as plt
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

# Zeitparameter
dt = 0.01  # Zeitschritt in ms
t_max = 50.0  # Maximale Zeit in ms
t = np.linspace(0, t_max, int(t_max / dt))  # Zeitvektor


# Funktionen für alpha und beta
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


# Differentialgleichungssystem
def hodgkin_huxley(y, t, I0):
    V, n, m, h = y
    I_Na = gNa * m ** 3 * h * (V - ENa)
    I_K = gK * n ** 4 * (V - EK)
    I_L = gL * (V - EL)

    dVdt = (I0 - I_Na - I_K - I_L) / Cm
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h

    return [dVdt, dndt, dmdt, dhdt]


# Anfangsbedingungen
V0 = -65.0  # Initiales Membranpotential in mV
n0 = 0.3177  # Initialer Wert von n
m0 = 0.0529  # Initialer Wert von m
h0 = 0.5961  # Initialer Wert von h
y0 = [V0, n0, m0, h0]

# Lösung des Systems
solution = odeint(hodgkin_huxley, y0, t, args=(I0,))

# Ergebnisse extrahieren
V_values = solution[:, 0]
n_values = solution[:, 1]
m_values = solution[:, 2]
h_values = solution[:, 3]

#Definieren der Klasse Neuron mit sämtlichen für das neuronale Netzwerk relevanten Funktionen
class Neuron:
    I_in=0 #Input Spannung
    V=-5
    m=0
    n=0
    h=0
    def __init__(self,V,n,m,h):
        self.V=V
        self.n=n
        self.m=m
        self.h=h

    def set_I_in(self,I):
        self.I_in

    def solve(self):
        y = [self.V, self.n, self.m, self.h]
        solution = odeint(hodgkin_huxley, y, t, args=(I0,))
        self.V = solution[:, 0]
        self.n = solution[:, 1]
        self.m = solution[:, 2]
        self.h = solution[:, 3]
    def init_solve(self,y0):
        solution = odeint(hodgkin_huxley, y0, t, args=(I0,))
        self.V = solution[:, 0]
        self.n = solution[:, 1]
        self.m = solution[:, 2]
        self.h = solution[:, 3]


weights =[1,1,1,1]
Network = [] #empty arrayleeres Netzwerk array

for x in range(5):  # appending von insg. 5 Neuronen
    Network.append(Neuron())


