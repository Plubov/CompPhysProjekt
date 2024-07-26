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
I0 = -5.0  # Konstanter angelegter Strom in uA/cm^2

# Zeitparameter
dt = 0.001  # Zeitschritt in ms
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


# Funktion für den zeitlich variablen Strom
def applied_current(t):
    if 10 <= t <= 11:
        return 50.0  # Stromimpuls von 50 nA/cm^2
    return I0


# Differentialgleichungssystem
def hodgkin_huxley(y, t):
    V, n, m, h = y
    I = applied_current(t)
    I_Na = gNa * m ** 3 * h * (V - ENa)
    I_K = gK * n ** 4 * (V - EK)
    I_L = gL * (V - EL)

    dVdt = (I - I_Na - I_K - I_L) / Cm
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
solution = odeint(hodgkin_huxley, y0, t)

# Ergebnisse extrahieren
V_values = solution[:, 0]
n_values = solution[:, 1]
m_values = solution[:, 2]
h_values = solution[:, 3]

# Ergebnisse plotten
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, V_values, label='Membranpotential (V)')
plt.xlabel('Zeit (ms)')
plt.ylabel('Membranpotential (mV)')
plt.title('Hodgkin-Huxley Modell (odeint)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, n_values, label='n (Kaliumaktivierung)')
plt.plot(t, m_values, label='m (Natriumaktivierung)')
plt.plot(t, h_values, label='h (Natriuminaktivierung)')
plt.xlabel('Zeit (ms)')
plt.ylabel('Gating-Variablen')
plt.legend()

plt.tight_layout()
plt.show()