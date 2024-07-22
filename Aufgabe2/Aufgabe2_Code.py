import numpy as np
import matplotlib.pyplot as plt

# Parameter
C_m = 1.0
g_Na = 120.0
g_K = 36.0
g_L = 0.3
V_Na = 115.0
V_K = -12.0
V_L = 10.6
I_0 = 10.0

# Zeitparameter
dt = 0.01
T = 50.0
time = np.arange(0, T + dt, dt)


# Funktionen zur Berechnung der alpha und beta Werte
def alpha_m(V): return 0.1 * (25 - V) / (np.exp((25 - V) / 10) - 1)


def beta_m(V): return 4 * np.exp(-V / 18)


def alpha_h(V): return 0.07 * np.exp(-V / 20)


def beta_h(V): return 1 / (np.exp((30 - V) / 10) + 1)


def alpha_n(V): return 0.01 * (10 - V) / (np.exp((10 - V) / 10) - 1)


def beta_n(V): return 0.125 * np.exp(-V / 80)


# Initialbedingungen
V = -65.0
m = alpha_m(V) / (alpha_m(V) + beta_m(V))
h = alpha_h(V) / (alpha_h(V) + beta_h(V))
n = alpha_n(V) / (alpha_n(V) + beta_n(V))

# Listen zur Speicherung der Ergebnisse
V_values = []
m_values = []
h_values = []
n_values = []

# Simulation im Euler Verfahren
for t in time:
    # Ströme
    I_Na = g_Na * (m ** 3) * h * (V - V_Na)
    I_K = g_K * (n ** 4) * (V - V_K)
    I_L = g_L * (V - V_L)

    # Membranpotential
    dVdt = (I_0 - I_Na - I_K - I_L) / C_m
    V += dt * dVdt

    # Torvariablen
    dm_dt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dh_dt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dn_dt = alpha_n(V) * (1 - n) - beta_n(V) * n

    m += dt * dm_dt
    h += dt * dh_dt
    n += dt * dn_dt

    # Speichern der Ergebnisse
    V_values.append(V)
    m_values.append(m)
    h_values.append(h)
    n_values.append(n)

# Plotten der Ergebnisse
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(time, V_values, label='Membranpotential V (mV)')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(time, m_values, label='m-Wert')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(time, h_values, label='h-Wert')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(time, n_values, label='n-Wert')
plt.legend()

plt.xlabel('Zeit (ms)')
plt.show()
