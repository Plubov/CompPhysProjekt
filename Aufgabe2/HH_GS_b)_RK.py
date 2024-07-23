import numpy as np
import matplotlib.pyplot as plt

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
num_steps = int(t_max / dt)  # Anzahl der Zeitschritte

# Anfangsbedingungen
V = -65.0  # Initiales Membranpotential in mV
n = 0.3177  # Initialer Wert von n
m = 0.0529  # Initialer Wert von m
h = 0.5961  # Initialer Wert von h


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


# Differentialgleichungen
def dVdt(V, n, m, h, I0):
    I_Na = gNa * m ** 3 * h * (V - ENa)
    I_K = gK * n ** 4 * (V - EK)
    I_L = gL * (V - EL)
    return (I0 - I_Na - I_K - I_L) / Cm


def dndt(V, n):
    return alpha_n(V) * (1 - n) - beta_n(V) * n


def dmdt(V, m):
    return alpha_m(V) * (1 - m) - beta_m(V) * m


def dhdt(V, h):
    return alpha_h(V) * (1 - h) - beta_h(V) * h


# Arrays zur Speicherung der Werte
V_values = np.zeros(num_steps)
n_values = np.zeros(num_steps)
m_values = np.zeros(num_steps)
h_values = np.zeros(num_steps)
t_values = np.linspace(0, t_max, num_steps)

# Initiale Werte
V_values[0] = V
n_values[0] = n
m_values[0] = m
h_values[0] = h

# Runge-Kutta 4. Ordnung
for i in range(1, num_steps):
    # Berechne k1
    k1_V = dt * dVdt(V, n, m, h, I0)
    k1_n = dt * dndt(V, n)
    k1_m = dt * dmdt(V, m)
    k1_h = dt * dhdt(V, h)

    # Berechne k2
    k2_V = dt * dVdt(V + 0.5 * k1_V, n + 0.5 * k1_n, m + 0.5 * k1_m, h + 0.5 * k1_h, I0)
    k2_n = dt * dndt(V + 0.5 * k1_V, n + 0.5 * k1_n)
    k2_m = dt * dmdt(V + 0.5 * k1_V, m + 0.5 * k1_m)
    k2_h = dt * dhdt(V + 0.5 * k1_V, h + 0.5 * k1_h)

    # Berechne k3
    k3_V = dt * dVdt(V + 0.5 * k2_V, n + 0.5 * k2_n, m + 0.5 * k2_m, h + 0.5 * k2_h, I0)
    k3_n = dt * dndt(V + 0.5 * k2_V, n + 0.5 * k2_n)
    k3_m = dt * dmdt(V + 0.5 * k2_V, m + 0.5 * k2_m)
    k3_h = dt * dhdt(V + 0.5 * k2_V, h + 0.5 * k2_h)

    # Berechne k4
    k4_V = dt * dVdt(V + k3_V, n + k3_n, m + k3_m, h + k3_h, I0)
    k4_n = dt * dndt(V + k3_V, n + k3_n)
    k4_m = dt * dmdt(V + k3_V, m + k3_m)
    k4_h = dt * dhdt(V + k3_V, h + k3_h)

    # Aktualisiere die Variablen
    V += (k1_V + 2 * k2_V + 2 * k3_V + k4_V) / 6
    n += (k1_n + 2 * k2_n + 2 * k3_n + k4_n) / 6
    m += (k1_m + 2 * k2_m + 2 * k3_m + k4_m) / 6
    h += (k1_h + 2 * k2_h + 2 * k3_h + k4_h) / 6

    # Werte speichern
    V_values[i] = V
    n_values[i] = n
    m_values[i] = m
    h_values[i] = h

# Ergebnisse plotten
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t_values, V_values, label='Membranpotential (V)')
plt.xlabel('Zeit (ms)')
plt.ylabel('Membranpotential (mV)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_values, n_values, label='n (Kaliumaktivierung)')
plt.plot(t_values, m_values, label='m (Natriumaktivierung)')
plt.plot(t_values, h_values, label='h (Natriuminaktivierung)')
plt.xlabel('Zeit (ms)')
plt.ylabel('Gating-Variablen')
plt.legend()

plt.tight_layout()
plt.show()