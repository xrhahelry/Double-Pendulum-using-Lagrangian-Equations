import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

g = 9.81
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0


def derivs(state, t):
    theta1, omega1, theta2, omega2 = state

    dtheta1dt = omega1
    domega1dt = (
        -g * (2 * m1 + m2) * np.sin(theta1)
        - m2 * g * np.sin(theta1 - 2 * theta2)
        - 2
        * np.sin(theta1 - theta2)
        * m2
        * (l2 * omega2**2 + l1 * omega1**2 * np.cos(theta1 - theta2))
    ) / (l1 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))

    dtheta2dt = omega2
    domega2dt = (
        2
        * np.sin(theta1 - theta2)
        * (
            l1 * (m1 + m2) * omega1**2
            + g * (m1 + m2) * np.cos(theta1)
            + l2 * m2 * omega2**2 * np.cos(theta1 - theta2)
        )
    ) / (l2 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))

    return [dtheta1dt, domega1dt, dtheta2dt, domega2dt]


theta1_init = np.pi / 2.0
omega1_init = 0.0
theta2_init = np.pi / 2.0
omega2_init = 0.0
state_init = [theta1_init, omega1_init, theta2_init, omega2_init]

t = np.linspace(0, 40, 2000)

state = odeint(derivs, state_init, t)
theta1 = state[:, 0]
omega1 = state[:, 1]
theta2 = state[:, 2]
omega2 = state[:, 3]

x1 = l1 * np.sin(theta1)
y1 = -l1 * np.cos(theta1)
x2 = x1 + l2 * np.sin(theta2)
y2 = y1 - l2 * np.cos(theta2)

fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 2])

ax_anim = fig.add_subplot(gs[:, 0])
ax_anim.set_aspect("equal", "box")
ax_anim.set_xlim(-2, 2)
ax_anim.set_ylim(-2, 2)
ax_anim.set_xlabel("x")
ax_anim.set_ylabel("y")
ax_anim.set_title("Double Pendulum Motion")

ax_energy1 = fig.add_subplot(gs[0, 1])
ax_energy1.set_xlim(0, t[-1])
ax_energy1.set_ylim(0, 15)
ax_energy1.set_xlabel("Time")
ax_energy1.set_ylabel("Energy")
ax_energy1.set_title("Bob 1 Energies")

ax_energy2 = fig.add_subplot(gs[1, 1])
ax_energy2.set_xlim(0, t[-1])
ax_energy2.set_ylim(0, 35)
ax_energy2.set_xlabel("Time")
ax_energy2.set_ylabel("Energy")
ax_energy2.set_title("Bob 2 Energies")

(line1,) = ax_anim.plot([], [], "k-", lw=2)
(line2,) = ax_anim.plot([], [], "k-", lw=2)
(bob1,) = ax_anim.plot([], [], "bo", markersize=10)
(bob2,) = ax_anim.plot([], [], "ro", markersize=10)
(trajectory,) = ax_anim.plot([], [], "k-", alpha=0.5)

(line_T1,) = ax_energy1.plot([], [], label="Kinetic 1", color="blue")
(line_U1,) = ax_energy1.plot([], [], label="Potential 1", color="orange")
(line_T2,) = ax_energy2.plot([], [], label="Kinetic 2", color="green")
(line_U2,) = ax_energy2.plot([], [], label="Potential 2", color="red")

ax_energy1.legend(loc="upper right")
ax_energy2.legend(loc="upper right")

T1_values = []
U1_values = []
T2_values = []
U2_values = []
time_values = []


def calculate_energies(theta1, omega1, theta2, omega2):
    T1 = 0.5 * m1 * (l1 * omega1) ** 2
    U1 = m1 * g * l1 * (1 - np.cos(theta1))
    T2 = (
        0.5
        * m2
        * (
            (l1 * omega1 * np.cos(theta1) + l2 * omega2 * np.cos(theta2)) ** 2
            + (l1 * omega1 * np.sin(theta1) + l2 * omega2 * np.sin(theta2)) ** 2
        )
    )
    U2 = m2 * g * (l1 * (1 - np.cos(theta1)) + l2 * (1 - np.cos(theta2)))
    return T1, U1, T2, U2


def update(i):
    line1.set_data([0, x1[i]], [0, y1[i]])
    bob1.set_data([x1[i]], [y1[i]])
    line2.set_data([x1[i], x2[i]], [y1[i], y2[i]])
    bob2.set_data([x2[i]], [y2[i]])
    trajectory.set_data(x2[: i + 1], y2[: i + 1])

    T1, U1, T2, U2 = calculate_energies(theta1[i], omega1[i], theta2[i], omega2[i])

    T1_values.append(T1)
    U1_values.append(U1)
    T2_values.append(T2)
    U2_values.append(U2)
    time_values.append(t[i])

    max_len = 1000
    if len(T1_values) > max_len:
        T1_values.pop(0)
        U1_values.pop(0)
        T2_values.pop(0)
        U2_values.pop(0)
        time_values.pop(0)

    line_T1.set_data(time_values, T1_values)
    line_U1.set_data(time_values, U1_values)
    line_T2.set_data(time_values, T2_values)
    line_U2.set_data(time_values, U2_values)

    ax_energy1.set_xlim(max(0, t[i] - 10), t[i] + 1)
    ax_energy2.set_xlim(max(0, t[i] - 10), t[i] + 1)

    return line1, bob1, line2, bob2, trajectory, line_T1, line_U1, line_T2, line_U2


ani = animation.FuncAnimation(fig, update, frames=len(t), interval=20, blit=True)

plt.tight_layout()
plt.show()
