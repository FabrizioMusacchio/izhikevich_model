# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
# set global font size for plots:
plt.rcParams.update({'font.size': 12})
# create a folder "figures" to save the plots (if it does not exist):
import os
if not os.path.exists('figures'):
    os.makedirs('figures')
# %% MAIN

# set simulation time and time step size:
T       = 400  # total simulation time in ms
dt      = 0.1  # time step size in ms
steps   = int(T / dt)  # number of simulation steps
t_start = 50  # start time for the input current
t_end   = T  # end time for the input current

# initialize parameters for one excitatory neuron:
p_TEST= [0.08, 0.2, -60, 8, "example"]
p_RS  = [0.02, 0.2, -65, 8, "regular spiking (RS)"] # regular spiking settings for excitatory neurons (RS)
p_IB  = [0.02, 0.2, -55, 4, "intrinsically bursting (IB)"] # intrinsically bursting (IB)
p_CH  = [0.02, 0.2, -51, 2, "chattering (CH)"] # chattering (CH)
p_FS  = [0.1, 0.2, -65, 2, "fast spiking (FS)"] # fast spiking (FS)
p_TC  = [0.02, 0.25, -65, 0.05, "thalamic-cortical (TC)"] # thalamic-cortical (TC) (doesn't work well)
p_LTS = [0.02, 0.25, -65, 2, "low-threshold spiking (LTS)"] # low-threshold spiking (LTS)
p_RZ  = [0.1, 0.26, -65, 2, "resonator (RZ)"] # resonator (RZ)
a, b, c, d, type = p_RS # just change the parameter set here to simulate different neuron types

# initial values of v and u:
v = -65 # mV # -65; -87 for TC 2nd type neurons
u = b * v

# initialize array to store the u, v, I and t values over time:
u_values = np.zeros(steps)
v_values = np.zeros(steps)
I_values = np.zeros(steps)
t_values = np.zeros(steps)

# set the baseline current:
I_baseline = 20 # 10 nA; -10 for TC 2nd type neurons

# simulation:
for t in range(steps):
    t_ms = t * dt  # current time in ms

    if t_ms >= t_start and t_ms <= t_end:
        I = I_baseline # change this to 0 for TC 2nd type neurons
    else:
        I = 0 # change this to I_baseline for TC 2nd type neurons
    # uncomment for an extra input current pulse:
    """ if t_ms >= 150 and t_ms <= 160:
        I = 2*I_baseline """

    # check for spike and reset if v >= 30 mV (reset-condition):
    if v >= 30:
        v = c  # reset membrane potential v to c
        u += d # increase recovery variable u by d

    # Euler's method for numerical integration:
    v += dt * 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
    v += dt * 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
    u += dt * a * (b * v - u)

    # store u, v, and I values:
    u_values[t] = u
    v_values[t] = v
    I_values[t] = I
    t_values[t] = t_ms

# ensure v_values do not exceed 30 mV in the plot:
v_values = np.clip(v_values, None, 30)

# plotting:
fig, ax1 = plt.subplots(figsize=(8,3.85))

# plot v_values on the left y-axis:
ax1.plot(t_values, v_values, label='Membrane potential v(t)', color='k', lw=1.3)
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('membrane potential $v$ [mV]', color='k')
ax1.tick_params(axis='y', colors='k')
ax1.set_yticks(np.arange(-90, 40, 15))
ax1.set_ylim(-85, 35)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

# create a second y-axis for u_values:
ax2 = ax1.twinx()
ax2.plot(t_values, u_values, label='Recovery variable u(t)', color='r', lw=2, alpha=1.0)
ax2.set_ylabel('recovery variable $u$ [a.u.]', color='r')
ax2.tick_params(axis='y', colors='r')
#ax2.set_ylim(min(u_values)*1.1,max(u_values) + np.abs(max(u_values))*1.5)
ax2.set_ylim(-20, 10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

# create a third y-axis for I_values:
ax3 = ax1.twinx()
# Offset the right spine of ax3. The ticks and label have already been placed on the right by twinx above.
ax3.spines['right'].set_position(('outward', 60))  
ax3.plot(t_values, I_values, label='Input Current I(t)', color='b', lw=2, alpha=0.75)
ax3.set_ylabel('input current $I$ [nA]', color='b')
ax3.tick_params(axis='y', colors='b')
ax3.set_ylim(-1,60)
ax3.set_yticks(np.arange(-10, 61, 10))
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)

# To make the border of the right-most axis visible, we need to turn the frame on. This hides the other plots, however, so we need to turn its fill off.
ax3.set_frame_on(True)
ax3.patch.set_visible(False)

plt.title(f'Membrane potential, recovery variable, and input current, {type}\n'
          f"Parameters: a={a}, b={b}, c={c}, d={d}", fontsize=12)
plt.tight_layout()
plt.savefig(f'figures/single_neuron_dynamics_{type}.png', dpi=300)
plt.show()

# %% SPECIAL PLOT FOR POST THUMB
plt.figure(figsize=(4.2, 4.2))

plt.plot(t_values, v_values, label='membrane potential v(t)', color='k', lw=1.3)
plt.title(f'Membrane potential, {type}\n'
          f"Parameters: a={a}, b={b}, c={c}, d={d}", fontsize=12)
plt.xlabel('Time (ms)')
plt.ylabel('membrane potential $v(t)$ [mV]')
plt.xlim(0, 210)
plt.ylim(-80, 40)
plt.yticks(np.arange(-80, 40, 20))
#plt.legend(loc='upper right')
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
""" plt.twinx()
plt.plot(u_values, label='recovery variable u(t)', color='red', lw=1.75)
plt.ylabel('recovery variable u [a.u.]')
#plt.legend(loc='upper left')
# mark the second y-axis labels and ticks in red:
plt.gca().yaxis.label.set_color('red')
plt.gca().tick_params(axis='y', colors='red')
#plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False) """

plt.tight_layout()
plt.savefig(f'figures/single_neuron_dynamics_{type}_thumb.png', dpi=300)
plt.show()
# %% SPECIAL PLOT TO EXPLAIN THE MODEL
plt.figure(figsize=(8, 7))
plt.subplot(2, 1, 1)
plt.plot(t_values, v_values, label='membrane potential v(t)', color='k', lw=2)
plt.title(f'Membrane potential, {type}\n'
          f"Parameters: a={a}, b={b}, c={c}, d={d}", fontsize=12)
#plt.xlabel('Time (ms)')
plt.ylabel('membrane potential $v$ [mV]')
plt.xlim(90,120)
plt.ylim(-80, 40)
plt.yticks(np.arange(-90, 40, 20))
#plt.legend(loc='upper right')
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

plt.subplot(2, 1, 2)
plt.plot(t_values, u_values, label='recovery variable u(t)', color='red', lw=2)
plt.ylabel('recovery variable $u$ [a.u.]')
plt.xlim(90,120)
plt.ylim(min(u_values)*1.1,max(u_values) + np.abs(max(u_values))*1.5)
plt.yticks(np.arange(-10, 20, 10))
plt.xlabel('Time (ms)')
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig(f'figures/single_neuron_dynamics_{type}_explain.png', dpi=300)
plt.show()
# %% PLOT a, b, c, d PARAMETER SPACE
# plot a vs. b and c vs. d parameter space:
plt.figure(figsize=(8, 3.5))
plt.subplot(1, 2, 1)
plt.plot([0.02, 0.1, 0.02, 0.1, 0.1, 0.02, 0.02], [0.2, 0.2, 0.2, 0.2, 0.26, 0.25, 0.25], 'ko', label='Neuron types',
         markersize=8)
dxy = 0.016
plt.text(0.02, 0.2-dxy, 'RS ', fontsize=12, ha='right', va='bottom', c='b')
plt.text(0.02, 0.2-2*dxy, 'CH ', fontsize=12, ha='right', va='bottom', c='r')
plt.text(0.02, 0.2-3*dxy, 'IB ', fontsize=12, ha='right', va='bottom', c='g')
plt.text(0.1, 0.2, 'FS ', fontsize=12, ha='right', va='bottom', c='orange')
plt.text(0.1, 0.26, 'RZ ', fontsize=12, ha='right', va='bottom', c='purple')
plt.text(0.02, 0.25, 'TC ', fontsize=12, ha='right', va='bottom', c='olive')
plt.text(0.02, 0.25+dxy, 'LTS ', fontsize=12, ha='right', va='bottom', c='deeppink')
plt.xticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.1])
plt.yticks([0.0, 0.05, 0.1,0.15, 0.2, 0.25, 0.3, 0.35])
plt.xlim([0, 0.12])
plt.ylim([0.1, 0.30])
plt.xlabel('parameter a\n(recovery time scale)')
plt.ylabel('parameter b\n(recovery sensitivity)')
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.title('Parameter a vs. b')

plt.subplot(1, 2, 2)
dxy = 0.85
plt.plot([-65, -65, -55, -51, -65, -65, -65], [8, 2, 4, 2, 2, 0.05, 2], 'ko', label='Neuron types',
         markersize=8)
plt.text(-65, 8, 'RS ', fontsize=12, ha='right', va='bottom', c='b')
plt.text(-55, 4, 'IB ', fontsize=12, ha='right', va='bottom', c='g')
plt.text(-51, 2, 'CH ', fontsize=12, ha='right', va='bottom', c='r')
plt.text(-65, 0.05, 'TC ', fontsize=12, ha='right', va='bottom', c='olive')
plt.text(-65, 2, 'FS ', fontsize=12, ha='right', va='bottom', c='orange')
plt.text(-65, 2+dxy, 'RZ ', fontsize=12, ha='right', va='bottom', c='purple')
plt.text(-65, 2+2*dxy, 'LTS ', fontsize=12, ha='right', va='bottom', c='deeppink')
plt.xticks([-70, -65, -60, -55, -50, -45])
plt.yticks([0.05, 2, 4, 6, 8, 10])
plt.xlim([-70, -45])
plt.ylim([-1, 10])
plt.xlabel('parameter c\n(after-spike reset value for v)')
plt.ylabel('parameter d\n(after-spike increment value for u)')
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.title('Parameter c vs. d')
plt.tight_layout()
plt.savefig('figures/izhikevich_neuron_types_parameter_space_abcd.png', dpi=300)
plt.show()

# %% WITH RK4 INTEGRATION
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# Set global font size for plots
plt.rcParams.update({'font.size': 12})

# Create a folder "figures" to save the plots (if it does not exist)
if not os.path.exists('figures'):
    os.makedirs('figures')

# Time settings
T = 1000
time = np.linspace(0, T, T + 1)  # Time array for solve_ivp

# Parameters for a neuron
p = [0.02, 0.2, -50, 2]  # Parameters for intrinsically bursting / chattering (IB/CH)
a, b, c, d = p

# Baseline current settings
I_baseline = 10
def I(t):
    return I_baseline if 50 <= t <= 500 else 0

# Initial conditions
v0 = -65
u0 = b * v0

# The system of equations
def model(t, y):
    v, u = y
    dvdt = 0.04 * v**2 + 5 * v + 140 - u + I(t)
    dudt = a * (b * v - u)
    return [dvdt, dudt]

# Event function to handle spikes
def reset(t, y):
    v, u = y
    return v - 30
reset.terminal = True
reset.direction = 1

# Solve the system
sol = solve_ivp(model, [0, T], [v0, u0], t_eval=time, events=reset, method='RK45', rtol=1e-6)

# Resetting mechanism
for t_idx, t in enumerate(sol.t_events[0]):
    if t_idx < len(sol.t_events[0]) - 1:
        sol.y[0, sol.t == t] = c
        sol.y[1, sol.t == t] += d

# Extract results
v_values = sol.y[0]
u_values = sol.y[1]
I_values = np.array([I(t) for t in sol.t])

# Define range for v and calculate nullclines
v_range = np.linspace(-80, 40, 300)
u_v_nullcline = 0.04 * v_range**2 + 5 * v_range + 140 + I_baseline  # v-nullcline
u_u_nullcline = b * v_range  # u-nullcline

# plotting:
plt.figure(figsize=(12, 10))

# membrane Potential v(t):
plt.subplot(2, 2, 1)
plt.plot(sol.t, v_values, label='Membrane Potential v(t)', color='blue')
plt.title('Membrane Potential v(t)')
plt.ylabel('Membrane potential v [mV]')
plt.legend()
plt.grid(True)

# Recovery Variable u(t)
plt.subplot(2, 2, 2)
plt.plot(sol.t, u_values, label='Recovery Variable u(t)', color='blue')
plt.title('Recovery Variable u(t)')
plt.ylabel('Recovery variable u [a.u.]')
plt.legend()
plt.grid(True)

# Input Current I(t)
plt.subplot(2, 2, 3)
plt.plot(sol.t, I_values, label='Input Current I(t)', color='blue')
plt.title('Input Current I(t)')
plt.xlabel('Time (ms)')
plt.ylabel('Input current I [nA]')
plt.legend()
plt.grid(True)

# Phase Space (v vs u)
plt.subplot(2, 2, 4)
plt.plot(v_range, u_v_nullcline, 'r--', label='v-nullcline')
plt.plot(v_range, u_u_nullcline, 'g--', label='u-nullcline')
plt.plot(v_values, u_values, 'b', label='Trajectory (v vs u)')
plt.title('Phase Space (v vs u)')
plt.xlabel('Membrane potential v [mV]')
plt.ylabel('Recovery variable u [a.u.]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('figures/neuron_dynamics_and_phase_space.png', dpi=120)
plt.show()


# %% MANUAL RK4 INTEGRATION
import numpy as np
import matplotlib.pyplot as plt
import os

# Set global font size for plots
plt.rcParams.update({'font.size': 12})

# Create a folder "figures" to save the plots (if it does not exist)
if not os.path.exists('figures'):
    os.makedirs('figures')

# Simulation time
T = 1000  # ms
dt = 1    # time step for RK4

# Initialize parameters for one neuron
p = [0.02, 0.2, -50, 2]  # Parameters for intrinsically bursting / chattering (IB/CH)
a, b, c, d = p

# Initial values of v and u
v = -65
u = b * v

# Initialize array to store the u and v values over time
u_values = np.zeros(T)
v_values = np.zeros(T)
I_values = np.zeros(T)

# Set the baseline current:
I_baseline = 10

# Functions for dv/dt and du/dt
def dv_dt(v, u, I):
    return 0.04 * v**2 + 5 * v + 140 - u + I

def du_dt(v, u, a, b):
    return a * (b * v - u)

# RK4 integration function
def rk4(v, u, I):
    # k1 for v and u
    kv1 = dv_dt(v, u, I)
    ku1 = du_dt(v, u, a, b)

    # k2 for v and u
    kv2 = dv_dt(v + 0.5 * kv1 * dt, u + 0.5 * ku1 * dt, I)
    ku2 = du_dt(v + 0.5 * kv1 * dt, u + 0.5 * ku1 * dt, a, b)

    # k3 for v and u
    kv3 = dv_dt(v + 0.5 * kv2 * dt, u + 0.5 * ku2 * dt, I)
    ku3 = du_dt(v + 0.5 * kv2 * dt, u + 0.5 * ku2 * dt, a, b)

    # k4 for v and u
    kv4 = dv_dt(v + kv3 * dt, u + ku3 * dt, I)
    ku4 = du_dt(v + kv3 * dt, u + ku3 * dt, a, b)

    # Update v and u
    v += (kv1 + 2 * kv2 + 2 * kv3 + kv4) / 6 * dt
    u += (ku1 + 2 * ku2 + 2 * ku3 + ku4) / 6 * dt

    return v, u

# Simulation of 1000 ms
for t in range(T):
    if t >= 50 and t <= 500:
        I = I_baseline 
    else:
        I = 0
    
    if v >= 30:
        v = c
        u += d
        
    v, u = rk4(v, u, I)

    # Store u and v values
    u_values[t] = u
    v_values[t] = v
    I_values[t] = I

# Define range for v and calculate nullclines
v_range = np.linspace(-80, 40, 300)
u_v_nullcline = 0.04 * v_range**2 + 5 * v_range + 140 + I_baseline  # v-nullcline
u_u_nullcline = b * v_range  # u-nullcline

# Plotting
plt.figure(figsize=(12, 10))

# Ensure v_values does not exceed 30 mV in the plot
v_values_clipped = np.clip(v_values, None, 30)

# Plotting v(t), u(t), and I(t) for the neuron
plt.subplot(2, 2, 1)
plt.plot(v_values_clipped, label='Membrane Potential v(t)', color='blue')
plt.title('Membrane Potential v(t)')
plt.ylabel('Membrane potential v [mV]')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(u_values, label='Recovery Variable u(t)', color='blue')
plt.title('Recovery Variable u(t)')
plt.ylabel('Recovery variable u [a.u.]')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(I_values, label='Input Current I(t)', color='blue')
plt.title('Input Current I(t)')
plt.xlabel('Time (ms)')
plt.ylabel('Input current I [nA]')
plt.legend()
plt.grid(True)

# Plotting phase space trajectory and nullclines
plt.subplot(2, 2, 4)
#plt.plot(v_range, u_v_nullcline, 'r--', label='v-nullcline')
#plt.plot(v_range, u_u_nullcline, 'g--', label='u-nullcline')
plt.plot(v_values, u_values, 'b', label='Trajectory (v vs u)')
plt.title('Phase Space (v vs u)')
plt.xlabel('Membrane potential v [mV]')
plt.ylabel('Recovery variable u [a.u.]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('figures/neuron_dynamics_and_phase_space.png', dpi=120)
plt.show()

# %% END