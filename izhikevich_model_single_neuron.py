"""
A single neuron model based on the Izhikevich model (2003).

author: Fabrizio Musacchio
date: Apr 20, 2024
"""
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
I_baseline = 10 # 10 nA; -10 for TC 2nd type neurons

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
ax3.set_yticks(np.arange(-10, 61, 10))
ax3.set_ylim(-11,60)
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
# %% END