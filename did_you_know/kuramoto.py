import numpy as np

# Set the parameters of the model
omegas = [1, 2, 3, 4]  # Natural frequencies of the oscillators
k = 0.5  # Coupling strength

# Initialize the phase of each oscillator
thetas = [0, 0, 0, 0]

# Set the time step for the simulation
dt = 0.01

# Set the total duration of the simulation
T = 100

# Loop through time steps and update the phase of each oscillator
for t in range(T):
  for i in range(4):
    thetas[i] += omegas[i] * dt + k * np.sum([np.sin(thetas[j] - thetas[i]) for j in range(4) if j != i]) * dt

# Print the final phase of each oscillator
print(thetas)
