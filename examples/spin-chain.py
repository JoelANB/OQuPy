# Copyright 2020 The TEMPO Collaboration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example of a 5-site XYZ Heisenberg spin chain.
"""

import os
import sys
sys.path.insert(0,'.')
from time import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt

import oqupy
import dill as pickle

# -----------------------------------------------------------------------------

sx = oqupy.operators.sigma("x")
sy = oqupy.operators.sigma("y")
sz = oqupy.operators.sigma("z")
up_dm = oqupy.operators.spin_dm("z+")
down_dm = oqupy.operators.spin_dm("z-")

def save_object(filename,obj):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

# --- Parameters --------------------------------------------------------------

# -- time steps --
dt = 0.2
num_steps = 200

# -- bath --
alpha = 0.08
omega_cutoff = 4.0
temperature = 1.6
pt_dkmax = 40
pt_epsrel = 1.0e-6

# -- chain --
N = 4
h = np.array([[0.0, 0.0, 1.0]]*N)
J = np.array([[0.3, 0.0, 0.0]]*(N-1))
tebd_order = 2
tebd_epsrel = 1.0e-5


# --- Compute process tensors -------------------------------------------------

correlations = oqupy.PowerLawSD(alpha=alpha,
                                zeta=3,
                                cutoff=omega_cutoff,
                                cutoff_type='exponential',
                                temperature=temperature)
bath = oqupy.Bath(0.5 * sz, correlations)
pt_tempo_parameters = oqupy.TempoParameters(dt=dt,
                                            dkmax=pt_dkmax,
                                            epsrel=pt_epsrel)

print("Process tensor (PT) computation:")
pt = oqupy.pt_tempo_compute(bath=bath,
                            start_time=0.0,
                            end_time=num_steps * dt,
                            parameters=pt_tempo_parameters,
                            progress_type='bar')


# --- PT-TEBD preperation -----------------------------------------------------

# -- initial state --
initial_augmented_mps = oqupy.AugmentedMPS([up_dm] + [down_dm] * (N-1))

# -- chain hamiltonian --
system_chain = oqupy.SystemChain(hilbert_space_dimensions=[2]*N)

for n in range(N):
    for i, xyz in enumerate(["x", "y", "z"]):
        system_chain.add_site_hamiltonian(
            site=n,
            hamiltonian=0.5*h[n, i]*oqupy.operators.sigma(xyz))
for n in range(N-1):
    for i, xyz in enumerate(["x", "y", "z"]):
        system_chain.add_nn_hamiltonian(
            site=n,
            hamiltonian_l=0.5*J[n, i]*oqupy.operators.sigma(xyz),
            hamiltonian_r=0.5*oqupy.operators.sigma(xyz))

pt_tebd_params = oqupy.PtTebdParameters(
    dt=dt,
    order=tebd_order,
    epsrel=tebd_epsrel)
dynamics_sites=list(range(N))


# -- PT-TEBD computation ------------------------------------------------------

pt_tebd_closed = oqupy.PtTebd(
    initial_augmented_mps=initial_augmented_mps,
    system_chain=system_chain,
    process_tensors=[None]*N,
    parameters=pt_tebd_params,
    dynamics_sites=dynamics_sites,
    chain_control=None)
pt_tebd_open = oqupy.PtTebd(
    initial_augmented_mps=initial_augmented_mps,
    system_chain=system_chain,
    process_tensors=[pt]*N,
    parameters=pt_tebd_params,
    dynamics_sites=dynamics_sites,
    chain_control=None)

print("PT-TEBD computation (closed spin chain):")
t = time()
results_closed = pt_tebd_closed.compute(num_steps, progress_type="bar")
results_closed["computation_time"] = time() - t
print("PT-TEBD computation (open spin chain):")
t = time()
results_open = pt_tebd_open.compute(num_steps, progress_type="bar")
results_open["computation_time"] = time() - t


all_results = {
    "N":N,
    "results_closed":results_closed,
    "results_closed":results_open}

save_object(f"./examples/results/N{N:03d}.pkl",all_results)

# -- plot results -------------------------------------------------------------

plt.figure(1)

for site, dyn in results_closed['dynamics'].items():
    plt.plot(*dyn.expectations(sz, real=True),
             color=f"C{site}", linestyle="solid",
             label=f"$<\\sigma_{site}^z>$")
for site, dyn in results_open['dynamics'].items():
    plt.plot(*dyn.expectations(sz, real=True),
             color=f"C{site}", linestyle="dotted")

plt.legend()
plt.show()
