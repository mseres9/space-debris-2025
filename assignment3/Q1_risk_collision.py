import time
import os
import numpy as np
from itertools import combinations
from datetime import datetime, timedelta
import pickle
import ConjunctionUtilities as ConjUtil
import TudatPropagator as prop

# Step 1: Retrieve the dataset
assignment_data_directory = 'data/group4'
rso_file = os.path.join(assignment_data_directory, 'estimated_rso_catalog.pkl')
rso_dict = ConjUtil.read_catalog_file(rso_file)

# Step 2: Define initial and final time
t0 = (datetime(2025, 4, 1, 12, 0, 0) - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
tf = t0 + 48 * 3600  # 48 hours later
trange = np.array([t0, tf])

# Step 3: Initialize Tudat bodies
bodies_to_create = ['Sun', 'Earth', 'Moon']
bodies = prop.tudat_initialize_bodies(bodies_to_create)

# Step 4: Define integrator parameters
int_params = {
    'tudat_integrator': 'rkf78',
    'step': 10.,
    'max_step': 1000.,
    'min_step': 1e-3,
    'rtol': 1e-12,
    'atol': 1e-12
}

# Step 5: Propagate state and covariance for all objects
propagated_states = {}
state_parameters = {}

# Define common body creation list
bodies_to_create = ['Sun', 'Earth', 'Moon']

for obj_id in list(rso_dict.keys())[:4]:
    print(f"Propagating object ID: {obj_id}")

    # Extract initial state and covariance from the catalog
    state_data = rso_dict[obj_id]
    Xo = state_data['state']
    Po = state_data['covar']

    # Set default state parameters as specified
    state_params = {}
    state_params['mass'] = state_data.get('mass', 100.0)  # Default mass if not provided
    state_params['area'] = state_data.get('area', 1.0)    # Default area if not provided
    state_params['Cd'] = state_data.get('Cd', 2.2)        # Default drag coefficient
    state_params['Cr'] = state_data.get('Cr', 1.3)        # Default reflectivity coefficient
    state_params['sph_deg'] = 8                          # Spherical harmonics degree
    state_params['sph_ord'] = 8                          # Spherical harmonics order
    state_params['central_bodies'] = ['Earth']            # Central body
    state_params['bodies_to_create'] = bodies_to_create   # Bodies to create for propagation

    try:
        # Propagate state and covariance
        tout, Xout, Pout = prop.propagate_state_and_covar(Xo, Po, trange, state_params, int_params, bodies)

        # Store the result
        propagated_states[obj_id] = {
            'time': tout,
            'state': Xout,
            'covar': Pout
        }

    except Exception as e:
        print(f"Error propagating object ID {obj_id}: {e}")
        continue
    state_parameters[obj_id] = state_params
print("State propagation completed for all objects.")

# Step 6: Compute TCA for every pair of objects
tca_results = {}
start_time = time.time()

print(f"state_params: {state_parameters}")

for obj1_id, obj2_id in combinations(propagated_states.keys(), 2):
    print(f"Computing TCA for pair: ({obj1_id}, {obj2_id})")

    # Retrieve states for the pair
    X1 = propagated_states[obj1_id]['state']
    X2 = propagated_states[obj2_id]['state']

    # Set up RSO parameters
    rso1_params = state_parameters[obj1_id]
    rso2_params = state_parameters[obj2_id]

    # Compute TCA
    T_list, rho_list = ConjUtil.compute_TCA(X1, X2, trange, rso1_params, rso2_params, int_params, bodies)

    # Store the result
    tca_results[(obj1_id, obj2_id)] = {
        'T_list': T_list,
        'rho_list': rho_list
    }

print(f"TCA computation completed in {time.time() - start_time:.2f} seconds.")

# Step 7: Identify High Interest Events (HIEs)
hie_threshold = 100  # meters
hie_results = {}

for pair, data in tca_results.items():
    min_distance = min(data['rho_list'])
    tca_time = data['T_list'][data['rho_list'].index(min_distance)]

    if min_distance < hie_threshold:
        decision = "Avoidance maneuver recommended"
    else:
        decision = "No maneuver required"

    hie_results[pair] = {
        'tca': tca_time,
        'miss_distance': min_distance,
        'decision': decision
    }

print("HIE analysis completed.")

# Step 8: Save the results
with open('tca_results.pkl', 'wb') as f:
    pickle.dump(tca_results, f)

with open('hie_results.pkl', 'wb') as f:
    pickle.dump(hie_results, f)

print("Results saved to files.")

# Step 9: Generate Summary Report
print("\nHigh Interest Event Summary:")
for pair, result in hie_results.items():
    print(
        f"Pair: {pair}, TCA: {result['tca']}, Miss Distance: {result['miss_distance']} m, Decision: {result['decision']}")

