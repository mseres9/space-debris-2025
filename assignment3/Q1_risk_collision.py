import time
import os
import numpy as np
from itertools import combinations
from datetime import datetime, timedelta
from itertools import combinations
import pickle
import ConjunctionUtilities as ConjUtil
import TudatPropagator as prop
from FilteringUtilites import *
from tudatpy.astro.element_conversion import cartesian_to_keplerian, keplerian_to_cartesian
import matplotlib.pyplot as plt
from tudatpy.astro.two_body_dynamics import propagate_kepler_orbit

# Function to convert from Cartesian to RTN coordinates
def cartesian_to_rtn(X1, X2):
    r1 = X1[:3]
    v1 = X1[3:]
    r2 = X2[:3]
    v2 = X2[3:]

    r_rel = r2 - r1
    v_rel = v2 - v1

    r_norm = np.linalg.norm(r1)
    r_hat = r1 / r_norm
    h = np.cross(r1, v1)
    h_hat = h / np.linalg.norm(h)
    t_hat = np.cross(h_hat, r_hat)

    R = np.vstack((r_hat, t_hat, h_hat)).T

    rel_pos_rtn = R @ r_rel
    rel_vel_rtn = R @ v_rel

    return rel_pos_rtn, rel_vel_rtn

# Function to format the TCA as a TDB calendar date
def convert_to_tdb(tca_seconds):
    base_date = datetime(2000, 1, 1, 12, 0, 0)  # J2000 epoch
    tca_date = base_date + timedelta(seconds=tca_seconds)
    return tca_date.isoformat()


# Step 1: Retrieve the dataset
assignment_data_directory = 'data/group4'
rso_file = os.path.join(assignment_data_directory, 'estimated_rso_catalog.pkl')
rso_dict = ConjUtil.read_catalog_file(rso_file)

# Step 2: Apply filtering to the catalog before propagation
D = 100  # Distance threshold (meters)
filtered_pairs = []

print("Applying filtering to the catalog...")
object_ids = list(rso_dict.keys())
for i in range(len(object_ids)):
    for j in range(i + 1, len(object_ids)):
        rso1 = rso_dict[object_ids[i]]
        rso2 = rso_dict[object_ids[j]]

        # Apply Apogee-Perigee Filter
        if apogee_perigee_filter(rso1['state'], rso2['state'], D):
            continue

        # # Apply Geometrical Filter
        # if geometrical_filter(rso1['state'], rso2['state'], D):
        #     continue

        # # Apply Time Filter
        # if time_filter(rso1['state'], rso2['state'], D):
        #     continue

        # If all filters pass, store the pair
        filtered_pairs.append((object_ids[i], object_ids[j]))

print(f"Filtering completed: {len(filtered_pairs)} pairs selected.")

# Step 3: Define initial and final time
t0 = (datetime(2025, 4, 1, 12, 0, 0) - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
tf = t0 + 48 * 3600  # 48 hours later
trange = np.array([t0, tf])

# Step 4: Initialize Tudat bodies
bodies_to_create = ['Sun', 'Earth', 'Moon']
bodies = prop.tudat_initialize_bodies(bodies_to_create)

# Step 5: Define integrator parameters
int_params = {
    'tudat_integrator': 'rkf78',
    'step': 10.,
    'max_step': 1000.,
    'min_step': 1e-3,
    'rtol': 1e-12,
    'atol': 1e-12
}

# Step 6: Propagate state and covariance for filtered objects
propagated_states = {}
state_parameters = {}

# Define common body creation list
bodies_to_create = ['Sun', 'Earth', 'Moon']

for obj1_id, obj2_id in filtered_pairs[:6]:
    for obj_id in [obj1_id, obj2_id]:
        if obj_id not in propagated_states:
            print(f"Propagating object ID: {obj_id}")

            # Extract initial state and covariance from the catalog
            state_data = rso_dict[obj_id]
            Xo = state_data['state']
            Po = state_data['covar']

            # Set default state parameters as specified
            state_params = {
                'mass': state_data.get('mass', 100.0),
                'area': state_data.get('area', 1.0),
                'Cd': state_data.get('Cd', 2.2),
                'Cr': state_data.get('Cr', 1.3),
                'sph_deg': 8,
                'sph_ord': 8,
                'central_bodies': ['Earth'],
                'bodies_to_create': bodies_to_create
            }

            try:
                # Propagate state and covariance
                tout, Xout, Pout = prop.propagate_state_and_covar(Xo, Po, trange, state_params, int_params, bodies)
                propagated_states[obj_id] = {'time': tout, 'state': Xout, 'covar': Pout}
            except Exception as e:
                print(f"Error propagating object ID {obj_id}: {e}")
                continue

            state_parameters[obj_id] = state_params

print("State propagation completed for all filtered objects.")

# Step 7: Compute TCA for each filtered pair
tca_results = {}
start_time = time.time()

for obj1_id, obj2_id in filtered_pairs[:6]:
    print(f"Computing TCA for pair: ({obj1_id}, {obj2_id})")

    X1 = propagated_states[obj1_id]['state']
    X2 = propagated_states[obj2_id]['state']

    rso1_params = state_parameters[obj1_id]
    rso2_params = state_parameters[obj2_id]

    T_list, rho_list = ConjUtil.compute_TCA(X1, X2, trange, rso1_params, rso2_params, int_params, bodies)

    if T_list:
        tca_results[(obj1_id, obj2_id)] = {'T_list':  [convert_to_tdb(t) for t in T_list], 'rho_list': rho_list}

print(f"TCA computation completed in {time.time() - start_time:.2f} seconds.")

# Step 8: Print results as a CDM
def print_cdm(pair, tca, miss_distance, mahalanobis, outer_pc, pc, rel_pos_rtn, rel_vel_rtn):
    print(f"\nCDM for pair {pair}:")
    print(f"Object 1 ID: {pair[0]}")
    print(f"Object 2 ID: {pair[1]}")
    print(f"TCA (TDB): {tca}")
    print(f"Miss Distance: {miss_distance:.3f} m")
    print(f"Mahalanobis Distance: {mahalanobis:.3f}")
    print(f"Outer Pc: {outer_pc:.3e}")
    print(f"Pc: {pc:.3e}")
    print(f"Relative Position RTN: {rel_pos_rtn}")
    print(f"Relative Velocity RTN: {rel_vel_rtn}")

# Step 9: Analyze TCA results and print CDM
for pair, result in tca_results.items():
    min_distance = min(result['rho_list'])
    tca_time = result['T_list'][result['rho_list'].index(min_distance)]

    X1 = propagated_states[obj1_id]['state']
    X2 = propagated_states[obj2_id]['state']

    P1 = propagated_states[obj1_id]['covar']
    P2 = propagated_states[obj2_id]['covar']

    # Risk analysis
    d2 = ConjUtil.compute_miss_distance(X1, X2)
    dM = ConjUtil.compute_mahalanobis_distance(X1, X2)
    Pc = ConjUtil.Pc2D_Foster(X1, P1, X2, P2, D)
    Uc = ConjUtil.Uc2D(X1, P1, X2, P2, D)

    # Relative position and velocity in RTN
    rel_pos_rtn, rel_vel_rtn = cartesian_to_rtn(X1, X2)

    # Print CDM
    print_cdm(pair, tca_time, min_distance, dM, Uc, Pc, rel_pos_rtn, rel_vel_rtn)

print("CDM generation completed.")

# Step 7: Identify High Interest Events (HIEs)
hie_threshold = 10  # meters
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

output_dir = "assignment3/output_Q1"
os.makedirs(output_dir, exist_ok=True)

# Step 8: Save the results
with open(os.path.join(output_dir, 'tca_results.pkl'), 'wb') as f:
    pickle.dump(tca_results, f)

with open(os.path.join(output_dir, 'hie_results.pkl'), 'wb') as f:
    pickle.dump(hie_results, f)

print("Results saved to:", output_dir)

print("Results saved to files.")

# Step 9: Generate Summary Report
print("\nHigh Interest Event Summary:")
for pair, result in hie_results.items():
    print(
        f"Pair: {pair}, TCA: {result['tca']}, Miss Distance: {result['miss_distance']} m, Decision: {result['decision']}")


# # Debug: Propagate using Keplerian motion for 48 hours and plot
# keplerian_positions = {}
# for obj_id in rso_dict.keys():
#     Xo = rso_dict[obj_id]['state']
#     t_kepler = np.linspace(0, 48 * 3600, 1000)
#     keplerian_positions[obj_id] = []
#     for t in t_kepler:
#         kepler_state = cartesian_to_keplerian(Xo, 3.986004418e14)
#         kepler_position_k = propagate_kepler_orbit(kepler_state, t,3.986004418e14 )
#         kepler_position = keplerian_to_cartesian(kepler_position_k, 3.986004418e14)
#         # Extract the position and convert to kilometers
#         position = np.array(kepler_position[:3]) / 1000.0
#         keplerian_positions[obj_id].append(position)
#     keplerian_positions[obj_id] = np.array(keplerian_positions[obj_id])
#
# # Plotting the Keplerian orbits
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for obj_id, positions in keplerian_positions.items():
#     # Check if positions array is non-empty and has the correct shape
#     if positions.shape[0] > 0:
#         ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=f'Keplerian Orbit {obj_id}')
# ax.set_title('Keplerian Orbit Propagation')
# ax.set_xlabel('X (km)')
# ax.set_ylabel('Y (km)')
# ax.set_zlabel('Z (km)')
# ax.legend()
# plt.show()


