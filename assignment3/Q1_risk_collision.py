import time
import os
import numpy as np
from itertools import combinations
from datetime import datetime, timedelta
from itertools import combinations
import pickle
import json
import ConjunctionUtilities as ConjUtil
import TudatPropagator as prop
from FilteringUtilites import *
from tudatpy.astro.element_conversion import cartesian_to_keplerian, keplerian_to_cartesian
import matplotlib.pyplot as plt
from tudatpy.astro.two_body_dynamics import propagate_kepler_orbit

from assignment3.ConjunctionUtilities import eci2ric, eci2ric_vel


def convert_to_tdb(tca_seconds):
    base_date = datetime(2000, 1, 1, 12, 0, 0)  # J2000 epoch
    tca_date = base_date + timedelta(seconds=tca_seconds)
    return tca_date.isoformat()


# Step 1: Retrieve the dataset
assignment_data_directory = 'data/group4'
rso_file = os.path.join(assignment_data_directory, 'estimated_rso_catalog.pkl')
rso_dict = ConjUtil.read_catalog_file(rso_file)

# Step 2: Apply filtering to the catalog before propagation
D = 100e3  # Distance threshold (meters)
protected_id = 31698
filtered_pairs = []


print("Filtering all potential threats to object", protected_id)
object_ids = list(rso_dict.keys())
object_ids.remove(protected_id)

for obj_id in object_ids:
    rso1 = rso_dict[protected_id]
    rso2 = rso_dict[obj_id]

    if apogee_perigee_filter(rso1['state'], rso2['state'], D):
        continue

    # if geometrical_filter(rso1['state'], rso2['state'], D):
    #     continue

    # if time_filter(rso1['state'], rso2['state'], D):
    #     continue

    filtered_pairs.append((protected_id, obj_id))

print(f"Filtering completed: {len(filtered_pairs)} potential threat pairs selected.")

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

# Step 6: Compute TCA for each filtered pair
tca_results = {}
start_time = time.time()

for obj1_id, obj2_id in filtered_pairs:
    print(f"Computing TCA for pair: ({obj1_id}, {obj2_id})")

    state_data1 = rso_dict[obj1_id]
    state_data2 = rso_dict[obj2_id]

    X1 = state_data1['state']
    X2 = state_data2['state']

    rso1_params = {
        'mass': state_data1.get('mass', 100.0),
        'area': state_data1.get('area', 1.0),
        'Cd': state_data1.get('Cd', 2.2),
        'Cr': state_data1.get('Cr', 1.3),
        'sph_deg': 8,
        'sph_ord': 8,
        'central_bodies': ['Earth'],
        'bodies_to_create': bodies_to_create
    }

    rso2_params = {
        'mass': state_data2.get('mass', 100.0),
        'area': state_data2.get('area', 1.0),
        'Cd': state_data2.get('Cd', 2.2),
        'Cr': state_data2.get('Cr', 1.3),
        'sph_deg': 8,
        'sph_ord': 8,
        'central_bodies': ['Earth'],
        'bodies_to_create': bodies_to_create
    }

    T_list, rho_list = ConjUtil.compute_TCA(X1, X2, trange, rso1_params, rso2_params, int_params, bodies)

    idx_min_rho = np.argmin(rho_list)
    tca_results[(obj1_id, obj2_id)] = {
        'T_list': T_list,
        'rho_list': rho_list,
        'tca_time': T_list[idx_min_rho]
    }
print(f"TCA computation completed in {time.time() - start_time:.2f} seconds.")

#
# # Step 7: Propagate state and covariance for filtered objects
# propagated_states = {}
# state_parameters = {}
#
# for (obj1_id, obj2_id), data in list(tca_results.items()):
#     tca_time = data['tca_time']
#     trange1 = np.array([t0, tca_time])
#
#     for obj_id in [obj1_id, obj2_id]:
#         if obj_id not in propagated_states:
#             print(f"Propagating object ID: {obj_id}")
#
#             state_data = rso_dict[obj_id]
#
#             Xo = state_data['state']
#             Po = state_data['covar']
#
#             state_params = {
#                 'mass': state_data.get('mass'),
#                 'area': state_data.get('area'),
#                 'Cd': state_data.get('Cd'),
#                 'Cr': state_data.get('Cr'),
#                 'sph_deg': 8,
#                 'sph_ord': 8,
#                 'central_bodies': ['Earth'],
#                 'bodies_to_create': bodies_to_create
#             }
#
#             try:
#                 tout, Xout, Pout = prop.propagate_state_and_covar(Xo, Po, trange1, state_params, int_params, bodies)
#                 propagated_states[obj_id] = {'time': tout, 'state': Xout, 'covar': Pout}
#             except Exception as e:
#                 print(f"Error propagating object ID {obj_id}: {e}")
#                 continue
#
#             state_parameters[obj_id] = state_params
#
# print("State propagation completed for all filtered objects.")
#
#
# # Step 8: Print results as a CDM
# def print_cdm(pair, tca, miss_distance, mahalanobis, outer_pc, pc, rel_pos_rtn, rel_vel_rtn):
#     print(f"\nCDM for pair {pair}:")
#     print(f"Object 1 ID: {pair[0]}")
#     print(f"Object 2 ID: {pair[1]}")
#     print(f"TCA (TDB): {convert_to_tdb(tca)}")
#     print(f"Miss Distance: {miss_distance:.3f} m")
#     print(f"Mahalanobis Distance: {mahalanobis:.3f}")
#     print(f"Outer Pc: {outer_pc:.3f}")
#     print(f"Pc: {pc:.3f}")
#     print(f"Relative Position RTN: {rel_pos_rtn}")
#     print(f"Relative Velocity RTN: {rel_vel_rtn}")
#
# # Step 9: Analyze TCA results and print CDM
# cdm_data = {}  # Initialize an empty dictionary to store CDM data
#
# for pair, result in tca_results.items():
#     min_distance = min(result['rho_list'])
#     tca_time = result['T_list'][result['rho_list'].index(min_distance)]
#
#     obj1_id, obj2_id = pair
#     X1 = propagated_states[obj1_id]['state']
#     X2 = propagated_states[obj2_id]['state']
#     P1 = propagated_states[obj1_id]['covar']
#     P2 = propagated_states[obj2_id]['covar']
#
#     r1 = np.sqrt(rso_dict[obj1_id]['area'] / (np.pi))
#     r2 = np.sqrt(rso_dict[obj2_id]['area'] / (np.pi))
#
#     dM = ConjUtil.compute_mahalanobis_distance(X1, X2, P1, P2)
#     Pc = ConjUtil.Pc2D_Foster(X1, P1, X2, P2, r1 + r2)
#     Uc = ConjUtil.Uc2D(X1, P1, X2, P2, r1 + r2)
#
#     x_diff = X1-X2
#     rel_pos_rtn = eci2ric(X1[:3],X1[3:],x_diff[:3])
#     rel_vel_rtn = eci2ric_vel(X1[:3],X1[3:],rel_pos_rtn,x_diff[3:])
#
#     if min_distance < 5000:  # 5000m
#         cdm_data[pair] = {
#             'State1': X1.tolist(),
#             'State2': X2.tolist(),
#             'Covariance1':P1.tolist(),
#             'Covariance2':P2.tolist(),
#             'TCA_Time': tca_time,
#             'dE': min_distance,
#             'dM': dM,
#             'Uc': Uc,
#             'Pc': Pc,
#             'Relative_Position_RTN': rel_pos_rtn.tolist(),  # Ensure this is a list
#             'Relative_Velocity_RTN': rel_vel_rtn.tolist()  # Ensure this is a list
#         }
#
#         print_cdm(pair, tca_time, min_distance, dM, Uc, Pc, rel_pos_rtn, rel_vel_rtn)
#
# print("CDM generation completed.")
#
# # Step 7: Identify High Interest Events (HIEs)
# hie_r_threshold = 1e4  # 1e3 meters, JAXA
# hie_Pc_trshold = 1e-6 # 1e-4 ESA
# #todo: add mahalanobis here minore 4.6 per assicurare sulla probabilità di impatto
# # TODO: what are delande and foster? Implement
#
# hie_results = {}
# #todo: maybe consider energy of the impact to determine whether or not to maneuver
#
# for pair, data in cdm_data.items():
#     min_distance = data['dE']
#     tca_time = data['TCA_Time']
#     Pc = data["Pc"]
#     dM = data["dM"]
#     Uc = data["Uc"]
#
#     if min_distance < hie_r_threshold or Pc > hie_Pc_trshold:
#         decision = f"HIE identified for pair: {pair}"
#     else:
#         decision = "Not an HIE"
#
#     hie_results[pair] = {
#         'TCA': tca_time,
#         'Eucledian Miss Distance': min_distance,
#         'Probabiltiy of Collision': Pc,
#         "Outer Probability of Collision":Uc,
#         'Mahalanobis Distance': dM,
#         'Decision': decision
#     }
#
# print("HIE analysis completed.")
#
output_dir = "assignment3/output_Q1"
os.makedirs(output_dir, exist_ok=True)

# # Step 8: Save the results
#
def convert_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj  # fallback
#
# with open(os.path.join(output_dir, 'cdm_results.json'), 'w') as f:
#     json.dump({str(k): {kk: convert_for_json(vv) for kk, vv in v.items()}
#                for k, v in cdm_data.items()}, f, indent=4)
#
with open(os.path.join(output_dir, 'tca_results.json'), 'w') as f:
    json.dump({str(k): {kk: convert_for_json(vv) for kk, vv in v.items()}
               for k, v in tca_results.items()}, f, indent=4)

# with open(os.path.join(output_dir, 'hie_results.json'), 'w') as f:
#     json.dump({str(k): {kk: convert_for_json(vv) for kk, vv in v.items()}
#                for k, v in hie_results.items()}, f, indent=4)
#
# print("All results have been saved as JSON.")
# print("Results saved to:", output_dir)
#
# print("Results saved to files.")

####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################

# # Generate summary report
# print("\nHigh Interest Event Summary:")
# for pair, result in hie_results.items():
#     if result['decision'] == not "Not an HIE":
#         print(
#             f"Pair: {pair}, TCA: {result['tca']}, Miss Distance: {result['miss_distance']} m, Decision: {result['decision']}")
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



# Function to convert from Cartesian to RTN coordinates
def cartesian_to_rtn(X1, X2):
    r1 = np.array(X1[:3])
    v1 = np.array(X1[3:])
    r2 = np.array(X2[:3])
    v2 = np.array(X2[3:])

    r1 = r1.flatten()
    v1 = v1.flatten()
    r2 = r2.flatten()
    v2 = v2.flatten()

    # print(r1, v1, r2, v2)
    # print("r1 shape:", r1.shape)
    # print("v1 shape:", v1.shape)

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


#### DEBUG ############################################################################################################

# t0 = (datetime(2025, 4, 1, 12, 0, 0) - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
# t_tca = 796888799.9962436
# trange = [t0, t_tca]
#
# id = 31698
# Xo = rso_dict[id]['state']
# Po = rso_dict[id]['covar']
#
# bodies_to_create = ['Sun', 'Earth', 'Moon']
# bodies = prop.tudat_initialize_bodies(bodies_to_create)
#
# state_params = {
#     'mass': rso_dict[id]['mass'],
#     'area': rso_dict[id]['area'],
#     'Cd': rso_dict[id]['Cd'],
#     'Cr': rso_dict[id]['Cd'],
#     'sph_deg': 8,
#     'sph_ord': 8,
#     'central_bodies': ['Earth'],
#     'bodies_to_create': bodies_to_create
#
# }
#
# int_params = {
#     'tudat_integrator': 'rkf78',
#     'step': 10.,
#     'max_step': 1000.,
#     'min_step': 1e-3,
#     'rtol': 1e-12,
#     'atol': 1e-12
# }
#
#
#
# tout, Xout, Pout = prop.propagate_state_and_covar(Xo, Po, trange, state_params, int_params, bodies)
#
# #### 91362
#
# id2 = 91362
# Xo2 = rso_dict[id2]['state']
# Po2= rso_dict[id2]['covar']
#
# state_params2 = {
#     'mass': rso_dict[id2]['mass'],
#     'area': rso_dict[id2]['area'],
#     'Cd': rso_dict[id2]['Cd'],
#     'Cr': rso_dict[id2]['Cd'],
#     'sph_deg': 8,
#     'sph_ord': 8,
#     'central_bodies': ['Earth'],
#     'bodies_to_create': bodies_to_create
#
# }
#
# tout2, Xout2, Pout2 = prop.propagate_state_and_covar(Xo2, Po2, trange, state_params2, int_params, bodies)


######################################