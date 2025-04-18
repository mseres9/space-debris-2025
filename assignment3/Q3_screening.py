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


# Function to print CDM data
def print_cdm(pair, tca, miss_distance, mahalanobis, outer_pc, pc, rel_pos_rtn, rel_vel_rtn,decision=None):
    print(f"\nCDM for pair {pair}:")
    # print(f"Object 1 ID: {pair[0]}")
    # print(f"Object 2 ID: {pair[1]}")
    print(f"TCA (TDB): {(tca)} s")
    print(f"Miss Distance: {miss_distance} m")
    print(f"Mahalanobis Distance: {mahalanobis:} ")
    print(f"Outer Pc: {outer_pc:}")
    print(f"Pc: {pc:}")
    print(f"Relative Position RTN: {rel_pos_rtn}")
    print(f"Relative Velocity RTN: {rel_vel_rtn}")
    print(f"Energy to Mass Ratio:{EMR}")
    if decision:
        print(f"Decision: {decision}")

# Step 1: Retrieve the dataset
assignment_data_directory = 'data/group4'
rso_file = os.path.join(assignment_data_directory, 'rso_dict_modified_Q3.pkl')
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


output_dir = "assignment3/output_Q3"
os.makedirs(output_dir, exist_ok=True)

# # Step 8: Save the results
def convert_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj  # fallback

with open(os.path.join(output_dir, 'tca_results.json'), 'w') as f:
    json.dump({str(k): {kk: convert_for_json(vv) for kk, vv in v.items()}
               for k, v in tca_results.items()}, f, indent=4)


# Initialize dictionaries to store propagated states and state parameters
cdm_data_new = {}
propagated_states = {}
state_parameters = {}
ID_HIE = {}

hie_r_threshold = 1e3  # 1e3 meters, JAXA
hie_Pc_trshold = 1e-4 # 1e-4 ESA
hie_Uc_trshold = 1e-4


hie_r_threshold = 1e3  # 1e3 meters, JAXA
hie_Pc_trshold = 1e-4 # 1e-4 ESA
hie_Uc_trshold = 1e-4

# Define initial time
t0 = (datetime(2025, 4, 1, 12, 0, 0) - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
bodies_to_create = ['Sun', 'Earth', 'Moon']
bodies = prop.tudat_initialize_bodies(bodies_to_create)

# Define integration parameters
int_params = {
    'tudat_integrator': 'rkf78',
    'step': 10.,
    'max_step': 1000.,
    'min_step': 1e-3,
    'rtol': 1e-12,
    'atol': 1e-12
}

# Iterate over pairs in the TCA data
for pair, data in tca_results.items():
    # Extract the object IDs and TCA time for the current pair
    obj1_id, obj2_id = map(int, pair.strip('()').split(','))
    tca_time = data['tca_time']
    trange1 = np.array([t0, tca_time])  # Time range for propagation
    min_distance = min(data['rho_list'])

    # Retrieve the state and covariance data for object 1 and object 2
    state_data1 = rso_dict[obj1_id]
    X1 = state_data1['state']
    P1 = state_data1['covar']

    state_data2 = rso_dict[obj2_id]
    X2 = state_data2['state']
    P2 = state_data2['covar']

    state_params1 = {
        'mass': state_data1.get('mass'),
        'area': state_data1.get('area'),
        'Cd': state_data1.get('Cd'),
        'Cr': state_data1.get('Cr'),
        'sph_deg': 8,
        'sph_ord': 8,
        'central_bodies': ['Earth'],
        'bodies_to_create': bodies_to_create
    }

    state_params2 = {
        'mass': state_data2.get('mass'),
        'area': state_data2.get('area'),
        'Cd': state_data2.get('Cd'),
        'Cr': state_data2.get('Cr'),
        'sph_deg': 8,
        'sph_ord': 8,
        'central_bodies': ['Earth'],
        'bodies_to_create': bodies_to_create
    }

    m1 = rso_dict[obj1_id]['mass']
    m2 = rso_dict[obj2_id]['mass']

    # Propagate state and covariance for both objects at the given TCA time
    try:
        # Propagate the state and covariance for object 1
        tout1, Xout1, Pout1 = prop.propagate_state_and_covar(X1, P1, trange1, state_params1, int_params, bodies)
        propagated_states[(obj1_id, tca_time)] = {'time': tout1, 'state': Xout1, 'covar': Pout1}

        # Propagate the state and covariance for object 2
        tout2, Xout2, Pout2 = prop.propagate_state_and_covar(X2, P2, trange1, state_params2, int_params, bodies)
        propagated_states[(obj2_id, tca_time)] = {'time': tout2, 'state': Xout2, 'covar': Pout2}

    except Exception as e:
        print(f"Error propagating pair {obj1_id}, {obj2_id} at TCA {tca_time}: {e}")
        continue


    # Calculate and store the risk metrics
    X1 = propagated_states[(obj1_id, tca_time)]['state']
    X2 = propagated_states[(obj2_id, tca_time)]['state']
    P1 = propagated_states[(obj1_id, tca_time)]['covar']
    P2 = propagated_states[(obj2_id, tca_time)]['covar']

    # Calculate relative position and velocity in RTN
    x_diff = X1 - X2
    rel_pos_rtn = eci2ric(X1[:3], X1[3:], x_diff[:3])
    rel_vel_rtn = eci2ric_vel(X1[:3], X1[3:], rel_pos_rtn, x_diff[3:])

    # Calculate Mahalanobis Distance
    dM = ConjUtil.compute_mahalanobis_distance(X1, X2, P1, P2)

    # Calculate Probability of Collision using Foster's method
    r1 = np.sqrt(rso_dict[obj1_id]['area']/(np.pi))  # Radius of object 1
    r2 = np.sqrt(rso_dict[obj2_id]['area']/(np.pi))  # Radius of object 2
    Pc = ConjUtil.Pc2D_Foster(X1, P1, X2, P2, r1 + r2)
    Uc = ConjUtil.Uc2D(X1, P1, X2, P2, r1 + r2)
    EMR = 0.5*m2/m1*np.linalg.norm(rel_vel_rtn)**2


    # Store the results in the new CDM data dictionary
    cdm_data_new[pair] = {
        'State1': X1,
        'State2': X2,
        'Covariance1': P1,
        'Covariance2': P2,
        'TCA_Time': tca_time,
        'dE': min_distance,
        'dM': dM,
        'Pc': Pc,
        'Uc': Uc,
        'Relative_Position_RTN': rel_pos_rtn.tolist(),
        'Relative_Velocity_RTN': rel_vel_rtn.tolist(),
        'Relative_Velocirt_norm': np.linalg.norm(rel_vel_rtn),
        'EMR': EMR
    }

    # Alert generation based on Pc
    if Pc > 1e-7 or min_distance < 5000:
        # Yellow alert level
        print(f"Yellow alert: Collision Risk identified for pair {pair}")
        # Store the alert
        cdm_data_new[pair]['Alert'] = 'Yellow Alert'

        # HIE classification only for Yellow Alerts
        decision = ""
        if min_distance < hie_r_threshold and Pc > hie_Pc_trshold:
            decision = "HIE found"
        elif min_distance < hie_r_threshold and Uc < hie_Uc_trshold:
            decision = "Not an HIE, safe"
        else:
            decision = "No immediate risk"

        # Store the HIE decision
        cdm_data_new[pair]['HIE_Decision'] = decision

        ID_HIE[pair] = {'State1': X1,
        'State2': X2,
        'Covariance1': P1,
        'Covariance2': P2,
        'TCA_Time': tca_time,
        'dE': min_distance,
        'dM': dM,
        'Pc': Pc,
        'Uc': Uc,
        'Relative_Position_RTN': rel_pos_rtn.tolist(),
        'Relative_Velocity_RTN': rel_vel_rtn.tolist(),
        'Relative_Velocirt_norm': np.linalg.norm(rel_vel_rtn),
        'EMR': EMR
                        }

        print_cdm(pair, tca_time, min_distance, dM, Uc, Pc, rel_pos_rtn, rel_vel_rtn, decision=decision)

print("CDM generation completed.")

output_dir = "assignment3/output_Q3"
os.makedirs(output_dir, exist_ok=True)

# Step 8: Save the results

def convert_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj  # fallback

with open(os.path.join(output_dir, 'cdm_results_with_HIE.json'), 'w') as f:
    json.dump({str(k): {kk: convert_for_json(vv) for kk, vv in v.items()}
               for k, v in cdm_data_new.items()}, f, indent=4)

with open(os.path.join(output_dir, 'ID_HIE.json'), 'w') as f:
    json.dump({str(k): {kk: convert_for_json(vv) for kk, vv in v.items()}
               for k, v in ID_HIE.items()}, f, indent=4)