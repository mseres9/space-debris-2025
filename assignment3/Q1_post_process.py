import json
import time
import os
import numpy as np
from itertools import combinations
from datetime import datetime, timedelta
import pickle
import json
import ConjunctionUtilities as ConjUtil
import TudatPropagator as prop
from FilteringUtilites import *
from assignment3.ConjunctionUtilities import eci2ric, eci2ric_vel

# Function to load JSON file as dictionary
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

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
    if decision:
        print(f"Decision: {decision}")
# Load datasets
assignment_data_directory = 'data/group4'
rso_file = os.path.join(assignment_data_directory, 'estimated_rso_catalog.pkl')
rso_dict = ConjUtil.read_catalog_file(rso_file)

cdm_data = load_json_file('assignment3/output_Q1/cdm_results.json')
hie_data = load_json_file('assignment3/output_Q1/hie_results.json')
tca_data = load_json_file('assignment3/output_Q1/tca_results.json')

# Initialize dictionaries to store propagated states and state parameters
cdm_data_new = {}
propagated_states = {}
state_parameters = {}

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
for pair, data in tca_data.items():
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
        'Relative_Velocity_RTN': rel_vel_rtn.tolist()
    }

    # Alert generation based on Pc
    if Pc > 1e-7:
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

        print_cdm(pair, tca_time, min_distance, dM, Uc, Pc, rel_pos_rtn, rel_vel_rtn, decision=decision)

print("CDM generation completed.")

output_dir = "assignment3/output_Q1"
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
