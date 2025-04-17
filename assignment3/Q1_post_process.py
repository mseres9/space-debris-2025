import json
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
from assignment3.ConjunctionUtilities import eci2ric, eci2ric_vel


def print_cdm(pair, tca, miss_distance, mahalanobis, outer_pc, pc, rel_pos_rtn, rel_vel_rtn):
    print(f"\nCDM for pair {pair}:")
    print(f"Object 1 ID: {pair[0]}")
    print(f"Object 2 ID: {pair[1]}")
    print(f"TCA (TDB): {(tca)}")
    print(f"Miss Distance: {miss_distance:.3f} m")
    print(f"Mahalanobis Distance: {mahalanobis:.3f}")
    print(f"Outer Pc: {outer_pc:.3f}")
    print(f"Pc: {pc:.3f}")
    print(f"Relative Position RTN: {rel_pos_rtn}")
    print(f"Relative Velocity RTN: {rel_vel_rtn}")

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Load datasets
assignment_data_directory = 'data/group4'
rso_file = os.path.join(assignment_data_directory, 'estimated_rso_catalog.pkl')
rso_dict = ConjUtil.read_catalog_file(rso_file)

cdm_data = load_json_file('assignment3/output_Q1/cdm_results.json')
hie_data = load_json_file('assignment3/output_Q1/hie_results.json')
tca_data = load_json_file('assignment3/output_Q1/tca_results.json')

# print(hie_data)
# print(tca_data)
# print(cdm_data)

cdm_data_new = {}

for pair, result in cdm_data.items():

    print(pair)
    print(result)
    obj1_id, obj2_id = map(int, pair.strip('()').split(','))

    tca_time = result['TCA_Time']
    min_distance = result['dE']

    X1 = np.array(result['State1'])
    X2 = np.array(result['State2'])

    P1 = np.array(result['Covariance1'])
    P2 = np.array(result['Covariance2'])

    x_diff = X1 - X2
    rel_pos_rtn = eci2ric(X1[:3], X1[3:], x_diff[:3])
    rel_vel_rtn = eci2ric_vel(X1[:3], X1[3:], rel_pos_rtn, x_diff[3:])

    dM = ConjUtil.compute_mahalanobis_distance(X1, X2, P1, P2)

    r1 = np.sqrt(rso_dict[obj1_id]['area']/(4*np.pi))  # Radius obj1
    r2 = np.sqrt(rso_dict[obj2_id]['area']/(4*np.pi))  # Radius obj2
    Pc = ConjUtil.Pc2D_Foster(X1, P1, X2, P2, r1 + r2)
    Uc = ConjUtil.Uc2D(X1, P1, X2, P2, r1 + r2)

    # Store CDM data
    cdm_data_new[pair] = {
        'TCA_Time': tca_time,
        'dE': min_distance,
        'dM': dM,
        'Pc': Pc,
        'Uc':Uc,
        'Relative_Position_RTN': rel_pos_rtn,
        'Relative_Velocity_RTN': rel_vel_rtn
    }

    # Optionally, print CDM data for each pair
    print_cdm(pair, tca_time, min_distance, dM, Uc, Pc, rel_pos_rtn, rel_vel_rtn)

print("CDM generation completed.")


rel_vel_norm = list(np.linalg( for x in cdm_data_new[x]['rel_pos_rtn'])






