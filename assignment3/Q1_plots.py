import matplotlib.pyplot as plt
import numpy as np
import os
import json
import TudatPropagator as prop
import ConjunctionUtilities as ConjUtil
from datetime import datetime

propagated_states = {}


def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# Define initial time
t0 = (datetime(2025, 4, 1, 12, 0, 0) - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
tf = t0 + 48 * 3600
bodies_to_create = ['Sun', 'Earth', 'Moon']
bodies = prop.tudat_initialize_bodies(bodies_to_create)

# Integration parameters
int_params = {
    'tudat_integrator': 'rkf78',
    'step': 10.,
    'max_step': 1000.,
    'min_step': 1e-3,
    'rtol': 1e-12,
    'atol': 1e-12
}

# Load datasets
assignment_data_directory = 'data/group4'
rso_file = os.path.join(assignment_data_directory, 'estimated_rso_catalog.pkl')
rso_dict = ConjUtil.read_catalog_file(rso_file)
cdm_data = load_json_file('assignment3/output_Q1/ID_HIE.json')

output_dir = "assignment3/output_Q1_plots"
os.makedirs(output_dir, exist_ok=True)

# Only process these object IDs
target_obj_ids = [91894, 91630, 91813]

# Iterate over pairs
for pair, data in cdm_data.items():
    obj1_id, obj2_id = map(int, pair.strip('()').split(','))

    # Process only specific object IDs
    if obj2_id not in target_obj_ids:
        continue

    tca_time = data['TCA_Time']
    min_distance = data['dE']

    state_data1 = rso_dict[obj1_id]
    X1 = state_data1['state']
    P1 = state_data1['covar']

    state_data2 = rso_dict[obj2_id]
    X2 = state_data2['state']
    P2 = state_data2['covar']

    state_params1 = {
        'mass': state_data1.get('mass'), 'area': state_data1.get('area'),
        'Cd': state_data1.get('Cd'), 'Cr': state_data1.get('Cr'),
        'sph_deg': 8, 'sph_ord': 8,
        'central_bodies': ['Earth'], 'bodies_to_create': bodies_to_create
    }

    state_params2 = {
        'mass': state_data2.get('mass'), 'area': state_data2.get('area'),
        'Cd': state_data2.get('Cd'), 'Cr': state_data2.get('Cr'),
        'sph_deg': 8, 'sph_ord': 8,
        'central_bodies': ['Earth'], 'bodies_to_create': bodies_to_create
    }

    trange1 = np.arange(t0, tf, 100)
    rel_pos_rtn_array = np.zeros((3, len(trange1)))

    # Set the initial propagation state to the current values from the previous iteration (if applicable)
    previous_t = t0
    previous_X1 = X1
    previous_X2 = X2
    previous_P1 = P1
    previous_P2 = P2

    for idx, t in enumerate(trange1):
        print(f"Processing time step {idx}, t = {t}")

        # Start propagation from previous state and time
        trange = [previous_t, t + 100]

        tout1_plot, Xout1_plot, _ = prop.propagate_state_and_covar(previous_X1, previous_P1, trange, state_params1,
                                                                   int_params, bodies)
        tout2_plot, Xout2_plot, _ = prop.propagate_state_and_covar(previous_X2, previous_P2, trange, state_params2,
                                                                   int_params, bodies)

        # Update the states for the next iteration
        previous_t = t + 100
        previous_X1 = Xout1_plot
        previous_X2 = Xout2_plot
        previous_P1 = _  # Assuming covariance is not needed here, but you may store it if necessary
        previous_P2 = _  # Same as above

        # Relative position calculation
        state1 = np.asarray(Xout1_plot).flatten()
        state2 = np.asarray(Xout2_plot).flatten()
        rel_pos = np.array(state1[:3]) - np.array(state2[:3])
        rel_pos_eci = rel_pos.reshape(3, 1)

        rel_pos_rtn = ConjUtil.eci2ric(state1[:3], state1[3:], rel_pos_eci).flatten()
        rel_pos_rtn_array[:, idx] = rel_pos_rtn

    time_rel = (trange1 - tca_time) / 3600
    rel_norm = np.linalg.norm(rel_pos_rtn_array, axis=0)

    # Plot RTN
    plt.figure(figsize=(10, 6))
    plt.plot(time_rel, rel_norm, label='Relative distance RTN')
    plt.scatter(0, min_distance, color='red', label='TCA (min distance)', zorder=5)
    plt.xlabel('Time relative to TCA [hours]')
    plt.ylabel('Relative Position RTN [m]')
    plt.title(f'Relative RTN Position (Pair {obj1_id}, {obj2_id})')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=0.8)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_filename = f"RTN_plot_pair_{obj1_id}_{obj2_id}.png"
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.show()
    plt.close()
