import EstimationUtilities as EstUtil
import os
import numpy as np
import seaborn as sns
import TudatPropagator as prop
import matplotlib.pyplot as plt
from tudatpy.astro import two_body_dynamics, time_conversion

# Set up function to transform celestial coordinates to ECI
def cel2eci(y_cel_list):
    '''
    This function converts celestial coordinates to ECI
    Parameters
    ----------
    y_cel : list
        states in celestial frame

    Returns
    -------
    y_ecef_list: array
        states in ecef frame
    '''

    # Preallocate array for ECI coordinates
    y_eci_list = np.zeros((len(y_cel_list), 3))

    # Loop through entries and convert coordinates
    for i, y_cel in enumerate(y_cel_list):
        # Extract celestial coordinates
        r = y_cel[0, 0]
        ra = y_cel[1, 0]
        dec = y_cel[2, 0]
        # Calculate ECI coordinates
        y_eci = r * np.array([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])
        y_eci_list[i, :] = y_eci

    return y_eci_list

# Set directory for assignment data
cwd = os.getcwd()
assignment_data_directory = os.path.join(cwd, 'data\group4')

# Load measurement data for estimation case
meas_file = os.path.join(assignment_data_directory, 'q2_meas_maneuver_91267.pkl')
state_params, meas_dict, sensor_params = EstUtil.read_measurement_file(meas_file)

# Setup filter parameters such as process noise
Qeci = 1e-12*np.diag([1., 1., 1.])
Qric = 1e-12*np.diag([1., 1., 1.])

filter_params = {}
filter_params['Qeci'] = Qeci
filter_params['Qric'] = Qric
filter_params['alpha'] = 1.
filter_params['gap_seconds'] = 600.

# Setup integration parameters
int_params = {}
int_params['tudat_integrator'] = 'rkf78'
int_params['step'] = 10.
int_params['max_step'] = 1000.
int_params['min_step'] = 1e-3
int_params['rtol'] = 1e-12
int_params['atol'] = 1e-12

# Initialize tudat bodies
bodies_to_create = ['Earth', 'Sun', 'Moon']
bodies = prop.tudat_initialize_bodies(bodies_to_create)

# Partition the measurement dictionary
tk = np.array(meas_dict['tk_list'])
Yk = np.array(meas_dict['Yk_list'])
Yk_reshaped = np.squeeze(Yk, axis=-1)
# Convert using predefined functions
Yk_eci = cel2eci(Yk)

# Run UKF filter
filter_output = EstUtil.ukf(state_params, meas_dict, sensor_params, int_params, filter_params, bodies)

# Extract times
times_ukf = sorted(filter_output.keys())

# Make lists for states and residuals
states_ukf = []
covs_ukf = []
resids_ukf = []

# Loop through each time and collect data
for t in times_ukf:
    states_ukf.append(filter_output[t]['state'])
    covs_ukf.append(filter_output[t]['covar'])
    resids_ukf.append(filter_output[t]['resids'])

states_ukf = np.array(states_ukf)
covs_ukf = np.array(covs_ukf)
resids_ukf = np.array(resids_ukf)
print(f"dV between: {tk[121]} and {tk[122]}")

t_days_start = time_conversion.seconds_since_epoch_to_julian_day(tk[121])
t_days_end = time_conversion.seconds_since_epoch_to_julian_day(tk[122])
t_cal_start = time_conversion.julian_day_to_calendar_date(t_days_start)
t_cal_end = time_conversion.julian_day_to_calendar_date(t_days_end)
print(f"Manoeuvre between {t_cal_start} and {t_cal_end}")

# Propagate forward from first UKF point
t_prop, states_prop = prop.propagate_orbit(states_ukf[0, :], [times_ukf[0], times_ukf[-1]], state_params, int_params, bodies)

# Transform measurements to eci
r_eci = Yk_eci.copy()
sensor_ecef = sensor_params['sensor_ecef'].flatten()
earth_rotation_model = bodies.get("Earth").rotation_model
for i, t in enumerate(tk):
    ecef2eci = earth_rotation_model.body_fixed_to_inertial_rotation(t)
    r_eci[i, :] += np.matmul(ecef2eci, sensor_ecef)

############# Plotting for initial analysis #############
plt.rcParams.update({'font.size': 16})
# Plot residuals from UKF to show where manoeuvre happens
plt.figure(figsize = (12,8))
plt.plot(times_ukf[:122]-times_ukf[0], np.linalg.norm(resids_ukf, axis=1)[:122])
plt.scatter(times_ukf[:122]-times_ukf[0], np.linalg.norm(resids_ukf, axis=1)[:122])
plt.grid()
plt.yscale('log')
plt.xlabel('Time [s]')
plt.ylabel('Residuals [m]')
plt.savefig('plots\Q2_pre_dv_resids1.png')
plt.show()

plt.figure(figsize = (12,8))
plt.plot(times_ukf[:146]-times_ukf[0], np.linalg.norm(resids_ukf, axis=1)[:146])
plt.scatter(times_ukf[:146]-times_ukf[0], np.linalg.norm(resids_ukf, axis=1)[:146])
plt.grid()
plt.yscale('log')
plt.xlabel('Time [s]')
plt.ylabel('Residuals [m]')
plt.savefig('plots\Q2_pre_dv_resids2.png')
plt.show()

# Set global font size
plt.rcParams.update({'font.size': 11})

# Create figure for observation and initial orbit estimate
fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# First subplot
axs[0].plot(t_prop, states_prop[:, 0]/1000, label='Pre-Manoeuvre Orbit', color='blue')
axs[0].scatter(tk, r_eci[:, 0]/1000, label='Observations', color='green')
axs[0].grid()
axs[0].set_ylabel('Inertial X Position [km]')
axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)

# Second subplot
axs[1].plot(t_prop, states_prop[:, 1]/1000, color='blue')
axs[1].scatter(tk, r_eci[:, 1]/1000, color='green')
axs[1].grid()
axs[1].set_ylabel('Inertial Y Position [km]')

# Third subplot
axs[2].plot(t_prop, states_prop[:, 2]/1000, color='blue')
axs[2].scatter(tk, r_eci[:, 2]/1000, color='green')
axs[2].grid()
axs[2].set_ylabel('Inertial Z Position [km]')

# Add a common x-label
plt.xlabel('Time since J2000 [s]')
# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig('plots/Q2_pre_dv_orbit.png')
# Show plot
plt.show()

############################################################################################
###################################### DETERMINING DV ######################################
############################################################################################
# Set unperturbed state parameters
state_params_unperturbed = {
    "Cd": 0.0,
    "Cr": 0.0,
    "area": 0.0,
    "mass": 0.0,
    "sph_deg": 0.0,
    "sph_ord": 0.0,
    "central_bodies": ["Earth"],
    "bodies_to_create": ["Earth"]
}

# Choose integration parameters for MC analysis (backwards propagation)
int_params_MC = {}
int_params_MC['tudat_integrator'] = 'rk4'
int_params_MC['step'] = -1.0

def get_lambert_problem_result(t1, t2, r1_eci, r2_eci):
    mu_earth = bodies.get("Earth").gravitational_parameter

    # Create Lambert targeter
    lambertTargeter = two_body_dynamics.LambertTargeterIzzo(
        r1_eci,
        r2_eci,
        t2 - t1,
        mu_earth,
        is_retrograde=True,
    )

    # Compute initial Cartesian state of Lambert arc
    lambert_arc_initial_state  = np.zeros(6)
    lambert_arc_initial_state[:3] = r1_eci.flatten()
    lambert_arc_initial_state[3:] = lambertTargeter.get_departure_velocity()

    lambert_arc_final_state = np.zeros(6)
    lambert_arc_final_state[:3] = r2_eci.flatten()
    lambert_arc_final_state[3:] = lambertTargeter.get_arrival_velocity()

    return lambert_arc_initial_state, lambert_arc_final_state

# Set np.random seed
np.random.seed(42)
# Define function to make noise measurement
def generate_noisy_eci_measurements(Yk_trimmed, tk_trimmed):
    sigma_r = sensor_params['sigma_dict']['rg']
    sigma_ra = sensor_params['sigma_dict']['ra']
    sigma_dec = sensor_params['sigma_dict']['dec']

    sensor_ecef = sensor_params['sensor_ecef'].flatten()
    earth_rotation_model = bodies.get("Earth").rotation_model

    r_eci_noisy = np.zeros((2, 3))

    for l in range(2):
        y = Yk_trimmed[l]

        # Add Gaussian noise
        r = y[0, 0] + np.random.normal(0, sigma_r)
        ra = y[1, 0] + np.random.normal(0, sigma_ra)
        dec = y[2, 0] + np.random.normal(0, sigma_dec)

        # Convert to ECI
        y_eci = r * np.array([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])

        # Transform sensor offset to ECI and add
        ecef2eci = earth_rotation_model.body_fixed_to_inertial_rotation(tk_trimmed[l])
        y_eci += np.matmul(ecef2eci, sensor_ecef)

        r_eci_noisy[l, :] = y_eci

    return r_eci_noisy[0], r_eci_noisy[1]

# Set up lists to store results of state at dv
states_at_dv = list()

# Set number of first, last, and MC to use
N_1, N_2, N_MC = 6, 6, 200

# Loop through points to use for lambert arc, and fit lambert arcs
for i in range(N_1):
    # Define first point of lambert arc
    idx1 = 122+i
    for j in range(N_2):
        print(f"#################### INIITAL POINT {i+1}, FINAL POINT {j+1} ####################")
        # Define second point of Lambert arc
        idx2 = 145-j

        # Set up MC for 100 runs
        for k in range(N_MC):
            if (k+1)%25 == 0:
                print(f"{k+1} MC runs done ...")
            # Retrieve relevant measurements
            Yk_trim = [Yk[idx1], Yk[idx2]]
            tk_trim = [tk[idx1], tk[idx2]]

            # Create noise data
            r1_eci_noisy, r2_eci_noisy = generate_noisy_eci_measurements(Yk_trim, tk_trim)

            # Fit lambert arc
            state1, state2 = get_lambert_problem_result(tk[idx1], tk[idx2], r1_eci_noisy, r2_eci_noisy)

            # If first point is not at idx 121 propagate state 1 back with unperturbed
            if idx1 != 121:
                _, states_inter1 = prop.propagate_orbit(state1, [tk[idx1], tk[121]], state_params_unperturbed, int_params_MC)
                state_at_dv_estimate = states_inter1[0]
            else:
                state_at_dv_estimate = state1

            # Calculate estimated dv
            dv_estimate = np.linalg.norm((state_at_dv_estimate - states_ukf[121].reshape(6))[3:])

            # Update state params for second point of lambert arc
            state_params_MC = state_params.copy()
            state_params_MC['mass'] = state_params['mass'] / np.exp(np.linalg.norm(dv_estimate) / (220 * 9.80665))
            state_params_MC['epoch_tdb'] = tk[idx2]
            state_params_MC['state'] = state2

            # Propagate second state from lambert arc backwards with perturbations
            _, states_inter2 = prop.propagate_orbit(state2, [tk[idx2], tk[121]], state_params_MC, int_params_MC)
            state_at_dv = states_inter2[0]

            # Save estimated state
            states_at_dv.append(state_at_dv)


# Convert post dv states to array
states_at_dv = np.array(states_at_dv)

# Get means and cov of states at dv and dv itself
mean_state_at_dv = np.mean(states_at_dv, axis=0)
cov_state_at_dv = np.cov(states_at_dv.T)
dv_array = np.array(states_at_dv - states_ukf[121].reshape(6))[:, 3:]
mean_dv = np.mean(dv_array, axis=0)
cov_dv = np.cov(dv_array.T)

# Update state parameters
state_params_after_dv = state_params.copy()
state_params_after_dv['mass'] = state_params['mass'] / np.exp(np.linalg.norm(mean_dv)/(220*9.80665))
state_params_after_dv['epoch_tdb'] = tk[121]
state_params_after_dv['state'] = mean_state_at_dv.reshape(6,1)
state_params_after_dv['covar'] = cov_state_at_dv

# Propagate final orbit estimate with perturbations and dv applied
t_after_dv, states_after_dv = prop.propagate_orbit(mean_state_at_dv, [tk[121], tk[-1]], state_params_after_dv, int_params)

# Propagate covariance to get uncertainty in final state
t_final, state_final, cov_final  = prop.propagate_state_and_covar(mean_state_at_dv.reshape(6,1), cov_state_at_dv, [tk[121], tk[-1]], state_params_after_dv, int_params)

# Print final output
print("####### FINAL OUTPUT #######")
print("Assumed time of manoeuvre:", tk[122])
print("Mean state at dv:", mean_state_at_dv)
print("Covariance of state at dv:\n", cov_state_at_dv)
print("Mean delta-V Mag:", np.linalg.norm(mean_dv))
print("Mean delta-V:", mean_dv)
print("Covariance of delta-V:\n", cov_dv)
print("Covariance of final state:\n", cov_final)
print(f"Sanity Check: {states_after_dv[-1] - state_final.T}")

# Calculate correlations
def correlation(cov):
    n = cov.shape[0]
    corr = np.zeros_like(cov)
    stds = np.zeros(n)
    # Calculate stds
    for x in range(n):
        stds[x] = np.sqrt(cov[x,x])

    # Calculate correlation matrix
    for x in range(n):
        for y in range(n):
            corr[x, y] = cov[x,y]/(stds[x]*stds[y])
    # Return correlation matrix
    return corr

corr_state_at_dv = correlation(cov_state_at_dv)
corr_dv = correlation(cov_dv)

###############################################################################################
######################################### Final Plots #########################################
###############################################################################################
# Set global font size
plt.rcParams.update({'font.size': 11})

# Plot comparison
fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

axs[0].scatter(tk, r_eci[:,0]/1000, color='green', marker='o', label='Observations')
axs[0].plot(t_after_dv, states_after_dv[:,0]/1000, 'b-', label='Post-Manoeuvre Orbit')
axs[0].plot(t_prop, states_prop[:, 0]/1000, 'r:', label='Pre-Manoeuvre Orbit')
axs[0].grid()
axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
axs[0].set_ylabel('Inertial X Position [km]')

axs[1].scatter(tk, r_eci[:,1]/1000, color='green', marker='o')
axs[1].plot(t_after_dv, states_after_dv[:,1]/1000, 'b-')
axs[1].plot(t_prop, states_prop[:, 1]/1000, 'r:')
axs[1].grid()
axs[1].set_ylabel('Inertial Y Position [km]')

axs[2].scatter(tk, r_eci[:,2]/1000, color='green', marker='o')
axs[2].plot(t_after_dv, states_after_dv[:,2]/1000, 'b-')
axs[2].plot(t_prop, states_prop[:, 2]/1000, 'r:')
axs[2].grid()
axs[2].set_ylabel('Inertial Z Position [km]')

plt.xlabel('Time since J2000 [s]')
plt.tight_layout()
plt.savefig('plots/Q2_post_dv_orbit.png')
plt.show()


######################################### HISTOGRAMS #########################################

# Set a style for the plots using seaborn
sns.set_theme(style='whitegrid')
colors = sns.color_palette("muted")[:3]

# Plotting histograms for states
plt.figure(figsize=(12, 5))

# Histograms for position and velocity components
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.title(f"{['X', 'Y', 'Z'][i]} Position Component", fontsize=16)
    plt.xlabel("Value", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)

    # Plot histogram with edges
    counts, bins, patches = plt.hist(states_at_dv[:, i], bins=30, alpha=0.5, edgecolor='black',
                                     color=colors[0])

    # Calculate mean and covariance
    mean_value = np.mean(states_at_dv[:, i])
    std_dev = np.std(states_at_dv[:, i])

    # Add mean line
    plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label='Mean' if i==0 else None)

    # Add covariance bounds
    plt.axvline(mean_value + std_dev, color='orange', linestyle=':', linewidth=2, label="$1\sigma$" if i==0 else None)
    plt.axvline(mean_value - std_dev, color='orange', linestyle=':', linewidth=2)
    if i==0:
        plt.legend()
    plt.grid()

plt.tight_layout()
plt.savefig('plots/Q2_post_dv_state_r_histogram.png')
plt.show()

plt.figure(figsize=(12, 5))

# Bottom 3 for Velocity
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.title(f"{['X', 'Y', 'Z'][i]} Velocity Component", fontsize=16)
    plt.xlabel("Value", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)

    # Plot histogram
    counts, bins, patches = plt.hist(states_at_dv[:, i + 3], bins=30, alpha=0.5, edgecolor='black',
                                     color=colors[1])

    # Calculate mean and covariance
    mean_value = np.mean(states_at_dv[:, i + 3])
    std_dev = np.std(states_at_dv[:, i + 3])

    # Add mean line
    plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label='Mean' if i==0 else None)

    # Add covariance bounds
    plt.axvline(mean_value + std_dev, color='orange', linestyle=':', linewidth=2, label="$1\sigma$" if i==0 else None)
    plt.axvline(mean_value - std_dev, color='orange', linestyle=':', linewidth=2)
    if i==0:
        plt.legend()
    plt.grid()

plt.tight_layout()
plt.savefig('plots/Q2_post_dv_state_v_histogram.png')
plt.show()

# Plotting histogram for delta velocity components
plt.figure(figsize=(12, 5))

# Create subplots for each delta velocity component
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.title(f"Delta-V {['X', 'Y', 'Z'][i]} Component", fontsize=16)
    plt.xlabel("Value", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)

    # Plot histogram with edges
    counts, bins, patches = plt.hist(dv_array[:, i], bins=30, alpha=0.5, edgecolor='black',
                                     color=colors[2])

    # Calculate mean and covariance
    mean_value = np.mean(dv_array[:, i])
    std_dev = np.std(dv_array[:, i])

    # Add mean line
    plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label='Mean' if i==0 else None)

    # Add covariance bounds
    plt.axvline(mean_value + std_dev, color='orange', linestyle=':', linewidth=2, label="$1\sigma$" if i==0 else None)
    plt.axvline(mean_value - std_dev, color='orange', linestyle=':', linewidth=2)
    if i==0:
        plt.legend()
    plt.grid()

plt.tight_layout()
plt.savefig('plots/Q2_dv_histogram.png')
plt.show()

# Plot correlation matrices
labels = ['$r_x$', '$r_y$', '$r_z$', '$v_x$', '$v_y$', '$v_z$']
plt.figure()
sns.heatmap(np.abs(corr_state_at_dv), xticklabels=labels, yticklabels=labels,annot=False, center=0, square=True, linewidths=.5)
plt.tight_layout()
plt.savefig('plots/Q2_corr_state_at_dv.png')
plt.show()

labels = ['$\Delta v_x$', '$\Delta v_y$', '$\Delta v_z$']
plt.figure()
sns.heatmap(np.abs(corr_dv), xticklabels=labels, yticklabels=labels,annot=False, center=0, square=True, linewidths=.5)
plt.tight_layout()
plt.savefig('plots/Q2_corr_dv.png')
plt.show()

####################################################################################################
############################################## PART C ##############################################
####################################################################################################

# Time of dv: tk[122]

# State after dv: mean_state_at_dv

# Covariance of state after dv: cov_state_at_dv
