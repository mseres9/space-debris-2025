# Import statements
import pickle
import EstimationUtilities as EstUtil
import os
import numpy as np
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt

# Set up functions
def cel2ecef(y_cel_list):
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
    y_ecef_list = np.zeros((len(y_cel_list), 3))

    # Loop through entries and convert coordinates
    for i, y_cel in enumerate(y_cel_list):
        # Extract celestial coordinates
        r = y_cel[0, 0]
        ra = y_cel[1, 0]
        dec = y_cel[2, 0]
        # Calculate ECI coordinates
        y_eci = r * np.array([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])
        y_ecef_list[i, :] = y_eci

    return y_ecef_list


def ecef2eci(ecef_coords, times_j2000_s):
    """
    Convert coordinates from ECEF to ECI frame using sidereal time rotation.

    Parameters
    ----------
    ecef_coords : array
        N x 3 array of ECEF coordinates in meters
    times_j2000_s : list or array
        N-element array of times in seconds since J2000 epoch (2000-01-01 12:00:00 TDB)

    Returns
    -------
    eci_coords : array
        N x 3 array of ECI coordinates in meters
    """
    # Convert seconds since J2000 to Astropy Time object
    j2000 = Time('2000-01-01 12:00:00', scale='tdb')
    times = j2000 + times_j2000_s * u.s

    eci_coords = np.zeros((len(times), 3))

    for i, (ecef, t) in enumerate(zip(ecef_coords, times)):
        # Get GMST in radians
        gmst = t.sidereal_time('mean', 'greenwich').rad

        # Rotate around Z-axis by GMST
        rot_matrix = np.array([
            [np.cos(gmst), -np.sin(gmst), 0],
            [np.sin(gmst), np.cos(gmst), 0],
            [0, 0, 1]
        ])

        eci_coords[i, :] = np.matmul(rot_matrix, ecef)

    return eci_coords


# Set directory for assignment data
cwd = os.getcwd()
assignment_data_directory = os.path.join(cwd, 'data\group4')

# Load measurement data for estimation case
meas_file = os.path.join(assignment_data_directory, 'q2_meas_maneuver_91267.pkl')
state_params, meas_dict, sensor_params = EstUtil.read_measurement_file(meas_file)

# Step 1: Convert measured states from celestial coordinates to ECEF
# Extract list of measurements
Yk_list = meas_dict['Yk_list']
tk_list = meas_dict['tk_list']
# Convert using predefined functions
Yk_ecef_list = cel2ecef(Yk_list)

# Step 2: Convert object positions and sensor position from ECEF to ECI frame
sensor_ecef = sensor_params['sensor_ecef'].flatten()
r_ecef = sensor_ecef + Yk_ecef_list
r_eci = ecef2eci(r_ecef, tk_list)


# Set global font size
plt.rcParams.update({'font.size': 11})

# Create figure
fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# First subplot
axs[0].scatter(tk_list-tk_list[0], r_eci[:, 0]/1000, color='blue')
axs[0].set_title('x component')
axs[0].grid()
axs[0].set_ylabel('x [km]')

# Second subplot
axs[1].scatter(tk_list-tk_list[0], r_eci[:, 1]/1000, color='green')
axs[1].set_title('y component')
axs[1].grid()
axs[1].set_ylabel('y [km]')

# Third subplot
axs[2].scatter(tk_list-tk_list[0], r_eci[:, 2]/1000, color='red')
axs[2].set_title('z component')
axs[2].grid()
axs[2].set_ylabel('z [km]')

# Add a common x-label
plt.xlabel('Time since start (s)')
# Adjust layout to prevent overlap
plt.tight_layout()
# Show plot
plt.show()