# Import statements
import pickle
import EstimationUtilities as EstUtil
import os
import numpy as np
from astropy.time import Time
import astropy.units as u

# Set up functions
def cel2eci(y_cel_list):
    '''
    This function converts celestial coordinates to ECI
    Parameters
    ----------
    y_cel : list of states in celestial frame

    Returns
    -------
    Array of ECI coordinates
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

    for i, t in enumerate(times):
        # Get GMST in radians
        gmst = t.sidereal_time('mean', 'greenwich').rad

        # Rotate around Z-axis by GMST
        rot_matrix = np.array([
            [np.cos(gmst), -np.sin(gmst), 0],
            [np.sin(gmst), np.cos(gmst), 0],
            [0, 0, 1]
        ])

        eci_coords[i, :] = np.matmul(rot_matrix, ecef_coords)

    return eci_coords


# Set directory for assignment data
cwd = os.getcwd()
assignment_data_directory = os.path.join(cwd, 'data\group4')

# Load measurement data for estimation case
meas_file = os.path.join(assignment_data_directory, 'q2_meas_maneuver_91267.pkl')
state_params, meas_dict, sensor_params = EstUtil.read_measurement_file(meas_file)

# Step 1: Convert measured states from celestial coordinates to ECI
# Extract list of measurements
Yk_list = meas_dict['Yk_list']
tk_list = meas_dict['tk_list']
# Convert using predefined functions
Yk_eci_list = cel2eci(Yk_list)

# Step 2: Convert sensor position from ECEF to ECI frame
sensor_ecef = sensor_params['sensor_ecef'].flatten()
sensor_eci = ecef2eci(sensor_ecef, tk_list)

# Step 3: Get absolute position of object
r_eci = Yk_eci_list + sensor_eci

