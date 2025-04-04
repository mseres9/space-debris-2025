import numpy as np
from astropy.coordinates.earth_orientation import eccentricity
import tudatpy.astro.element_conversion as ec

def apogee_perigee_filter(rso1, rso2, D):
    """
    Apogee-Perigee Filter to check if two objects can be dismissed.
    """
    # Extract perigee and apogee of the primary and secondary object
    q1 = rso1['perigee']
    Q1 = rso1['apogee']
    q2 = rso2['perigee']
    Q2 = rso2['apogee']

    # Largest perigee and smallest apogee
    q = max(q1, q2)
    Q = min(Q1, Q2)

    # Check if the objects can be filtered out
    if q - Q > D:
        return True  # Filter out
    return False  # Keep for further analysis


def geometrical_filter(rso1, rso2, D):
    """
    Geometrical Filter to check if two objects can be dismissed.
    """
    # Extract position vectors and inclination difference
    rp = np.linalg.norm(rso1['position'])
    rs = np.linalg.norm(rso2['position'])
    IR = np.abs(rso1['inclination'] - rso2['inclination'])

    # Calculate uR for both objects
    urp = rso1['true_anomaly'] + rso1['argument_of_perigee'] - rso1['raan']
    urs = rso2['true_anomaly'] + rso2['argument_of_perigee'] - rso2['raan']

    # Compute the cosine of gamma
    cos_gamma = np.cos(urp) * np.cos(urs) + np.sin(urp) * np.sin(urs) * np.cos(IR)

    # Calculate the relative distance squared
    r_rel_sq = rp**2 + rs**2 - 2 * rp * rs * cos_gamma

    # Check if the relative distance is less than D
    if np.sqrt(r_rel_sq) < D:
        return False  # Keep for further analysis
    return True  # Filter out


def time_filter(rso1, rso2, D):
    IR = np.abs(rso1['inclination'] - rso2['inclination'])
    mu = 3.986 * 10 ** 14
    ######  Object one #######
    a_1 = (rso1['perigee'] + rso1['apogee'])/2
    e_1 = (rso1['perigee'] - rso1['apogee'])/(rso1['perigee'] + rso1['apogee'])

    #Constants
    alpha_1 = a_1* (1-e_1**2) * np.sin(IR)
    ax_1 = e_1 * np.cos(rso1['argument_of_perigee']- rso1['raan'])
    ay_1 = e_1 * np.sin(rso1['argument_of_perigee']- rso1['raan'])
    Q_1 = alpha_1 * (alpha_1 - 2*D*ay_1) - (1-e_1**2)*D**2

    to_solve_1 = (-D**2*ax_1 + (alpha_1-D*ay_1)*np.sqrt(Q_1))/(alpha_1*(alpha_1 - 2*D*ay_1)+D**2*e_1**2)
    to_solve_2 = (-D**2*ax_1 - (alpha_1-D*ay_1)*np.sqrt(Q_1))/(alpha_1*(alpha_1 - 2*D*ay_1)+D**2*e_1**2)
    if Q_1 < 0 or np.abs(to_solve_1) > 1 or np.abs(to_solve_2) > 1 :
        return False
    ur1 = np.arccos(to_solve_1)
    ur2 = np.arccos(to_solve_2)

    ######  Object two #######
    a_2 = (rso2['perigee'] + rso2['apogee']) / 2
    e_2 = (rso2['perigee'] - rso2['apogee']) / (rso2['perigee'] + rso2['apogee'])

    # Constants
    alpha_2 = a_2*(1 - e_2 ** 2) * np.sin(IR)
    ax_2 = e_2 * np.cos(rso2['argument_of_perigee'] - rso2['raan'])
    ay_2 = e_2 * np.sin(rso2['argument_of_perigee'] - rso2['raan'])
    Q_2 = alpha_2 * (alpha_2 - 2 * D * ay_2) - (1 - e_2 ** 2) * D ** 2


    to_solve_3 = (-D ** 2 * ax_2 + (alpha_2 - D * ay_2) * np.sqrt(Q_2)) / (
                alpha_2 * (alpha_2 - 2 * D * ay_2) + D ** 2 * e_2 ** 2)
    to_solve_4 = (-D ** 2 * ax_2 - (alpha_2 - D * ay_2) * np.sqrt(Q_2)) / (
                alpha_2 * (alpha_2 - 2 * D * ay_2) + D ** 2 * e_2 ** 2)

    if Q_2 < 0 or np.abs(to_solve_3) > 1 or np.abs(to_solve_4) > 1 :
        return False

    ur3 = np.arccos(to_solve_3)
    ur4 = np.arccos(to_solve_4)

    #
    f1 = ur1 - rso1['argument_of_perigee'] - rso1['raan']
    f2 = ur2 - rso1['argument_of_perigee'] - rso1['raan']
    f3 = ur3 - rso2['argument_of_perigee'] - rso2['raan']
    f4 = ur4 - rso2['argument_of_perigee'] - rso2['raan']


    M1 = ec.true_to_mean_anomaly(e_1,f1)
    t1 = ec.delta_mean_anomaly_to_elapsed_time(M1,mu,a_1*10**3)

    M2 = ec.true_to_mean_anomaly(e_1, f2)
    t2 = ec.delta_mean_anomaly_to_elapsed_time(M2, mu, a_1*10**3)

    M3 = ec.true_to_mean_anomaly(e_2, f3)
    t3 = ec.delta_mean_anomaly_to_elapsed_time(M3, mu, a_2*10**3)

    M4 = ec.true_to_mean_anomaly(e_2, f4)
    t4 = ec.delta_mean_anomaly_to_elapsed_time(M4, mu, a_2*10**3)

    T1 = 2*np.pi*np.sqrt((a_1*10**3)**3/mu)
    T2 = 2*np.pi*np.sqrt((a_2*10**3)**3/mu)

    n = 0
    while n*T1 < 48 * 3600:
        if max(T1*n + t1,T2*n + t3) < min(T1*n + t2,T2*n +t4):
            return False
        else:
            n = n + 1
    return True

def filter_object_pairs(rso_catalog, D):
    """
    Apply filtering to object pairs in the catalog.
    """
    filtered_pairs = []

    # Iterate through all combinations of primary and secondary objects
    object_ids = list(rso_catalog.keys())
    for i in range(len(object_ids)):
        for j in range(i + 1, len(object_ids)):
            rso1 = rso_catalog[object_ids[i]]
            rso2 = rso_catalog[object_ids[j]]

            # Apply Apogee-Perigee Filter
            if apogee_perigee_filter(rso1, rso2, D):
                continue  # Skip this pair

            # Apply Geometrical Filter
            if geometrical_filter(rso1, rso2, D):
                continue  # Skip this pair

            # If both filters fail, keep the pair
            filtered_pairs.append((object_ids[i], object_ids[j]))

    print(f"Filtered down to {len(filtered_pairs)} object pairs for TCA assessment.")
    return filtered_pairs
