import numpy as np
from astropy.coordinates.earth_orientation import eccentricity
import tudatpy.astro.element_conversion as ec

def apogee_perigee_filter(X1, X2, D):
    """
    Apogee-Perigee Filter to check if two objects can be dismissed.
    X1, X2: State vectors (6x1 numpy array) - [x, y, z, vx, vy, vz]
    D: Distance threshold (float)
    """
    # Convert Cartesian states to Keplerian elements
    kep1 = ec.cartesian_to_keplerian(X1, 3.986004418e14)
    kep2 = ec.cartesian_to_keplerian(X2, 3.986004418e14)

    # print(f"kep1: {kep1}, kep2: {kep2}")

    # Extract perigee and apogee
    q1 = kep1[0] * (1 - kep1[1])
    Q1 = kep1[0] * (1 + kep1[1])
    q2 = kep2[0] * (1 - kep2[1])
    Q2 = kep2[0] * (1 + kep2[1])

    # Define the bigger of the two perigees
    q = max(q1, q2)
    # Define the smaller of the two apogees
    Q = min(Q1, Q2)
    #print(f"q:{q}, Q: {Q},q - Q: {q-Q}")

    # Check if the objects can be filtered out
    if q - Q > D:
        return True  # Filter out
    return False  # Keep for further analysis


def geometrical_filter(X1, X2, D):
    """
    Geometrical Filter to check if two objects can be dismissed.
    X1, X2: State vectors (6x1 numpy array) - [x, y, z, vx, vy, vz]
    D: Distance threshold (float)
    """
    #todo: add exception of the coplanar method

    # Convert Cartesian states to Keplerian elements
    kep1 = ec.cartesian_to_keplerian(X1, 3.986004418e14)
    kep2 = ec.cartesian_to_keplerian(X2, 3.986004418e14)

    # Extract inclination difference
    IR = np.abs(kep1[2] - kep2[2])

    # Relative distance and velocity
    rp = np.linalg.norm(X1[:3])
    rs = np.linalg.norm(X2[:3])

    # True anomalies and argument of perigee
    urp = kep1[5] + kep1[4] - kep1[3]
    urs = kep2[5] + kep2[4] - kep2[3]

    # Compute the cosine of gamma
    cos_gamma = np.cos(urp) * np.cos(urs) + np.sin(urp) * np.sin(urs) * np.cos(IR)

    # Calculate the relative distance squared
    r_rel_sq = rp**2 + rs**2 - 2 * rp * rs * cos_gamma

    # Check if the relative distance is less than D
    if np.sqrt(r_rel_sq) < D:
        return False  # Keep for further analysis
    return True  # Filter out


def time_filter(X1, X2, D):
    """
    Time Filter to check if two objects can be dismissed.
    X1, X2: State vectors (6x1 numpy array) - [x, y, z, vx, vy, vz]
    D: Distance threshold (float)
    """
    mu = 3.986004418e14

    # Convert Cartesian states to Keplerian elements
    kep1 = ec.cartesian_to_keplerian(X1, mu)
    kep2 = ec.cartesian_to_keplerian(X2, mu)

    # Extract inclination difference
    IR = np.abs(kep1[2] - kep2[2])

    # Semi-major axes and eccentricities
    a_1, e_1 = kep1[0], kep1[1]

    # Constants for object one
    alpha_1 = a_1 * (1 - e_1**2) * np.sin(IR)
    ax_1 = e_1 * np.cos(kep1[4] - kep1[3])
    ay_1 = e_1 * np.sin(kep1[4] - kep1[3])
    Q_1 = alpha_1 * (alpha_1 - 2*D*ay_1) - (1-e_1**2)*D**2

    to_solve_1 = (-D**2*ax_1 + (alpha_1-D*ay_1)*np.sqrt(Q_1))/(alpha_1*(alpha_1 - 2*D*ay_1)+D**2*e_1**2)
    to_solve_2 = (-D**2*ax_1 - (alpha_1-D*ay_1)*np.sqrt(Q_1))/(alpha_1*(alpha_1 - 2*D*ay_1)+D**2*e_1**2)
    if Q_1 < 0 or np.abs(to_solve_1) > 1 or np.abs(to_solve_2) > 1 :
        return False
    ur1 = np.arccos(to_solve_1)
    ur2 = np.arccos(to_solve_2)

    ######  Object two #######
    a_2, e_2 = kep2[0], kep2[1]

    # Constants
    alpha_2 = a_2*(1 - e_2 ** 2) * np.sin(IR)
    ax_2 = e_2 * np.cos(kep2[4] - kep2[3])
    ay_2 = e_2 * np.sin(kep2[4] - kep2[3])
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
    f1 = ur1 - kep1[4] - kep1[3]
    f2 = ur2 - kep1[4] - kep1[3]
    f3 = ur3 - kep2[4] - kep2[3]
    f4 = ur4 - kep2[4] - kep2[3]


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
