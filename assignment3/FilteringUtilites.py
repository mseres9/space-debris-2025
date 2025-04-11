import numpy as np
from astropy.coordinates.earth_orientation import eccentricity
import tudatpy.astro.element_conversion as ec
from pandas.core.nanops import na_accum_func
from breakup_modeling import *
from breakup_modeling.BreakupUtilities import *



def pertubations(X):
    """
    Calculates mean change over orbit of keplerian elements.
    X State vector (6x1 numpy array) - [x, y, z, vx, vy, vz]
    """
    state_params = {
                    'mass':  100.0,
                    'area': 1.0,
                    'Cd':  2.2,
                    'Nquad': 8,
                }
    dX = int_salt_grav_drag(X,state_params)
    return dX
def compute_dr2dt(dX1,dX2,rp,rs,kep1,kep2,IR,cos_gamma,urp,urs):
    mu = 3.986004418e14
    #Determine mean motion and n dot for object 1
    n1 = np.sqrt(mu / kep1[0] ** 3)
    ndot1 = -3 / 2 * dX1[0] * np.sqrt(mu / kep1[0] ** 5)
    #Determine mean motion and n dot for object 2
    n2 = np.sqrt(mu / kep2[0] ** 3)
    ndot2 = -3 / 2 * dX2[0] * np.sqrt(mu / kep2[0] ** 5)

    rp_dot = 2/3 * ndot1/n1 *(kep1[0] * (1 - kep1[1]) * np.cos(kep1[5]) - rp )
    rs_dot = 2/3 * ndot2/n2 *(kep2[0] * (1 - kep2[1]) * np.cos(kep2[5]) - rs )

    urp_dot = 1 / (1 - kep1[1]**2) * (2 + kep1[1] * np.cos(kep1[5])) * np.sin(kep1[5] * dX1[1]) + dX1[4]
    urs_dot = 1 / (1 - kep2[1]**2) * (2 + kep2[1] * np.cos(kep2[5])) * np.sin(kep2[5] * dX2[1]) + dX2[4]

    dr2dt = (2 * rp_dot * (rp - rs * cos_gamma) +
        2 * rs_dot * (rs - rs * cos_gamma) -
        2 * rp * rs * ( (np.cos(urp) * np.sin(urs) * np.cos(IR) - np.sin(urp) * np.cos(urs) ) * urp_dot +
                       (np.sin(urp) * np.cos(urs) * np.cos(IR) - np.cos(urp) * np.sin(urs) ) * urs_dot   -
                       (np.sin(urp) * np.sin(urs) * np.sin(IR) ))

    )
    return dr2dt


def new_period(K,kep,dX,deltadot):
    """
    Calculates new orbital period
    K: number of revolutions
    kep: keplerian state (6x1 numpy array)
    dX: Mean change over orbit  (5x1 numpy array) - [dadt, dedt, didt, draandt, daop/dt]
    deltadot: change in delta angle
    """
    Re = 6371 * 10 ** 3
    mu = 3.986004418e14
    J2 = -0.108 * 10 ** (-2)

    a = kep[0]
    e = kep[1]
    i = kep[2]

    n = np.sqrt(mu/a**3)

    ndot = -3/2 * dX[0] * np.sqrt(mu/a**5)
    Mdot = 3/4 * J2 * n * (Re**2)/(a**2 * (1-e**2)**(3/2)) * (3 * np.cos(i)**2 - 1)
    aopdot = dX[4]

    TDF = 2*np.pi / (n + Mdot + aopdot - deltadot)
    T = TDF * (1 - 2 * np.pi/n * ndot/n * K )
    return T

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
    else:
        return False
    #return False  # Keep for further analysis


def calc_ur(IR,kep1,kep2):
    A = 1/np.sin(IR) * (np.sin(kep2[2]) * np.sin(kep1[4] - kep2[4]))
    B = 1/np.sin(IR) * ( np.sin(kep1[2]) * np.cos(kep2[2]) - np.sin(kep2[2]) * np.cos(kep1[2])  * np.cos(kep1[4] - kep2[4])  )
    deltap = np.arctan(A / B)

    C = 1 / np.sin(IR) * (np.sin(kep1[2]) * np.sin(kep1[4] - kep2[4]))
    d = 1 / np.sin(IR) * (np.sin(kep1[2]) * np.cos(kep2[2]) * np.cos(kep1[4] - kep2[4]) - np.sin(kep2[2]) * np.cos(kep1[2]) )
    deltas = np.arctan(C / d)

    urp = kep1[5] + kep1[3] - deltap
    urs = kep2[5] + kep2[3] - deltas
    return urp, urs, deltap, deltas


def geometrical_filter(X1, X2, Dthres):
    """
    Geometrical Filter to check if two objects can be dismissed.
    X1, X2: State vectors (6x1 numpy array) - [x, y, z, vx, vy, vz]
    Dthres: Distance threshold (float)
    """
    #todo: add exception of the coplanar method

    def Newton(kep1, kep2,urp,urs,fp,fs,rp,rs,cos_gamma):
        #Eccentricity
        ep = kep1[1]
        es = kep2[1]

        #Eccentric anomaly
        Ep = ec.true_to_eccentric_anomaly(fp,ep)
        Es = ec.true_to_eccentric_anomaly(fs, es)

        #Constants ax and ay
        axp = ep * np.cos(kep1[3]-  kep1[4])
        ayp = ep * np.sin(kep1[3] - kep1[4])

        axs = es * np.cos(kep2[3] - kep2[4])
        ays = es * np.sin(kep2[3] - kep2[4])

        #Constant A to G
        A = np.sin(urp) + ayp
        B = np.cos(urp) + axp
        C = np.sin(urs) + ays
        D = np.cos(urs) + axs
        F = rp * ep * np.sin(fp) + rs * ( A * np.cos(urs) - B * np.cos(IR) * np.sin(urs))
        G = rs * es * np.sin(fs) + rp * ( C * np.cos(urp) - D * np.cos(IR) * np.sin(urp))
        Ffp = rp * ep * np.cos(Ep) + rs * cos_gamma
        Ffs = - rs / (1 + es * np.cos(fs)) * (A * C + B * D * np.cos(IR))
        Gfp = - rp / (1 + ep * np.cos(fp)) * (A * C + B * D * np.cos(IR))
        Gfs = rs * es * np.cos(Es) + rp * cos_gamma

        #Compute true anomaly increments
        h = (F * Gfs - G * Ffs) / (Ffs * Gfp - Ffp * Gfs)
        k = (G * Ffp - F * Gfp) / (Ffs * Gfp - Ffp * Gfs)

        #Update fp and fs
        fp = fp + h
        fs = fs + k

        return fp, fs, h , k

    # Convert Cartesian states to Keplerian elements
    kep1 = ec.cartesian_to_keplerian(X1, 3.986004418e14)
    kep2 = ec.cartesian_to_keplerian(X2, 3.986004418e14)

    # Compute vector normal to orbital planes
    ws = np.array([np.sin(kep1[4]) * np.cos(kep1[2]),np.cos(kep1[4]) * np.sin(kep1[2]),np.cos(kep1[2])])
    wp = np.array([np.sin(kep2[4]) * np.cos(kep2[2]),np.cos(kep2[4]) * np.sin(kep2[2]),np.cos(kep2[2])])
    K = np.cross(ws, wp)
    if np.linalg.norm(K) > 1:
        return False

    # Inclination of the two orbital planes
    IR = np.arcsin(np.linalg.norm(K))

    # Extract distance and velocity
    rp = np.linalg.norm(X1[:3])
    rs = np.linalg.norm(X2[:3])

    # Calculate urp and urs
    urp = calc_ur(IR, kep1, kep2)[0]
    urs = calc_ur(IR, kep1, kep2)[1]

    # Calculate urp and urs
    deltap = calc_ur(IR, kep1, kep2)[2]
    deltas = calc_ur(IR, kep1, kep2)[3]

    #Initial true anomalies
    fp = 0
    fs = 0
    list_fp = [deltap - kep1[3],fp + np.pi]
    list_fs = [deltas - kep2[3], fs + np.pi]
    #
    list_r_rel = []

    # Compute the cosine of gamma
    cos_gamma = np.cos(urp) * np.cos(urs) + np.sin(urp) * np.sin(urs) * np.cos(IR)

    for i in range(len(list_fp)):
        #Iteration loop using Newton's method
        not_converged = True
        iterations = 0
        fp = list_fp[i]
        fs = list_fs[i]
        while not_converged:

            #Compute new true anomalies
            fp = Newton(kep1, kep2, urp, urs, fp, fs,rp,rs,cos_gamma)[0]
            fs = Newton(kep1, kep2, urp, urs, fp, fs,rp,rs,cos_gamma)[1]

            #Extract increments
            h = Newton(kep1, kep2, urp, urs, fp, fs,rp,rs,cos_gamma)[2]
            k = Newton(kep1, kep2, urp, urs, fp, fs,rp,rs,cos_gamma)[3]

            #Udpate keplerian state with new true anomaly
            kep1[5] = fp
            kep2[5] = fs

            #Update urs and urp
            urp = calc_ur(IR, kep1, kep2)[0]
            urs = calc_ur(IR, kep1, kep2)[1]


            #Updtae rs and rp
            rp = kep1[0] * (1 - kep1[1] ** 2) / (1 + kep1[1] * np.cos(kep1[5]))
            rs = kep2[0] * (1 - kep2[1] ** 2) / (1 + kep2[1] * np.cos(kep2[5]))

            # Update the cosine of gamma
            cos_gamma = np.cos(urp) * np.cos(urs) + np.sin(urp) * np.sin(urs) * np.cos(IR)

            #Start iterations
            iterations += 1
            tol = 0.001
            if iterations > 100:
                #print("Too many iterations")
                not_converged = False
            if h < tol * np.pi/180 and k < tol * np.pi/180:
                #print("New object")
                not_converged = False

        # Calculate the relative distance squared
        r_rel_sq = rp ** 2 + rs ** 2 - 2 * rp * rs * cos_gamma
        use_perturbations = False
        if use_perturbations:
            dX1 = pertubations(kep1)
            dX2 = pertubations(kep2)
            dr2dt = compute_dr2dt(dX1, dX2, rp, rs, kep1, kep2, IR, cos_gamma, urp, urs)
            r_rel_sq_1 = r_rel_sq + dr2dt * 24 * 3600
            r_rel_sq_2 = r_rel_sq - dr2dt * 24 * 3600


            list_r_rel.append(r_rel_sq_1)
            list_r_rel.append(r_rel_sq_2)
        else:
            list_r_rel.append(r_rel_sq)


    # Check if the relative distance is less than D
    if all(x>Dthres**2 for x in list_r_rel):
        return True  #Filter out
    else:

        return False # Keep for further analysis


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

    # Compute vector normal to orbital planes
    ws = np.array([np.sin(kep1[4]) * np.cos(kep1[2]), np.cos(kep1[4]) * np.sin(kep1[2]), np.cos(kep1[2])])
    wp = np.array([np.sin(kep2[4]) * np.cos(kep2[2]), np.cos(kep2[4]) * np.sin(kep2[2]), np.cos(kep2[2])])
    K = np.cross(ws, wp)

    #If objects are coplanar eliminate them
    if np.linalg.norm(K) > 1:
        return False

    # Inclination of the two orbital planes
    IR = np.arcsin(np.linalg.norm(K))

    # Semi-major axis and eccentricity
    a_1, e_1 = kep1[0], kep1[1]

    #Constants for object one
    alpha_1 = a_1 * (1 - e_1**2) * np.sin(IR)
    ax_1 = e_1 * np.cos(kep1[4] - kep1[3])
    ay_1 = e_1 * np.sin(kep1[4] - kep1[3])
    Q_1 = alpha_1 * (alpha_1 - 2*D*ay_1) - (1-e_1**2)*D**2

    # Test from Hoots
    if Q_1 < 0:
        return False

    #To simplify calculations
    to_solve_1 = (-D**2*ax_1 + (alpha_1-D*ay_1)*np.sqrt(Q_1))/(alpha_1*(alpha_1 - 2*D*ay_1)+D**2*e_1**2)
    to_solve_2 = (-D**2*ax_1 - (alpha_1-D*ay_1)*np.sqrt(Q_1))/(alpha_1*(alpha_1 - 2*D*ay_1)+D**2*e_1**2)

    if np.abs(to_solve_1) > 1 or np.abs(to_solve_2) > 1:
        return False



    #Determine ur1 and ur2
    ur1 = np.arccos(to_solve_1)
    ur2 = np.arccos(to_solve_2)

    ##### Repeat for object two ####

    # Semi-major axis and eccentricity
    a_2, e_2 = kep2[0], kep2[1]

    # Constants for object two
    alpha_2 = a_2*(1 - e_2 ** 2) * np.sin(IR)
    ax_2 = e_2 * np.cos(kep2[4] - kep2[3])
    ay_2 = e_2 * np.sin(kep2[4] - kep2[3])
    Q_2 = alpha_2 * (alpha_2 - 2 * D * ay_2) - (1 - e_2 ** 2) * D ** 2

    if Q_2 < 0:
        return False

    #To simplify calculations
    to_solve_3 = (-D ** 2 * ax_2 + (alpha_2 - D * ay_2) * np.sqrt(Q_2)) / (alpha_2 * (alpha_2 - 2 * D * ay_2) + D ** 2 * e_2 ** 2)
    to_solve_4 = (-D ** 2 * ax_2 - (alpha_2 - D * ay_2) * np.sqrt(Q_2)) / (alpha_2 * (alpha_2 - 2 * D * ay_2) + D ** 2 * e_2 ** 2)

    #Test from Hoots
    if Q_2 < 0 or np.abs(to_solve_3) > 1 or np.abs(to_solve_4) > 1 :
        return False

    # Determine ur3 and ur4
    ur3 = np.arccos(to_solve_3)
    ur4 = np.arccos(to_solve_4)

    #Determine true anomalies window
    f1 = ur1 - kep1[4] - kep1[3]
    f2 = ur2 - kep1[4] - kep1[3]
    f3 = ur3 - kep2[4] - kep2[3]
    f4 = ur4 - kep2[4] - kep2[3]

    #Transform into time
    M1 = ec.true_to_mean_anomaly(e_1,f1)
    t1 = ec.delta_mean_anomaly_to_elapsed_time(M1,mu,a_1*10**3)
    #
    M2 = ec.true_to_mean_anomaly(e_1, f2)
    t2 = ec.delta_mean_anomaly_to_elapsed_time(M2, mu, a_1*10**3)
    #
    M3 = ec.true_to_mean_anomaly(e_2, f3)
    t3 = ec.delta_mean_anomaly_to_elapsed_time(M3, mu, a_2*10**3)
    #
    M4 = ec.true_to_mean_anomaly(e_2, f4)
    t4 = ec.delta_mean_anomaly_to_elapsed_time(M4, mu, a_2*10**3)

    #Compute period of two objects
    T1 = 2*np.pi*np.sqrt((a_1*10**3)**3/mu)
    T2 = 2*np.pi*np.sqrt((a_2*10**3)**3/mu)

    #Compute delta p and s
    deltap = calc_ur(IR, kep1, kep2)[2]
    deltas = calc_ur(IR, kep1, kep2)[3]

    #Begin iteration to check whether time interval overlap
    K = 0
    dt = 0
    while dt < 48 * 3600:
        if max(T1*K + t1, T2*K + t3) < min(T1*K + t2, T2*K +t4):
            return False #Keep for further analysis
        else:
            K = K + 1
            dt = dt + max(T1, T2)
            use_pertubations = True
            if use_pertubations:
                dX1 = pertubations(kep1)
                dX2 = pertubations(kep2)

                deltadot1 = 1 / np.sin(IR) * np.sin(kep2[2]) * np.cos(deltas) * (dX1[3]-dX2[3])
                deltadot2 = 1 / np.sin(IR) * np.sin(kep1[2]) * np.cos(deltap) * (dX1[3]-dX2[3])

                T1 = new_period(K,kep1,dX1,deltadot1)
                T2 = new_period(K,kep2,dX2,deltadot2)



    return True #Filter out

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
