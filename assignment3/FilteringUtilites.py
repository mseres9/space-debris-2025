
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
