import pickle
import os
import numpy as np
from scipy.optimize import fsolve
from astropy.constants import G, M_earth

# Load Data
current_dir = os.getcwd()
with open(os.path.join(current_dir, 'q4_meas_iod_99004.pkl'), 'rb') as f:
    data = pickle.load(f)

# Constants
mu = G.value * M_earth.value  # Earth's gravitational parameter (m^3/s^2)

# Extract data
tk_list = data[2]["tk_list"]          # Time list
Yk_list = data[2]["Yk_list"]          # Measurement list
sensor_ecef = data[1]['sensor_ecef'].flatten()  # Sensor position in ECEF

# Extract first measurement
rg, ra, dec = Yk_list[0].flatten()

# Convert measurement to ECI position
def ra_dec_to_eci(rg, ra, dec, sensor_ecef):
    # Convert to Cartesian relative to sensor
    r_rel = rg * np.array([
        np.cos(ra) * np.cos(dec),
        np.sin(ra) * np.cos(dec),
        np.sin(dec)
    ])
    # Approximate ECEF = ECI (neglecting Earth's rotation here)
    r_eci = sensor_ecef + r_rel
    return r_eci

r_eci_1 = ra_dec_to_eci(rg, ra, dec, sensor_ecef)

# Estimate velocity from second observation
rg2, ra2, dec2 = Yk_list[1].flatten()
r_eci_2 = ra_dec_to_eci(rg2, ra2, dec2, sensor_ecef)
dt = tk_list[1] - tk_list[0]
v_eci_1 = (r_eci_2 - r_eci_1) / dt

# Convert Cartesian state to Keplerian orbital elements
def cartesian_to_keplerian(r, v, mu):
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    e_vec = (np.cross(v, h) / mu) - (r / r_norm)
    e = np.linalg.norm(e_vec)
    a = 1 / ((2 / r_norm) - (v_norm**2 / mu))
    i = np.arccos(h[2] / h_norm)
    n = np.cross([0, 0, 1], h)
    RAAN = np.arctan2(n[1], n[0])
    omega = np.arctan2(e_vec[2], np.dot(n, e_vec))
    theta = np.arccos(np.dot(e_vec, r) / (e * r_norm))
    return a, e, np.degrees(i), np.degrees(RAAN), np.degrees(omega), np.degrees(theta)

# Compute orbital elements
keplerian_elements = cartesian_to_keplerian(r_eci_1, v_eci_1, mu)

# Display results
print("Keplerian Elements at Initial Measurement Time:")
print(f"Semi-major axis (a): {keplerian_elements[0]:.2f} m")
print(f"Eccentricity (e): {keplerian_elements[1]:.5f}")
print(f"Inclination (i): {keplerian_elements[2]:.2f} degrees")
print(f"RAAN: {keplerian_elements[3]:.2f} degrees")
print(f"Argument of Periapsis (omega): {keplerian_elements[4]:.2f} degrees")
print(f"True Anomaly (theta): {keplerian_elements[5]:.2f} degrees")

# Monte Carlo for Uncertainty Estimation
num_samples = 1000
perturbation_scale = np.array([10.0, 0.001, 0.001])  # Std dev of noise for [range, RA, Dec]
kepler_samples = []

for i in range(num_samples):
    noise = np.random.normal(0, perturbation_scale, size=3)
    r_noisy = ra_dec_to_eci(rg + noise[0], ra + noise[1], dec + noise[2], sensor_ecef)
    v_noisy = (
        ra_dec_to_eci(rg2 + noise[0], ra2 + noise[1], dec2 + noise[2], sensor_ecef)
        - r_noisy
    ) / dt
    kepler_samples.append(cartesian_to_keplerian(r_noisy, v_noisy, mu))

kepler_samples = np.array(kepler_samples)
covariance_matrix = np.cov(kepler_samples.T)

print("\nUncertainty (Covariance Matrix of Keplerian Elements):")
print(covariance_matrix)
