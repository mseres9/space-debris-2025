import os
import numpy as np
import EstimationUtilities as EstUtil
import ConjunctionUtilities as ConjUtil
import TudatPropagator as prop
import matplotlib.pyplot as plt
import pickle

# Directory where group data is stored
data_dir = os.path.join("data", "group4")

# Load optical and radar measurement data
optical_file = os.path.join(data_dir, "q3_optical_meas_objchar_91991.pkl")
radar_file = os.path.join(data_dir, "q3_radar_meas_objchar_91991.pkl")

state_params_optical, meas_dict_optical, sensor_params_optical = EstUtil.read_measurement_file(optical_file)
state_params_radar, meas_dict_radar, sensor_params_radar = EstUtil.read_measurement_file(radar_file)

# Ensure each Yk is a column vector
meas_dict_optical["Yk_list"] = [yk.reshape(-1, 1) for yk in meas_dict_optical["Yk_list"]]
meas_dict_radar["Yk_list"] = [yk.reshape(-1, 1) for yk in meas_dict_radar["Yk_list"]]

# Merge measurements from radar and optical
tk_list_combined = np.concatenate([meas_dict_radar["tk_list"], meas_dict_optical["tk_list"]])
Yk_list_combined = meas_dict_radar["Yk_list"] + meas_dict_optical["Yk_list"]
sensor_list_combined = ["radar"] * len(meas_dict_radar["tk_list"]) + ["optical"] * len(meas_dict_optical["tk_list"])

# Sort measurements by time
sorted_indices = np.argsort(tk_list_combined)
tk_list_sorted = tk_list_combined[sorted_indices]
Yk_list_sorted = [Yk_list_combined[i] for i in sorted_indices]
sensor_list_sorted = [sensor_list_combined[i] for i in sorted_indices]

# Create a combined measurement dictionary
meas_dict_combined = {
    "tk_list": tk_list_sorted,
    "Yk_list": Yk_list_sorted,
    "sensor_list": sensor_list_sorted
}

# Store both sensor parameter dictionaries
sensor_params_combined = {
    "radar": sensor_params_radar,
    "optical": sensor_params_optical
}

# Load the RSO catalog (use estimated values, not the truth)
catalog_file = os.path.join(data_dir, "estimated_rso_catalog.pkl")
rso_dict = ConjUtil.read_catalog_file(catalog_file)
rso = rso_dict[91991]

# Print the estimated physical parameters from the catalog
for key in ["Cd", "Cr", "area", "mass"]:
    print(f"{key} = {rso[key]}")

# Create a full copy of the original RSO dictionary
rso_dict_modified = rso_dict.copy()

# Modify only the object with ID 91991
rso_dict_modified[91991] = rso_dict[91991].copy()
rso_dict_modified[91991]["area"] = 5.241
rso_dict_modified[91991]["mass"] = 10

# Make sure the directory exists
os.makedirs("data/group4", exist_ok=True)

# Save the modified dictionary as a .pkl file
with open("data/group4/rso_dict_modified.pkl", "wb") as f:
    pickle.dump({0: rso_dict_modified}, f)

catalog_file2 = os.path.join(data_dir, "rso_dict_modified.pkl")
rso_dict2 = ConjUtil.read_catalog_file(catalog_file2)
rso2 = rso_dict2[91991]

# Print the estimated physical parameters from the catalog
for key in ["Cd", "Cr", "area", "mass"]:
    print(f"{key} = {rso2[key]}")

# Define initial state and physical parameters
state_params = state_params_radar

X_dyn = state_params["state"] 
Cd = np.array([[state_params["Cd"]]])
Cr = np.array([[state_params["Cr"]]])
Area = np.array([[state_params["area"]]])
mass = np.array([[state_params["mass"]]])

# Stack into extended state vector (10x1)
Xo = np.vstack((X_dyn, Area/mass))
# Xo = np.vstack((X_dyn, mass))

# ---------------------- Tuning

# Extend initial covariance
P_dyn = state_params["covar"]
# P_param = np.diag([5e-2, 1e-5])
# Po = np.block([
#     [P_dyn, np.zeros((6,2))],
#     [np.zeros((2,6)), P_param]
# ])
P_param = np.diag([1e-3])
Po = np.block([
    [P_dyn, np.zeros((6,1))],
    [np.zeros((1,6)), P_param]
])

state_params["state"] = Xo
state_params["covar"] = Po
# state_params["area"] = 10

# UKF tuning
Qeci = 1e-12 * np.diag([1., 1., 1.])
Qric = 1e-12 * np.diag([1., 1., 1.])
#Qparam = 1e-9 * np.diag([1., 1.])
Qparam = 1e-9 * np.diag([1])
filter_params = {
    "Qeci": Qeci,
    "Qric": Qric,
    "Qparam": Qparam,
    "alpha": 1e-3,
    "gap_seconds": 600.0
}

# Integration settings
int_params = {
    "tudat_integrator": "rkf78",
    "step": 1.0,
    "max_step": 1.0,
    "min_step": 1e-3,
    "rtol": 1e-8,
    "atol": 1e-8
}

# ---------------------- Plots

# Measurement times (in seconds)
tk_optical = np.array(meas_dict_optical["tk_list"])
tk_radar = np.array(meas_dict_radar["tk_list"])

Yk_optical = np.array([y.flatten() for y in meas_dict_optical["Yk_list"]])
Yk_radar = np.array([y.flatten() for y in meas_dict_radar["Yk_list"]])

# Shift times relative to first optical and radar measurement separately
tk_optical_shifted = tk_optical - tk_optical.min()
tk_radar_shifted = tk_radar - tk_radar.min()

fig, axs = plt.subplots(2, 1, figsize=(14, 10))

# ---------------------- Optical subplot 

axs[0].plot(tk_optical_shifted/3600, Yk_optical[:, 0], "ro", markersize=4)
axs[0].set_ylabel("Apparent Magnitude", fontsize=16)
axs[0].set_xlabel(f"Time + {tk_optical.min()/3600:.2f} [hours]", fontsize=16)
axs[0].set_title("Optical Measurements", fontsize=16)
axs[0].tick_params(axis="both", labelsize=16)
axs[0].grid(True)

# ---------------------- Radar subplot 

# Primary axis = Range
axs[1].plot(tk_radar_shifted/3600, Yk_radar[:, 0], "bo", markersize=4)

# Secondary axis = RA
ax2 = axs[1].twinx()
ax2.plot(tk_radar_shifted/3600, np.rad2deg(Yk_radar[:, 1]), "go", markersize=4)

# Tertiary axis = Dec
ax3 = axs[1].twinx()
ax3.spines["right"].set_position(("axes", 1.075))  
ax3.plot(tk_radar_shifted/3600, np.rad2deg(Yk_radar[:, 2]), "mo", markersize=4)

# Set axis labels
axs[1].set_ylabel("Range [m]", color="blue", fontsize=14)
ax2.set_ylabel("RA [deg]", color="green", fontsize=14)
ax3.set_ylabel("Dec [deg]", color="magenta", fontsize=14)

axs[1].set_xlabel(f"Time + {tk_radar.min()/3600:.2f} [seconds]", fontsize=14)
axs[1].set_title("Radar Measurements", fontsize=16)
axs[1].tick_params(axis="both", labelsize=12)
ax2.tick_params(axis="y", labelsize=12)
ax3.tick_params(axis="y", labelsize=12)
axs[1].grid(True)
# NO legend needed now
fig.tight_layout()
plt.show()

# ---------------------- Filter

# Initialize Tudat environment
bodies_to_create = ["Earth", "Sun", "Moon"]
bodies = prop.tudat_initialize_bodies(bodies_to_create)

# Run the Unscented Kalman Filter
filter_output = EstUtil.ukf(state_params, meas_dict_combined, sensor_params_combined, int_params, filter_params, bodies)

# ---------------------- Plots

# Extract times (in hours)
tk_list = sorted(filter_output.keys())
times = np.array([(tk - tk_list[0]) / 3600.0 for tk in tk_list])  # in hours

# ---------- Optical residuals (Magnitude)
mag_residuals = []
time_mag = []

for i, tk in enumerate(tk_list):
    if meas_dict_combined["sensor_list"][i] == "optical":
        resid = filter_output[tk]["resids"].flatten()
        mag_residuals.append(resid[0])
        time_mag.append(times[i])

# Convert to arrays
time_mag = np.array(time_mag)
mag_residuals = np.array(mag_residuals)

# Compute stats
mag_mean = np.mean(mag_residuals[4:])
mag_std = np.std(mag_residuals[4:])

# Plot optical residuals
plt.figure(figsize=(10, 5))
plt.plot(time_mag[4:], mag_residuals[4:], "r.", markersize=5)
plt.axhline(mag_mean, color="blue", linestyle="--", label="Mean")
plt.axhline(mag_mean + mag_std, color="red", linestyle=":", label="+1σ")
plt.axhline(mag_mean - mag_std, color="red", linestyle=":", label="-1σ")
plt.xlabel("Time since first measurement [hours]", fontsize=16)
plt.ylabel("Magnitude Residuals", fontsize=16)
plt.title("Optical Measurement Residuals (Magnitude)", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()

# ---------- Radar residuals (Range, RA, DEC)
range_residuals = []
ra_residuals = []
dec_residuals = []
times_radar = []

for i, tk in enumerate(tk_list):
    if meas_dict_combined["sensor_list"][i] == "radar":
        resids = filter_output[tk]["resids"].flatten()
        range_residuals.append(resids[0])
        ra_residuals.append(np.rad2deg(resids[1]))
        dec_residuals.append(np.rad2deg(resids[2]))
        times_radar.append((tk - tk_list[0]) / 3600.0)  # in hours

# Convert to arrays
range_residuals = np.array(range_residuals)
ra_residuals = np.array(ra_residuals)
dec_residuals = np.array(dec_residuals)
times_radar = np.array(times_radar)

# Compute stats
range_mean, range_std = np.mean(range_residuals), np.std(range_residuals)
ra_mean, ra_std = np.mean(ra_residuals), np.std(ra_residuals)
dec_mean, dec_std = np.mean(dec_residuals), np.std(dec_residuals)

# Plot radar residuals
fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

axs[0].plot(times_radar, range_residuals, "b.", markersize=4)
axs[0].axhline(range_mean, color="blue", linestyle="--")
axs[0].axhline(range_mean + range_std, color="red", linestyle=":")
axs[0].axhline(range_mean - range_std, color="red", linestyle=":")
axs[0].set_ylabel("Range Residual [m]", fontsize=16)
axs[0].set_title("Radar Measurement Residuals", fontsize=16)
axs[0].tick_params(axis="both", labelsize=16)
axs[0].grid(True)
axs[0].legend()

axs[1].plot(times_radar, ra_residuals, "g.", markersize=4)
axs[1].axhline(ra_mean, color="blue", linestyle="--")
axs[1].axhline(ra_mean + ra_std, color="red", linestyle=":")
axs[1].axhline(ra_mean - ra_std, color="red", linestyle=":")
axs[1].set_ylabel("RA Residual [deg]", fontsize=16)
axs[1].tick_params(axis="both", labelsize=16)
axs[1].grid(True)
axs[1].legend()

axs[2].plot(times_radar, dec_residuals, "m.", markersize=4)
axs[2].axhline(dec_mean, color="blue", linestyle="--", label="Mean")
axs[2].axhline(dec_mean + dec_std, color="red", linestyle=":", label="+1σ")
axs[2].axhline(dec_mean - dec_std, color="red", linestyle=":", label="-1σ")
axs[2].set_ylabel("DEC Residual [deg]", fontsize=16)
axs[2].set_xlabel("Time since first measurement [hours]", fontsize=16)
axs[2].tick_params(axis="both", labelsize=16)
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()

# Extract final estimated parameters
X_final = filter_output[tk_list[-1]]["state"]
P_final = filter_output[tk_list[-1]]["covar"]

Cd_hat = X_final[6, 0]

Cd_std = np.sqrt(P_final[6, 6])

# Print estimated parameters
print("\nEstimated Physical Parameters at Final Epoch:")
print(f"Ratio   = {Cd_hat:.4f} ± {Cd_std:.4f}")

# Count measurements
n_optical = sum(1 for s in meas_dict_combined["sensor_list"] if s == "optical")
n_radar = sum(1 for s in meas_dict_combined["sensor_list"] if s == "radar")

# Compute RMS for all residuals
mag_rms = np.sqrt(np.mean(np.square(mag_residuals)))
range_rms = np.sqrt(np.mean(np.square(range_residuals)))
ra_rms = np.sqrt(np.mean(np.square(ra_residuals)))
dec_rms = np.sqrt(np.mean(np.square(dec_residuals)))

# Print everything
print(f"Number of optical observations: {n_optical}")
print(f"Number of radar observations: {n_radar}")
print("\nPost-fit Residual RMS:")
print(f"Magnitude RMS: {mag_rms:.4f}")
print(f"Range RMS:     {range_rms:.4f} m")
print(f"RA RMS:        {ra_rms:.6f} deg")
print(f"DEC RMS:       {dec_rms:.6f} deg")