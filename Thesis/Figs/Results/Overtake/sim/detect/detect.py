import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
import yaml
import os
from matplotlib.ticker import FuncFormatter
import skimage

# root_path = "/home/chris/masters_report/Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/Real"
root_path = "C:/Users/bbdnet2985/Desktop/Matsters/Report/Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/Overtake/sim/detect"

# Configure matplotlib to use LaTeX-style fonts and formatting
plt.rcParams.update({
    # Use LaTeX to render text (optional - requires LaTeX installed)
    'text.usetex': False,  # Set to True if you have LaTeX installed
    
    # Font settings to match LaTeX documents
    'font.family': 'serif',
    'font.serif': ['Caladea', 'TeX Gyre Termes', 'Times New Roman', 'DejaVu Serif'],
    'font.sans-serif': ['Carlito', 'TeX Gyre Heros', 'DejaVu Sans'],
    'font.monospace': ['Computer Modern Typewriter', 'DejaVu Sans Mono'],
    
    # Font sizes (smaller to match your original setup)
    'font.size': 10,        # Reduced from 12
    # 'axes.labelsize': 10,   # Reduced from 12
    # 'axes.titlesize': 11,   # Reduced from 14
    # 'xtick.labelsize': 9,   # Reduced from 11
    # 'ytick.labelsize': 9,   # Reduced from 11
    'legend.fontsize': 9,   # Reduced from 11
    # 'figure.titlesize': 12, # Reduced from 16
    
    # Figure quality and format
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    
    # Grid and axes styling
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    
    # Legend styling
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    
    # LaTeX preamble for additional packages if needed (only if text.usetex=True)
    'text.latex.preamble': r'\usepackage{amsmath,amssymb,amsfonts}'
})

print("Matplotlib configured for LaTeX-style formatting")
print("Text rendering:", plt.rcParams['text.usetex'])
print("Font family:", plt.rcParams['font.family'])

# Functions
def nearest_point(point, trajectory):
	"""
	Return the nearest point along the given piecewise linear trajectory.

	Args:
		point (numpy.ndarray, (2, )): (x, y) of current pose
		trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
			NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world

	Returns:
		nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
		nearest_dist (float): distance to the nearest point (negative if point is on the right)
		t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
		i (int): index of nearest point in the array of trajectory waypoints
	"""
	diffs = trajectory[1:] - trajectory[:-1]
	l2s = np.sum(diffs**2, axis=1)
	dots = np.einsum('ij,ij->i', point - trajectory[:-1], diffs)
	t = np.clip(dots / l2s, 0.0, 1.0)
	projections = trajectory[:-1] + t[:, np.newaxis] * diffs
	dists = np.linalg.norm(point - projections, axis=1)
	
	# Determine if the point is on the right
	cross_products = np.cross(diffs, point - trajectory[:-1])
	dists = np.where(cross_products < 0, -dists, dists)
	
	min_dist_segment = np.argmin(np.abs(dists))
	return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

def progress_along_trajectory(points, trajectory):
    """
    Return the progress along the trajectory.

    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints

    Returns:
        progress (float): progress along the trajectory in meters
    """
    track_length = np.insert(np.cumsum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)), 0, 0)
    progress = np.zeros(points.shape[0])
    for idx, point in enumerate(points):
        _, _, t, i = nearest_point(point, trajectory)
        progress[idx] = track_length[i] + t * (track_length[i+1] - track_length[i])
    return progress

def get_laps_indices(points, trajectory):
    """
    Return the lap indices along the trajectory.

    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints

    Returns:
        laps (numpy.ndarray, (M, )): lap indices along the trajectory
    """
    progress = progress_along_trajectory(points, trajectory)
    laps = np.zeros(points.shape[0], dtype=int)
    lap_count = 0
    track_length = np.insert(np.cumsum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)), 0, 0)[-1]
    for i in range(1, points.shape[0]):
        if progress[i-1] - progress[i] > track_length/3:  # Detect lap completion
            lap_count += 1
        laps[i] = lap_count
    return laps

# Load map to get boundaries and raceline
map_path = 'C:/Users/bbdnet2985/Desktop/Matsters/Report/Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/maps'
map_yaml_path = f"{map_path}/sep2.yaml"
map_img = plt.imread(f'{map_path}/sep2.png')
map_img = np.flipud(map_img)
map_img = scipy.ndimage.distance_transform_edt(map_img)
map_img = np.abs(map_img - 1)
map_img[map_img!=0]=1
bx,by = np.where(map_img==0)


with open(map_yaml_path, 'r') as yaml_stream:
    try:
        map_metadata = yaml.safe_load(yaml_stream)
        map_resolution = map_metadata['resolution']
        origin = map_metadata['origin']
    except yaml.YAMLError as ex:
        print(ex)

bx = bx * map_resolution + origin[1]
by = by * map_resolution + origin[0]

raceline = np.loadtxt(f'{map_path}/sep2_minCurve.csv', delimiter=',')

# def plot_map():
#     plot_map() #plt.scatter(by, bx, c='black',marker='s',s=5)  # Plot the boundaries
#     plt.axis('equal')
#     plt.axis('off')

def plot_map(img=map_img):
		boundaries = skimage.measure.find_contours(img, 0.5)
		print(f"Found {len(boundaries)} boundaries")
		# print(boundaries)
		# for boundary in boundaries:
		# 	plt.plot(boundary[:,1]*map_resolution + origin[0], boundary[:,0]*map_resolution + origin[1], 'black', linewidth=0.1)
		plt.plot(boundaries[0][:,1]*map_resolution + origin[0], boundaries[0][:,0]*map_resolution + origin[1], 'black', linewidth=0.5)
		plt.plot(boundaries[-1][:,1]*map_resolution + origin[0], boundaries[-1][:,0]*map_resolution + origin[1], 'black', linewidth=0.5)


# plt.figure()
# plt.plot(raceline[:,0], raceline[:,1], c='red', linewidth=0.5)  # Plot the raceline
# plot_map()
# plt.show()
# Detection
detected_positions_path = f'{root_path}/detect.csv'
ego_positions_path = f'{root_path}/ego_pose.csv'
opp_positions_path = f'{root_path}/opp_pose.csv'
detected_positions = np.loadtxt(detected_positions_path, delimiter=',')
ego_positions = np.loadtxt(ego_positions_path, delimiter=',')
opp_positions = np.loadtxt(opp_positions_path, delimiter=',')
save_folder = f'{root_path}/Figs'

#only use times that exist in detected positions
valid_times = detected_positions[:,0]
interpolated_actual_positions = np.zeros_like(detected_positions)
for i,t in enumerate(valid_times):
    #find closest time in opp_positions
    idx = np.argmin(np.abs(opp_positions[:,0]-t))
    interpolated_actual_positions[i,:] = opp_positions[idx,:-1]




error = np.linalg.norm(interpolated_actual_positions[:,1:] - detected_positions[:,1:], axis=1)
mean_error = np.mean(error)
std_error = np.std(error)
max_error = np.max(error)
min_error = np.min(error)
rmse = np.sqrt(np.mean(error**2))
print(f'Mean error: {mean_error:.4f} m')
print(f'Standard deviation of error: {std_error:.4f} m')
print(f'Max error: {max_error:.4f} m')
print(f'Min error: {min_error:.4f} m')
print(f'RMSE: {rmse:.4f} m')

x_error = detected_positions[:, 1] - interpolated_actual_positions[:, 1]
y_error = detected_positions[:, 2] - interpolated_actual_positions[:, 2]
outlier_idx = np.where(error >  2*std_error)[0]
print(f'Number of outliers: {len(outlier_idx)} out of {len(error)}')
not_outlier_idx = np.where(error <= 2*std_error)[0]
mean_x_error = np.mean(x_error)
mean_y_error = np.mean(y_error)
print(f'Mean X Error: {mean_x_error:.4f} m, Mean Y Error: {mean_y_error:.4f} m')

#rmse without outliers
error_no_outliers = error[not_outlier_idx]
mean_error_no_outliers = np.mean(error_no_outliers)
std_error_no_outliers = np.std(error_no_outliers)
rmse_no_outliers = np.sqrt(np.mean(error_no_outliers**2))
print(f'Mean error (no outliers): {mean_error_no_outliers:.4f} m')
print(f'Standard deviation of error (no outliers): {std_error_no_outliers:.4f} m')
print(f'RMSE (no outliers): {rmse_no_outliers:.4f} m')

print(f'number of detections: {detected_positions.shape[0]}')
TP = detected_positions[error < 2*std_error].shape[0]
print(f'TP: {TP}')
FP = detected_positions[error >= 2*std_error].shape[0]
print(f'FP: {FP}')
FN = opp_positions.shape[0] - TP - FP
print(f'FN: {FN}')
TPR = TP / (TP + FN)
print(f'TPR: {TPR:.4f}')
FPR = FP / (FP + TP)
print(f'FPR: {FPR:.4f}')

laps = get_laps_indices(interpolated_actual_positions[:,1:], raceline[:, :2])
print(f'unique laps: {np.unique(laps)}')
print(map_img.shape)
mask_in = np.where((laps==2)&(error<=2*std_error))[0]
mask_out = np.where((laps==2)&(error>2*std_error))[0]
lap_mask = (laps==2)
width = 18/2.54
height = width*map_img.shape[0]/map_img.shape[1]
print(f'fig size: {width} x {height}')
plt.figure(figsize=(width, height))  # Make figure square
plot_map()
plt.plot(interpolated_actual_positions[lap_mask, 1], interpolated_actual_positions[lap_mask, 2], label='True Opponent Trajectory', color='blue', linewidth=0.5)
plt.scatter(detected_positions[mask_in, 1], detected_positions[mask_in, 2], label='Detected Positions', color='red', marker='o', s=5, edgecolors='black')
plt.scatter(detected_positions[mask_out, 1], detected_positions[mask_out, 2], label='False Positive Detections', color='orange', marker='o', s=5, edgecolors='black')
plt.legend(loc='upper right')
plt.axis('equal')
plt.axis('image')
plt.xlim(-4,7.5)
# plt.ylim(-1,4)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.savefig(f'{save_folder}/detection_results_on_map.pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True, format='pdf', )
plt.show()



plt.figure(figsize=(3.5,3.5))
plt.scatter(x_error[not_outlier_idx], y_error[not_outlier_idx], label='Detection Error', color='#1f77b4', marker='o', s=30, edgecolors='#222222', alpha=0.7)
plt.scatter(x_error[outlier_idx], y_error[outlier_idx], label='False Positive Error', color='#ff7f0e', marker='o', s=30, edgecolors='#222222', alpha=0.7)
plt.scatter(mean_x_error, mean_y_error,  label='Mean Error', color='red', marker='+', s=80, edgecolors='black')
# plot circle around mean x, y error with radius 2*std_error
circle = plt.Circle((mean_x_error, mean_y_error), std_error, color='red', fill=False, linestyle='--', label='Std Dev')
plt.gca().add_artist(circle)
plt.xscale('symlog', linthresh=np.max([std_error*1.2, 0.1]))
plt.yscale('symlog', linthresh=np.max([std_error*1.2, 0.1]))
plt.xlabel('SymLog Detection Error X (m)')
plt.ylabel('SymLog Detection Error Y (m)')
plt.grid(True, alpha=0.4)
plt.axis('equal')
plt.legend()
def format_func(value, tick_number):
    if value == 0:
        return "0"
    elif abs(value) < 1:
        if abs(value) >= 0.1:
            return f"{value:.1f}"

    else:
        return f"{value:.0f}"
    
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func))
plt.savefig(f'{save_folder}/detection_error_scatter.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=True, format='pdf')
# plt.show()
# #remove outliers


idx_no_outliers = np.where(error < mean_error + 2*std_error)[0]
idx_outliers = np.where(error >= mean_error + 2*std_error)[0]
print(f'outlier threshold: {mean_error + 2*std_error}')
print(f'Number of outliers: {idx_outliers} out of {len(error)}')
error_no_outliers = error[idx_no_outliers]
mean_error_no_outliers = np.mean(error_no_outliers)
std_error_no_outliers = np.std(error_no_outliers)
max_error_no_outliers = np.max(error_no_outliers)
min_error_no_outliers = np.min(error_no_outliers)
rmse_no_outliers = np.sqrt(np.mean(error_no_outliers**2))
print(f'Without outliers:')
print(f'Mean error: {mean_error_no_outliers:.4f} m')
print(f'Standard deviation of error: {std_error_no_outliers:.4f} m')
print(f'Max error: {max_error_no_outliers:.4f} m')
print(f'Min error: {min_error_no_outliers:.4f} m')
print(f'RMSE: {rmse_no_outliers:.4f} m')

x_error = detected_positions[:, 1] - interpolated_actual_positions[:, 1]
y_error = detected_positions[:, 2] - interpolated_actual_positions[:, 2]
mean_x_error = np.mean(x_error)
mean_y_error = np.mean(y_error)
print(f'Mean X Error: {mean_x_error:.4f} m')
print(f'Mean Y Error: {mean_y_error:.4f} m')
# standard_deviation = np.std(np.sqrt(x_error**2 + y_error**2))
# print(f'Standard Deviation of Error: {standard_deviation:.4f} m')

unique_scan_times, counts = np.unique(detected_positions[:, 0], return_counts=True)
print(f'unique_scan_times: {len(unique_scan_times)}')
times = np.arange(detected_positions[0, 0], detected_positions[-1, 0]+1/15, 1/15)
print(f'times: {len(times)}')

# TP: error<2*rmse_no_outliers
# rmse = rmse_no_outliers
# std_error = std_error_no_outliers


TP = detected_positions[error < 2*rmse]
all_tps = len(TP)
# print(f'True Positives: {len(TP)}')
# # if mutiple detections at same timestamp, keep only one with smallest error
unique_times_tp, indices_tp, counts = np.unique(TP[:, 0], return_index=True, return_counts=True)
TP = TP[indices_tp]
rem = all_tps - len(TP)
print(f'True Positives: {len(TP)}')
# # include detections from detected positions with error >= 0.3 as false positives and extra detections from TP
FP = detected_positions[error >= 2*rmse]
print(f'False Positives: {len(FP)+rem}')
FN = len(times) - len(TP)-len(FP)-rem
print(f'False Negatives: {FN}')

TPR = (len(TP) / (len(TP) + FN) if (len(TP) + FN) > 0 else 0)*100
print(f'True Positive Rate: {TPR:.2f}')
FPR = ((len(FP)+rem) / (len(FP) +rem + len(TP)) if (len(FP) + len(TP)) > 0 else 0)*100
print(f'False Positive Rate: {FPR:.2f}')


# # Plotting
# pc = ['red']*len(error)
# for i in idx_outliers:
#     pc[i] = '#ff7f0e'
# d=1
# plt.figure(figsize=(6.5, 3.5))
# # plot_map() #plt.scatter(by, bx, c='black',marker='s',s=5)  # Plot the boundaries
# plot_map()
# plt.scatter(detected_positions[::d, 1], detected_positions[::d, 2], label='Detected Position', color=pc, marker='o', s=20, edgecolors='black')
# plt.plot(interpolated_actual_positions[::d, 1], interpolated_actual_positions[::d, 2], label='Actual Position', color='blue')
# plt.legend(loc='upper left')
# plt.axis('equal')
# plt.axis('off')
# # plt.savefig(f'{save_folder}/detection_results_on_map.pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True, format='pdf', )
# plt.show()

# plt.figure(figsize=(3.5,3.5))
# plt.scatter(x_error[idx_no_outliers], y_error[idx_no_outliers], label='Error', color='#1f77b4', marker='o', s=30, edgecolors='#222222', alpha=0.7)
# plt.scatter(x_error[idx_outliers], y_error[idx_outliers], label='Error outliers', color='#ff7f0e', marker='o', s=30, edgecolors='#222222', alpha=0.7)
# plt.plot(mean_x_error, mean_y_error, 'o', color='#d62728', label='Mean Error', markersize=8, markeredgecolor='black')
# circle = plt.Circle((mean_x_error, mean_y_error), std_error, color='#d62728', fill=False, linestyle='--', linewidth=2, label='1 Std Dev')
# plt.gca().add_artist(circle)
# plt.xscale('symlog', linthresh=np.max([std_error*1.2, 0.1]))
# plt.yscale('symlog', linthresh=np.max([std_error*1.2, 0.1]))
# plt.xlabel('SymLog Detection Error X (m)')
# plt.ylabel('SymLog Detection Error Y (m)')
# plt.grid(True, alpha=0.4)
# plt.axis('equal')
# plt.legend(loc='upper right')
# plt.tight_layout()

# def format_func(value, tick_number):
#     if value == 0:
#         return "0"
#     elif abs(value) < 1:
#         if abs(value) >= 0.1:
#             return f"{value:.1f}"

#     else:
#         return f"{value:.0f}"
    
# plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
# plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func))
# # plt.savefig(f'{save_folder}/detection_error_scatter.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=True, format='pdf', )
# plt.show()

