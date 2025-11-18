import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
import yaml
import os
from matplotlib.ticker import FuncFormatter
import skimage
import matplotlib.patches as mpatches

# root_path = "/home/chris/masters_report/Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/Real"
root_path = "C:/Users/bbdnet2985/Desktop/Matsters/Report/Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/Overtake/real/tracking"

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

# Create a custom legend entry that shows an arrow
from matplotlib.lines import Line2D

def create_arrow_legend_entry(color, label):
    # Create a custom Line2D with a custom marker to represent an arrow
    return Line2D([0], [0], color=color, marker='>',
                 linestyle='-', markersize=10, 
                 markeredgewidth=1, label=label)

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
# #plt.show()
# Detection
detected_positions_path = f'{root_path}/track.csv'
ego_positions_path = f'{root_path}/ego_pose.csv'
opp_positions_path = f'{root_path}/opp_pose.csv'
estimated = np.loadtxt(detected_positions_path, delimiter=',')
ego_positions = np.loadtxt(ego_positions_path, delimiter=',')
ego_positions[:,1:3]  += np.random.normal(0, 0.005, ego_positions[:,1:3].shape)
opp_positions = np.loadtxt(opp_positions_path, delimiter=',')
opp_positions[:,1:3]  += np.random.normal(0, 0.005, opp_positions[:,1:3].shape)
# opp_positions[:,-1] += np.random.normal(0, 0.01, opp_positions[:,-1].shape)
save_folder = f'{root_path}/Figs'


# Create interpolation functions
interp_func_x = interp1d(opp_positions[:, 0], opp_positions[:, 1], kind='linear', fill_value="extrapolate")
interp_func_y = interp1d(opp_positions[:, 0], opp_positions[:, 2], kind='linear', fill_value="extrapolate")
interp_func_heading = interp1d(opp_positions[:, 0], opp_positions[:, 3], kind='linear', fill_value="extrapolate")
interp_func_speed = interp1d(opp_positions[:, 0], opp_positions[:, 4], kind='linear', fill_value="extrapolate")

for i in range(1,len(estimated)):
    if estimated[i, 0] - estimated[i-1, 0] <= 1/30:
        estimated[i, 0] = estimated[i-1, 0]

# Apply interpolation to detected positions
interpolated_x = interp_func_x(estimated[:, 0])
interpolated_y = interp_func_y(estimated[:, 0])
interpolated_heading = interp_func_heading(estimated[:, 0])
interpolated_speed = interp_func_speed(estimated[:, 0])

interpolated_actual_positions = np.column_stack((estimated[:, 0], interpolated_x, interpolated_y, interpolated_heading, interpolated_speed))

estimated[:,3] =np.arctan2(np.sin(estimated[:,3]), np.cos(estimated[:,3]))
interpolated_actual_positions[:,3] =np.arctan2(np.sin(interpolated_actual_positions[:,3]), np.cos(interpolated_actual_positions[:,3]))
print(f'number of detections: {estimated.shape[0]}')
print(f'number of actual positions: {interpolated_actual_positions.shape[0]}')

laps = get_laps_indices(interpolated_actual_positions[:,1:3], raceline[:, :2])
print(f'unique laps: {np.unique(laps)}')
# plt.plot(laps)
# #plt.show()

mask = np.where((laps==1)|(laps==2))[0]

width = 18/2.54
height = width*map_img.shape[0]/map_img.shape[1]
print(f'fig size: {width} x {height}')

# Plot trajectories with arrows to show orientation
plt.figure(figsize=(width, height))
plot_map()

# Plot continuous trajectories
plt.plot(estimated[mask,1], estimated[mask,2], c='blue', linewidth=0.5, label='Estimated Pose') 
plt.plot(interpolated_actual_positions[mask,1], interpolated_actual_positions[mask,2], c='red', linewidth=0.5, label='Actual Pose')
plt.legend()
plt.axis('equal')
plt.axis('off')
plt.savefig(f'{save_folder}/estimated_results_on_map.pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True, format='pdf')
#plt.show()

def plot_vehicle_arrows(ax, positions, color, sample_rate=10, label=None):
	"""
	Plot vehicle positions as arrows with directions from the heading angle
	
	Args:
		ax: Matplotlib axes
		positions: Array with columns [time, x, y, heading_rad, ...]
		color: Arrow color
		sample_rate: Sample every N points to avoid crowding
		label: Legend label for the arrows
	"""
	# Sample positions to avoid overcrowding
	indices = np.arange(0, len(positions), sample_rate)
	positions = positions[indices]
	
	# Extract x, y, and heading info
	x = positions[:, 1]
	y = positions[:, 2]
	headings = positions[:, 3]

	# Calculate direction vectors from heading angles
	arrow_length = 0.4  # Length of the arrows
	u = np.cos(headings) * arrow_length  # Scale for visualization
	v = np.sin(headings) * arrow_length

	# Set consistent alpha
	alphas = np.ones(len(positions)) * 0.7
	
	# Plot arrows
	for i in range(len(positions)):
		ax.quiver(x[i], y[i], u[i], v[i], 
				 color=color, alpha=alphas[i], 
				 scale=1, scale_units='xy',
				 width=0.005, headwidth=3, 
				 label=label if i == len(positions)-1 else None)
	
	# return arrows, label
      
mask = np.where((laps==4))[0]
plt.figure(figsize=(width, height))
plot_map()
# Plot vehicle arrows 
ax = plt.gca()
plot_vehicle_arrows(ax, estimated[mask], color='blue', sample_rate=5, label='Estimated Pose')
plot_vehicle_arrows(ax, interpolated_actual_positions[mask], color='red', sample_rate=5, label='Actual Pose')
plt.legend(loc='upper left')
plt.axis('equal')
plt.axis('off')
# plt.savefig(f'{save_folder}/estimated_results_on_map_arrows.pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True, format='pdf')
# #plt.show()

estimated[:,0] -= estimated[0,0]  # Set start time to 0
time = estimated[mask,0]
time -= time[0]  # Set start time to 0
width = 8/2.54
# x
plt.figure(figsize=(width, width))
plt.plot(time, estimated[mask,1], label='Estimated', color='blue')
plt.plot(time, interpolated_actual_positions[mask,1], label='Actual', color='red')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('X Position (m)')
# plt.title('X Position over Time')
plt.grid(True)
# plt.savefig(f'{save_folder}/x_position_over_time.pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True, format='pdf')
# #plt.show()
# y
plt.figure(figsize=(width, width))
plt.plot(time, estimated[mask,2], label='Estimated Y Position', color='blue')
plt.plot(time, interpolated_actual_positions[mask,2], label='Actual Y Position', color='red')
# plt.legend(loc='upper left')
plt.xlabel('Time (s)')
plt.ylabel('Y Position (m)')
# plt.title('Y Position over Time')
plt.grid(True)
# plt.savefig(f'{save_folder}/y_position_over_time.pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True, format='pdf')
# #plt.show()
# yaw
plt.figure(figsize=(width, width))
plt.plot(time, np.rad2deg(estimated[mask,3]), label='Estimated Yaw', color='blue')
plt.plot(time, np.rad2deg(interpolated_actual_positions[mask,3]), label='Actual Yaw', color='red')
# plt.legend(loc='upper left')
plt.xlabel('Time (s)')
plt.ylabel('Yaw ($^\circ$)')
# plt.title('Yaw over Time')
plt.grid(True)
plt.savefig(f'{save_folder}/yaw_over_time.pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True, format='pdf')
# #plt.show()
# speed

def ma(speed):
    window_size =2
    weights = np.ones(window_size) / window_size
    smoothed_speed = np.convolve(speed, weights, mode='same')
    return smoothed_speed

plt.figure(figsize=(width, width))
plt.plot(time, estimated[mask,4], label='Estimated Speed', color='blue')
plt.plot(time, interpolated_actual_positions[mask,4], label='Actual Speed', color='red')
# plt.plot(time, ma(estimated[mask,4]), label='Smoothed Estimated Speed', color='cyan')
# plt.legend(loc='upper left')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
# plt.title('Speed over Time')
plt.grid(True)
# plt.savefig(f'{save_folder}/speed_over_time.pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True, format='pdf')
# #plt.show()

# mask = np.where((laps==2)|(laps==3))[0]
# Error plots
position_error = np.linalg.norm(estimated[:,1:3] - interpolated_actual_positions[:,1:3], axis=1)
mean_position_error = np.mean(position_error)
std_position_error = np.std(position_error)
rmse_position_error = np.sqrt(np.mean(position_error**2))
print(f'Position Error - Mean: {mean_position_error:.4f} m, Std: {std_position_error:.4f} m, RMSE: {rmse_position_error:.4f} m')

yaw_error = estimated[:,3] - interpolated_actual_positions[:,3]
yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
cos = np.cos(yaw_error)
sin = np.sin(yaw_error)
mean_yaw_error = np.arctan2(np.mean(sin), np.mean(cos))
std_yaw_error = np.sqrt(-2 * np.log(np.sqrt(np.mean(cos)**2 + np.mean(sin)**2)))
rmse_yaw_error = np.sqrt(np.mean(yaw_error**2))
print(f'Yaw Error - Mean: {np.degrees(mean_yaw_error):.4f} deg, Std: {np.degrees(std_yaw_error):.4f} deg, RMSE: {np.degrees(rmse_yaw_error):.4f} deg')

print('circular std = ', np.degrees(np.sqrt(-2 * np.log(np.sqrt(np.mean(cos)**2 + np.mean(sin)**2)))))

speed_error = estimated[:,4] - interpolated_actual_positions[:,4]
mean_speed_error = np.mean(speed_error)
std_speed_error = np.std(speed_error)
rmse_speed_error = np.sqrt(np.mean(speed_error**2))
print(f'Speed Error - Mean: {mean_speed_error:.4f} m/s, Std: {std_speed_error:.4f} m/s, RMSE: {rmse_speed_error:.4f} m/s')

# pos error
plt.figure(figsize=(width, width))
plt.axhline(y=rmse_position_error, color='black', linestyle='--', label='Mean Position Error',linewidth=0.5)
plt.plot(time, position_error[mask], label='Position Error', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Position Error (m)')
# plot mean line
plt.grid(True)
# plt.savefig(f'{save_folder}/position_error.pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True, format='pdf')
# #plt.show()
# yaw error
plt.figure(figsize=(width, width))
plt.axhline(y=np.rad2deg(rmse_yaw_error), color='black', linestyle='--', label='Mean Yaw Error', linewidth=0.5)
plt.plot(time, np.abs(np.rad2deg(yaw_error[mask])), label='Yaw Error', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Absolute Yaw Error ($^\circ$)')
# plot mean
plt.grid(True)
# plt.savefig(f'{save_folder}/yaw_error.pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True, format='pdf')
# #plt.show()
# speed error
plt.figure(figsize=(width, width))
plt.axhline(y=rmse_speed_error, color='black', linestyle='--', label='Mean Speed Error', linewidth=0.5)
plt.plot(time, np.abs(speed_error[mask]), label='Speed Error', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Absolute Speed Error (m/s)')
# plot mean
plt.grid(True)
# plt.savefig(f'{save_folder}/speed_error.pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True, format='pdf')
# #plt.show()
plt.close()
##################################################################################################################
# Trailing
##################################################################################################################
mask
raceline = np.loadtxt(f'{map_path}/sep2_minCurve.csv', delimiter=',')
# Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/Overtake/sim/tracking/trail/ego_pose.csv
ego_pose = np.loadtxt(f'{root_path}/trail/ego_pose.csv', delimiter=',')
ego_pose[:,1:3]  += np.random.normal(0, 0.02, ego_pose[:,1:3].shape)
# ego_pose[:,-1] += np.random.normal(0, 0.01, ego_pose[:,-1].shape)
opp_pose = np.loadtxt(f'{root_path}/trail/opp_pose.csv', delimiter=',')
opp_pose[:,1:3]  += np.random.normal(0, 0.02, opp_pose[:,1:3].shape)
# opp_pose[:,-1] += np.random.normal(0, 0.01, opp_pose[:,-1].shape)
# opp_pose[]
print(f'ego positions shape: {ego_pose.shape}')
print(f'opp positions shape: {opp_pose.shape}')
valid_times = opp_pose[:,0]
ego = ego_pose[np.isin(ego_pose[:,0], valid_times)]
print(f'filtered ego positions shape: {ego.shape}')

laps = get_laps_indices(ego_pose[:,1:3], raceline[:, :2])
print(f'laps: {laps}')
unique_laps = np.unique(laps)
print(f'unique laps: {unique_laps}')
# mask = np.where((laps==1)|(laps==2)| (laps==3)| (laps==4)| (laps==5))[0]
# mask = np.where((laps>1)&(laps<=5))[0]
mask = np.where((laps==2)|(laps==3))[0]
print(f'number of points in mask: {np.unique(laps[mask]).shape[0]}')

ego_progress = progress_along_trajectory(ego[:,1:3], raceline[:,:2])
opp_progress = progress_along_trajectory(opp_pose[:,1:3], raceline[:,:2])
times = opp_pose[:,0] - opp_pose[0,0]
times = times[mask]
times -= times[0]
cum_dist = np.insert(np.cumsum(np.linalg.norm(np.diff(raceline[:,:2], axis=0), axis=1)), 0, 0)
total_length = cum_dist[-1]
dist = np.ones_like(ego_progress)
ds = np.mod(opp_progress - ego_progress, total_length)
mean_ds = np.mean(ds)
print(f'Mean Distance between vehicles along raceline: {mean_ds} m')
mean_ds_error = np.mean(np.abs(1.5 - ds))
print(f'Mean Absolute Error Distance between vehicles along raceline: {mean_ds_error} m')
std_ds = np.std(1.5-ds)
print(f'Std Distance between vehicles along raceline: {std_ds} m')
rmse_ds = np.sqrt(np.mean((1.5-ds)**2))
print(f'RMSE Distance between vehicles along raceline: {rmse_ds} m')
width = 8/2.54
height = 8/2.54
plt.figure(figsize=(width, height))
plt.plot(times, ds[mask], color='blue',label='Distance between Vehicles')
plt.axhline(y=1.5, color='red', linestyle='--', label='Desired Distance (1.5 m)')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.legend()
plt.grid(True)
plt.savefig(f'{save_folder}/distance_over_time.pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True, format='pdf')
# #plt.show()

width = 8/2.54
height = 8/2.54
plt.figure(figsize=(width, height))
plt.plot(times, ego_pose[mask,4], color='blue',label='Ego Vehicle Speed')
plt.plot(times, opp_pose[mask,4], color='red',label='Opponent Vehicle Speed')
# plt.plot(times, ds[mask], color='blue',label='Distance between Vehicles along the raceline')
# plt.axhline(y=1.0, color='red', linestyle='--', label='Desired Distance (1 m)')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.legend()
plt.grid(True)
plt.savefig(f'{save_folder}/trail_speed_over_time.pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True, format='pdf')
# #plt.show()
