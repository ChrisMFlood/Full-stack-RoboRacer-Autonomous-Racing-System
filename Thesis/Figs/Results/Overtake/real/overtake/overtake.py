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
root_path = "C:/Users/bbdnet2985/Desktop/Matsters/Report/Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/Overtake/sim/tracking"

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
		# print(f"Found {len(boundaries)} boundaries")
		# print(boundaries)
		# for boundary in boundaries:
		# 	plt.plot(boundary[:,1]*map_resolution + origin[0], boundary[:,0]*map_resolution + origin[1], 'black', linewidth=0.1)
		plt.plot(boundaries[0][:,1]*map_resolution + origin[0], boundaries[0][:,0]*map_resolution + origin[1], 'black', linewidth=0.5)
		plt.plot(boundaries[-1][:,1]*map_resolution + origin[0], boundaries[-1][:,0]*map_resolution + origin[1], 'black', linewidth=0.5)

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
	if len(positions) > 1:
		alphas = np.linspace(0.01, 1.0, len(positions))
		alphas /= alphas[-1]  # Ensure last arrow is fully opaque
	else:
		alphas = np.ones(len(positions))

	# Plot arrows
	for i in range(len(positions)):
		ax.quiver(x[i], y[i], u[i], v[i], 
				 color=color, alpha=alphas[i], 
				 scale=1, scale_units='xy',
				 width=0.005, headwidth=3, 
				 label=label if i == len(positions)-1 else None)

def plot_vehicles(t0, t1, ego_pose, opp_pose, estimated_opp_pose, ax, ego_sample_rate=11, opp_sample_rate=11):
	plot_map()
	plt.plot(raceline[:,0], raceline[:,1], 'k--', linewidth=0.5, label='Raceline')
	ego_mask = (ego_pose[:,0]>=t0) & (ego_pose[:,0]<=t1)
	opp_mask = (opp_pose[:,0]>=t0) & (opp_pose[:,0]<=t1)
	esti_mask = (estimated_opp_pose[:,0]>=t0) & (estimated_opp_pose[:,0]<=t1)
	plot_vehicle_arrows(ax, opp_pose[opp_mask,:], color='red', sample_rate=opp_sample_rate, label='Opponent Vehicle')
	plot_vehicle_arrows(ax, estimated_opp_pose[esti_mask,:], color='green', sample_rate=opp_sample_rate, label='Estimated Opponent')
	plot_vehicle_arrows(ax, ego_pose[ego_mask,:], color='blue', sample_rate=ego_sample_rate, label='Ego Vehicle')
	ax.set_aspect('equal', 'box')

def get_ref_speed(pose, raceline):
	speed = np.zeros(pose.shape[0])
	
	for i, p in enumerate(pose):
		_, _, t, idx = nearest_point(p[1:3], raceline[:,0:2])
		if idx < raceline.shape[0]-1:
			if raceline.shape[1] > 7:
				speed[i] = raceline[idx, 7] + t * (raceline[idx+1, 7] - raceline[idx, 7])
			else:
				speed[i] = raceline[idx, 3] + t * (raceline[idx+1, 3] - raceline[idx, 3])

	return speed

def plot_velocity_profiles(time, ego_states, ego_pose, opp_pose,  estimated_opp_pose, ax, local_path):
	ego_mask = (ego_pose[:,0]>=time[0]-1) & (ego_pose[:,0]<=time[2]+1)
	opp_mask = (opp_pose[:,0]>=time[0]-1) & (opp_pose[:,0]<=time[2]+1)
	ego_time = ego_pose[ego_mask,0]
	opp_time = opp_pose[opp_mask,0]
	ax.plot(ego_time, ego_pose[ego_mask,4], 'b-', label='Ego Vehicle Speed',linewidth=0.5)
	ax.plot(opp_time, opp_pose[opp_mask,4], 'r-', label='Opponent Vehicle Speed',linewidth=0.5)
	raceline = np.loadtxt(f'{map_path}/sep2_minCurve.csv', delimiter=',')
	ego_ref_speed = get_ref_speed(ego_pose[ego_mask,:], raceline)
	local_ref_speed = get_ref_speed(ego_pose[ego_mask,:], local_path)
	ax.plot(ego_time, ego_ref_speed, '--', label='Raceline Speed',linewidth=0.5)
	ax.plot(ego_time, local_ref_speed, '--', label='Local Planner Speed',linewidth=0.5)
	t0 = time[0]
	t1 = time[1]
	t2 = time[2]
	ax.axvline(t0, color='black', linestyle='--', linewidth=0.5)
	ax.axvline(t1, color='black', linestyle='--', linewidth=0.5)
	ax.axvline(t2, color='black', linestyle='--', linewidth=0.5)
	plt.xticks([t0, t1, t2], ['$t_1$', '$t_2$', '$t_3$'])
	plt.ylabel('Speed (m/s)')
	plt.xlabel('Time (s)')


	

def plot_progress(time, ego_states, ego_pose, opp_pose,  estimated_opp_pose, ax):
	raceline = np.loadtxt(f'{map_path}/sep2_minCurve.csv', delimiter=',')
	raceline = np.roll(raceline, -np.argmin(raceline[:,7]), axis=0)
	track_length = np.insert(np.cumsum(np.linalg.norm(np.diff(raceline[:,0:2], axis=0), axis=1)), 0, 0)
	total_length = track_length[-1]

	ego_mask = (ego_pose[:,0]>=time[0]-1) & (ego_pose[:,0]<=time[2]+1)

	opp_x_func = interp1d(opp_pose[:,0], opp_pose[:,1], kind='linear', fill_value="extrapolate")
	opp_y_func = interp1d(opp_pose[:,0], opp_pose[:,2], kind='linear', fill_value="extrapolate")
	opp_head_func = interp1d(opp_pose[:,0], opp_pose[:,3], kind='linear', fill_value="extrapolate")
	opp_v_func = interp1d(opp_pose[:,0], opp_pose[:,4], kind='linear', fill_value="extrapolate")

	opp_x = opp_x_func(ego_pose[ego_mask,0])
	opp_y = opp_y_func(ego_pose[ego_mask,0])
	opp_head = opp_head_func(ego_pose[ego_mask,0])
	opp_v = opp_v_func(ego_pose[ego_mask,0])
	opp_pose = np.vstack((ego_pose[ego_mask,0], opp_x, opp_y, opp_head, opp_v)).T
	opp_mask = (opp_pose[:,0]>=time[0]-1) & (opp_pose[:,0]<=time[2]+1)

	ego_time = ego_pose[ego_mask,0]
	opp_time = opp_pose[opp_mask,0]
	print(opp_pose[opp_mask].shape)
	ego_progress = progress_along_trajectory(ego_pose[ego_mask,1:3], raceline[:,0:2])
	opp_progress = progress_along_trajectory(opp_pose[opp_mask,1:3], raceline[:,0:2])
	print(f'ego progress: {ego_progress.shape}')
	print(f'opp progress: {opp_progress.shape}')
	diff_progress =  np.zeros(ego_progress.shape)
	for i in range(ego_progress.shape[0]):
		diff_progress[i] = (opp_progress[i] - ego_progress[i])%total_length if (opp_progress[i] - ego_progress[i])%total_length < total_length/2 else -((ego_progress[i] - opp_progress[i])%total_length)
	ax.plot(ego_time, np.abs(diff_progress), 'purple', label='Raceline Distance')
	t0 = time[0]
	t1 = time[1]
	t2 = time[2]
	ax.axvline(t0, color='black', linestyle='--', linewidth=0.5)
	ax.axvline(t1, color='black', linestyle='--', linewidth=0.5)
	ax.axvline(t2, color='black', linestyle='--', linewidth=0.5)
	plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
	ax.set_xticks([t0, t1, t2])
	ax.set_xticklabels(['$t_1$', '$t_2$', '$t_3$'])
	plt.ylabel('Distance between vehicles (m)')
	plt.xlabel('Time (s)')
	plt.plot(ego_time, np.linalg.norm(ego_pose[ego_mask,1:3]-opp_pose[opp_mask,1:3], axis=1), 'orange', label='Euclidean Distance')
	plt.legend()

def ittc(time, ego_states, ego_pose, opp_pose, ax):
	raceline = np.loadtxt(f'{map_path}/sep2_minCurve.csv', delimiter=',')
	raceline = np.roll(raceline, -np.argmin(raceline[:,7]), axis=0)
	track_length = np.insert(np.cumsum(np.linalg.norm(np.diff(raceline[:,0:2], axis=0), axis=1)), 0, 0)
	total_length = track_length[-1]

	ego_mask = (ego_pose[:,0]>=time[0]-1) & (ego_pose[:,0]<=time[2]+1)

	opp_x_func = interp1d(opp_pose[:,0], opp_pose[:,1], kind='linear', fill_value="extrapolate")
	opp_y_func = interp1d(opp_pose[:,0], opp_pose[:,2], kind='linear', fill_value="extrapolate")
	opp_head_func = interp1d(opp_pose[:,0], opp_pose[:,3], kind='linear', fill_value="extrapolate")
	opp_v_func = interp1d(opp_pose[:,0], opp_pose[:,4], kind='linear', fill_value="extrapolate")

	opp_x = opp_x_func(ego_pose[ego_mask,0])
	opp_y = opp_y_func(ego_pose[ego_mask,0])
	opp_head = opp_head_func(ego_pose[ego_mask,0])
	opp_v = opp_v_func(ego_pose[ego_mask,0])
	opp_pose = np.vstack((ego_pose[ego_mask,0], opp_x, opp_y, opp_head, opp_v)).T
	opp_mask = (opp_pose[:,0]>=time[0]-1) & (opp_pose[:,0]<=time[2]+1)

	ego_time = ego_pose[ego_mask,0]
	opp_time = opp_pose[opp_mask,0]
	print(opp_pose[opp_mask].shape)
	ego_progress = progress_along_trajectory(ego_pose[ego_mask,1:3], raceline[:,0:2])
	opp_progress = progress_along_trajectory(opp_pose[opp_mask,1:3], raceline[:,0:2])
	print(f'ego progress: {ego_progress.shape}')
	print(f'opp progress: {opp_progress.shape}')
	diff_progress =  np.zeros(ego_progress.shape)
	for i in range(ego_progress.shape[0]):
		diff_progress[i] = (opp_progress[i] - ego_progress[i])%total_length if (opp_progress[i] - ego_progress[i])%total_length < total_length/2 else -((ego_progress[i] - opp_progress[i])%total_length)

	ittc_values = np.abs(diff_progress) / (ego_pose[ego_mask,4] - opp_pose[opp_mask,4] + 1e-5)  # Add small value to avoid division by zero
	ittc_values = np.clip(ittc_values, -10, 10)  # Clip negative values
	ax.plot(ego_time, ittc_values, 'magenta', label='ITTC (s)', linewidth=0.5)
	return ittc_values
	

	

speeds = [0,25,50,55]
# speeds= [55]
root_path = "C:/Users/bbdnet2985/Desktop/Matsters/Report/Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/Overtake/real/overtake"
for speed in speeds:
	save_path = f"{root_path}/{speed}/"
	ego_pose = np.loadtxt(f"{root_path}/{speed}/ego_pose.csv", delimiter=',')
	opp_pose = np.loadtxt(f"{root_path}/{speed}/opp_pose.csv", delimiter=',')
	ego_states = np.loadtxt(f"{root_path}/{speed}/state.csv", delimiter=',')
	estimated_opp_pose = np.loadtxt(f"{root_path}/{speed}/track.csv", delimiter=',')
	# if speed == 70:
	# 	# roll opp pose 1 time step back
	# 	opp_pose[:,0] = np.roll(opp_pose[:,0], -1, axis=0)
	# 	estimated_opp_pose[:,0] = np.roll(estimated_opp_pose[:,0], -1, axis=0)
	waypoints_path = f"{root_path}/{speed}/waypoints"
	wp = {}
	wp_times = []
	for file in os.listdir(f'{waypoints_path}/'):
		if file.endswith('.csv'):
			f=file.split('.csv')[0].split('_')[-1]
			wp[f] = np.loadtxt(f'{waypoints_path}/{file}', delimiter=',')
			# plt.plot(wp[f][:,0], wp[f][:,1], 'orange', linewidth=2)
			
			wp_times.append(float(f))
	# #plt.show()
	wp_times.sort()
	
	times = []
	for i in range(ego_states.shape[0]):
		if ego_states[i,1]==1:
			if i+2<ego_states.shape[0]:
				times += [[ego_states[i,0], ego_states[i+1,0], ego_states[i+2,0]]]
			
	print(f'number of overtakes at {speed} km/h: {len(times)}')

	# if True:
	# 	for idx, time in enumerate(times):
	# 		print(f"Time interval {idx}: {time}")
	# 		fig = plt.figure()
	# 		fig.suptitle(f'Overtake Maneuver at {speed} km/h - Overtake {idx}')
	# 		gs = fig.add_gridspec(2, 3)
	# 		ego_sample_rate = 11
	# 		opp_sample_rate = 11
	# 		ax1 = fig.add_subplot(gs[0,0])
	# 		t0 = time[0]-5
	# 		t1 = time[0]
	# 		plot_vehicles(t0, t1, ego_pose, opp_pose, estimated_opp_pose, ax1, ego_sample_rate, opp_sample_rate)

	# 		ax2 = fig.add_subplot(gs[0,1])
	# 		t0 = time[0]
	# 		t1 = time[1]
	# 		plot_vehicles(t0, t1, ego_pose, opp_pose, estimated_opp_pose, ax2, ego_sample_rate, opp_sample_rate)

	# 		ax3 = fig.add_subplot(gs[1,0])
	# 		t0 = time[1]
	# 		t1 = time[2]
	# 		plot_vehicles(t0, t1, ego_pose, opp_pose, estimated_opp_pose, ax3, ego_sample_rate, opp_sample_rate)
	# 		w = []
	# 		for wp_time in wp_times:
	# 			print(f'Checking waypoint time: {wp_time} between {t0-0.1} and {t1}')
	# 			if float(wp_time)>=t0-0.1 and float(wp_time)<=t1-0.1:
	# 				w.append(float(wp_time))
	# 		w.sort()
	# 		print(w)
	# 		for i, wt in enumerate(w):
	# 			key = f'{wt:.6f}'
	# 			print(f'Plotting waypoints for time {key}')
	# 			wp_ = wp[key]
	# 			# print(wp_)
	# 			ax3.plot(wp_[:,0], wp_[:,1], 'orange', linewidth=2, label='Overtake Waypoints' if i==0 else None)
			
	# 		ax4 = fig.add_subplot(gs[1,1])
	# 		t0 = time[2]
	# 		t1 = time[2]+5
	# 		plot_vehicles(t0, t1, ego_pose, opp_pose, estimated_opp_pose, ax4, ego_sample_rate, opp_sample_rate)

	# 		ax5 = fig.add_subplot(gs[0,2])
	# 		plot_velocity_profiles(time, ego_states, ego_pose, opp_pose, estimated_opp_pose, ax5, wp[key])
	# 		ax6 = fig.add_subplot(gs[1,2])
	# 		plot_progress(time, ego_states, ego_pose, opp_pose, estimated_opp_pose, ax6)

	# 	# #plt.show()
		

	if speed == 0:
		time_id = 24
	if speed == 25:
		time_id = 20
	if speed == 50:
		time_id = 4
	if speed == 55:
		time_id = 6

	# times = times[time_id]
	ego_sample_rate = 6
	opp_sample_rate = 3
	time = times[time_id]
	size=(10/2.54, (10/2.54)*3.5/6.5)
	plt.figure(figsize=size)
	t0 = time[0]-1
	t1 = time[0]
	plt.axis('equal')
	plt.grid(False)
	plt.axis('off')
	plot_map()
	plot_vehicles(t0, t1, ego_pose, opp_pose, estimated_opp_pose, plt.gca(), ego_sample_rate=ego_sample_rate, opp_sample_rate=opp_sample_rate)
	# add raceline, opponent vehicle and ego vehicle labels to legend

	# add dotted line raceline to legend

	# plt.legend(handles=[blue_patch, red_patch, green_patch], loc='lower right')
	
	plt.savefig(f'{save_path}/global.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0,  transparent=True, format='pdf')
	# #plt.show()

	plt.figure(figsize=size)
	t0 = time[0]
	t1 = time[1]
	plot_map()
	plt.axis('equal')
	plt.grid(False)
	plt.axis('off')
	plot_vehicles(t0, t1, ego_pose, opp_pose, estimated_opp_pose, plt.gca(), ego_sample_rate=ego_sample_rate, opp_sample_rate=opp_sample_rate)
	plt.savefig(f'{save_path}/planning.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0,  transparent=True, format='pdf')
	# #plt.show()

	plt.figure(figsize=size)
	t0 = time[1]
	t1 = time[2]
	w = []
	for wp_time in wp_times:
		print(f'Checking waypoint time: {wp_time} between {t0-0.1} and {t1}')
		if float(wp_time)>=t0-0.1 and float(wp_time)<=t1-0.1:
			w.append(float(wp_time))
	w.sort()
	print(w)
	for i, wt in enumerate(w):
		key = f'{wt:.6f}'
		print(f'Plotting waypoints for time {key}')
		wp_ = wp[key]
		# print(wp_)
		plt.plot(wp_[:,0], wp_[:,1], linestyle='-.', color='orange', linewidth=1, label='Overtake Path' if i==0 else None)
	plot_map()
	plot_vehicles(t0, t1, ego_pose, opp_pose, estimated_opp_pose, plt.gca(), ego_sample_rate=ego_sample_rate, opp_sample_rate=opp_sample_rate)
	plt.legend(loc='upper center')
	plt.axis('equal')
	plt.grid(False)
	plt.axis('off')
	plt.savefig(f'{save_path}/overtake.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0,  transparent=True, format='pdf')
	# #plt.show()

	plt.figure(figsize=size)
	t0 = time[2]
	t1 = time[2]+1
	plot_map()
	plt.axis('equal')
	plt.grid(False)
	plt.axis('off')
	plot_vehicles(t0, t1, ego_pose, opp_pose, estimated_opp_pose, plt.gca(), ego_sample_rate=ego_sample_rate, opp_sample_rate=opp_sample_rate)
	# plt.legend(loc ='lower right')
	#  added the overtake waypoints to the legend
	# orange_patch = mpatches.Patch(color='orange', label='Overtake Waypoints')
	# plt.legend(handles=[blue_patch, red_patch, green_patch, orange_patch], loc='lower right')
	plt.savefig(f'{save_path}/post_overtake.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0,  transparent=True, format='pdf')
	#plt.show()


	plt.figure(figsize=(8/2.54, 8/2.54))
	for wt in w:
		key = f'{wt:.6f}'
		plot_velocity_profiles(time, ego_states, ego_pose, opp_pose, estimated_opp_pose, plt.gca(), wp[key])
	if speed == 0:
		plt.legend(loc='lower right')
	plt.savefig(f'{save_path}/speed_tracking.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0,  transparent=True, format='pdf')
	#plt.show()

	plt.figure(figsize=(8/2.54, 8/2.54))
	plot_progress(time, ego_states, ego_pose, opp_pose, estimated_opp_pose, plt.gca())
	plt.savefig(f'{save_path}/progress_difference.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0,  transparent=True, format='pdf')
	#plt.show()





	# fig, ax1 = plt.subplots(figsize=(8/2.54, 8/2.54))

	# # --- Front (velocity profiles) ---
	# plot_velocity_profiles(time, ego_states, ego_pose, opp_pose, estimated_opp_pose, ax1, wp[key])

	# # --- Back (ITTC curve) ---
	# ax2 = ax1.twinx()
	# ax2.set_ylabel('ITTC (s)')
	# ax2.patch.set_alpha(0)  # make background transparent
	# ittc_values = ittc(time, ego_states, ego_pose, opp_pose, ax2)

	# # --- Layering: ax2 behind ax1 ---
	# ax1.set_zorder(ax2.get_zorder() + 1)
	# ax1.patch.set_visible(False)

	# # --- Combine legends from both axes ---
	# handles1, labels1 = ax1.get_legend_handles_labels()
	# handles2, labels2 = ax2.get_legend_handles_labels()
	# ax1.legend(handles1 + handles2, labels1 + labels2)

	# fig.tight_layout()
	# plt.savefig(f'{save_path}/speed_tracking.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0,  transparent=True, format='pdf')
	# #plt.show()
