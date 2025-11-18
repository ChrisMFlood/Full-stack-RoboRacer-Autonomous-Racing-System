import numpy as np 
import matplotlib.pyplot as plt
import skimage
import yaml
import scipy

map_names = ['esp', 'gbr', 'aut', 'mco']
map_display_names = {'esp': 'Spain', 'gbr': 'Silverstone', 'aut': 'Austria', 'mco': 'Monaco'}

# Store results for summary table
all_results = {}

plt.rcParams.update({
		# Base font and size settings
		'font.size': 10,           # Base font size
		'font.family': 'serif',
		'font.serif': ['Caladea', 'TeX Gyre Termes', 'Times New Roman', 'DejaVu Serif'],
		'font.sans-serif': ['Carlito', 'TeX Gyre Heros', 'DejaVu Sans'],
		'font.monospace': ['Computer Modern Typewriter', 'DejaVu Sans Mono'],
		
		# Text rendering
		'text.usetex': False,      # Set to True if you have LaTeX installed
		'text.latex.preamble': r'\usepackage{amsmath,amssymb,amsfonts}',
		
		# Element sizes
		'axes.titlesize': 10,      # Title font size
		'axes.labelsize': 10,      # Axis label font size
		'xtick.labelsize': 9,      # X-axis tick label size
		'ytick.labelsize': 9,      # Y-axis tick label size
		'legend.fontsize': 8,      # Legend font size
		'figure.titlesize': 11,    # Figure title size
		
		# Figure properties
		'figure.autolayout': True, # Automatically adjust subplot params
		'figure.dpi': 300,
		
		# Save settings
		'savefig.dpi': 300,        # High DPI for PDF exports
		'savefig.format': 'pdf',   # Default save format
		'savefig.bbox': 'tight',   # Tight bounding box
		'savefig.pad_inches': 0.1, # Small padding
		'savefig.transparent': True, # Transparent background
		
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
		'legend.fancybox': False
	})

print("Matplotlib configured for LaTeX-style formatting")
print("Text rendering:", plt.rcParams['text.usetex'])
print("Font family:", plt.rcParams['font.family'])

def load_parameter_file(yaml_file):
	with open(yaml_file, 'r') as file:
		map_metadata = yaml.safe_load(file)
		map_resolution = map_metadata['resolution']
		origin = map_metadata['origin']
	return map_resolution, origin

def nearest_point(point, trajectory):
	"""
	Return the nearest point along the given piecewise linear trajectory.

	Args:
		point (numpy.ndarray, (2, )): (x, y) of current pose
		trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
			NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world

	Returns:
		nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
		nearest_dist (float): distance to the nearest point
		t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
		i (int): index of nearest point in the array of trajectory waypoints
	"""

	trajectory = trajectory[:, :2]
	diffs = trajectory[1:] - trajectory[:-1]
	l2s = np.sum(diffs**2, axis=1)
	dots = np.einsum('ij,ij->i', point - trajectory[:-1], diffs)
	t = np.clip(dots / l2s, 0.0, 1.0)
	projections = trajectory[:-1] + t[:, np.newaxis] * diffs
	dists = np.linalg.norm(point - projections, axis=1)
	min_dist_segment = np.argmin(dists)
	return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

for map_name in map_names:
	print(f"\n{'='*60}")
	print(f"Processing map: {map_name.upper()}")
	print(f"{'='*60}")
	
	# Load map to get boundaries and raceline
	map_path = 'C:/Users/bbdnet2985/Desktop/Matsters/Report/Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/maps'
	map_yaml_path = f"{map_path}/{map_name}.yaml"
	map_img = plt.imread(f'{map_path}/{map_name}.png')
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

	raceline = np.loadtxt(f'{map_path}/{map_name}_minCurve.csv', delimiter=',')

	# def plot_map():
	#     plt.scatter(by, bx, c='black',marker='s',s=1)  # Plot the boundaries
	#     # plt.plot(raceline[:,0], raceline[:,1], label='Raceline')  # Plot the raceline
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


	# plt.figure()
	# # plt.plot(raceline[:,0], raceline[:,1], c='red', linewidth=2)  # Plot the raceline
	# plot_map()
	# plt.show()

	file_path = f"C:/Users/bbdnet2985/Desktop/Matsters/Report/Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/pf/{map_name}"
	true_pose = np.loadtxt(f'{file_path}/ego_pose.csv',skiprows=1,delimiter=',')
	naive_pose = np.loadtxt(f'{file_path}/naive_pose.csv',skiprows=1,delimiter=',')
	diff_pose = np.loadtxt(f'{file_path}/diff_pose.csv',skiprows=1,delimiter=',')

	#find where true pose has the same time stamps as naive pose
	true_indices = np.searchsorted(true_pose[:,0], naive_pose[:,0])
	true_pose = true_pose[true_indices]

	true_naive_x_func = scipy.interpolate.interp1d(true_pose[:,0], true_pose[:,1], fill_value="extrapolate")
	true_naive_y_func = scipy.interpolate.interp1d(true_pose[:,0], true_pose[:,2], fill_value="extrapolate")
	true_naive_h_func = scipy.interpolate.interp1d(true_pose[:,0], true_pose[:,3], fill_value="extrapolate")
	interp_x = true_naive_x_func(naive_pose[:,0])
	interp_y = true_naive_y_func(naive_pose[:,0])
	interp_h = true_naive_h_func(naive_pose[:,0])
	true_naive_pose = np.column_stack((naive_pose[:,0], interp_x, interp_y, interp_h))

	interp_x = true_naive_x_func(diff_pose[:,0])
	interp_y = true_naive_y_func(diff_pose[:,0])
	interp_h = true_naive_h_func(diff_pose[:,0])
	true_diff_pose = np.column_stack((diff_pose[:,0], interp_x, interp_y, interp_h))

	# print(f'True pose shape: {true_pose.shape}')
	# print(f'Naive pose shape: {naive_pose.shape}')
	# print(f'Diff pose shape: {diff_pose.shape}')

	# true_fn_x = scipy.interpolate.interp1d(true_pose[:,0], true_pose[:,1], fill_value="extrapolate")
	# true_fn_y = scipy.interpolate.interp1d(true_pose[:,0], true_pose[:,2], fill_value="extrapolate")
	# true_fn_h = scipy.interpolate.interp1d(true_pose[:,0], true_pose[:,3], fill_value="extrapolate")

	# interp_x = true_fn_x(naive_pose[:,0])
	# interp_y = true_fn_y(naive_pose[:,0])
	# interp_h = true_fn_h(naive_pose[:,0])

	# true_pose = np.column_stack((naive_pose[:,0], interp_x, interp_y, interp_h))

	# rotate naive and diff to be the the true frame


	naive_abs_pose_error = np.abs(np.linalg.norm(naive_pose[:,1:3]-true_naive_pose[:,1:3],axis=1))
	diff_abs_pose_error = np.abs(np.linalg.norm(diff_pose[:,1:3]-true_diff_pose[:,1:3],axis=1))
	#sort errors
	naive_abs_pose_error = np.sort(naive_abs_pose_error)
	diff_abs_pose_error = np.sort(diff_abs_pose_error)

	naive_pose_rmse = np.sqrt(np.mean(naive_abs_pose_error**2))
	diff_pose_rmse = np.sqrt(np.mean(diff_abs_pose_error**2))
	# print(f'Naive pose RMSE: {naive_pose_rmse:.3f} m')
	# print(f'Diff pose RMSE: {diff_pose_rmse:.3f} m')
	# IQR for outliers
	naive_q75, naive_q25 = np.percentile(naive_abs_pose_error, [75 ,25])
	naive_iqr = naive_q75 - naive_q25
	diff_q75, diff_q25 = np.percentile(diff_abs_pose_error, [75 ,25])
	diff_iqr = diff_q75 - diff_q25
	diff_outlier_threshold_high = diff_q75 + 1.5 * diff_iqr
	naive_outlier_threshold = naive_q75 + 1.5 * naive_iqr

	#histogram of errors
	max_error = max(np.max(naive_abs_pose_error), np.max(diff_abs_pose_error))
	bins = np.linspace(0, max_error, 50)
	plt.figure(figsize=(8/2.54,8/2.54))
	plt.hist(naive_abs_pose_error, bins=bins, alpha=0.5, label='Naive Pose Error', density=True)
	plt.hist(diff_abs_pose_error, bins=bins, alpha=0.5, label='Diff Pose Error', density=True)
	plt.xlabel('Absolute Position Error (m)')
	plt.ylabel('Probability Density')
	# show rmse
	plt.axvline(naive_pose_rmse, color='blue', linestyle='dashed', linewidth=1, label='Naive Pose RMSE')
	plt.axvline(diff_pose_rmse, color='red', linestyle='dashed', linewidth=1, label='Diff Pose RMSE')
	plt.legend()
	# plt.grid()
	plt.savefig(f'{file_path}/pose_error_histogram.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0, transparent=True, format='pdf')
	plt.close()


	# print(f'Naive pose IQR: {naive_iqr:.3f} m')
	# print(f'Diff pose IQR: {diff_iqr:.3f} m')
	# print(f'Diff pose outlier threshold: {diff_outlier_threshold_high:.3f} m')
	# print(f'Naive pose outlier threshold: {naive_outlier_threshold:.3f} m')

	# #outliers
	# print(f'Number of naive outliers: {np.sum(naive_abs_pose_error>naive_outlier_threshold)}')
	# print(f'Number of diff outliers: {np.sum(diff_abs_pose_error>diff_outlier_threshold_high)}')
	# #outlier percentage
	# print(f'Naive outlier percentage: {np.sum(naive_abs_pose_error>naive_outlier_threshold)/len(naive_abs_pose_error)*100:.3f} %')
	# print(f'Diff outlier percentage: {np.sum(diff_abs_pose_error>diff_outlier_threshold_high)/len(diff_abs_pose_error)*100:.3f} %')

	# # error greater than 0.3m
	# print(f'Number of naive errors > 0.3m: {np.sum(naive_abs_pose_error>0.3)}')
	# print(f'Number of diff errors > 0.3m: {np.sum(diff_abs_pose_error>0.3)}')
	# print(f'Naive errors > 0.3m percentage: {np.sum(naive_abs_pose_error>0.3)/len(naive_abs_pose_error)*100:.3f} %')
	# print(f'Diff errors > 0.3m percentage: {np.sum(diff_abs_pose_error>0.3)/len(diff_abs_pose_error)*100:.3f} %')

	# print(f'Mean naive error: {np.mean(naive_abs_pose_error):.3f} m')
	# print(f'Mean diff error: {np.mean(diff_abs_pose_error):.3f} m')
	# print(f'Max naive error: {np.max(naive_abs_pose_error):.3f} m')
	# print(f'Max diff error: {np.max(diff_abs_pose_error):.3f} m')
	# print(f'Standard deviation naive error: {np.std(naive_abs_pose_error):.3f} m')
	# print(f'Standard deviation diff error: {np.std(diff_abs_pose_error):.3f} m')

	naive_heading_error = true_naive_pose[:,3]-naive_pose[:,3]
	naive_heading_error = np.arctan2(np.sin(naive_heading_error), np.cos(naive_heading_error))
	diff_heading_error = true_diff_pose[:,3]-diff_pose[:,3]
	diff_heading_error = np.arctan2(np.sin(diff_heading_error), np.cos(diff_heading_error))
	#sort errors
	naive_heading_error = np.sort(np.abs(naive_heading_error))
	diff_heading_error = np.sort(np.abs(diff_heading_error))
	naive_heading_rmse = np.sqrt(np.mean(naive_heading_error**2))
	diff_heading_rmse = np.sqrt(np.mean(diff_heading_error**2))
	#histogram of errors
	max_error = max(np.max(np.degrees(naive_heading_error)), np.max(np.degrees(diff_heading_error)))
	bins = np.linspace(0, max_error, 50)
	# bins=50
	plt.figure(figsize=(8/2.54,8/2.54))
	plt.hist(np.degrees(naive_heading_error), bins=bins, alpha=0.5, label='Naive Heading Error', density=True)
	plt.hist(np.degrees(diff_heading_error), bins=bins, alpha=0.5, label='Diff Heading Error', density=True)
	# show rmse
	plt.axvline(np.degrees(naive_heading_rmse), color='blue', linestyle='dashed', linewidth=1, label='Naive Heading RMSE')
	plt.axvline(np.degrees(diff_heading_rmse), color='red', linestyle='dashed', linewidth=1, label='Diff Heading RMSE')
	plt.xlabel('Absolute Heading Error ($^\circ$)')
	plt.ylabel('Probability Density')
	plt.legend()
	# plt.grid()
	plt.savefig(f'{file_path}/heading_error_histogram.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0, transparent=True, format='pdf')
	plt.close()
	#rmse
	# print(f'Naive heading RMSE: {np.degrees(naive_heading_rmse):.3f} deg')
	# print(f'Diff heading RMSE: {np.degrees(diff_heading_rmse):.3f} deg')
	# IQR for outliers
	naive_heading_iqr = np.percentile(naive_heading_error, 75) - np.percentile(naive_heading_error, 25)
	diff_heading_iqr = np.percentile(diff_heading_error, 75) - np.percentile(diff_heading_error, 25)
	# print(f'Naive heading IQR: {np.degrees(naive_heading_iqr):.3f} deg')
	# print(f'Diff heading IQR: {np.degrees(diff_heading_iqr):.3f} deg')
	#outliers
	naive_heading_outlier_threshold_high = np.percentile(naive_heading_error, 75) + 1.5 * naive_heading_iqr
	diff_heading_outlier_threshold_high = np.percentile(diff_heading_error, 75) + 1.5 * diff_heading_iqr
	# print(f'Naive heading outlier threshold: {np.degrees(naive_heading_outlier_threshold_high):.3f} deg')
	# print(f'Diff heading outlier threshold: {np.degrees(diff_heading_outlier_threshold_high):.3f} deg')
	# print(f'Number of naive heading outliers: {np.sum(naive_heading_error>naive_heading_outlier_threshold_high)}')
	# print(f'Number of diff heading outliers: {np.sum(diff_heading_error>diff_heading_outlier_threshold_high)}')
	# #outlier percentage
	# print(f'Naive heading outlier percentage: {np.sum(naive_heading_error>naive_heading_outlier_threshold_high)/len(naive_heading_error)*100:.3f} %')
	# print(f'Diff heading outlier percentage: {np.sum(diff_heading_error>diff_heading_outlier_threshold_high)/len(diff_heading_error)*100:.3f} %')
	# print(f'Standard deviation naive heading error: {np.degrees(np.std(naive_heading_error)):.3f} deg')
	# print(f'Standard deviation diff heading error: {np.degrees(np.std(diff_heading_error)):.3f} deg')

	# print a table model, pose: rmse std max, heading: rmse std max
	print(f'{"Model":<20} {"Pose RMSE (m)":<15} {"Pose Std (m)":<15} {"Pose Max (m)":<15} {"Heading RMSE (deg)":<20} {"Heading Std (deg)":<20} {"Heading Max (deg)":<20}')
	print(f'{"Naive":<20} {naive_pose_rmse:<15.3f} {np.std(naive_abs_pose_error):<15.3f} {np.max(naive_abs_pose_error):<15.3f} {np.degrees(naive_heading_rmse):<20.3f} {np.degrees(np.std(naive_heading_error)):<20.3f} {np.degrees(np.max(naive_heading_error)):<20.3f}')
	print(f'{"Differential Drive":<20} {diff_pose_rmse:<15.3f} {np.std(diff_abs_pose_error):<15.3f} {np.max(diff_abs_pose_error):<15.3f} {np.degrees(diff_heading_rmse):<20.3f} {np.degrees(np.std(diff_heading_error)):<20.3f} {np.degrees(np.max(diff_heading_error)):<20.3f}')

	naive_out = np.zeros(true_naive_pose.shape[0], dtype=bool)
	diff_out = np.zeros(true_diff_pose.shape[0], dtype=bool)
	for i in range(true_naive_pose.shape[0]):
		if naive_heading_error[i] > 2*naive_heading_rmse:
			naive_out[i] = True
		if naive_abs_pose_error[i] > 2*naive_pose_rmse:
			naive_out[i] = True
	
	for i in range(true_diff_pose.shape[0]):
		if diff_heading_error[i] > 2*diff_heading_rmse:
			diff_out[i] = True
		if diff_abs_pose_error[i] > 2*diff_pose_rmse:
			diff_out[i] = True
		
	# for i in range(true_pose.shape[0]):
	# 	if naive_heading_error[i] > 2*naive_heading_rmse:
	# 		naive_out[i] = True
	# 	if diff_heading_error[i] > 2*diff_heading_rmse:
	# 		diff_out[i] = True

	# 	if naive_abs_pose_error[i] > 2*naive_pose_rmse:
	# 		naive_out[i] = True
	# 	if diff_abs_pose_error[i] > 2*diff_pose_rmse:
	# 		diff_out[i] = True

	# print(f'Naive outlier indices: {np.sum(naive_out)}')
	# print(f'Differential Drive outlier indices: {np.sum(diff_out)}')
	#percentage of outliers
	print(f'Naive outlier percentage: {np.sum(naive_out)/len(naive_out)*100:.3f} %')
	print(f'Differential Drive outlier percentage: {np.sum(diff_out)/len(diff_out)*100:.3f} %')
	
	# Store results for this map
	all_results[map_name] = {
		'naive': {
			'pose_rmse': naive_pose_rmse,
			'pose_std': np.std(naive_abs_pose_error),
			'pose_max': np.max(naive_abs_pose_error),
			'heading_rmse': np.degrees(naive_heading_rmse),
			'heading_std': np.degrees(np.std(naive_heading_error)),
			'heading_max': np.degrees(np.max(naive_heading_error)),
			'outlier_pct': np.sum(naive_out)/len(naive_out)*100
		},
		'diff': {
			'pose_rmse': diff_pose_rmse,
			'pose_std': np.std(diff_abs_pose_error),
			'pose_max': np.max(diff_abs_pose_error),
			'heading_rmse': np.degrees(diff_heading_rmse),
			'heading_std': np.degrees(np.std(diff_heading_error)),
			'heading_max': np.degrees(np.max(diff_heading_error)),
			'outlier_pct': np.sum(diff_out)/len(diff_out)*100
		}
	}
	
	# get laps
	# divide the pose into laps around a racetrack
	# find the nearest point on the raceline to the start point
	def find_lap_indices(poses, raceline):
		start_point = poses[0, 1:3]
		_, _, _, start_index = nearest_point(start_point, raceline[:, :2])

		laps = np.zeros(poses.shape[0], dtype=int)
		l = 1
		prev_index = start_index

		for i in range(poses.shape[0]):
			point = poses[i, 1:3]
			_, _, _, index = nearest_point(point, raceline[:, :2])
			# Detect crossing the start line (from high index to low index)
			if index < prev_index:  # Use a threshold to avoid noise
				l += 1
			laps[i] = l
			prev_index = index
		return laps

	true_laps = find_lap_indices(true_pose, raceline)
	# plt.plot(true_laps)
	# plt.show()

	laps = np.unique(true_laps)

	# for l in laps[1:-1]:
	#     plt.figure(figsize=(15/2.54,15/2.54))
	#     # plt.title(f'Lap {l}')
	#     idx = np.where(true_laps==l)[0]
	#     plt.plot(true_pose[idx,1], true_pose[idx,2], label='True Pose', alpha=0.5, linestyle='-')
	#     plt.plot(naive_pose[idx,1], naive_pose[idx,2], label='Naive Pose', alpha=0.5, linestyle='--')
	#     plt.plot(diff_pose[idx,1], diff_pose[idx,2], label='Diff Pose', alpha=0.5, linestyle='-.')
	#     plot_map()
	#     plt.legend()
	#     if l == 4:
	#         plt.savefig(f'{file_path}/lap_{l}_comparison.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0, transparent=True, format='pdf')
	#     # plt.show()

	#     plt.figure()
	#     plt.plot(true_pose[idx,0], np.linalg.norm(true_pose[idx,1:3]-naive_pose[idx,1:3],axis=1), label='Naive Pose', linestyle='--')
	#     plt.plot(true_pose[idx,0], np.linalg.norm(true_pose[idx,1:3]-diff_pose[idx,1:3],axis=1), label='Diff Pose', linestyle='-.')
	#     plt.xlabel('Time (s)')
	#     plt.ylabel('Position Error (m)')
	#     plt.legend()
	#     if l == 4:
	#         plt.savefig(f'{file_path}/lap_{l}_errors.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0, transparent=True, format='pdf')
	#     # plt.show()

	#     plt.figure()
	#     plt.plot(true_pose[idx,0], np.abs((true_pose[idx,3]-naive_pose[idx,3]+np.pi)%(2*np.pi)-np.pi), label='Naive Pose', linestyle='--')
	#     plt.plot(true_pose[idx,0], np.abs((true_pose[idx,3]-diff_pose[idx,3]+np.pi)%(2*np.pi)-np.pi), label='Diff Pose', linestyle='-.')
	#     plt.xlabel('Time (s)')
	#     plt.ylabel('Heading Error (rad)')
	#     plt.legend()
	#     if l == 4:
	#         plt.savefig(f'{file_path}/lap_{l}_heading_errors.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0, transparent=True, format='pdf')    

	#     plt.show()

	# plt.figure()
	# plt.plot(true_pose[:,1], true_pose[:,2], label='True Pose', linewidth=2)
	# plt.plot(naive_pose[:,1], naive_pose[:,2], label='Naive Pose', linewidth=2)
	# plt.plot(diff_pose[:,1], diff_pose[:,2], label='Diff Pose', linewidth=2)
	# # plot_map()
	# plt.show()

# Print summary table for LaTeX
print(f"\n{'='*120}")
print("LATEX TABLE DATA")
print(f"{'='*120}\n")

for map_name in map_names:
	results = all_results[map_name]
	display_name = map_display_names[map_name]
	
	# Determine best values (lower is better for all metrics)
	metrics = ['pose_rmse', 'pose_std', 'pose_max', 'heading_rmse', 'heading_std', 'heading_max', 'outlier_pct']
	
	# Format values with bold for best
	def format_value(naive_val, diff_val):
		if naive_val < diff_val:
			return f"\\textbf{{{naive_val:.3f}}}", f"{diff_val:.3f}"
		elif diff_val < naive_val:
			return f"{naive_val:.3f}", f"\\textbf{{{diff_val:.3f}}}"
		else:  # equal
			return f"{naive_val:.3f}", f"{diff_val:.3f}"
	
	naive_formatted = []
	diff_formatted = []
	
	for metric in metrics:
		naive_str, diff_str = format_value(results['naive'][metric], results['diff'][metric])
		naive_formatted.append(naive_str)
		diff_formatted.append(diff_str)
	
	# Print naive row
	print(f"    \\multirow{{2}}{{*}}{{{display_name}}} ")
	print(f"     & Naive & {' & '.join(naive_formatted)} \\\\")
	
	# Print diff drive row
	print(f"     & Diff. Drive & {' & '.join(diff_formatted)} \\\\")
	print("    \\hline\\hline")

# Calculate and print averages
avg_naive = {
	'pose_rmse': np.mean([all_results[m]['naive']['pose_rmse'] for m in map_names]),
	'pose_std': np.mean([all_results[m]['naive']['pose_std'] for m in map_names]),
	'pose_max': np.mean([all_results[m]['naive']['pose_max'] for m in map_names]),
	'heading_rmse': np.mean([all_results[m]['naive']['heading_rmse'] for m in map_names]),
	'heading_std': np.mean([all_results[m]['naive']['heading_std'] for m in map_names]),
	'heading_max': np.mean([all_results[m]['naive']['heading_max'] for m in map_names]),
	'outlier_pct': np.mean([all_results[m]['naive']['outlier_pct'] for m in map_names])
}

avg_diff = {
	'pose_rmse': np.mean([all_results[m]['diff']['pose_rmse'] for m in map_names]),
	'pose_std': np.mean([all_results[m]['diff']['pose_std'] for m in map_names]),
	'pose_max': np.mean([all_results[m]['diff']['pose_max'] for m in map_names]),
	'heading_rmse': np.mean([all_results[m]['diff']['heading_rmse'] for m in map_names]),
	'heading_std': np.mean([all_results[m]['diff']['heading_std'] for m in map_names]),
	'heading_max': np.mean([all_results[m]['diff']['heading_max'] for m in map_names]),
	'outlier_pct': np.mean([all_results[m]['diff']['outlier_pct'] for m in map_names])
}

# Format average values with bold for best
avg_naive_formatted = []
avg_diff_formatted = []

for metric in metrics:
	naive_str, diff_str = format_value(avg_naive[metric], avg_diff[metric])
	avg_naive_formatted.append(naive_str)
	avg_diff_formatted.append(diff_str)

print("     \\multirow{2}{*}{\\textbf{Average}}")
print(f"     & Naive & {' & '.join(avg_naive_formatted)} \\\\")
print(f"     & Diff. Drive & {' & '.join(avg_diff_formatted)} \\\\")
print("     \\hline\\hline")

print(f"\n{'='*120}")
print("END LATEX TABLE DATA")
print(f"{'='*120}\n")