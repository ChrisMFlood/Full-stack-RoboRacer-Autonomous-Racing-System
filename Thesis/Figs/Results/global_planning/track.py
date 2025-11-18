import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage
import yaml
from PIL import Image

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
		'savefig.pad_inches': 0.0, # Small padding
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

def plot_trajectory(map_name):
	print(f"Extracting centre line for: {map_name}")
	# Load map image
	map_path = 'Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/maps/'
	map_yaml_path = f"{map_path}/{map_name}.yaml"
	map_img = plt.imread(f'{map_path}/{map_name}.png')
	map_img = np.flipud(map_img)
	# map_img = scipy.ndimage.binary_dilation(map_img, iterations=1).astype(map_img.dtype)
	map_img = scipy.ndimage.distance_transform_edt(map_img)
	map_img = np.abs(map_img - 1)
	map_img[map_img!=0]=1
	# bx,by = np.where(map_img==0)
	with open(map_yaml_path, 'r') as yaml_stream:
		try:
			map_metadata = yaml.safe_load(yaml_stream)
			map_resolution = map_metadata['resolution']
			origin = map_metadata['origin']
		except yaml.YAMLError as ex:
			print(ex)
	# bx = bx * map_resolution + origin[1]
	# by = by * map_resolution + origin[0]

	if map_name == 'esp':
		#rotate map by 45 degrees
		theta = np.radians(45)
		c, s = np.cos(theta), np.sin(theta)
		R = np.array([[c, -s], [s, c]])
		# bx, by = (R @ np.vstack((bx-origin[1], by-origin[0]))).T + np.array([[origin[1]], [origin[0]]])
		# bx = bx.flatten()
		# by = by.flatten()
		map_img1 = scipy.ndimage.rotate(map_img, -45, reshape=True, order=1, mode='nearest', cval=0.0)
		# map_img[map_img!=0]=1
		print(np.unique(map_img1))
		plt.imshow(map_img1, cmap='gray', origin='lower')

		plt.show()


	# def find_lines(img, start, border):
	# 	DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
	# 	stack = [start]
	# 	while stack:
	# 		current = stack.pop()
	# 		for direction in DIRECTIONS:
	# 			neighbor = current + direction
	# 			# Check bounds
	# 			if (0 <= neighbor[0] < img.shape[0]) and (0 <= neighbor[1] < img.shape[1]):
	# 				if img[neighbor[0], neighbor[1]] == 0:
	# 					border.append([neighbor[0]*map_resolution + origin[1], neighbor[1]*map_resolution + origin[0]])
	# 					img[neighbor[0], neighbor[1]] = 1
	# 					stack.append(neighbor)
	# 	border = np.array(border)
	# 	return border

	# # img = np.copy(map_img)
	# # start = np.array(np.where(img==0)).T[0]
	# # border = [[start[0]*map_resolution + origin[1], start[1]*map_resolution + origin[0]]]
	# # border_out = find_lines(img, start, border)
	# # print(border_out)
	# # start = np.array(np.where(img==0)).T[0]
	# # border = [[start[0]*map_resolution + origin[1], start[1]*map_resolution + origin[0]]]
	# # border_in = find_lines(img, start, border)
	# # plt.imshow(img, cmap='gray', origin='lower', extent=[origin[0], origin[0]+map_img.shape[1]*map_resolution, origin[1], origin[1]+map_img.shape[0]*map_resolution])
	# # plt.plot(border_out[:,1],border_out[:,0])
	# # plt.plot(border_in[:,1],border_in[:,0])
	# # plt.show()

	centerline = np.loadtxt(f'Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/maps/{map_name}_centreline.csv',skiprows=1,delimiter=',')
	min_curve = np.loadtxt(f'Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/maps/{map_name}_minCurve.csv',skiprows=1,delimiter=',')
	short = np.loadtxt(f'Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/maps/{map_name}_short.csv',skiprows=1,delimiter=',')

	def plot_map(img):
		boundaries = skimage.measure.find_contours(img, 0.5)
		print(f"Found {len(boundaries)} boundaries")
		# print(boundaries)
		# for boundary in boundaries:
		# 	plt.plot(boundary[:,1]*map_resolution + origin[0], boundary[:,0]*map_resolution + origin[1], 'black', linewidth=0.1)
		boundary_out = boundaries[0]
		boundary_out[:,1] = boundary_out[:,1]*map_resolution + origin[0]
		boundary_out[:,0] = boundary_out[:,0]*map_resolution + origin[1]
		boundary_in = boundaries[-1]
		boundary_in[:,1] = boundary_in[:,1]*map_resolution + origin[0]
		boundary_in[:,0] = boundary_in[:,0]*map_resolution + origin[1]
		if map_name == 'esp':
			#rotate boundaries by 45 degrees
			theta = np.radians(45)
			c, s = np.cos(theta), np.sin(theta)
			R = np.array([[c, -s], [s, c]])
			boundary_out = boundary_out @ R
			boundary_in = boundary_in @ R
			# for boundary in boundaries:
			# 	plt.plot(boundary[:,1]*map_resolution + origin[0], boundary[:,0]*map_resolution + origin[1], 'black', linewidth=0.1)
		
		plt.plot(boundary_out[:,1], boundary_out[:,0], 'black', linewidth=0.5)
		plt.plot(boundary_in[:,1], boundary_in[:,0], 'black', linewidth=0.5)

	# plot_map(map_img)
	# plt.show()
	
	width = 8/2.54
	height = np.max([width*(map_img.shape[0]/map_img.shape[1]),width])
	if map_name == 'esp':
		width = 8/2.54
		height = np.max([width*(map_img1.shape[0]/map_img1.shape[1]),width])
	if map_name == 'map3':
		width = 8/2.54
		height = width*(map_img.shape[0]/map_img.shape[1])
	plt.figure(figsize=(width, height))
	# plt.imshow(map_img, cmap='gray', origin='lower', extent=[origin[0], origin[0]+map_img.shape[1]*map_resolution, origin[1], origin[1]+map_img.shape[0]*map_resolution])
	# plt.scatter(by, bx, s=0.01, c='black', label='Track Boundary')
	cx = centerline[:,0]
	cy = centerline[:,1]
	mcx = min_curve[:,0]
	mcy = min_curve[:,1]
	sx = short[:,0]
	sy = short[:,1]
	if map_name == 'esp':
		#rotate trajectories by 45 degrees
		theta = np.radians(-45)
		c, s = np.cos(theta), np.sin(theta)
		R = np.array([[c, -s], [s, c]])
		centerline[:,:2] = centerline[:,:2] @ R
		min_curve[:,:2] = min_curve[:,:2] @ R
		short[:,:2] = short[:,:2] @ R
		cx = centerline[:,0]
		cy = centerline[:,1]
		mcx = min_curve[:,0]
		mcy = min_curve[:,1]
		sx = short[:,0]
		sy = short[:,1]
	if map_name != 'map3':
		plt.plot(cx, cy, linewidth=0.5, linestyle='--', label=f'Centre Line (Time: {centerline[-1,9]:.2f} s)')
		plt.plot(sx, sy, linewidth=1, linestyle='-.', label=f'Shortest Path (Time: {short[-1,9]:.2f} s)')
		plt.plot(mcx, mcy, linewidth=1, label=f'Minimum Curvature (Time: {min_curve[-1,9]:.2f} s)')
	else:
		plt.plot(cx, cy, linewidth=0.5, linestyle='--', label=f'Centre Line')
		# plt.plot(sx, sy, linewidth=1, linestyle='-.', label=f'Shortest Path (Time: {short[-1,9]:.2f} s)')
		plt.plot(mcx, mcy, linewidth=1, label=f'Minimum Curvature')
	plot_map(map_img)
	# plt.title(f'Centre Line Extraction for {map_name.upper()} Map')
	plt.axis('equal')
	plt.axis('off')
	plt.legend()
	plt.savefig(f'Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/global_planning/{map_name}_traj.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0, transparent=True, format='pdf')
	plt.show()

	# print('min curvature stats:')
	# print(f'Mean: {np.mean(min_curve[:,5]):.2f} m^-1')
	# print(f'Max: {np.max(min_curve[:,5]):.2f} m^-1')
	# print(f'Std: {np.std(min_curve[:,5]):.2f} m^-1')

	# #distance to track boundaries
	# print('------------------------------------------------------------------------------------------------ ')
	# print('Distance to track boundaries:')
	# print(f'Centre line min right: {np.min(centerline[:,2]):.2f} m, mean: {np.mean(centerline[:,2]):.2f} m, max: {np.max(centerline[:,2]):.2f} m, std: {np.std(centerline[:,2]):.2f} m')
	# print(f'Centre line min left: {np.min(centerline[:,3]):.2f} m, mean: {np.mean(centerline[:,3]):.2f} m, max: {np.max(centerline[:,3]):.2f} m, std: {np.std(centerline[:,3]):.2f} m')
	# print(f'Minimum curvature min right: {np.min(min_curve[:,2]):.2f} m, mean: {np.mean(min_curve[:,2]):.2f} m, max: {np.max(min_curve[:,2]):.2f} m, std: {np.std(min_curve[:,2]):.2f} m')
	# print(f'Minimum curvature min left: {np.min(min_curve[:,3]):.2f} m, mean: {np.mean(min_curve[:,3]):.2f} m, max: {np.max(min_curve[:,3]):.2f} m, std: {np.std(min_curve[:,3]):.2f} m')
	# print(f'Shortest path min right: {np.min(short[:,2]):.2f} m, mean: {np.mean(short[:,2]):.2f} m, max: {np.max(short[:,2]):.2f} m, std: {np.std(short[:,2]):.2f} m')
	# print(f'Shortest path min left: {np.min(short[:,3]):.2f} m, mean: {np.mean(short[:,3]):.2f} m, max: {np.max(short[:,3]):.2f} m, std: {np.std(short[:,3]):.2f} m')
	# print('------------------------------------------------------------------------------------------------ ')
	# #print table for boundary distances
	# print(f'	# print('------------------------------------------------------------------------------------------------ ')
	# print(f'| {"Trajectory":<20} | {"Min Right (m)":<15} | {"Mean Right (m)":<15} | {"Max Right (m)":<15} | {"Std Right (m)":<15} | {"Min Left (m)":<15} | {"Mean Left (m)":<15} | {"Max Left (m)":<15} | {"Std Left (m)":<15} |')
	# print(f'| {"-"*20} | {"-"*15} | {"-"*15} | {"-"*15} | {"-"*15} | {"-"*15} | {"-"*15} | {"-"*15} | {"-"*15} |')
	# print(f'| {"Centre Line":<20} | {np.min(centerline[:,2]):<15.2f} | {np.mean(centerline[:,2]):<15.2f} | {np.max(centerline[:,2]):<15.2f} | {np.std(centerline[:,2]):<15.2f} | {np.min(centerline[:,3]):<15.2f} | {np.mean(centerline[:,3]):<15.2f} | {np.max(centerline[:,3]):<15.2f} | {np.std(centerline[:,3]):<15.2f} |')
	# print(f'| {"Minimum Curvature":<20} | {np.min(min_curve[:,2]):<15.2f} | {np.mean(min_curve[:,2]):<15.2f} | {np.max(min_curve[:,2]):<15.2f} | {np.std(min_curve[:,2]):<15.2f} | {np.min(min_curve[:,3]):<15.2f} | {np.mean(min_curve[:,3]):<15.2f} | {np.max(min_curve[:,3]):<15.2f} | {np.std(min_curve[:,3]):<15.2f} |')
	# print(f'| {"Shortest Path":<20} | {np.min(short[:,2]):<15.2f} | {np.mean(short[:,2]):<15.2f} | {np.max(short[:,2]):<15.2f} | {np.std(short[:,2]):<15.2f} | {np.min(short[:,3]):<15.2f} | {np.mean(short[:,3]):<15.2f} | {np.max(short[:,3]):<15.2f} | {np.std(short[:,3]):<15.2f} |')
	# print('------------------------------------------------------------------------------------------------ ')

	# Print trajectory comparison table in LaTeX format
	# print("\\begin{table}[htbp]")
	# print("\\centering")
	# print("\\caption{Trajectory comparison for the " + map_name.upper() + " track}")
	# print("\\label{tab:trajectory_" + map_name + "}")
	# print("\\begin{tabular}{lcccc}")
	# print("\\hline")
	# print("Trajectory & Lap Time (s) & Max Speed (m/s) & Min Speed (m/s) & Distance (m) \\\\")
	# print("\\hline")
	# print(f"Centre Line & {centerline[-1,9]:.2f} & {np.max(centerline[:,7]):.2f} & {np.min(centerline[:,7]):.2f} & {np.sum(np.linalg.norm(np.diff(centerline[:,:2], axis=0), axis=1)):.2f} \\\\")
	# print(f"Min Curvature & {min_curve[-1,9]:.2f} & {np.max(min_curve[:,7]):.2f} & {np.min(min_curve[:,7]):.2f} & {np.sum(np.linalg.norm(np.diff(min_curve[:,:2], axis=0), axis=1)):.2f} \\\\")
	# print(f"Shortest Path & {short[-1,9]:.2f} & {np.max(short[:,7]):.2f} & {np.min(short[:,7]):.2f} & {np.sum(np.linalg.norm(np.diff(short[:,:2], axis=0), axis=1)):.2f} \\\\")
	# print("\\hline")
	# print("\\end{tabular}")
	# print("\\end{table}")
	
	# Also print a table for markdown format (for documentation or README)
	print("\n\n# Markdown Table")
	print('Trajectory comparison for the ' + map_name.upper() + ' track')
	print('------------------------------------------------------------------------------------------------ ')
	print(f"| Trajectory | Lap Time (s) | Max Speed (m/s) | Min Speed (m/s) | Distance (m) |")
	print(f"| --- | --- | --- | --- | --- |")
	print(f"| Centre Line | {centerline[-1,9]:.2f} | {np.max(centerline[:,7]):.2f} | {np.min(centerline[:,7]):.2f} | {np.sum(np.linalg.norm(np.diff(centerline[:,:2], axis=0), axis=1)):.2f} |")
	print(f"| Min Curvature | {min_curve[-1,9]:.2f} | {np.max(min_curve[:,7]):.2f} | {np.min(min_curve[:,7]):.2f} | {np.sum(np.linalg.norm(np.diff(min_curve[:,:2], axis=0), axis=1)):.2f} |")
	print(f"| Shortest Path | {short[-1,9]:.2f} | {np.max(short[:,7]):.2f} | {np.min(short[:,7]):.2f} | {np.sum(np.linalg.norm(np.diff(short[:,:2], axis=0), axis=1)):.2f} |")
	# print(f'| {"-"*20} | {"-"*15} | {"-"*15} | {"-"*15} | {"-"*15} | {"-"*15} | {"-"*15} | {"-"*15} | {"-"*15} |')
	# print(f'| {"Centre Line":<20} | {np.min(centerline[:,2]):<15.2f} | {np.mean(centerline[:,2]):<15.2f} | {np.max(centerline[:,2]):<15.2f} | {np.std(centerline[:,2]):<15.2f} | {np.min(centerline[:,3]):<15.2f} | {np.mean(centerline[:,3]):<15.2f} | {np.max(centerline[:,3]):<15.2f} | {np.std(centerline[:,3]):<15.2f} |')
	# print(f'| {"Minimum Curvature":<20} | {np.min(min_curve[:,2]):<15.2f} | {np.mean(min_curve[:,2]):<15.2f} | {np.max(min_curve[:,2]):<15.2f} | {np.std(min_curve[:,2]):<15.2f} | {np.min(min_curve[:,3]):<15.2f} | {np.mean(min_curve[:,3]):<15.2f} | {np.max(min_curve[:,3]):<15.2f} | {np.std(min_curve[:,3]):<15.2f} |')
	# print(f'| {"Shortest Path":<20} | {np.min(short[:,2]):<15.2f} | {np.mean(short[:,2]):<15.2f} | {np.max(short[:,2]):<15.2f} | {np.std(short[:,2]):<15.2f} | {np.min(short[:,3]):<15.2f} | {np.mean(short[:,3]):<15.2f} | {np.max(short[:,3]):<15.2f} | {np.std(short[:,3]):<15.2f} |')
	# print('------------------------------------------------------------------------------------------------ ')

	# print table: track, centerline: lap time, max speed, min speed, distance, min curvature: lap time, max speed, min speed, distance, shortest distance: lap time, max speed, min speed, distance


	# def nearest_point(point, trajectory):
	# 	"""
	# 	Return the nearest point along the given piecewise linear trajectory.

	# 	Args:
	# 		point (numpy.ndarray, (2, )): (x, y) of current pose
	# 		trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
	# 			NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world

	# 	Returns:
	# 		nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
	# 		nearest_dist (float): distance to the nearest point (negative if point is on the right)
	# 		t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
	# 		i (int): index of nearest point in the array of trajectory waypoints
	# 	"""
	# 	diffs = trajectory[1:] - trajectory[:-1]
	# 	l2s = np.sum(diffs**2, axis=1)
	# 	dots = np.einsum('ij,ij->i', point - trajectory[:-1], diffs)
	# 	t = np.clip(dots / l2s, 0.0, 1.0)
	# 	projections = trajectory[:-1] + t[:, np.newaxis] * diffs
	# 	dists = np.linalg.norm(point - projections, axis=1)
		
	# 	# Determine if the point is on the right
	# 	cross_products = np.cross(diffs, point - trajectory[:-1])
	# 	dists = np.where(cross_products < 0, -dists, dists)
		
	# 	min_dist_segment = np.argmin(np.abs(dists))
	# 	return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

	# def get_s_and_speed(path, center, cs):
	# 	speed = path[:,7]
	# 	s = np.zeros(speed.shape[0])
	# 	for n in range(speed.shape[0]):
	# 		_,_,t,i = nearest_point(path[n,:2], center[:,:2])
	# 		s[n] = cs[i] + t * np.linalg.norm(center[i+1,:2]-center[i,:2])
	# 	return s, speed
	# centerS = np.insert(np.linalg.norm(np.diff(centerline[:,:2],axis=0),axis=1).cumsum(),0,0)
	# center_speed = centerline[:,7]
	# curveS, curve_speed = get_s_and_speed(min_curve, centerline, centerS)
	# shortS, short_speed = get_s_and_speed(short, centerline, centerS)

	# width = 16
	# plt.figure(figsize=(width/2.54, width/2.54*(map_img.shape[0]/map_img.shape[1])))
	# ax1 = plt.gca()
	# # ax2 = ax1.twinx()

	# ax1.plot(centerS, center_speed, label='Center Speed Profile')
	# ax1.plot(shortS, short_speed, label='Short Speed Profile')
	# ax1.plot(curveS, curve_speed, label='Curve Speed Profile')
	# ax1.set_xlabel('Distance along track (m)')
	# ax1.set_ylabel('Speed (m/s)')
	# ax1.tick_params(axis='y')

	# # ax2.plot(centerS, np.abs(centerline[:,5]), ':', label='Centerline Curvature')
	# # ax2.plot(shortS, np.abs(short[:,5]), '-.', label='Shortest Path Curvature')
	# # ax2.plot(curveS, np.abs(min_curve[:,5]), '--', label='Min Curvature')
	# # ax2.set_ylabel('Curvature (1/m)')
	# # ax2.tick_params(axis='y')

	# ax1.legend(loc='upper left')
	# # ax2.legend(loc='upper right')

	# ax1.grid(False)
	# plt.savefig(f'Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/centerline/{map_name}_speed_profile.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0, transparent=True, format='pdf')
	# plt.show()

def main():
	maps = ['esp', 'gbr', 'aut', 'mco','map3']
	# maps = ['esp']
	for map_name in maps:
		plot_trajectory(map_name)

if __name__ == "__main__":
	main()