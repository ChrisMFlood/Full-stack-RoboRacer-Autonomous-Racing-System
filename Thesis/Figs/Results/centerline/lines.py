import numpy as np
import matplotlib.pyplot as plt
import scipy
import yaml
from PIL import Image

map_name = 'esp'
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
print(f"Extracting centre line for: {map_name}")

# map_img = np.flipud(map_img)
# map_img = scipy.ndimage.distance_transform_edt(map_img)
# map_img = np.abs(map_img - 1)
# map_img[map_img!=0]=1
# bx,by = np.where(map_img==0)

map_path = 'Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/maps/'
map_yaml_path = f"{map_path}/{map_name}.yaml"
map_img = plt.imread(f'{map_path}/{map_name}.png')
map_img = np.flipud(map_img)
map_img = scipy.ndimage.binary_dilation(map_img, iterations=5).astype(map_img.dtype)
# plt.imshow(map_img, cmap='gray', origin='lower')
# plt.colorbar()
# plt.show()

map_img = scipy.ndimage.distance_transform_edt(map_img)
map_img = np.abs(map_img - 1)
map_img[map_img!=0]=1
# print(map_img)
# plt.imshow(map_img, cmap='gray', origin='lower')
# plt.show()
#make track wider

bx,by = np.where(map_img==0)
print(bx.shape, by.shape)


with open(map_yaml_path, 'r') as yaml_stream:
	try:
		map_metadata = yaml.safe_load(yaml_stream)
		map_resolution = map_metadata['resolution']
		origin = map_metadata['origin']
	except yaml.YAMLError as ex:
		print(ex)

bx = bx * map_resolution + origin[1]
by = by * map_resolution + origin[0]

centerline = np.loadtxt(f'Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/maps/{map_name}_centreline.csv',skiprows=1,delimiter=',')
center_time = centerline[-1,9]
center_speed = centerline[:,7]
centerS = np.insert(np.linalg.norm(np.diff(centerline[:,:2],axis=0),axis=1).cumsum(),0,0)

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

def get_s_and_speed(path, center, cs):
	speed = path[:,7]
	s = np.zeros(speed.shape[0])
	for n in range(speed.shape[0]):
		_,_,t,i = nearest_point(path[n,:2], center[:,:2])
		s[n] = cs[i] + t * np.linalg.norm(center[i+1,:2]-center[i,:2])
	return s, speed
	 


min_curve = np.loadtxt(f'Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/maps/{map_name}_minCurve.csv',skiprows=1,delimiter=',')
curve_time = min_curve[-1,9]
curveS, curve_speed = get_s_and_speed(min_curve, centerline, centerS)
# curveS = min_curve[:,6]
short = np.loadtxt(f'Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/maps/{map_name}_short.csv',skiprows=1,delimiter=',')
short_time = short[-1,9]
shortS, short_speed = get_s_and_speed(short, centerline, centerS)
# shortS = short[:,6]

print(map_img.shape)

width = 17
plt.figure(figsize=(width/2.54, width/2.54*(map_img.shape[0]/map_img.shape[1])))
plt.scatter(by, bx, s=1, c='k')
plt.plot(centerline[:,0], centerline[:,1], 'b--', label='Centerline (t={:.2f}s)'.format(center_time))
plt.plot(min_curve[:,0], min_curve[:,1], 'g-', label='Min Curvature (t={:.2f}s)'.format(curve_time))
plt.plot(short[:,0], short[:,1], 'r-.', label='Shortest (t={:.2f}s)'.format(short_time))
plt.axis('off')
plt.legend()
plt.savefig(f'Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/centerline/_lines.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0, transparent=True, format='pdf')
plt.show()

# plt.figure()
# # plt.plot(centerS, center_speed, 'b--', label='Centerline (t={:.2f}s)'.format(center_time))
# # plt.plot(curveS, curve_speed, 'g-', label='Min Curvature (t={:.2f}s)'.format(curve_time))
# # plt.plot(shortS, short_speed, 'r-.', label='Shortest (t={:.2f}s)'.format(short_time))
# plt.plot(curveS, curve_speed, 'g-', label='Speed Profile')
# plt.plot(curveS, np.abs(min_curve[:,5]), 'k--', label='Curvature')
# plt.xlabel('Distance along track (m)')
# plt.ylabel('Speed (m/s)')
# plt.legend()
# plt.grid('off')
# plt.savefig(f'Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/centerline/_speed_profile.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0, transparent=True, format='pdf')
# plt.show()

width = 16
plt.figure(figsize=(width/2.54, width/2.54*(map_img.shape[0]/map_img.shape[1])))
ax1 = plt.gca()
ax2 = ax1.twinx()

ax1.plot(curveS, curve_speed, 'g-', label='Speed Profile')
ax1.set_xlabel('Distance along track (m)')
ax1.set_ylabel('Speed (m/s)')
ax1.tick_params(axis='y')

ax2.plot(curveS, np.abs(min_curve[:,5]), '--', label='Curvature')
ax2.set_ylabel('Curvature (1/m)')
ax2.tick_params(axis='y')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

ax1.grid(False)
plt.savefig(f'Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/centerline/_speed_profile.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0, transparent=True, format='pdf')
plt.show()
