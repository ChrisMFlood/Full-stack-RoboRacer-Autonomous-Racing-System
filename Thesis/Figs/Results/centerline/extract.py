import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import yaml
import scipy
from scipy.ndimage import distance_transform_edt as edt
from PIL import Image
import os
import pandas as pd
import sys
import utils
import matplotlib.pyplot as plt
import tph

# Constants
TRACK_WIDTH_MARGIN = 0.0 # Extra Safety margin, in meters

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.parent = None
		self.start = False
		self.end = False

	def __eq__(self, other):
		return isinstance(other, Point) and self.x == other.x and self.y == other.y

	def __hash__(self):
		return hash((self.x, self.y))  # Allow using Point as a dictionary key

# Modified from https://github.com/CL2-UWaterloo/Head-to-Head-Autonomous-Racing/blob/main/gym/f110_gym/envs/laser_models.py
# load map image
def getCentreLine(map_name):
	'''Extracts the centreline from the map image and saves it as a csv file'''

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

	path = 'Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/maps/'
	if os.path.exists(f"{path}{map_name}.png"):
		map_img_path = f"{path}{map_name}.png"
	elif os.path.exists(f"{path}{map_name}.pgm"):
		map_img_path = f"{path}{map_name}.pgm"
	else:
		raise Exception("Map not found!")

	map_yaml_path = f"{path}{map_name}.yaml"
	raw_map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
	raw_map_img = raw_map_img.astype(np.float64)

	# load map yaml
	with open(map_yaml_path, 'r') as yaml_stream:
		try:
			map_metadata = yaml.safe_load(yaml_stream)
			map_resolution = map_metadata['resolution']
			origin = map_metadata['origin']
		except yaml.YAMLError as ex:
			print(ex)

	orig_x = origin[0]
	orig_y = origin[1]

	# grayscale -> binary. Converts grey to black
	map_img = raw_map_img.copy()
	map_img[map_img <= 210.] = 0
	map_img[map_img > 210.] = 1

	map_height = map_img.shape[0]
	map_width = map_img.shape[1]

	# add a black border to the map to avoid edge cases
	map_img_with_border = np.zeros((map_height + 20, map_width + 20))
	map_img_with_border[10:map_height + 10, 10:map_width + 10] = map_img
	print(map_img_with_border)

	# make track wider
	# map_img_with_border = np.abs(1 - map_img_with_border)
	# map_img_with_border = scipy.ndimage.binary_dilation(map_img_with_border, iterations=1).astype(map_img_with_border.dtype)
	# # # invert the map, so that walls are 1 and free space is 0
	# map_img_with_border_edt = scipy.ndimage.distance_transform_edt(map_img_with_border)
	# print(f"Max distance to wall: {np.max(map_img_with_border_edt)} meters")
	# map_img_with_border[map_img_with_border_edt > 3] = 0
	# map_img_with_border = np.abs(1 - map_img_with_border)


	# Calculate Euclidean Distance Transform (tells us distance to nearest wall)
	dist_transform_b = scipy.ndimage.distance_transform_edt(map_img_with_border)
	dist_transform = np.zeros((map_height, map_width))
	dist_transform = dist_transform_b[10:map_height + 10, 10:map_width + 10]
	dist_img = np.copy(dist_transform)
	plt.figure()
	plt.imshow(dist_img, origin='lower', cmap='gray')
	# plt.axis('equal')
	plt.axis('off')
	plt.xlim(80, map_width-80)
	plt.ylim(70, map_height-30)
	root_path = "C:/Users/bbdnet2985/Desktop/Matsters/Report/Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/centerline/centerline_process"
	plt.savefig(f'{root_path}/edt.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0, transparent=True, format='pdf')
	plt.show()

	# Threshold the distance transform to create a binary image
	# You should play around with this number. Is you say hairy lines generated, either clean the map so it is more curvy or increase this number
	THRESHOLD = 0.0/map_resolution

	mask = dist_transform < THRESHOLD
	dist_transform[mask] = 0.0
	centers = dist_transform
	# centers = dist_transform > THRESHOLD
	print(centers)
	plt.figure()
	plt.imshow(centers, origin='lower', cmap='gray')
	plt.axis('off')
	plt.show()

	centerline = skeletonize(centers)
	skel_img = np.copy(centerline)
	plt.figure()
	plt.imshow(skel_img, origin='lower', cmap='gray')
	plt.axis('off')
	plt.xlim(80, map_width-80)
	plt.ylim(70, map_height-30)
	plt.savefig(f'{root_path}/skeleton.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0, transparent=True, format='pdf')
	plt.show()

	centerline_dist = np.where(centerline, dist_transform, 0.0) #distance to closest edge
	
	startX = int((0-orig_x)/map_resolution)
	startY = int((0-orig_y)/map_resolution)
	start = (startY, startX)
	# Distance transform to get point closest to start on centreline
	distanceToStart_img = np.ones_like(dist_transform)
	distanceToStart_img[startY, startX] = 0
	distanceToStartTransform = scipy.ndimage.distance_transform_edt(distanceToStart_img)
	distanceToStart = np.where(centerline, distanceToStartTransform, distanceToStartTransform+200)
	start_point = np.unravel_index(np.argmin(distanceToStart, axis=None), distanceToStart.shape)
	starting_point = Point(start_point[1], start_point[0])
	starting_point.start = True
	print('-------------------------')
	print(f'starting_point: {starting_point.x} {starting_point.y}')
	print('-------------------------')

	sys.setrecursionlimit(20000)

	NON_EDGE = 0.0
	visited = {}
	centerline_nodes = []
	track_widths = []
	DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
	# If you want the other direction first
	# DIRECTIONS = [(0, -1), (-1, 0),  (0, 1), (1, 0), (-1, 1), (-1, -1), (1, 1), (1, -1) ]

	
	def findEndPoint(starting_point: Point):
		for direction in DIRECTIONS[::-1]:
			newPoint = Point(starting_point.x + direction[0], starting_point.y + direction[1])
			if centerline_dist[newPoint.y][newPoint.x] != NON_EDGE:
				newPoint.end = True
				return newPoint
			
	ending_point = findEndPoint(starting_point)
	print('-------------------------')
	print(f'ending_point: {ending_point.x} {ending_point.y}')
	print('-------------------------')

	def dfs(startPoint: Point, endPoint: Point):
		visited[(starting_point.x,starting_point.y)] = True  # Track visited points and their parents
		stack = [startPoint]  # Stack stores points to visit
	
		while stack:
			point = stack.pop()
	
			if not (0 <= point.x < len(centerline_dist[0]) and 0 <= point.y < len(centerline_dist)):
				continue
			if centerline_dist[point.y][point.x] == NON_EDGE:
				continue
			centerline_nodes.append(point)
	
			for direction in DIRECTIONS:
				newPoint = Point(point.x + direction[0], point.y + direction[1])
	
				if point.start and newPoint.x == endPoint.x and newPoint.y == endPoint.y:
					continue
				elif not (0 <= newPoint.x < len(centerline_dist[0]) and 0 <= newPoint.y < len(centerline_dist)):
					continue
				elif centerline_dist[newPoint.y][newPoint.x] == NON_EDGE:
					continue
				elif visited.get((newPoint.x, newPoint.y)):
					continue
				elif point.parent and newPoint.x == point.parent.x and newPoint.y == point.parent.y:
					continue
				elif newPoint.x == endPoint.x and newPoint.y == endPoint.y:
					ending_point.parent = point
				else:
					newPoint.parent = point
					stack.append(newPoint)
					visited[(newPoint.x, newPoint.y)] = True

	dfs(starting_point, ending_point)

	print(f"Number of centerline points: {len(centerline_nodes)}")

	centerline_points = [(starting_point.x, starting_point.y)]
	track_widths = [np.array([centerline_dist[starting_point.y][starting_point.x], centerline_dist[starting_point.y][starting_point.x]])]
	centerline_points.append((ending_point.x, ending_point.y))
	track_widths.append(np.array([centerline_dist[ending_point.y][ending_point.x], centerline_dist[ending_point.y][ending_point.x]]))

	while ending_point.parent is not None:
		print(ending_point.x, ending_point.y)
		ending_point = ending_point.parent
		centerline_points.append((ending_point.x, ending_point.y))
		track_widths.append(np.array([centerline_dist[ending_point.y][ending_point.x], centerline_dist[ending_point.y][ending_point.x]]))

	centerline_points.append((starting_point.x, starting_point.y))
	track_widths.append(np.array([centerline_dist[starting_point.y][starting_point.x], centerline_dist[starting_point.y][starting_point.x]]))

	print(f"Final Centerline Path Length: {len(centerline_points)}")

	track_widths_np = np.array(track_widths)[::]
	waypoints = np.array(centerline_points)[::]
	print(f"Track widths shape: {track_widths_np.shape}, waypoints shape: {waypoints.shape}")

	plt.figure()
	plt.plot(np.array(centerline_points)[:,0], np.array(centerline_points)[:,1], 'r.')
	plt.imshow(map_img, origin='lower', cmap='gray')
	plt.axis('off')
	plt.xlim(80, map_width-80)
	plt.ylim(70, map_height-30)
	plt.savefig(f'{root_path}/centerline_points.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0, transparent=True, format='pdf')
	plt.show()

	# Merge track widths with waypoints
	data = np.concatenate((waypoints, track_widths_np), axis=1)[::]

	# plt.imshow(map_img, origin='lower', cmap='gray')
	# plt.plot(data[:, 0], data[:, 1], 'r.')
	#plt.show()

	# calculate map parameters
	orig_x = origin[0]
	orig_y = origin[1]
	# ??? Should be 0
	orig_s = np.sin(origin[2])
	orig_c = np.cos(origin[2])

	# get the distance transform
	transformed_data = data
	transformed_data *= map_resolution
	transformed_data += np.array([orig_x, orig_y, 0, 0])

	# Safety margin
	transformed_data -= np.array([0, 0, TRACK_WIDTH_MARGIN, TRACK_WIDTH_MARGIN])

	transformed_data = tph.interp_track(transformed_data, 0.05)
	transformed_data = utils.spline_approximation(transformed_data,3, 5, 0.05, 0.05, True)
	transformed_data = tph.interp_track(transformed_data, 0.1)
	transformed_data = np.vstack((transformed_data,transformed_data[0]))

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

	# plt.imshow(map_img, origin='lower', cmap='gray')
	plt.figure()
	plt.scatter(by, bx, color='k', s=5)
	plt.plot(transformed_data[:, 0], transformed_data[:, 1], 'r-')
	plt.xlim(80*map_resolution+origin[0], (map_width-80)*map_resolution+origin[0])
	plt.ylim(70*map_resolution+origin[1], (map_height-30)*map_resolution+origin[1])
	plt.xlabel('X [m]')
	plt.ylabel('Y [m]')
	plt.axis('off')
	plt.grid(True)
	plt.savefig(f'{root_path}/final_centerline.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0, transparent=True, format='pdf')
	plt.show()

	print(f'transformed data shape:{transformed_data.shape}')

	# trajectory = utils.trajectory(transformed_data)
	# print(np.linalg.norm(np.diff(trajectory.track.path, axis=0), axis=1))

	# plt.imshow(map_img, origin='lower', cmap='gray')
	# plt.plot((trajectory.track.path[:, 0]-orig_x)/map_resolution, (trajectory.track.path[:, 1]-orig_y)/map_resolution, 'r.')
	# #plt.show()

	# # save trajectory
	# utils.saveTrajectroy(trajectory, map_name, 'centreline')

def main():
	# for file in os.listdir('/home/chris/sim2_ws/src/maps/'):
	# 	if file.endswith('.png'):
	# 		map_name = file.split('.')[0]
	# 		# if not os.path.exists(f"maps/{map_name}_centreline.csv"):
	# 		print(f"Extracting centre line for: {map_name}")
	# 		getCentreLine(map_name)
	# getCentreLine('aut')
	# getCentreLine('berlin')
	getCentreLine('map3')


if __name__ == "__main__":
	main()

