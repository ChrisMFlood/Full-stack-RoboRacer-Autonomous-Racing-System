import numpy as np
import yaml
import matplotlib.pyplot as plt
import sys
from scipy import interpolate
from scipy import optimize
from scipy import spatial
import math
import tph
import os

class Track:
	def __init__(self, track):
		# Open
		self.track = track
		self.path = track[:, :2]
		self.widths = track[:, 2:4]
		self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
		self.s_path = np.insert(np.cumsum(self.el_lengths),0,0)
		self.n_path = np.zeros(len(self.path))
		# Closed
		self.path_closed = np.row_stack([self.path, self.path[0]])
		self.width_closed = np.row_stack([self.widths, self.widths[0]])
		self.el_lengths_closed = np.linalg.norm(np.diff(self.path_closed, axis=0), axis=1)
		self.track_closed = np.column_stack([self.path_closed, self.width_closed])

		# Heading and curvature
		self.x_coeff, self.y_coeff, self.M, normvec_normalized = tph.calc_splines(self.path_closed)#returns unclosed
		print(f'x_coeff: {self.x_coeff.shape}, y_coeff: {self.y_coeff.shape}, M: {self.M.shape}, normvec_normalized: {normvec_normalized.shape}')
		spline_lengths = tph.calc_spline_lengths(self.x_coeff, self.y_coeff)
		# make a list of ones of length len(spline_lengths) 
		stepnum_fixed = [2] * len(spline_lengths)
		print(f'stepnum_fixed: {len(stepnum_fixed)}')
		path_interp, spline_inds, t_values, dists_interp = tph.interp_splines(self.x_coeff, self.y_coeff, spline_lengths, incl_last_point=False, stepnum_fixed=stepnum_fixed)
		# path_interp, spline_inds, t_values, dists_interp = tph.interp_splines(x_coeff, y_coeff, spline_lengths, incl_last_point=False, stepsize_approx=0.05)
		print(f'path interp: {path_interp.shape}, self.path: {self.path_closed.shape}')
		self.path = path_interp
		print(f'spline inds: {spline_inds}')
		print(f't_values: {t_values}')
		# Interpolate path is open
		print(f'path_interp: {path_interp.shape}, spline_inds: {spline_inds.shape}, t_values: {t_values.shape}')
		print(f'first point: {path_interp[0]}, last point: {path_interp[-1]}, second last point: {path_interp[-2]}')
		self.psi, self.kappa, self.dkappa = tph.calc_head_curv_an(self.y_coeff, self.x_coeff, spline_inds, t_values, True, True)
		print(f'psi: {self.psi.shape}, kappa: {self.kappa.shape}, dkappa: {self.dkappa.shape}')
		self.normvectors = tph.calc_normal_vectors(self.psi)
		# print(f'normvectors: {self.normvectors}, {normvec_normalized}')


class trajectory:
	def __init__(self, track):
		self.track = Track(track)
		self.v, self.a, self.t = generateVelocityProfile(self.track)
		self.data_save = np.column_stack((self.track.path, self.track.widths, -self.track.psi, -self.track.kappa, self.track.s_path, self.v, self.a, self.t, self.track.n_path))



def generateVelocityProfile(track: Track):
	'''
	generate velocity profile for the given track
	'''
	
	racetrack_params = load_parameter_file("RaceTrackGenerator")
	vehicle_params = load_parameter_file("vehicle_params")
	# mu = racetrack_params["mu"]* np.ones(len(track.path))
	mu = 0.3 * np.ones(len(track.path))
	ax_max_machine = np.array([[0, racetrack_params["max_longitudinal_acc"]], 
							[vehicle_params["max_speed"], racetrack_params["max_longitudinal_acc"]]])
	ggv = np.array([[0, racetrack_params["max_longitudinal_acc"], racetrack_params["max_lateral_acc"]], 
				 [vehicle_params["max_speed"], racetrack_params["max_longitudinal_acc"], racetrack_params["max_lateral_acc"]]])

	print(f'kappa length {track.kappa.shape[0]}')
	print(f'psi length {track.psi.shape[0]}')
	print(f'path length {track.el_lengths_closed.shape[0]}')
	print(f'mu length {mu.shape[0]}')
	speeds = tph.calc_vel_profile(ax_max_machines=ax_max_machine, kappa=track.kappa, el_lengths=track.el_lengths_closed,
												closed=True, drag_coeff=0, m_veh=vehicle_params["vehicle_mass"], ggv=ggv, mu=mu)
	print(f'speed length {speeds.shape[0]}')
	acceleration = tph.calc_ax_profile(speeds, track.el_lengths, True)
	t = tph.calc_t_profile(speeds, track.el_lengths, 0, acceleration)
	# print(t[-1])
	return speeds, acceleration, t

# class Trajectroy:
# 	def __init__(self, track, map_name):
# 		self.track = track
# 		self.path = track[:, :2]
# 		self.widths = track[:, 2:4]
# 		self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
# 		print(f'Calculating trajectory for {map_name}')
# 		self.s_path, self.n_path = getSN(map_name, self.path)
# 		self.psi, self.kappa = tph.calc_head_curv_num(np.column_stack((self.path[:,1],self.path[:,0])), self.el_lengths, False)
# 		self.normvectors = tph.calc_normal_vectors(self.psi)
# 		self.v, self.a, self.t = generateVelocityProfile(np.column_stack((self.path, self.widths)))
# 		self.data_save = np.column_stack((self.path, self.widths, -self.psi, -self.kappa, self.s_path, self.v, self.a, self.t, self.n_path))

# class Trajectory_an:
# 	def __init__(self, track):
# 		self.track = track
# 		self.path = self.track[:, :2]
# 		self.widths = self.track[:, 2:4]
# 		self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
# 		self.closed_path = np.row_stack([self.path, self.path[0]])
# 		self.closed_el_lengths = np.linalg.norm(np.diff(self.closed_path, axis=0), axis=1)
# 		self.coeffs_x, self.coeffs_y, self.A, self.normvec_normalized = tph.calc_splines(self.closed_path, self.closed_el_lengths)
# 		self.spline_lengths = tph.calc_spline_lengths(self.coeffs_x, self.coeffs_y)
# 		self.path_interp, self.spline_inds, self.t_values, self.dists_interp = tph.interp_splines(self.coeffs_x, self.coeffs_y, self.spline_lengths, False, 0.1)
# 		self.psi, self.kappa, self.dkappa = tph.calc_head_curv_an(self.coeffs_x, self.coeffs_y, self.spline_inds, self.t_values, True, True)
# 		self.normvectors = tph.calc_normal_vectors(self.psi)

# class Track:
# 	def __init__(self, track):
# 		self.path = track[:, :2]
# 		self.widths = track[:, 2:4]
# 		self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
# 		# For closed tracks, add the distance from last point back to first point
# 		closing_distance = np.linalg.norm(self.path[0] - self.path[-1])
# 		self.el_lengths = np.append(self.el_lengths, closing_distance)
# 		self.psi, self.kappa = tph.calc_head_curv_num(np.column_stack((self.path[:,0],self.path[:,1])), self.el_lengths, True)
# 		self.normvectors = tph.calc_normal_vectors(self.psi)

def load_parameter_file(paramFile):
	file_name = f"/home/chris/sim2_ws/src/global_planning/params/{paramFile}.yaml"
	with open(file_name, 'r') as file:
		params = yaml.load(file, Loader=yaml.FullLoader)    
	return params

# def generateVelocityProfile(track):
# 	'''
# 	generate velocity profile for the given track
	
# 	.. inputs::
# 	:param track:           track in the format [x, y, w_tr_right, w_tr_left, (banking)].
# 	:type track:            np.ndarray
# 	'''
# 	track = Track(track)
# 	racetrack_params = load_parameter_file("RaceTrackGenerator")
# 	vehicle_params = load_parameter_file("vehicle_params")
# 	ax_max_machine = np.array([[0, racetrack_params["max_longitudinal_acc"]], [vehicle_params["max_speed"], racetrack_params["max_longitudinal_acc"]]])
# 	mu = racetrack_params["mu"]* np.ones(len(track.path))
# 	ggv = np.array([[0, racetrack_params["max_longitudinal_acc"], racetrack_params["max_lateral_acc"]], [vehicle_params["max_speed"], racetrack_params["max_longitudinal_acc"], racetrack_params["max_lateral_acc"]]])

# 	print(f'kappa length {track.kappa.shape[0]}')
# 	print(f'psi length {track.psi.shape[0]}')
# 	print(f'path length {track.path.shape[0]}')
# 	speeds = tph.calc_vel_profile(ax_max_machines=ax_max_machine, kappa=track.kappa, el_lengths=track.el_lengths, 
# 												closed=True, drag_coeff=0, m_veh=vehicle_params["vehicle_mass"], ggv=ggv, mu=0.50* np.ones(len(track.path)), 
# 												v_max=4.0, v_start=4.0)
# 	print(f'speed length {speeds.shape[0]}')
# 	acceleration = tph.calc_ax_profile(speeds, track.el_lengths, True)
# 	t = tph.calc_t_profile(speeds, track.el_lengths, 0, acceleration)
# 	# print(t[-1])
# 	return speeds, acceleration, t

def saveTrajectroy(trajectroy: trajectory, map_name, pathType):
	savePath = f"/home/chris/sim2_ws/src/maps/{map_name}_{pathType}.csv"
	print(f"Saving {savePath}")
	save = trajectroy.data_save
	with open(savePath, 'wb') as fh:
		np.savetxt(fh, save, fmt='%0.16f', delimiter=',', header='x_m,y_m,w_tr_right_m,w_tr_left_m,psi,kappa,s,velocity,acceleration,time,n')

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

def getSN(map_name, path):
	'''
	Get the S and N values for the given map name
	'''
	centreline_file = f"/home/chris/sim2_ws/src/global_planning/maps/{map_name}_centreline.csv"
	# check if centreline exists
	if not os.path.exists(centreline_file):
		print(f"Centreline file {centreline_file} does not exist.")
		s = np.zeros(len(path))
		n = np.zeros(len(path))
		
	centreLine = np.loadtxt(centreline_file, delimiter=',', skiprows=1)
	s = np.zeros(len(path))
	n = np.zeros(len(path))
	for i,point in enumerate(path):
		_, n[i], t, index = nearest_point(point, centreLine)
		s[i] = centreLine[index, 6] + t * np.linalg.norm(centreLine[index+1, :2] - centreLine[index, :2])
	return s,n

def spline_approximation(track: np.ndarray,
						 k_reg: int = 3,
						 s_reg: int = 10,
						 stepsize_prep: float = 1.0,
						 stepsize_reg: float = 3.0,
						 debug: bool = False) -> np.ndarray:
	"""
	author:
	Fabian Christ

	modified by:
	Alexander Heilmeier, Christopher Flood

	.. description::
	Smooth spline approximation for a track (e.g. centerline, reference line).

	.. inputs::
	:param track:           [x, y, w_tr_right, w_tr_left, (banking)] (always unclosed).
	:type track:            np.ndarray
	:param k_reg:           order of B splines.
	:type k_reg:            int
	:param s_reg:           smoothing factor (usually between 5 and 100).
	:type s_reg:            int
	:param stepsize_prep:   stepsize used for linear track interpolation before spline approximation.
	:type stepsize_prep:    float
	:param stepsize_reg:    stepsize after smoothing.
	:type stepsize_reg:     float
	:param debug:           flag for printing debug messages
	:type debug:            bool

	.. outputs::
	:return track_reg:      [x, y, w_tr_right, w_tr_left, (banking)] (always unclosed).
	:rtype track_reg:       np.ndarray

	.. notes::
	The function can only be used for closable tracks, i.e. track is closed at the beginning!
	The banking angle is optional and must not be provided!
	"""

	# ------------------------------------------------------------------------------------------------------------------
	# LINEAR INTERPOLATION BEFORE SMOOTHING ----------------------------------------------------------------------------
	# ------------------------------------------------------------------------------------------------------------------

	track_interp = tph.interp_track(track=track,stepsize=stepsize_prep)
	track_interp_cl = np.vstack((track_interp, track_interp[0]))

	# ------------------------------------------------------------------------------------------------------------------
	# SPLINE APPROXIMATION / PATH SMOOTHING ----------------------------------------------------------------------------
	# ------------------------------------------------------------------------------------------------------------------

	# create closed track (original track)
	track_cl = np.vstack((track, track[0]))
	no_points_track_cl = track_cl.shape[0]
	el_lengths_cl = np.sqrt(np.sum(np.power(np.diff(track_cl[:, :2], axis=0), 2), axis=1))
	dists_cum_cl = np.cumsum(el_lengths_cl)
	dists_cum_cl = np.insert(dists_cum_cl, 0, 0.0)

	# find B spline representation of the inserted path and smooth it in this process
	# (tck_cl: tuple (vector of knots, the B-spline coefficients, and the degree of the spline))
	tck_cl, t_glob_cl = interpolate.splprep([track_interp_cl[:, 0], track_interp_cl[:, 1]],
											k=k_reg,
											s=s_reg,
											per=1)[:2]

	# calculate total length of smooth approximating spline based on euclidian distance with points at every 0.25m
	no_points_lencalc_cl = math.ceil(dists_cum_cl[-1]) * 4
	path_smoothed_tmp = np.array(interpolate.splev(np.linspace(0.0, 1.0, no_points_lencalc_cl), tck_cl)).T
	len_path_smoothed_tmp = np.sum(np.sqrt(np.sum(np.power(np.diff(path_smoothed_tmp, axis=0), 2), axis=1)))

	# get smoothed path
	no_points_reg_cl = math.ceil(len_path_smoothed_tmp / stepsize_reg) + 1
	path_smoothed = np.array(interpolate.splev(np.linspace(0.0, 1.0, no_points_reg_cl), tck_cl)).T[:-1]

	# ------------------------------------------------------------------------------------------------------------------
	# PROCESS TRACK WIDTHS (AND BANKING ANGLE IF GIVEN) ----------------------------------------------------------------
	# ------------------------------------------------------------------------------------------------------------------

	# find the closest points on the B spline to input points
	dists_cl = np.zeros(no_points_track_cl)                 # contains (min) distances between input points and spline
	closest_point_cl = np.zeros((no_points_track_cl, 2))    # contains the closest points on the spline
	closest_t_glob_cl = np.zeros(no_points_track_cl)        # containts the t_glob values for closest points
	t_glob_guess_cl = dists_cum_cl / dists_cum_cl[-1]       # start guess for the minimization

	for i in range(no_points_track_cl):
		# get t_glob value for the point on the B spline with a minimum distance to the input points
		closest_t_glob_cl[i] = optimize.fmin(dist_to_p,
											 x0=t_glob_guess_cl[i],
											 args=(tck_cl, track_cl[i, :2]),
											 disp=False)

		# evaluate B spline on the basis of t_glob to obtain the closest point
		closest_point_cl[i] = interpolate.splev(closest_t_glob_cl[i], tck_cl)

		# save distance from closest point to input point
		dists_cl[i] = math.sqrt(math.pow(closest_point_cl[i, 0] - track_cl[i, 0], 2)
								+ math.pow(closest_point_cl[i, 1] - track_cl[i, 1], 2))

	if debug:
		print("Spline approximation: mean deviation %.2fm, maximum deviation %.2fm"
			  % (float(np.mean(dists_cl)), float(np.amax(np.abs(dists_cl)))))

	# get side of smoothed track compared to the inserted track
	sides = np.zeros(no_points_track_cl - 1)

	for i in range(no_points_track_cl - 1):
		sides[i] = tph.side_of_line(a=track_cl[i, :2],
												 b=track_cl[i+1, :2],
												 z=closest_point_cl[i])

	sides_cl = np.hstack((sides, sides[0]))

	# calculate new track widths on the basis of the new reference line, but not interpolated to new stepsize yet
	w_tr_right_new_cl = track_cl[:, 2] + sides_cl * dists_cl
	w_tr_left_new_cl = track_cl[:, 3] - sides_cl * dists_cl

	# interpolate track widths after smoothing (linear)
	w_tr_right_smoothed_cl = np.interp(np.linspace(0.0, 1.0, no_points_reg_cl), closest_t_glob_cl, w_tr_right_new_cl)
	w_tr_left_smoothed_cl = np.interp(np.linspace(0.0, 1.0, no_points_reg_cl), closest_t_glob_cl, w_tr_left_new_cl)

	track_reg = np.column_stack((path_smoothed, w_tr_right_smoothed_cl[:-1], w_tr_left_smoothed_cl[:-1]))

	# interpolate banking if given (linear)
	if track_cl.shape[1] == 5:
		banking_smoothed_cl = np.interp(np.linspace(0.0, 1.0, no_points_reg_cl), closest_t_glob_cl, track_cl[:, 4])
		track_reg = np.column_stack((track_reg, banking_smoothed_cl[:-1]))

	return track_reg


# ----------------------------------------------------------------------------------------------------------------------
# DISTANCE CALCULATION FOR OPTIMIZATION --------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# return distance from point p to a point on the spline at spline parameter t_glob
def dist_to_p(t_glob: np.ndarray, path: list, p: np.ndarray):
	s = interpolate.splev(t_glob, path)
	s = np.asarray(s).flatten()
	p = np.asarray(p).flatten()
	return spatial.distance.euclidean(p, s)