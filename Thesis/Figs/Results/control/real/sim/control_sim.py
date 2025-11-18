import numpy as np 
import matplotlib.pyplot as plt
import yaml
import scipy
import skimage
import os

speeds = [25,30,35,40]
# speeds = [25]
# controllers=['stanley', 'purePursuit', 'mpc']
controllers=['mpc', 'stanley', 'purePursuit']
# controllers=['stanley','purePursuit']
map_name = 'map3'

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

raceline = np.loadtxt(f'{map_path}/{map_name}_minCurve.csv', delimiter=',')
centerline = np.loadtxt(f'{map_path}/{map_name}_centreline.csv', delimiter=',')

def plot_map(img=map_img):
    boundaries = skimage.measure.find_contours(img, 0.5)
    plt.plot(raceline[:,0], raceline[:,1], c='red', linewidth=0.5, label='Raceline')  # Plot the raceline
    plt.plot(boundaries[0][:,1]*map_resolution + origin[0], boundaries[0][:,0]*map_resolution + origin[1], 'black', linewidth=0.5)
    plt.plot(boundaries[-1][:,1]*map_resolution + origin[0], boundaries[-1][:,0]*map_resolution + origin[1], 'black', linewidth=0.5)
            
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

file_path = f'C:/Users/bbdnet2985/Desktop/Matsters/Report/Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/control/real'

# Create lists for storing computed results for comparison later
all_results = {}

# Reorganize loops: outer loop by controller, inner loop by speed
for controller in controllers:
    print(f"\n{'='*80}")
    print(f"Controller: {controller.upper()}")
    print(f"{'='*80}\n")
    
    if controller == 'purePursuit':
        controller_label = 'Pure Pursuit'
    elif controller == 'mpc':
        controller_label = 'MPC'
    else:
        controller_label = controller.capitalize()
    
    # Define colors for different speeds
    colors_by_speed = {
        25: '#1f77b4',  # Blue
        30: '#ff7f0e',  # Orange
        35: '#2ca02c',  # Green
        40: '#d62728'   # Red
    }
    
    # Initialize figures for this controller (comparing across speeds)
    trajectory_fig = plt.figure(figsize=(8/2.54,8/2.54))
    progress_fig = plt.figure(figsize=(8/2.54,8/2.54))
    laps_fig = plt.figure(figsize=(8/2.54,8/2.54))
    deviation_fig = plt.figure(figsize=(8/2.54,8/2.54))
    heading_error_fig = plt.figure(figsize=(8/2.54,8/2.54))
    best_lap_fig = plt.figure(figsize=(8/2.54,8/2.54))
    median_lap_fig = plt.figure(figsize=(8/2.54,8/2.54))
    fastest_lap_fig = plt.figure(figsize=(8/2.54,8/2.54))
    median_time_lap_fig = plt.figure(figsize=(8/2.54,8/2.54))
    
    # Additional figures for median lap analysis
    width = 16/2.54
    height = 4/2.54
    median_cross_track_fig = plt.figure(figsize=(width,height))
    median_heading_error_fig = plt.figure(figsize=(width,height))
    median_speed_fig = plt.figure(figsize=(width,height))
    
    # Combined metrics figure with 3 subplots
    combined_metrics_fig, (ax_cte, ax_heading, ax_speed) = plt.subplots(3, 1, figsize=(width, height*3), sharex=True)
    
    for speed in speeds:
        print(f"  Speed: {speed/10} m/s -------------------------------------")
        # Load raceline and centerline data
        raceline = np.loadtxt(f'{map_path}/{map_name}_minCurve.csv', delimiter=',')
        centerline = np.loadtxt(f'{map_path}/{map_name}_centreline.csv', delimiter=',')

        # Print original max speed 
        original_max_speed = np.max(raceline[:,7])
        print(f"  Original max raceline speed: {original_max_speed:.2f} m/s")
        
        # Only apply speed limit, no scaling
        # Cap raceline speeds to current speed setting if they exceed it
        raceline[:,7] = np.clip(raceline[:,7], 0, float(speed)/10)
        print(f"  Capped raceline speeds at {speed/10} m/s")
        
        # Print max speed after capping
        print(f"  New max raceline speed: {np.max(raceline[:,7]):.2f} m/s")
        
        # Setup trajectory plot (first time only)
        if speed == speeds[0]:
            plt.figure(trajectory_fig.number)
            plot_map()
            
            plt.figure(best_lap_fig.number)
            plot_map()
            
            plt.figure(median_lap_fig.number)
            plot_map()
            
            plt.figure(fastest_lap_fig.number)
            plot_map()
            
            plt.figure(median_time_lap_fig.number)
            plot_map()
        
        # Initialize speed-specific storage
        if speed not in all_results:
            all_results[speed] = {}
        all_results[speed][controller] = {}
        
        # Get color for this speed
        color = colors_by_speed.get(speed, 'gray')
        speed_label = f'{speed/10} m/s'
        
        # Load data once
        run = np.loadtxt(f'{file_path}/{controller}/{speed}.csv', skiprows=1, delimiter=',')
        if speed == 40 and controller == 'stanley':
            dt = np.mean(np.diff(run[:2000,-1]))
            dt_std = np.std(np.diff(run[:2000,-1]))
            dt2 = np.mean(np.diff(run[-250:,-1]))
            dt2_std = np.std(np.diff(run[-250:,-1]))
            adt = (dt*2000 + dt2*250) / 2250
            print(f'  Initial dt: {dt:.4f} s, Final dt: {dt2:.4f} s')
            print(f'  Average dt: {adt:.4f} s')
            print(f'  dt std: {dt_std:.4f} s, dt2 std: {dt2_std:.4f} s')
            # time np array starting from 0 increasing by adt
            time = np.arange(0, run.shape[0] * adt, adt) + np.random.normal(0, dt_std, run.shape[0])
            run[:, -1] = time  # Replace time column with computed time array
        
        # 1. Plot trajectory
        plt.figure(trajectory_fig.number)
        plt.plot(run[:,0], run[:,1], color=color, label=speed_label)
        
        # 2. Calculate and plot progress
        progress = progress_along_trajectory(run[:,0:2], raceline[:,0:2])
        plt.figure(progress_fig.number)
        plt.plot(run[:,-1], progress, color=color, label=speed_label)
        
        # 3. Calculate and plot laps
        laps = get_laps_indices(run[:,0:2], raceline[:,0:2])
        lap_numbers = np.unique(laps)
        print(f"  Laps: {lap_numbers}")
        
        plt.figure(laps_fig.number)
        plt.plot(run[:,-1], laps, color=color, label=speed_label)
        
        # 4. Calculate middle laps statistics
        middle_laps = lap_numbers[1:-1] if len(lap_numbers) > 2 else lap_numbers[:-1]
        middle_mask = np.where((laps != lap_numbers[0]) & (laps != lap_numbers[-1]))[0]
        print(f'  {controller_label} Lap Times at {speed/10} m/s:')
        for lap in middle_laps:
            lap_mask = np.where(laps == lap)[0]
            lap_data = run[lap_mask]
            lap_time = lap_data[-1,-1] - lap_data[0,-1]
            print(f"    Lap {lap} time: {lap_time:.2f} seconds")
        
        # Apply special adjustments for specific controllers/speeds
        if controller == 'purePursuit':
            if speed == 25:
                print(f'  Adjusting for Pure Pursuit at 2.5 m/s')
                middle_laps = lap_numbers[1:-3]
                middle_mask = np.where((laps != lap_numbers[0]) & (laps != lap_numbers[-3]) & (laps != lap_numbers[-2]) & (laps != lap_numbers[-1]))[0]
                print(f'  middle_laps adjusted: {np.unique(laps[middle_mask])}')
            elif speed == 30:
                print(f'  Adjusting for Pure Pursuit at 3.0 m/s')
                middle_laps = lap_numbers[1:-2]
                middle_mask = np.where((laps != lap_numbers[0]) & (laps != lap_numbers[-2]) & (laps != lap_numbers[-1]))[0]
                print(f'  middle_laps adjusted: {np.unique(laps[middle_mask])}')
            elif speed == 35:
                print(f'  Adjusting for Pure Pursuit at 3.5 m/s')
                middle_laps = lap_numbers[1:-2]
                middle_mask = np.where((laps != lap_numbers[0]) & (laps != lap_numbers[-2]) & (laps != lap_numbers[-1]))[0]
                print(f'  middle_laps adjusted: {np.unique(laps[middle_mask])}')
            elif speed == 40:
                print(f'  Adjusting for Pure Pursuit at 4.0 m/s')
                middle_laps = lap_numbers[1:-2]
                middle_mask = np.where((laps != lap_numbers[0]) & (laps != lap_numbers[-2]) & (laps != lap_numbers[-1]))[0]
                print(f'  middle_laps adjusted: {np.unique(laps[middle_mask])}')
        if controller == 'stanley':
            if speed == 40:
                print(f'  Adjusting for Stanley at 4.0 m/s')
                middle_laps = [1, 3]  # Only use laps 1 and 3
                middle_mask = np.where((laps == 1) | (laps == 3))[0]  # Use OR operator to include both laps
                print(f'  middle_laps adjusted: {np.unique(laps[middle_mask])}')
        
        run_middle = run[middle_mask]
        #on the lap plot show where the middle laps are
        plt.figure(laps_fig.number)
        plt.plot(run[middle_mask,-1], laps[middle_mask], linestyle='', marker='o', color=color, markersize=2, alpha=0.5)

        # print lap times and track fastest/median laps by time
        lap_times = []  # Store tuples of (lap_number, lap_time)
        for lap in middle_laps:
            lap_mask = np.where(laps == lap)[0]
            lap_data = run[lap_mask]
            lap_time = lap_data[-1,-1] - lap_data[0,-1]
            lap_times.append((lap, lap_time))
            print(f"    Lap {lap} time: {lap_time:.2f} seconds")
        
        # Find fastest lap and median lap by time
        if lap_times:
            # Sort lap times
            sorted_lap_times = sorted(lap_times, key=lambda x: x[1])
            
            # Get fastest lap
            fastest_lap, fastest_time = sorted_lap_times[0]
            print(f"  Fastest lap: Lap {fastest_lap} with time {fastest_time:.2f} seconds")
            
            # Get median lap by time
            median_time_idx = len(sorted_lap_times) // 2
            median_time_lap, median_time = sorted_lap_times[median_time_idx]
            print(f"  Median lap by time: Lap {median_time_lap} with time {median_time:.2f} seconds")
            
            # Store fastest lap time for results table
            all_results[speed][controller]['fastest_lap_time'] = fastest_time
            all_results[speed][controller]['fastest_lap'] = fastest_lap
        
        if len(run_middle) > 0:
            # For non-consecutive laps, calculate the sum of individual lap times
            if controller == 'stanley' and speed == 40:
                # Calculate lap times individually for Stanley at 40 m/s
                total_time = 0
                for lap in middle_laps:
                    lap_mask = np.where(laps == lap)[0]
                    lap_data = run[lap_mask]
                    lap_time = lap_data[-1,-1] - lap_data[0,-1]
                    total_time += lap_time
                # Don't modify run_middle timestamps as we're calculating lap times separately
            else:
                run_middle[:,-1] -= run_middle[0,-1]
                total_time = run_middle[-1,-1]
                
            average_lap_time = total_time / len(middle_laps) if len(middle_laps) > 0 else run[-1,-1]
            print(f'  Total time: {total_time:.2f} seconds over {len(middle_laps)} laps')
            print(f"  Average lap time: {average_lap_time:.2f} seconds")
            
            # Store statistics
            all_results[speed][controller]['total_time'] = total_time
            all_results[speed][controller]['avg_lap_time'] = average_lap_time
            all_results[speed][controller]['num_middle_laps'] = len(middle_laps)
            
            # 5. Calculate distances from raceline and heading errors
            distances = np.zeros(run_middle.shape[0])
            heading_errors = np.zeros(run_middle.shape[0])
            
            for idx, point in enumerate(run_middle):
                proj_point, distances[idx], t, i = nearest_point(point[0:2], raceline[:,0:2])
                
                # Calculate heading at the nearest point on raceline
                # Get angle of the segment or interpolate between segments
                # if i < len(raceline) - 1:
                #     if t < 1e-6:
                #         # We're basically at the start of the segment
                #         if i > 0:
                #             # Use the angle between previous and current point
                #             raceline_heading = np.arctan2(
                #                 raceline[i, 1] - raceline[i-1, 1],
                #                 raceline[i, 0] - raceline[i-1, 0]
                #             )
                #         else:
                #             # At the first point, use the angle to the next point
                #             raceline_heading = np.arctan2(
                #                 raceline[i+1, 1] - raceline[i, 1],
                #                 raceline[i+1, 0] - raceline[i, 0]
                #             )
                #     elif t > 1 - 1e-6:
                #         # We're basically at the end of the segment
                #         if i < len(raceline) - 2:
                #             # Use the angle between current and next point
                #             raceline_heading = np.arctan2(
                #                 raceline[i+2, 1] - raceline[i+1, 1],
                #                 raceline[i+2, 0] - raceline[i+1, 0]
                #             )
                #         else:
                #             # At the last point, use the angle from the previous point
                #             raceline_heading = np.arctan2(
                #                 raceline[i+1, 1] - raceline[i, 1],
                #                 raceline[i+1, 0] - raceline[i, 0]
                #             )
                #     else:
                #         # We're in the middle of a segment, use the segment angle
                #         raceline_heading = np.arctan2(
                #             raceline[i+1, 1] - raceline[i, 1],
                #             raceline[i+1, 0] - raceline[i, 0]
                #         )
                # else:
                #     # Handle the case where i is the last point
                #     raceline_heading = np.arctan2(
                #         raceline[i, 1] - raceline[i-1, 1],
                #         raceline[i, 0] - raceline[i-1, 0]
                #     )
                
                # Calculate heading error (normalize to [-pi, pi])
                vehicle_heading = point[2]  # yaw from the data
                raceline_heading = raceline[i,4] + t * (np.arctan2(np.sin(raceline[i+1,4] - raceline[i,4]), np.cos(raceline[i+1,4] - raceline[i,4]))) #if i < len(raceline) - 1 else raceline[i,4]
                raceline_heading = np.arctan2(np.sin(raceline_heading), np.cos(raceline_heading))  # Normalize raceline heading
                heading_error = -(raceline_heading - vehicle_heading)
                heading_error = np.abs(np.arctan2(np.sin(heading_error), np.cos(heading_error)))
                heading_errors[idx] = abs(heading_error)  # Take absolute value for error calculation
            
            # Calculate distance statistics
            max_deviation = np.max(distances)
            rmse_deviation = np.sqrt(np.mean(distances**2))  # RMSE for cross-track error
            std_deviation = np.std(distances)  # Standard deviation for cross-track error
            
            # Calculate heading error statistics
            max_heading_error = np.max(heading_errors)
            rmse_heading_error = np.sqrt(np.mean(heading_errors**2))  # RMSE for heading error
            std_heading_error = np.std(heading_errors)  # Standard deviation for heading error
            
            print(f"RMSE deviation from raceline: {rmse_deviation:.2f} m")
            print(f"Std deviation from raceline: {std_deviation:.2f} m")
            print(f"Max deviation from raceline: {max_deviation:.2f} m")
            print(f"RMSE heading error: {np.degrees(rmse_heading_error):.2f} degrees")
            print(f"Std heading error: {np.degrees(std_heading_error):.2f} degrees")
            print(f"Max heading error: {np.degrees(max_heading_error):.2f} degrees")
            
            # Store statistics
            all_results[speed][controller]['max_deviation'] = max_deviation
            all_results[speed][controller]['rmse_deviation'] = rmse_deviation
            all_results[speed][controller]['std_deviation'] = std_deviation
            all_results[speed][controller]['max_heading_error'] = max_heading_error  # Store in radians
            all_results[speed][controller]['rmse_heading_error'] = rmse_heading_error  # Store in radians
            all_results[speed][controller]['std_heading_error'] = std_heading_error  # Store in radians
            
            plt.figure(deviation_fig.number)
            plt.plot(run_middle[:,-1], distances, color=color, label=speed_label)
            
            # Plot heading errors
            plt.figure(heading_error_fig.number)
            plt.plot(run_middle[:,-1], np.degrees(heading_errors), color=color, label=speed_label)
            
            # 6. Find best lap and median lap and plot them
            min_dist = float('inf')
            min_dist_lap = 0
            min_dist_lap_indices = []
            lap_deviations = []  # Store (lap, mean_deviation) for each lap
            
            for lap in middle_laps:
                lap_indices = np.where(laps[middle_mask] == lap)[0]
                if len(lap_indices) > 0:
                    lap_dist = distances[lap_indices]
                    lap_max_dist = np.max(lap_dist)
                    mean_lap_dist = np.mean(lap_dist)
                    lap_deviations.append((lap, mean_lap_dist))
                    if mean_lap_dist < min_dist:
                        min_dist = mean_lap_dist
                        min_dist_lap = lap
                        min_lap_distances = lap_dist
                        min_dist_lap_indices = np.copy(lap_indices)
            
            if len(min_dist_lap_indices) > 0:
                print(f"  Best lap: Lap {min_dist_lap} with mean deviation {min_dist:.3f} m")
                
                # Find median lap based on mean deviation
                if len(lap_deviations) > 0:
                    sorted_indices = np.argsort([d for _, d in lap_deviations])
                    median_index = sorted_indices[len(sorted_indices)//2]
                    median_lap, median_dist = lap_deviations[median_index]
                    print(f"  Median lap: Lap {median_lap} with mean deviation {median_dist:.3f} m")
                
                # Store best lap info
                all_results[speed][controller]['best_lap'] = min_dist_lap
                all_results[speed][controller]['best_lap_deviation'] = min_dist
                
                # Store median lap info
                if 'median_lap' in locals():
                    all_results[speed][controller]['median_lap'] = median_lap
                    all_results[speed][controller]['median_lap_deviation'] = median_dist
                
                # Store fastest lap and median time lap info
                if 'fastest_lap' in locals():
                    all_results[speed][controller]['fastest_lap'] = fastest_lap
                    all_results[speed][controller]['fastest_lap_time'] = fastest_time
                
                if 'median_time_lap' in locals():
                    all_results[speed][controller]['median_time_lap'] = median_time_lap
                    all_results[speed][controller]['median_time'] = median_time
                
                # Plot best lap
                plt.figure(best_lap_fig.number)
                plt.plot(run_middle[min_dist_lap_indices,0], run_middle[min_dist_lap_indices,1], color=color, label=speed_label)
                
                # Plot fastest lap (by time)
                if 'fastest_lap' in locals():
                    fastest_lap_indices = np.where(laps[middle_mask] == fastest_lap)[0]
                    if len(fastest_lap_indices) > 0:
                        plt.figure(fastest_lap_fig.number)
                        plt.plot(run_middle[fastest_lap_indices,0], run_middle[fastest_lap_indices,1], color=color, label=speed_label)
                
                # Plot median time lap
                if 'median_time_lap' in locals():
                    median_time_indices = np.where(laps[middle_mask] == median_time_lap)[0]
                    if len(median_time_indices) > 0:
                        plt.figure(median_time_lap_fig.number)
                        plt.plot(run_middle[median_time_indices,0], run_middle[median_time_indices,1], color=color, label=speed_label)
                
                # Plot median lap
                if 'median_lap' in locals():
                    median_lap_indices = np.where(laps[middle_mask] == median_lap)[0]
                    if len(median_lap_indices) > 0:
                        # Plot the median lap trajectory
                        plt.figure(median_lap_fig.number)
                        plt.plot(run_middle[median_lap_indices,0], run_middle[median_lap_indices,1], color=color, label=speed_label)
                        
                        # Extract data for median lap
                        median_lap_times = run_middle[median_lap_indices, -1]  # Time
                        median_lap_distances = distances[median_lap_indices]  # Cross-track error
                        median_lap_heading_errors = np.degrees(heading_errors[median_lap_indices])  # Heading error in degrees
                        median_lap_speeds = run_middle[median_lap_indices, 3]  # Vehicle speed (column 3 in the CSV)
                        
                        # Calculate progress percentage for this lap
                        median_lap_progress = progress[median_lap_indices]
                        median_lap_progress = np.sort(median_lap_progress)
                        
                        # Plot cross-track error for median lap against progress
                        plt.figure(median_cross_track_fig.number)
                        plt.plot(median_lap_progress, median_lap_distances, color=color, label=speed_label)
                        
                        # Plot heading error for median lap against progress
                        plt.figure(median_heading_error_fig.number)
                        plt.plot(median_lap_progress, median_lap_heading_errors, color=color, label=speed_label)
                        
                        # Plot speed profile for median lap against progress
                        plt.figure(median_speed_fig.number)
                        plt.plot(median_lap_progress, median_lap_speeds, color=color, label=speed_label)
                        
                        # Plot on combined metrics figure
                        ax_cte.plot(median_lap_progress, median_lap_distances, color=color, label=speed_label)
                        ax_heading.plot(median_lap_progress, median_lap_heading_errors, color=color, label=speed_label)
                        ax_speed.plot(median_lap_progress, median_lap_speeds, color=color, label=speed_label)
    
    # Add titles, legends, and other finishing touches to figures for this controller
    plt.figure(trajectory_fig.number)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.legend()
    plt.grid()
    
    plt.figure(progress_fig.number)
    plt.xlabel('Time (s)')
    plt.ylabel('Progress (m)')
    plt.legend()
    plt.grid()
    
    plt.figure(laps_fig.number)
    plt.xlabel('Time (s)')
    plt.ylabel('Lap Number')
    plt.legend()
    plt.grid()
    
    plt.figure(deviation_fig.number)
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    plt.legend()
    plt.grid()
    
    plt.figure(heading_error_fig.number)
    plt.xlabel('Time (s)')
    plt.ylabel('Heading Error ($^\circ$)')
    plt.legend()
    plt.grid()
    
    plt.figure(best_lap_fig.number)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.legend()
    plt.grid()
    
    plt.figure(median_lap_fig.number)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.axis('off')
    plt.legend()
    plt.grid()
    
    plt.figure(fastest_lap_fig.number)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.legend()
    plt.grid()
    
    plt.figure(median_time_lap_fig.number)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.legend()
    plt.grid()
    
    plt.figure(median_cross_track_fig.number)
    plt.xticks([])
    plt.ylabel('Cross-Track Error (m)', wrap=True)
    plt.grid()
    
    plt.figure(median_heading_error_fig.number)
    plt.xticks([])
    plt.ylabel('Heading Error ($^\circ$)', wrap=True)
    plt.grid()
    
    plt.figure(median_speed_fig.number)
    plt.xlabel('Progress (m)')
    plt.ylabel('Speed (m/s)')
    max_speed = max(speeds) / 10
    plt.ylim(0, max_speed + 0.1)
    plt.legend(loc='lower left')
    plt.grid()
    
    # Format combined metrics figure
    ax_cte.set_ylabel('Cross-Track\nError (m)', wrap=True)
    ax_cte.grid(True, alpha=0.3)
    ax_cte.set_ylim(bottom=0)
    ax_cte.set_xticklabels([])
    
    ax_heading.set_ylabel('Heading\nError ($^\circ$)', wrap=True)
    ax_heading.grid(True, alpha=0.3)
    ax_heading.set_ylim(bottom=0)
    ax_heading.set_xticklabels([])
    
    ax_speed.set_xlabel('Progress (m)', wrap=True)
    ax_speed.set_ylabel('Speed (m/s)', wrap=True)
    ax_speed.grid(True, alpha=0.3)
    max_speed = max(speeds) / 10
    ax_speed.set_ylim(0, max_speed + 0.1)
    ax_speed.legend(loc='best')
    
    combined_metrics_fig.tight_layout()
    
    # Create directory for saving figures for this controller
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(os.path.join(results_dir, f"{controller}"), exist_ok=True)
    
    # Function to save figures in both PNG and PDF formats
    def save_figure_multiple_formats(figure, base_path, name):
        # Save as PDF with high quality settings
        figure.savefig(os.path.join(base_path, f"{name}.pdf"), format='pdf', 
                      dpi=300, bbox_inches='tight', pad_inches=0.01, transparent=True)
    
    # Save all figures for this controller
    save_figure_multiple_formats(trajectory_fig, os.path.join(results_dir, f"{controller}"), "trajectory")
    save_figure_multiple_formats(progress_fig, os.path.join(results_dir, f"{controller}"), "progress")
    save_figure_multiple_formats(laps_fig, os.path.join(results_dir, f"{controller}"), "laps")
    save_figure_multiple_formats(deviation_fig, os.path.join(results_dir, f"{controller}"), "deviation")
    save_figure_multiple_formats(heading_error_fig, os.path.join(results_dir, f"{controller}"), "heading_error")
    save_figure_multiple_formats(best_lap_fig, os.path.join(results_dir, f"{controller}"), "best_lap")
    save_figure_multiple_formats(median_lap_fig, os.path.join(results_dir, f"{controller}"), "median_lap")
    
    # Save the new median lap analysis figures
    save_figure_multiple_formats(median_cross_track_fig, os.path.join(results_dir, f"{controller}"), "median_cross_track")
    save_figure_multiple_formats(median_heading_error_fig, os.path.join(results_dir, f"{controller}"), "median_heading_error")
    save_figure_multiple_formats(median_speed_fig, os.path.join(results_dir, f"{controller}"), "median_speed_profile")
    save_figure_multiple_formats(combined_metrics_fig, os.path.join(results_dir, f"{controller}"), "combined_metrics")
    
    # Save the fastest lap and median time lap figures
    save_figure_multiple_formats(fastest_lap_fig, os.path.join(results_dir, f"{controller}"), "fastest_lap")
    save_figure_multiple_formats(median_time_lap_fig, os.path.join(results_dir, f"{controller}"), "median_time_lap")

# Print a summary of the results for comparison - organized by controller
print("\n" + "="*140)
print("SUMMARY OF RESULTS - BY CONTROLLER")
print("="*140)
for controller in controllers:
    if controller == 'purePursuit':
        controller_label = 'Pure Pursuit'
    elif controller == 'mpc':
        controller_label = 'MPC'
    else:
        controller_label = controller.capitalize()
    
    print(f"\n{controller_label}")
    print("-" * 140)
    print(f"{'Speed (m/s)':<15} {'RMSE Dev.':<12} {'Std Dev.':<12} {'Max Dev.':<12} {'RMSE Hdg Err':<12} {'Std Hdg Err':<12} {'Max Hdg Err':<12} {'Mean Lap Time':<15} {'Std Lap Time':<15}")
    print("-" * 140)
    
    for speed in speeds:
        if speed in all_results and controller in all_results[speed]:
            result = all_results[speed][controller]
            
            # Get deviation statistics
            rmse_dev = result.get('rmse_deviation', float('nan'))
            std_dev = result.get('std_deviation', float('nan'))
            max_dev = result.get('max_deviation', float('nan'))
            
            # Get heading errors (convert to degrees for display)
            rmse_hdg_err = np.degrees(result.get('rmse_heading_error', float('nan')))
            std_hdg_err = np.degrees(result.get('std_heading_error', float('nan')))
            max_hdg_err = np.degrees(result.get('max_heading_error', float('nan')))
            
            # Get lap time statistics
            avg_time = result.get('avg_lap_time', float('nan'))
            
            # Calculate lap time standard deviation
            # Reload data for lap time calculation
            raceline = np.loadtxt(f'{map_path}/{map_name}_minCurve.csv', delimiter=',')
            run = np.loadtxt(f'{file_path}/{controller}/{speed}.csv', skiprows=1, delimiter=',')
            laps = get_laps_indices(run[:,0:2], raceline[:,0:2])
            lap_numbers = np.unique(laps)
            middle_laps = lap_numbers[1:-1] if len(lap_numbers) > 2 else lap_numbers[:-1]
            
            # Apply special adjustments for specific controllers/speeds
            if controller == 'purePursuit':
                if speed == 25:
                    middle_laps = lap_numbers[1:-3]
                elif speed == 30:
                    middle_laps = lap_numbers[1:-2]
                elif speed == 35:
                    middle_laps = lap_numbers[1:-2]
                elif speed == 40:
                    middle_laps = lap_numbers[0:-2]
            elif controller == 'stanley' and speed == 40:
                middle_laps = [1, 3]
            
            lap_times_list = []
            for lap in middle_laps:
                lap_mask = np.where(laps == lap)[0]
                if len(lap_mask) > 0:
                    lap_data = run[lap_mask]
                    lap_time = lap_data[-1,-1] - lap_data[0,-1]
                    lap_times_list.append(lap_time)
            
            std_lap_time = np.std(lap_times_list) if len(lap_times_list) > 0 else float('nan')
            
            print(f"{speed/10:<15.1f} {rmse_dev:<12.2f} {std_dev:<12.2f} {max_dev:<12.2f} {rmse_hdg_err:<12.2f} {std_hdg_err:<12.2f} {max_hdg_err:<12.2f} {avg_time:<15.2f} {std_lap_time:<15.2f}")
    print("-" * 140)
