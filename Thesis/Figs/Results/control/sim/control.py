import numpy as np 
import matplotlib.pyplot as plt
import yaml
import scipy
import skimage
import os

maps = ['esp','gbr','aut','mco']
# maps = ['aut']
# speeds = [25]
controllers=['mpc', 'stanley', 'purePursuit']
# controllers=['stanley', 'modifiedStanley']

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

def load_map_and_raceline(map_name):
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

        return map_img, origin, map_resolution, raceline, centerline

def plot_map(map_name):
    map_img, origin, map_resolution, raceline, centerline = load_map_and_raceline(map_name)
    boundaries = skimage.measure.find_contours(map_img, 0.5)
    plt.plot(raceline[:,0], raceline[:,1], c='red', linewidth=0.3, label='Raceline')  # Plot the raceline
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

file_path = f'C:/Users/bbdnet2985/Desktop/Matsters/Report/Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/control/sim'

# Create lists for storing computed results for comparison later
all_results = {}

# Single set of nested loops for all analysis
map_path = 'C:/Users/bbdnet2985/Desktop/Matsters/Report/Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/maps'
for map_name in maps:
    print(f"Map: {map_name}----------------------------------------------------")

    map_img, _, _, _, _ = load_map_and_raceline(map_name)
    # Load raceline and centerline data
    raceline = np.loadtxt(f'{map_path}/{map_name}_minCurve.csv', delimiter=',')
    centerline = np.loadtxt(f'{map_path}/{map_name}_centreline.csv', delimiter=',')

    # Print original max speed 
    original_max_speed = np.max(raceline[:,7])
    print(f"Original max raceline speed: {original_max_speed:.2f} m/s")

    # Initialize figures for this speed
    trajectory_fig = plt.figure(figsize=(8/2.54,8/2.54))
    progress_fig = plt.figure(figsize=(8/2.54,8/2.54))
    laps_fig = plt.figure(figsize=(8/2.54,8/2.54))
    deviation_fig = plt.figure(figsize=(8/2.54,8/2.54))
    heading_error_fig = plt.figure(figsize=(8/2.54,8/2.54))  # New figure for heading errors
    best_lap_fig = plt.figure(figsize=(8/2.54,8/2.54))
    median_lap_fig = plt.figure(figsize=(15/2.54,15/2.54*(map_img.shape[0]/map_img.shape[1])))  # New figure for median lap

    # Additional figures for median lap analysis
    width = 16/2.54
    height = 4/2.54
    median_cross_track_fig = plt.figure(figsize=(width,height))
    median_heading_error_fig = plt.figure(figsize=(width,height))
    median_speed_fig = plt.figure(figsize=(width,height))

    # Add raceline speed profile to the speed figure (once per speed setting)
    plt.figure(median_speed_fig.number)
    # Calculate raceline progress in meters
    raceline_progress = np.insert(np.cumsum(np.linalg.norm(np.diff(raceline[:,0:2], axis=0), axis=1)), 0, 0)
    # Plot raceline velocity from column 7 (capped at current speed)
    plt.plot(raceline_progress, raceline[:,7], 'k--', linewidth=1, alpha=0.5, 
             label=f'Reference Speed')
    
    # Setup trajectory plot
    plt.figure(trajectory_fig.number)
    plot_map(map_name)
    
    # Setup best lap plot
    plt.figure(best_lap_fig.number)
    plot_map(map_name)  
    
    # Setup median lap plot
    plt.figure(median_lap_fig.number)
    plot_map(map_name)
    
    all_results[map_name] = {}
    
    for controller in controllers:
        print(f"Controller: {controller.capitalize()}-------------------------------------------------")
        if controller =='purePursuit':
            controller_label = 'Pure Pursuit'
        elif controller == 'mpc':
            controller_label = 'MPC'
        else:
            controller_label = controller.capitalize()
        
        # Load data once
        run = np.loadtxt(f'{file_path}/{map_name}/{controller}.csv', skiprows=1, delimiter=',')
        temp = run[:,0]
        temp -= temp[0]
        temp = np.linspace(0, temp.shape[0]/50+0.02, temp.shape[0])
        # plt.plot(temp)
        # plt.show()
        run[:,0:-1] = run[:,1:]
        run[:,-1] = temp
        print(f"Loaded data from {file_path}/{map_name}/{controller}.csv")
        print(f"Run shape: {run.shape}")
        
        # Store results for this controller
        all_results[map_name][controller] = {}

        # Define line style for this controller
        color = {'stanley': '#1f77b4', 'purePursuit': '#ff7f0e', 'mpc': '#2ca02c', 'modifiedStanley': '#d62728'}.get(controller, 'gray')
        line_style = {'stanley': '-', 'purePursuit': '--', 'mpc': '-.', 'modifiedStanley': ':'}.get(controller, '-')
        line_width = {'stanley': 0.5, 'purePursuit': 0.5, 'mpc': 0.5, 'modifiedStanley': 0.5}.get(controller, 1.0)
        
        # 1. Plot trajectory
        plt.figure(trajectory_fig.number)
        plt.plot(run[:,0], run[:,1], linestyle=line_style, color=color, label=f'{controller_label}',linewidth=line_width)
        
        # 2. Calculate and plot progress
        progress = progress_along_trajectory(run[:,0:2], raceline[:,0:2])
        plt.figure(progress_fig.number)
        plt.plot(run[:,-1], progress, linestyle=line_style, color=color, label=f'{controller_label}',linewidth=line_width)
        
        # 3. Calculate and plot laps
        laps = get_laps_indices(run[:,0:2], raceline[:,0:2])
        lap_numbers = np.unique(laps)
        print(lap_numbers)
        
        plt.figure(laps_fig.number)
        plt.plot(run[:,-1], laps, linestyle=line_style, color=color, label=f'{controller_label}', linewidth=line_width)
        
        # 4. Calculate middle laps statistics
        middle_laps = lap_numbers[1:-1] if len(lap_numbers) > 2 else lap_numbers[:-1]
        middle_mask = np.where((laps != lap_numbers[0]) & (laps != lap_numbers[-1]))[0]
        run_middle = run[middle_mask]
        
        if len(run_middle) > 0:
            run_middle[:,-1] -= run_middle[0,-1]
            total_time = run_middle[-1,-1]
            average_lap_time = total_time / len(middle_laps) if len(middle_laps) > 0 else run[-1,-1]
            print(f'Total time: {total_time:.2f} seconds over {len(middle_laps)} laps')
            print(f"Average lap time (excluding first and last lap): {average_lap_time:.2f} seconds over {len(middle_laps)} laps")
            
            # Store statistics
            all_results[map_name][controller]['total_time'] = total_time
            all_results[map_name][controller]['avg_lap_time'] = average_lap_time
            all_results[map_name][controller]['num_middle_laps'] = len(middle_laps)

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
            rmse_deviation = np.sqrt(np.mean(distances**2))
            std_deviation = np.std(distances)
            max_deviation = np.max(distances)
            
            # Calculate heading error statistics
            rmse_heading_error = np.sqrt(np.mean(heading_errors**2))
            std_heading_error = np.std(heading_errors)
            max_heading_error = np.max(heading_errors)
            
            # Calculate lap time statistics
            lap_times = []
            for lap in middle_laps:
                lap_indices = np.where(laps[middle_mask] == lap)[0]
                if len(lap_indices) > 0:
                    lap_time = run_middle[lap_indices[-1], -1] - run_middle[lap_indices[0], -1]
                    lap_times.append(lap_time)
            
            mean_lap_time = np.mean(lap_times) if len(lap_times) > 0 else average_lap_time
            std_lap_time = np.std(lap_times) if len(lap_times) > 0 else 0.0
            
            print(f"Deviation - RMSE: {rmse_deviation:.2f} m, Std: {std_deviation:.2f} m, Max: {max_deviation:.2f} m")
            print(f"Heading Error - RMSE: {np.degrees(rmse_heading_error):.2f} deg, Std: {np.degrees(std_heading_error):.2f} deg, Max: {np.degrees(max_heading_error):.2f} deg")
            print(f"Lap Time - Mean: {mean_lap_time:.2f} s, Std: {std_lap_time:.2f} s")
            
            # Store statistics
            all_results[map_name][controller]['rmse_deviation'] = rmse_deviation
            all_results[map_name][controller]['std_deviation'] = std_deviation
            all_results[map_name][controller]['max_deviation'] = max_deviation
            all_results[map_name][controller]['rmse_heading_error'] = rmse_heading_error  # Store in radians
            all_results[map_name][controller]['std_heading_error'] = std_heading_error  # Store in radians
            all_results[map_name][controller]['max_heading_error'] = max_heading_error  # Store in radians
            all_results[map_name][controller]['mean_lap_time'] = mean_lap_time
            all_results[map_name][controller]['std_lap_time'] = std_lap_time

            plt.figure(deviation_fig.number)
            plt.plot(run_middle[:,-1], distances, linestyle=line_style, color=color, label=f'{controller_label}')
            
            # Plot heading errors
            plt.figure(heading_error_fig.number)
            plt.plot(run_middle[:,-1], np.degrees(heading_errors), linestyle=line_style, color=color, label=f'{controller_label}')
            
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
                print(f"Lap with minimum mean deviation from raceline: Lap {min_dist_lap} with mean deviation {min_dist:.2f} m")
                
                # Find median lap based on mean deviation
                if len(lap_deviations) > 0:
                    sorted_indices = np.argsort([d for _, d in lap_deviations])
                    median_index = sorted_indices[len(sorted_indices)//2]
                    median_lap, median_dist = lap_deviations[median_index]
                    print(f"Median lap based on mean deviation: Lap {median_lap} with mean deviation {median_dist:.2f} m")
                
                # Store best lap info
                all_results[map_name][controller]['best_lap'] = min_dist_lap
                all_results[map_name][controller]['best_lap_deviation'] = min_dist

                # Store median lap info
                if 'median_lap' in locals():
                    all_results[map_name][controller]['median_lap'] = median_lap
                    all_results[map_name][controller]['median_lap_deviation'] = median_dist

                # Plot best lap
                plt.figure(best_lap_fig.number)
                plt.plot(run_middle[min_dist_lap_indices,0], run_middle[min_dist_lap_indices,1], 
                         linestyle=line_style, color=color, label=f'{controller_label}', alpha=0.33)
                
                # Plot median lap
                if 'median_lap' in locals():
                    median_lap_indices = np.where(laps[middle_mask] == median_lap)[0]
                    if len(median_lap_indices) > 0:
                        # Plot the median lap trajectory
                        plt.figure(median_lap_fig.number)
                        plt.plot(run_middle[median_lap_indices,0], run_middle[median_lap_indices,1], 
                                 linestyle=line_style, color=color, label=f'{controller_label}', alpha=0.33, linewidth=line_width)
                        
                        # Extract data for median lap
                        median_lap_times = run_middle[median_lap_indices, -1]  # Time
                        median_lap_distances = distances[median_lap_indices]  # Cross-track error
                        median_lap_heading_errors = np.degrees(heading_errors[median_lap_indices])  # Heading error in degrees
                        median_lap_speeds = run_middle[median_lap_indices, 3]  # Vehicle speed (column 3 in the CSV)
                        
                        # Calculate progress percentage for this lap
                        median_lap_progress = progress[median_lap_indices]
                        median_lap_progress = np.sort(median_lap_progress)
                        # Normalize to percentage (0-100%)
                        # median_lap_progress_pct = (median_lap_progress - np.min(median_lap_progress)) / (np.max(median_lap_progress) - np.min(median_lap_progress)) * 100
                        
                        # Plot cross-track error for median lap against progress
                        plt.figure(median_cross_track_fig.number)
                        plt.plot(median_lap_progress, median_lap_distances, 
                                 linestyle=line_style, color=color, label=f'{controller_label}')
                        
                        # Plot heading error for median lap against progress
                        plt.figure(median_heading_error_fig.number)
                        plt.plot(median_lap_progress, median_lap_heading_errors, 
                                 linestyle=line_style, color=color, label=f'{controller_label}')
                        
                        # Plot speed profile for median lap against progress
                        plt.figure(median_speed_fig.number)
                        plt.plot(median_lap_progress, median_lap_speeds, 
                                 linestyle=line_style, color=color, label=f'{controller_label}')
    
    # Add titles, legends, and other finishing touches to figures
    plt.figure(trajectory_fig.number)
    # plt.title(f'Vehicle Trajectory at {speed} m/s')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.legend()
    plt.grid()
    
    plt.figure(progress_fig.number)
    # plt.title(f'Progress Along Raceline at {speed} m/s')
    plt.xlabel('Time (s)')
    plt.ylabel('Progress (m)')
    plt.legend()
    plt.grid()
    
    plt.figure(laps_fig.number)
    # plt.title(f'Lap Progress at {speed} m/s')
    plt.xlabel('Time (s)')
    plt.ylabel('Lap Number')
    plt.legend()
    plt.grid()
    
    plt.figure(deviation_fig.number)
    # plt.title(f'Deviation from Raceline at {speed} m/s')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    plt.legend()
    plt.grid()
    
    plt.figure(heading_error_fig.number)
    # plt.title(f'Heading Error at {speed} m/s')
    plt.xlabel('Time (s)')
    plt.ylabel('Heading Error ($^\circ$)', wrap=True)
    plt.legend()
    plt.grid()
    
    plt.figure(best_lap_fig.number)
    # plt.title(f'Best Lap Trajectories at {speed} m/s')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.legend()
    plt.grid()
    
    plt.figure(median_lap_fig.number)
    # plt.title(f'Median Lap Trajectories at {speed} m/s')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.axis('off')
    # if map_name == 'esp':
    plt.legend()
    plt.grid()
    
    plt.figure(median_cross_track_fig.number)
    # plt.title(f'Cross-Track Error for Median Laps at {speed} m/s')
    # plt.xlabel('Progress (m)')
    plt.xticks([])
    plt.ylabel('Cross-Track Error (m)', wrap=True)
    # plt.legend()
    plt.grid()
    
    plt.figure(median_heading_error_fig.number)
    # plt.title(f'Heading Error for Median Laps at {speed} m/s')
    # plt.xlabel('Progress (m)')
    plt.xticks([])
    plt.ylabel('Heading Error ($^\circ$)', wrap=True)
    # plt.legend()
    plt.grid()
    
    plt.figure(median_speed_fig.number)
    # plt.title(f'Speed Profiles for Median Laps at {speed} m/s')
    plt.xlabel('Progress (m)')
    plt.ylabel('Speed (m/s)')
    plt.ylim(0, 8.5)
    plt.legend(loc='lower left')
    plt.grid()
    
    # Create directory for saving figures
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(os.path.join(results_dir, f"{map_name}"), exist_ok=True)
    
    # Function to save figures in both PNG and PDF formats
    def save_figure_multiple_formats(figure, base_path, name):
        # Save as PNG (default)
        # figure.savefig(os.path.join(base_path, f"{name}.png"))
        # Save as PDF with high quality settings
        figure.savefig(os.path.join(base_path, f"{name}.pdf"), format='pdf', 
                      dpi=300, bbox_inches='tight', pad_inches=0.01, transparent=True)
    
    # Save all figures in both PNG and PDF formats
    save_figure_multiple_formats(trajectory_fig, os.path.join(results_dir, f"{map_name}"), "trajectory")
    save_figure_multiple_formats(progress_fig, os.path.join(results_dir, f"{map_name}"), "progress")
    save_figure_multiple_formats(laps_fig, os.path.join(results_dir, f"{map_name}"), "laps")
    save_figure_multiple_formats(deviation_fig, os.path.join(results_dir, f"{map_name}"), "deviation")
    save_figure_multiple_formats(heading_error_fig, os.path.join(results_dir, f"{map_name}"), "heading_error")
    save_figure_multiple_formats(best_lap_fig, os.path.join(results_dir, f"{map_name}"), "best_lap")
    save_figure_multiple_formats(median_lap_fig, os.path.join(results_dir, f"{map_name}"), "median_lap")

    # Save the new median lap analysis figures in both PNG and PDF formats
    save_figure_multiple_formats(median_cross_track_fig, os.path.join(results_dir, f"{map_name}"), "median_cross_track")
    save_figure_multiple_formats(median_heading_error_fig, os.path.join(results_dir, f"{map_name}"), "median_heading_error")
    save_figure_multiple_formats(median_speed_fig, os.path.join(results_dir, f"{map_name}"), "median_speed_profile")

    # Show all figures for this speed
    # plt.figure(trajectory_fig.number)
    # plt.show()
    
    # plt.figure(progress_fig.number)
    # plt.show()
    
    # plt.figure(laps_fig.number)
    # plt.show()
    
    # plt.figure(deviation_fig.number)
    # plt.show()
    
    # plt.figure(heading_error_fig.number)
    # plt.show()
    
    # plt.figure(best_lap_fig.number)
    # plt.show()
    
    # plt.figure(median_lap_fig.number)
    # plt.show()
    
    # plt.figure(median_cross_track_fig.number)
    # plt.show()
    
    # plt.figure(median_heading_error_fig.number)
    # plt.show()
    
    # plt.figure(median_speed_fig.number)
    # plt.show()

# Print a summary of the results for comparison
print("\nSUMMARY OF RESULTS")
print("=================")
for map_name in maps:
    print(f"\nMap: {map_name.upper()}")
    print("-" * 120)
    print(f"{'Controller':<15} {'Cross-Track Error (m)':<40} {'Heading Error (deg)':<40} {'Lap Time (s)':<20}")
    print(f"{'':15} {'RMSE':<12} {'Std':<12} {'Max':<12} {'RMSE':<12} {'Std':<12} {'Max':<12} {'Mean':<10} {'Std':<10}")
    print("-" * 120)
    for controller in controllers:
        if controller =='purePursuit':
            controller_label = 'Pure Pursuit'
        elif controller == 'mpc':
            controller_label = 'MPC'
        else:
            controller_label = controller.capitalize()
        if map_name in all_results and controller in all_results[map_name]:
            result = all_results[map_name][controller]
            
            # Cross-track error metrics
            rmse_dev = result.get('rmse_deviation', float('nan'))
            std_dev = result.get('std_deviation', float('nan'))
            max_dev = result.get('max_deviation', float('nan'))
            
            # Heading error metrics (convert to degrees for display)
            rmse_hdg = np.degrees(result.get('rmse_heading_error', float('nan')))
            std_hdg = np.degrees(result.get('std_heading_error', float('nan')))
            max_hdg = np.degrees(result.get('max_heading_error', float('nan')))
            
            # Lap time metrics
            mean_time = result.get('mean_lap_time', float('nan'))
            std_time = result.get('std_lap_time', float('nan'))
            
            print(f"{controller_label:<15} {rmse_dev:<12.2f} {std_dev:<12.2f} {max_dev:<12.2f} "
                  f"{rmse_hdg:<12.2f} {std_hdg:<12.2f} {max_hdg:<12.2f} "
                  f"{mean_time:<10.2f} {std_time:<10.2f}")
    print("-" * 120)

    def create_combined_metrics_plot(controllers, map_name, all_results, file_path):
        """
        Creates a combined plot with cross-track error, heading error, and speed profile
        stacked vertically with a shared x-axis showing track position.
        """
        
        # Get the track length for x-axis normalization
        raceline = np.loadtxt(f'{map_path}/{map_name}_minCurve.csv', delimiter=',')
        # raceline[:,7] = np.clip(raceline[:,7], 0, float(speed)/10)
        track_length = np.insert(np.cumsum(np.linalg.norm(np.diff(raceline[:,0:2], axis=0), axis=1)), 0, 0)[-1]
        # Custom color scheme for controllers
        colors = {
            'stanley': '#1f77b4',     # Blue
            'purePursuit': '#ff7f0e', # Orange
            'mpc': '#2ca02c',         # Green
            'modifiedStanley': '#d62728'  # Red
        }
        
        # Line styles for different controllers - use distinct patterns to improve visibility
        line_styles = {
            'stanley': '-',           # Solid line
            'purePursuit': '--',      # Dashed line
            'mpc': '-.',              # Dash-dot line
            'modifiedStanley': ':'    # Dotted line
        }
        line_widths = {
            'stanley': 0.5,
            'purePursuit': 0.5,
            'mpc': 0.5,
            'modifiedStanley': 0.5
        }
        
        # Create figure with 3 subplots sharing x-axis
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16/2.54, 11/2.54), sharex=True)
        
        # Set up the cross-track error plot (top)
        ax1.set_ylabel('Cross-Track\nError (m)', wrap=True)
        ax1.grid(True, alpha=0.3)
        
        # Set up the heading error plot (middle)
        ax2.set_ylabel('Heading\nError ($^\circ$)', wrap=True)
        ax2.grid(True, alpha=0.3)
        
        # Set up the speed profile plot (bottom)
        ax3.set_xlabel('Lap progress (%)', wrap=True)
        ax3.set_ylabel('Speed (m/s)', wrap=True)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, 100])
        
        # Plot the raceline speed profile on the bottom subplot
        raceline_progress = np.insert(np.cumsum(np.linalg.norm(np.diff(raceline[:,0:2], axis=0), axis=1)), 0, 0)
        norm_raceline_progress = raceline_progress / track_length * 100
        ax3.plot(norm_raceline_progress, raceline[:,7], 'k--', linewidth=1, alpha=0.5, label='Reference')
        
        # Set y-axis limit for speed plot to ensure visibility
        # ax3.set_ylim(0, float(speed)/10 + 0.1)
        
        # Process and plot each controller's data
        for controller in controllers:
            if controller =='purePursuit':
                controller_label = 'Pure Pursuit'
            elif controller == 'mpc':
                controller_label = 'MPC'
            else:
                controller_label = controller.capitalize()
            if controller not in all_results[map_name]:
                continue
            
            # Get the median lap information
            median_lap = all_results[map_name][controller].get('median_lap')
            if median_lap is None:
                continue
            
            # Get the data for this controller
            run = np.loadtxt(f'{file_path}/{map_name}/{controller}.csv', skiprows=1, delimiter=',')
            temp = run[:,0]
            temp = np.linspace(0, temp.shape[0]/50+0.02, temp.shape[0])
            run[:,0:-1] = run[:,1:]
            run[:,-1] = temp
            laps = get_laps_indices(run[:,0:2], raceline[:,0:2])
            middle_mask = np.where((laps != min(laps)) & (laps != max(laps)))[0]
            run_middle = run[middle_mask]
            
            # Extract median lap data
            median_lap_indices = np.where(laps[middle_mask] == median_lap)[0]
            if len(median_lap_indices) == 0:
                continue
                
            # Calculate metrics for median lap
            median_run = run_middle[median_lap_indices]
            median_progress = progress_along_trajectory(median_run[:,0:2], raceline[:,0:2])
            
            # Normalize progress to 0-100%
            norm_progress = (median_progress / track_length) * 100
            
            # Sort data by progress
            sort_indices = np.argsort(norm_progress)
            norm_progress = norm_progress[sort_indices]
            
            # Calculate cross-track errors
            distances = np.zeros(len(median_lap_indices))
            heading_errors = np.zeros(len(median_lap_indices))
            speed_errors = np.zeros(len(median_lap_indices))
            
            for idx, point in enumerate(median_run):
                proj_point, distances[idx], t, i = nearest_point(point[0:2], raceline[:,0:2])
                
                # Calculate heading error as in your original code
                # if i < len(raceline) - 1:
                #     if t < 1e-6:
                #         if i > 0:
                #             raceline_heading = np.arctan2(
                #                 raceline[i, 1] - raceline[i-1, 1],
                #                 raceline[i, 0] - raceline[i-1, 0]
                #             )
                #         else:
                #             raceline_heading = np.arctan2(
                #                 raceline[i+1, 1] - raceline[i, 1],
                #                 raceline[i+1, 0] - raceline[i, 0]
                #             )
                #     elif t > 1 - 1e-6:
                #         if i < len(raceline) - 2:
                #             raceline_heading = np.arctan2(
                #                 raceline[i+2, 1] - raceline[i+1, 1],
                #                 raceline[i+2, 0] - raceline[i+1, 0]
                #             )
                #         else:
                #             raceline_heading = np.arctan2(
                #                 raceline[i+1, 1] - raceline[i, 1],
                #                 raceline[i+1, 0] - raceline[i, 0]
                #             )
                #     else:
                #         raceline_heading = np.arctan2(
                #             raceline[i+1, 1] - raceline[i, 1],
                #             raceline[i+1, 0] - raceline[i, 0]
                #         )
                # else:
                #     raceline_heading = np.arctan2(
                #         raceline[i, 1] - raceline[i-1, 1],
                #         raceline[i, 0] - raceline[i-1, 0]
                #     )

                # # print(np.degrees(np.arctan2(np.sin(raceline_heading-raceline[i,4]), np.cos(raceline_heading-raceline[i,4]))))
                
                
                vehicle_heading = point[2]  # yaw from the data
                raceline_heading = raceline[i,4] + t * (np.arctan2(np.sin(raceline[i+1,4] - raceline[i,4]), np.cos(raceline[i+1,4] - raceline[i,4]))) #if i < len(raceline) - 1 else raceline[i,4]
                raceline_heading = np.arctan2(np.sin(raceline_heading), np.cos(raceline_heading))  # Normalize raceline heading
                heading_error = -(raceline_heading - vehicle_heading)
                heading_error = np.abs(np.arctan2(np.sin(heading_error), np.cos(heading_error)))
                heading_errors[idx] = abs(heading_error)  # Take absolute value for error calculation
                speed_errors[idx] = point[3] - raceline[i,7]  # Vehicle speed (column 3) - raceline speed (column 7)
            
            # Sort the arrays based on progress
            distances = distances[sort_indices]
            heading_errors = np.degrees(heading_errors[sort_indices])
            speeds = median_run[sort_indices, 3]
            
            # No smoothing - use the raw data directly
            smooth_distances = distances
            smooth_heading_errors = heading_errors
            smooth_speeds = speeds
            smooth_progress = norm_progress
            
            # Plot the data with the custom styling
            label = f"{controller_label}"
            color = colors.get(controller, 'gray')
            linestyle = line_styles.get(controller, '-')
            linewidth = line_widths.get(controller, 1.0)
            
            # Plot cross-track error
            ax1.plot(smooth_progress, smooth_distances, linestyle=linestyle, color=color, 
                    linewidth=linewidth, label=label)
            
            # Plot heading error
            ax2.plot(smooth_progress, smooth_heading_errors, linestyle=linestyle, color=color, 
                    linewidth=linewidth, label=label)
            
            # Plot speed profile
            ax3.plot(smooth_progress, smooth_speeds, linestyle=linestyle, color=color, 
                    linewidth=linewidth, label=label)
            
            # print(f"Plotted {controller_label} median lap metrics.")
            # print(f'mean speed error: {np.mean(np.abs(speed_errors)):.2f} m/s')
            # print(f'max speed error: {np.max(np.abs(speed_errors)):.2f} m/s')
            # print(f'std speed error: {np.std(speed_errors):.2f} m/s')
            # plt.figure()
            # plt.hist(speed_errors, bins=30, alpha=0.7)
            # plt.show()
        
        # Add a single legend for the entire figure
        # Combine handles from all subplots to ensure reference speed is included
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles3, labels3 = ax3.get_legend_handles_labels()
        all_handles = handles1 + handles3
        all_labels = labels1 + labels3
        
        # Remove duplicates while preserving order
        unique_labels = []
        unique_handles = []
        for handle, label in zip(all_handles, all_labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        
        fig.legend(unique_handles, unique_labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                fancybox=True, shadow=True, ncol=len(unique_labels), frameon=True, framealpha=0.9)
        
        # Set y-axis limits based on data ranges with some padding
        ax1.set_ylim(bottom=0)  # Cross-track error starts at 0
        ax2.set_ylim(bottom=0)  # Heading error starts at 0
        
        # Add a title for the entire figure
        # fig.suptitle(f'Performance Metrics at {speed} m/s', fontsize=12)
        
        # Adjust the layout
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15, hspace=0.05)
        
        return fig


    # In your main loop where you process each speed:
    combined_metrics_fig = create_combined_metrics_plot(controllers, map_name, all_results, file_path)
    save_figure_multiple_formats(combined_metrics_fig, os.path.join(results_dir, f"{map_name}"), "combined_metrics")
