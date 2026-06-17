import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.widgets import Slider  

# ==========================================
# SIMULATION CONFIGURATION (Global Variables)
# ==========================================
sa = 30           # Steering angle limit in degrees
wb = 2            # Wheelbase in meters
k_gain = 0.5      # Default position gain (used when testing speeds)
ks_gain = 1.0     # Default softening gain
v_target_default = 10.0  # Default target speed (m/s) (used when testing gains)
lat_offset = 20.0 # Initial lateral offset from path (meters)
init_yaw_deg = 45.0 # Initial vehicle heading in DEGREES (0 = East, 90 = North)
time_limit = 20.0 # Simulation run time (seconds)
dt = 0.05         # Physics time step (seconds)

# Option Arrays
gains_to_test = [0.1, 0.5, 1.5, 2.0]   # Configuration A: Test different gains
speeds_to_test = [2.0, 5.0, 8.0, 12.0] # Configuration B: Test different target speeds (m/s)

# ==========================================
# VEHICLE & CONTROLLER CLASSES
# ==========================================
class KinematicBicycleModel:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, L=2.0): 
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.L = L  

    def update(self, a, delta, dt, max_steer_deg): 
        delta = np.clip(delta, -np.radians(max_steer_deg), np.radians(max_steer_deg))
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += (self.v / self.L) * np.tan(delta) * dt
        self.v += a * dt

class StanleyController:
    def __init__(self, k=0.5, ks=1.0): 
        self.k = k
        self.ks = ks

    def compute_steering(self, vehicle, path_x, path_y, path_yaw):
        # 1. Project to the FRONT axle position
        fx = vehicle.x + vehicle.L * np.cos(vehicle.yaw)
        fy = vehicle.y + vehicle.L * np.sin(vehicle.yaw)
        
        # 2. Find the closest trajectory waypoint index
        dx = [fx - x for x in path_x]
        dy = [fy - y for y in path_y]
        distances = np.hypot(dx, dy)
        target_idx = np.argmin(distances)

        # 3. Compute clean error vectors relative to the path's direction
        dx_front = fx - path_x[target_idx]
        dy_front = fy - path_y[target_idx]
        
        path_vec_x = np.cos(path_yaw[target_idx])
        path_vec_y = np.sin(path_yaw[target_idx])
        
        # 4. Cross-Product for Orientation Side Selection
        cross_product = path_vec_x * dy_front - path_vec_y * dx_front
        
        # FIXED SIGN: Added a negative sign here to convert the positive feedback loop 
        # into a smooth corrective negative feedback loop.
        if cross_product >= 0:
            error_front_axle = -distances[target_idx]
        else:
            error_front_axle = distances[target_idx]

        # 5. Heading Error Calculation (Wrapped cleanly within [-pi, pi])
        yaw_error = path_yaw[target_idx] - vehicle.yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

        # 6. Smooth Stanley Control Law
        crosstrack_steering = np.arctan2(self.k * error_front_axle, vehicle.v + self.ks)
        delta = yaw_error + crosstrack_steering
        
        return delta, error_front_axle

# ==========================================
# SIMULATION CORE ENGINE (FIXED BOUNDARIES)
# ==========================================
def run_simulation(path_x, path_y, path_yaw, k_param, ks_param, v_target_param):
    vehicle = KinematicBicycleModel(x=path_x[0], y=path_y[0] - lat_offset, yaw=np.radians(init_yaw_deg), v=2.0, L=wb)
    controller = StanleyController(k=k_param, ks=ks_param)
    
    time = 0.0
    logs = []
    
    # Convert steering limit to radians once for performance
    max_steer_rad = np.radians(sa)

    while time < time_limit:
        raw_delta, cte = controller.compute_steering(vehicle, path_x, path_y, path_yaw)
        
        # FIXED: Enforce physical steering saturation limits immediately
        delta = np.clip(raw_delta, -max_steer_rad, max_steer_rad)
        
        # Throttle and tire scrub calculations now use the real, clipped delta
        dynamic_v_target = v_target_param * np.cos(delta) 
        tire_scrub_penalty = 1.2 * np.sin(abs(delta)) * vehicle.v
        a = 0.5 * (dynamic_v_target - vehicle.v) - tire_scrub_penalty
        
        # Log the real, bounded delta
        logs.append({
            'time': time, 'x': vehicle.x, 'y': vehicle.y, 
            'yaw': vehicle.yaw, 'v': vehicle.v, 'cte': cte, 'delta': delta
        })
        
        vehicle.update(a, delta, dt, max_steer_deg=sa)
        time += dt
        
    return pd.DataFrame(logs)

# ==========================================
# EXECUTION & DATA GENERATION
# ==========================================
waypoints_x = [0.0, 20.0,  40.0, 60.0,  80.0, 100.0, 110.0, 100.0, 80.0]
waypoints_y = [0.0,  5.0, -10.0,  0.0,  15.0,  10.0,  -5.0, -20.0, -25.0]

t_waypoints = np.arange(len(waypoints_x))
spline_x = CubicSpline(t_waypoints, waypoints_x, bc_type='natural')
spline_y = CubicSpline(t_waypoints, waypoints_y, bc_type='natural')

t_fine = np.linspace(0, len(waypoints_x) - 1, 1000)
path_x = spline_x(t_fine)
path_y = spline_y(t_fine)

dx_dt = spline_x(t_fine, 1)
dy_dt = spline_y(t_fine, 1)
path_yaw = np.arctan2(dy_dt, dx_dt)

results = {}
active_study = ""

# -------------------------------------------------------------
# CHOOSE YOUR OBSERVATION STUDY HERE (Comment/Uncomment blocks)
# -------------------------------------------------------------

# --- STUDY OPTION 1: Test different Controller Position Gains ---
active_study = "Gains"
for g in gains_to_test:
    run_id = f"Gain_k_{g}"
    results[run_id] = run_simulation(path_x, path_y, path_yaw, k_param=g, ks_param=ks_gain, v_target_param=v_target_default)

# --- STUDY OPTION 2: Test different Target Velocities ---
# Uncomment the block below and comment out Option 1 above to test speeds!
# active_study = "Speeds"
# for v in speeds_to_test:
#     run_id = f"V_target_{v}m/s"
#    results[run_id] = run_simulation(path_x, path_y, path_yaw, k_param=k_gain, ks_param=ks_gain, v_target_param=v)

# -------------------------------------------------------------

# ==========================================
# DESKTOP STANDARD PLOTTING ENGINE (.py Friendly)
# ==========================================
fig = plt.figure(figsize=(18, 10))
plt.subplots_adjust(bottom=0.18) 

ax_map = fig.add_subplot(2, 3, 1)
ax_cte = fig.add_subplot(2, 3, 2)
ax_compass = fig.add_subplot(2, 3, 3, projection='polar')
ax_steer = fig.add_subplot(2, 3, 4)
ax_vel = fig.add_subplot(2, 3, 5)
ax_blank = fig.add_subplot(2, 3, 6)
ax_blank.axis('off')

run_colors = {}

def build_static_background():
    ax_map.plot(path_x, path_y, 'r--', linewidth=2, label='Smoothed Reference Path')
    ax_map.plot(waypoints_x, waypoints_y, 'ko', markersize=6, label='Raw Waypoints')
    
    for run_id, df in results.items():
        line, = ax_map.plot(df['x'], df['y'], label=run_id, alpha=0.4)
        run_colors[run_id] = line.get_color()
        ax_cte.plot(df['time'], df['cte'], color=run_colors[run_id])
        ax_steer.plot(df['time'], np.degrees(df['delta']), color=run_colors[run_id])
        ax_vel.plot(df['time'], df['v'], color=run_colors[run_id])

    ax_map.set_xlabel('X Position (m)')
    ax_map.set_ylabel('Y Position (m)')
    ax_map.legend(loc='upper left')
    ax_map.grid(True)
    ax_map.axis('equal')

    ax_cte.set_xlabel('Time (s)')
    ax_cte.set_ylabel('Cross-Track Error (m)')
    ax_cte.grid(True)

    ax_compass.set_theta_zero_location("E") 
    ax_compass.set_theta_direction(1) 
    ax_compass.set_yticklabels([]) 

    ax_steer.axhline(y=sa, color='grey', linestyle=':')
    ax_steer.axhline(y=-sa, color='grey', linestyle=':')
    ax_steer.set_xlabel('Time (s)')
    ax_steer.set_ylabel('Steering Angle (Degrees)')
    ax_steer.grid(True)

    ax_vel.set_xlabel('Time (s)')
    ax_vel.set_ylabel('Velocity (m/s)')
    ax_vel.grid(True)
    
    # MODIFIED: Dynamically update reference targets depending on the active study type
    if active_study == "Gains":
        ax_vel.axhline(y=v_target_default, color='r', linestyle='--', linewidth=1.5, label='Target Speed')
        ax_vel.legend(loc='lower right')
    elif active_study == "Speeds":
        # Plot corresponding target limit lines for clarity if inspecting variations
        for v in speeds_to_test:
            ax_vel.axhline(y=v, linestyle=':', alpha=0.5, color='grey')

build_static_background()

dynamic_artists = []

def update_plots(target_time):
    global dynamic_artists
    for artist in dynamic_artists:
        artist.remove()
    dynamic_artists.clear()

    ax_map.set_title(f'Vehicle Positions at t = {target_time:.2f}s')
    
    v1 = ax_cte.axvline(x=target_time, color='k', linestyle='--', linewidth=1.2)
    v2 = ax_steer.axvline(x=target_time, color='k', linestyle='--', linewidth=1.2)
    v3 = ax_vel.axvline(x=target_time, color='k', linestyle='--', linewidth=1.2)
    dynamic_artists.extend([v1, v2, v3])

    for run_id, df in results.items():
        closest_idx = (df['time'] - target_time).abs().argmin()
        row = df.iloc[closest_idx]
        color = run_colors[run_id]

        p_dot, = ax_map.plot(row['x'], row['y'], marker='o', markersize=10, color=color, markeredgecolor='black')
        c_line, = ax_compass.plot([0, row['yaw']], [0, 1.0], color=color, linewidth=2.5)
        c_dot, = ax_compass.plot(row['yaw'], 1.0, marker='o', markersize=10, color=color, markeredgecolor='black', label=run_id)
        
        dynamic_artists.extend([p_dot, c_line, c_dot])

    ax_compass.set_title("Current Vehicle Heading ($\psi$)", pad=20)
    handles, labels = ax_compass.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    leg = ax_compass.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)
    dynamic_artists.append(leg)
    
    fig.canvas.draw_idle()

update_plots(0.0)

slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
time_slider = Slider(ax=slider_ax, label='Sim Time (s)', valmin=0.0, valmax=time_limit, valinit=0.0, valstep=dt)

def on_slider_change(val):
    update_plots(val)

time_slider.on_changed(on_slider_change)

plt.show()