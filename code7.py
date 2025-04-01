import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib
matplotlib.use("TkAgg")

# ====================================================
# CONFIGURABLE PARAMETERS
# ====================================================

# Robot parameters
ROBOT_INITIAL_POSE = [0.0, 0.0, 0.0]  # [x, y, theta] - Initial pose of the robot
ROBOT_WHEELBASE = 0.5  # Distance between the robot's wheels
ROBOT_BODY_RADIUS = 0.3  # Radius of the robot's body
ROBOT_WHEEL_WIDTH = 0.2  # Width of the robot's wheels
ROBOT_WHEEL_HEIGHT = 0.1  # Height of the robot's wheels

# Control inputs (keyboard commands)
CONTROL_FORWARD_VELOCITY = 2.0  # Velocity when moving forward/backward
CONTROL_TURN_VELOCITY = 1.0  # Velocity difference when turning left/right

# Landmarks
LANDMARKS = np.array([
    [-20.0, 10.0],  # Landmark 1
    [-20.0, -10.0],  # Landmark 2
    [20.0, 10.0],  # Landmark 3
    [20.0, -10.0],  # Landmark 4
    [5.0, -3.0]   # Landmark 5
])


# EKF parameters
# Initial state: [x, y, theta, lx1, ly1, lx2, ly2, lx3, ly3, lx4, ly4, lx5, ly5]
EKF_INITIAL_STATE = [0.0, 0.0, 0.0, LANDMARKS[0, 0], LANDMARKS[0, 1], LANDMARKS[1, 0], LANDMARKS[1, 1], LANDMARKS[2, 0], LANDMARKS[2, 1], LANDMARKS[3, 0], LANDMARKS[3, 1], LANDMARKS[4, 0], LANDMARKS[4, 1]]
EKF_INITIAL_COVARIANCE = np.eye(13) * 0.1  # Initial uncertainty in the state estimate
EKF_PROCESS_NOISE = np.eye(13) * 1  # Process noise covariance (Q)
EKF_MEASUREMENT_NOISE = np.eye(2) * 1  # Measurement noise covariance (R)


# Simulation parameters
TIME_STEP = 0.05  # Time step for simulation (dt)
SIMULATION_DURATION = 1000  # Number of animation frames

# Visualization parameters
PLOT_LIMITS_x = [-20, 20]  # Plot limits for x axes
PLOT_LIMITS_y = [-10, 10]  # Plot limits for y axes
TRAJECTORY_COLOR = 'r--'  # Color/style for the robot's trajectory
DETECTION_COLOR = 'blue'  # Color for landmark detection lines
EKF_PREDICTION_COLOR = 'orange'  # Color for EKF prediction line

# ====================================================
# ROBOT AND EKF IMPLEMENTATION
# ====================================================

class UnicycleRobot:
    """
    A class to represent a unicycle robot.
    The robot has a position (x, y), orientation (theta), and can move based on wheel velocities.
    """
    def __init__(self, x=0.0, y=0.0, theta=0.0, wheelbase=1.0):
        self.x = x  # x position
        self.y = y  # y position
        self.theta = theta  # Orientation (theta)
        self.wheelbase = wheelbase  # Distance between wheels
        self.v = 0.0  # Linear velocity
        self.omega = 0.0  # Angular velocity

    def set_wheel_velocities(self, v_l, v_r):
        """
        Set the robot's linear and angular velocities based on left and right wheel velocities.
        """
        self.v = (v_r + v_l) / 2  # Linear velocity is the average of the two wheels
        self.omega = (v_r - v_l) / self.wheelbase  # Angular velocity depends on the difference between the wheels

    def update(self, dt):
        """
        Update the robot's pose based on its current velocities and the time step.
        """
        self.theta += self.omega * dt  # Update orientation
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))  # Normalize theta to [-pi, pi]
        self.x += self.v * np.cos(self.theta) * dt  # Update x position
        self.y += self.v * np.sin(self.theta) * dt  # Update y position

    def get_state(self):
        """
        Get the robot's current state (x, y, theta).
        """
        return self.x, self.y, self.theta

class EKFLocalization:
    """
    A class to implement the Extended Kalman Filter (EKF) for robot localization.
    The EKF estimates the robot's pose and landmark positions based on sensor measurements.
    """
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.state = initial_state  # State vector [x, y, theta, lx1, ly1, lx2, ly2, ...]
        self.covariance = initial_covariance  # Covariance matrix of the state estimate
        self.process_noise = process_noise  # Process noise covariance (Q)
        self.measurement_noise = measurement_noise  # Measurement noise covariance (R)

    def predict(self, control_input, dt):
        """
        Predict the robot's state and covariance based on the control input and time step.
        """
        # Extract robot state
        x, y, theta = self.state[:3]
        v, omega = control_input

        # Predict robot state
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + omega * dt

        # Update state
        self.state[:3] = np.array([x_new, y_new, theta_new])

        # Compute Jacobian F (linearization of the motion model)
        F = np.eye(len(self.state))  # Start with identity matrix
        F[0, 2] = -v * np.sin(theta) * dt  # Partial derivative of x with respect to theta
        F[1, 2] = v * np.cos(theta) * dt  # Partial derivative of y with respect to theta

        # Update covariance
        self.covariance = F @ self.covariance @ F.T + self.process_noise

    def update(self, measurements):
        """
        Update the state and covariance based on sensor measurements.
        """
        for measurement in measurements:
            landmark_id, z = measurement
            lx, ly = self.state[3 + 2 * landmark_id:5 + 2 * landmark_id]  # Landmark position
            
            # Predicted measurement
            dx = lx - self.state[0]  # Difference in x
            dy = ly - self.state[1]  # Difference in y
            r_pred = np.sqrt(dx**2 + dy**2)  # Predicted range
            phi_pred = np.arctan2(dy, dx) - self.state[2]  # Predicted bearing

            # Measurement Jacobian H (linearization of the measurement model)
            H = np.zeros((2, len(self.state)))
            H[0, 0] = -dx / r_pred  # Partial derivative of range with respect to x
            H[0, 1] = -dy / r_pred  # Partial derivative of range with respect to y
            H[0, 3 + 2 * landmark_id] = dx / r_pred  # Partial derivative of range with respect to landmark x
            H[0, 4 + 2 * landmark_id] = dy / r_pred  # Partial derivative of range with respect to landmark y
            H[1, 0] = dy / (r_pred**2)  # Partial derivative of bearing with respect to x
            H[1, 1] = -dx / (r_pred**2)  # Partial derivative of bearing with respect to y
            H[1, 2] = -1  # Partial derivative of bearing with respect to theta
            H[1, 3 + 2 * landmark_id] = -dy / (r_pred**2)  # Partial derivative of bearing with respect to landmark x
            H[1, 4 + 2 * landmark_id] = dx / (r_pred**2)  # Partial derivative of bearing with respect to landmark y

            # Kalman Gain
            S = H @ self.covariance @ H.T + self.measurement_noise  # Innovation covariance
            K = self.covariance @ H.T @ np.linalg.inv(S)  # Kalman Gain

            # Update state and covariance
            innovation = z - np.array([r_pred, phi_pred])  # Innovation (difference between predicted and actual measurement)
            self.state += K @ innovation  # Update state
            self.covariance = (np.eye(len(self.state)) - K @ H) @ self.covariance  # Update covariance

# ====================================================
# VISUALIZATION
# ====================================================

class RobotVisualizer:
    """
    A class to handle the visualization of the robot, landmarks, and EKF estimates.
    """
    def __init__(self, ax, robot, ekf, landmarks, wheelbase=0.5, body_radius=0.3):
        self.ax = ax  # Matplotlib axis
        self.robot = robot  # Robot object
        self.ekf = ekf  # EKF object
        self.landmarks = landmarks  # Landmark positions
        self.wheelbase = wheelbase  # Distance between wheels
        self.body_radius = body_radius  # Radius of the robot's body

        # Create patches for the robot's body, wheels, and orientation arrow
        self.body = patches.Circle((0, 0), body_radius, color='deepskyblue', ec="black", lw=2, alpha=0.8)
        self.wheel_left = patches.Rectangle((-0.1, -0.05), ROBOT_WHEEL_WIDTH, ROBOT_WHEEL_HEIGHT, color='black')
        self.wheel_right = patches.Rectangle((-0.1, -0.05), ROBOT_WHEEL_WIDTH, ROBOT_WHEEL_HEIGHT, color='black')
        self.orientation_arrow = self.create_arrow(0, 0, 0.3, 0)

        # Add trajectory line
        self.trajectory, = ax.plot([], [], TRAJECTORY_COLOR, lw=1)  # Trajectory line
        self.x_data, self.y_data = [], []  # Store trajectory points

        # Add EKF prediction line
        self.ekf_prediction, = ax.plot([], [], linestyle='--', color=EKF_PREDICTION_COLOR, lw=1)  # EKF prediction line
        self.ekf_x_data, self.ekf_y_data = [], []  # Store EKF prediction points

        # Add patches to the axis
        self.ax.add_patch(self.body)
        self.ax.add_patch(self.wheel_left)
        self.ax.add_patch(self.wheel_right)
        self.ax.add_patch(self.orientation_arrow)

        # Plot landmarks
        for i, (lx, ly) in enumerate(self.landmarks):
            self.ax.scatter(lx, ly, color='green', label=f'Landmark {i+1}')
            self.ax.text(lx, ly, f'L{i+1}', fontsize=12, ha='right')

        # Add detection lines
        self.detection_lines = [ax.plot([], [], color=DETECTION_COLOR, linestyle='--')[0] for _ in range(len(landmarks))]

    def create_arrow(self, x, y, dx, dy):
        """
        Create an arrow representing the robot's orientation.
        """
        return patches.Arrow(x, y, dx, dy, width=0.05, color='red')

    def update(self):
        """
        Update the visualization with the robot's current state and EKF estimates.
        """
        x, y, theta = self.robot.get_state()

        # Update trajectory
        self.x_data.append(x)
        self.y_data.append(y)
        self.trajectory.set_data(self.x_data, self.y_data)

        # Update EKF prediction line
        ekf_x, ekf_y, _ = self.ekf.state[:3]
        self.ekf_x_data.append(ekf_x)
        self.ekf_y_data.append(ekf_y)
        self.ekf_prediction.set_data(self.ekf_x_data, self.ekf_y_data)

        # Update body position
        self.body.center = (x, y)

        # Compute wheel positions
        wheel_offset = self.wheelbase / 2
        wheel_x_offset = 0.1  # Adjust for better positioning

        left_wheel_x = x - wheel_offset * np.cos(theta) - wheel_x_offset
        left_wheel_y = y - wheel_offset * np.sin(theta) - 0.05
        right_wheel_x = x + wheel_offset * np.cos(theta) - wheel_x_offset
        right_wheel_y = y + wheel_offset * np.sin(theta) - 0.05

        self.wheel_left.set_xy((left_wheel_x, left_wheel_y))
        self.wheel_right.set_xy((right_wheel_x, right_wheel_y))

        # Update orientation arrow
        arrow_length = self.body_radius * 1.8

        # Remove the old arrow
        self.orientation_arrow.remove()

        # Create and add the new arrow
        self.orientation_arrow = self.create_arrow(
            x, y, arrow_length * np.cos(theta), arrow_length * np.sin(theta)
        )
        self.ax.add_patch(self.orientation_arrow)

        # Update detection lines
        for i, (lx, ly) in enumerate(self.landmarks):
            self.detection_lines[i].set_data([x, lx], [y, ly])

# ====================================================
# MAIN SIMULATION
# ====================================================

# Initialize robot
robot = UnicycleRobot(*ROBOT_INITIAL_POSE, ROBOT_WHEELBASE)

# Initialize EKF
ekf = EKFLocalization(EKF_INITIAL_STATE, EKF_INITIAL_COVARIANCE, EKF_PROCESS_NOISE, EKF_MEASUREMENT_NOISE)

# Set up figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(*PLOT_LIMITS_x)
ax.set_ylim(*PLOT_LIMITS_y)
#ax.set_aspect('equal')
ax.grid()

# Create visualizer
visualizer = RobotVisualizer(ax, robot, ekf, LANDMARKS, ROBOT_WHEELBASE, ROBOT_BODY_RADIUS)

# Keyboard control function
def on_key_press(event):
    """
    Handle keyboard input to control the robot.
    """
    if event.key == 'up':
        robot.set_wheel_velocities(CONTROL_FORWARD_VELOCITY, CONTROL_FORWARD_VELOCITY)  # Move forward
    elif event.key == 'down':
        robot.set_wheel_velocities(-CONTROL_FORWARD_VELOCITY, -CONTROL_FORWARD_VELOCITY)  # Move backward
    elif event.key == 'left':
        robot.set_wheel_velocities(CONTROL_TURN_VELOCITY, CONTROL_FORWARD_VELOCITY)  # Turn left
    elif event.key == 'right':
        robot.set_wheel_velocities(CONTROL_FORWARD_VELOCITY, CONTROL_TURN_VELOCITY)  # Turn right

# Connect keyboard event
fig.canvas.mpl_connect('key_press_event', on_key_press)

# Animation function
def animate(i):
    """
    Update the simulation at each time step.
    """
    # Simulate sensor measurements
    measurements = []
    for j, (lx, ly) in enumerate(LANDMARKS):
        dx = lx - robot.x
        dy = ly - robot.y
        r = np.sqrt(dx**2 + dy**2)
        phi = np.arctan2(dy, dx) - robot.theta
        measurements.append((j, np.array([r, phi]) + np.random.normal(0, 0.1, 2)))  # Add noise

    # Update EKF
    ekf.predict((robot.v, robot.omega), TIME_STEP)
    ekf.update(measurements)

    # Update robot
    robot.update(TIME_STEP)
    visualizer.update()
    return []

# Run animation
ani = animation.FuncAnimation(fig, animate, frames=SIMULATION_DURATION, init_func=lambda: [], blit=False, interval=100)
plt.show()
