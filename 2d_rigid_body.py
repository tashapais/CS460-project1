import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from collision_checking import collides
from scipy.spatial import ConvexHull

# Load polygons
polygons = np.load('2d_rigid_body.npy', allow_pickle=True)

# Rectangle dimensions
rect_width = 0.2
rect_height = 0.1

# Define function to create rectangle polygon at a certain position and orientation
def create_rectangle(x, y, width, height, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    rot_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                           [np.sin(angle_radians), np.cos(angle_radians)]])
    rect_corners = np.array([[x, y],
                             [x + width, y],
                             [x + width, y + height],
                             [x, y + height]])
    center = np.array([x + width/2, y + height/2])
    rect_corners_centered = rect_corners - center
    rotated_corners = np.dot(rect_corners_centered, rot_matrix.T) + center
    return rotated_corners

def initialize_rectangle(rect_width, rect_height, polygons):
    max_attempts = 1000  # To avoid infinite loop, set a maximum number of attempts
    for _ in range(max_attempts):
        # Generate random position and orientation
        x = np.random.uniform(0, 2 - rect_width)
        y = np.random.uniform(0, 2 - rect_height)
        angle = np.random.uniform(0, 360)  # Angle in degrees
        
        # Create rectangle
        rect_poly = create_rectangle(x, y, rect_width, rect_height, angle)
        
        # Check for collisions with all polygons and boundaries
        collision_detected = False
        for poly in polygons:
            if collides(rect_poly, poly):
                collision_detected = True
                break
        
        if not collision_detected:
            # If no collision, return the position and orientation
            return x, y, angle
        
    # If no valid position found after max_attempts, raise an exception or handle appropriately
    raise Exception("Could not initialize rectangle in a collision-free position after {} attempts".format(max_attempts))

# Use the function to initialize your rectangle
rect_pos_x, rect_pos_y, rect_angle = initialize_rectangle(rect_width, rect_height, polygons)
speed = 0.02  
angular_speed = 5  

# Create a global variable to keep track of the rectangle patch
rect_patch = None

def on_key_press(event):
    print(f"Key pressed: {event.key}")
    global rect_pos_x, rect_pos_y, rect_angle, rect_patch
    
    # Save current state in case we need to revert due to collision
    prev_x, prev_y, prev_angle = rect_pos_x, rect_pos_y, rect_angle
    
    if event.key == 'w':
        # Logic for moving forward
        dx = speed * np.cos(np.radians(rect_angle))
        dy = speed * np.sin(np.radians(rect_angle))
        rect_pos_x += dx
        rect_pos_y += dy
        
    elif event.key == 'z':
        # Logic for moving backward
        dx = speed * np.cos(np.radians(rect_angle))
        dy = speed * np.sin(np.radians(rect_angle))
        rect_pos_x -= dx
        rect_pos_y -= dy
        
    elif event.key == 'a':
        # Logic for rotating anticlockwise
        rect_angle += angular_speed
        
    elif event.key == 'd':
        # Logic for rotating clockwise
        rect_angle -= angular_speed
    
    # Ensure the angle stays within [0, 360)
    rect_angle = rect_angle % 360
    
    # Create a new rectangle with the updated position and orientation
    new_rect = create_rectangle(rect_pos_x, rect_pos_y, rect_width, rect_height, rect_angle)
    print(f"Updated Position: ({rect_pos_x}, {rect_pos_y}), Angle: {rect_angle}")

    # Check if rectangle is within boundaries
    if np.any(new_rect < 0) or np.any(new_rect[:, 0] > 2) or np.any(new_rect[:, 1] > 2):
        rect_pos_x, rect_pos_y, rect_angle = prev_x, prev_y, prev_angle
        print("Out of bounds! Move reverted.")
        return

    # Check for collisions with all polygons and boundaries
    for poly in polygons:
        if collides(new_rect, poly):
            # If collision, revert to the previous state
            rect_pos_x, rect_pos_y, rect_angle = prev_x, prev_y, prev_angle
            print("Collision detected! Move reverted.")
            return
    
    
    # Remove the old rectangle patch if it exists
    if rect_patch is not None:
        rect_patch.remove()
    
    # Draw the new rectangle and update the patch
    rect_patch = Polygon(new_rect, edgecolor='blue', facecolor='none')
    plt.gca().add_patch(rect_patch)
    plt.draw()

fig, ax = plt.subplots(dpi=100)
plt.axis('equal')

# Plotting all polygons
for poly in polygons:
    ax.add_patch(Polygon(poly, edgecolor='black', facecolor='none'))

# Plotting the initial position of the rectangle
#ax.add_patch(Polygon(create_rectangle(rect_pos_x, rect_pos_y, rect_width, rect_height, rect_angle), edgecolor='blue', facecolor='none'))

# Connecting the event handler
plt.gcf().canvas.mpl_connect('key_press_event', on_key_press)

# Adjusting the spines to make x=0 intersect with y=0
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

def compute_minkowski_sum(polygon_a, polygon_b, angle_degrees):
    # Rotate polygon_b by angle_degrees around its centroid
    angle_radians = np.radians(angle_degrees)
    rot_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                           [np.sin(angle_radians), np.cos(angle_radians)]])
    centroid_b = np.mean(polygon_b, axis=0)
    polygon_b_rotated = np.dot(polygon_b - centroid_b, rot_matrix.T) + centroid_b
    
    # Initialize an empty array to hold the vertices of the Minkowski sum.
    minkowski_polygon_vertices = []
    
    # Iterate through all vertices in each polygon.
    for vertex_a in polygon_a:
        for vertex_b in polygon_b_rotated:
            # Add together the current vertices from each polygon and append to the list.
            minkowski_polygon_vertices.append(vertex_a + vertex_b)
    
    # Convert the list to a NumPy array for easier manipulation and scipy compatibility.
    minkowski_polygon_vertices = np.array(minkowski_polygon_vertices)
    
    # Compute the convex hull of the Minkowski sum vertices.
    hull = ConvexHull(minkowski_polygon_vertices)
    
    # Return the vertices of the Minkowski sum in order.
    return minkowski_polygon_vertices[hull.vertices]


rect_orientations = [0, 45, 90]  
num_scenes = 5

# Initialize rect_patch at the beginning
rect_pos_x, rect_pos_y, rect_angle = initialize_rectangle(rect_width, rect_height, polygons)
rect_vertices = create_rectangle(rect_pos_x, rect_pos_y, rect_width, rect_height, rect_angle)
rect_patch = patches.Polygon(rect_vertices, edgecolor='blue', facecolor='none')

for i in range(num_scenes):
    # Load each scene
    scene = np.load(f'polygon_scene{i+1}.npy', allow_pickle=True)
    
    for j, orientation in enumerate(rect_orientations):
        plt.figure(figsize=(6,6))
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        
        for obstacle in scene:
            # Visualize original obstacle
            ax.add_patch(patches.Polygon(obstacle, edgecolor='black', facecolor='grey'))
            
            # Ensure rect_patch is initialized
            if rect_patch is None:
                rect_pos_x, rect_pos_y, rect_angle = initialize_rectangle(rect_width, rect_height, polygons)
                rect_vertices = create_rectangle(rect_pos_x, rect_pos_y, rect_width, rect_height, rect_angle)
                rect_patch = patches.Polygon(rect_vertices, edgecolor='blue', facecolor='none')
            
            # Extract vertices from rect_patch and calculate Minkowski sum
            rect_vertices = rect_patch.get_xy()

            # Compute and visualize the Minkowski sum
            minkowski_polygon = compute_minkowski_sum(obstacle, rect_vertices, orientation)
            ax.add_patch(patches.Polygon(minkowski_polygon, edgecolor='red', fill=None))
        
        plt.title(f'Scene {i+1}, Rectangle Orientation: {orientation} degrees')
        plt.xlim([0, 6])
        plt.ylim([0, 6])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.savefig(f'scene_{i+1}_orientation_{j+1}.png')
        plt.close()

