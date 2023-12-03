import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from collision_checking import collides, is_vertex_inside_polygon

# Parameters
LINK_1 = 0.4
LINK_2 = 0.25
WIDTH = 0.1
RADIUS = 0.05

angle1 = 0  # global angle of joint 1
angle2 = 0  # global angle of joint 2
joint_0 = [1, 1]  # origin joint
polygons = np.load("polygon_scene5.npy", allow_pickle=True)

def update_joints():
    global joint_1, joint_2
    joint_1 = [joint_0[0] + LINK_1 * math.cos(angle1),
               joint_0[1] + LINK_1 * math.sin(angle1)]
    joint_2 = [joint_1[0] + LINK_2 * math.cos(angle1 + angle2),
               joint_1[1] + LINK_2 * math.sin(angle1 + angle2)]
    return joint_1, joint_2

# Arm and Joint Configuration
joint_0 = [1, 1]
joint_1 = [joint_0[0] + LINK_1, joint_0[1]]
joint_2 = [joint_1[0] + LINK_2, joint_1[1]]

# Check and remove collision
polygons_to_remove = []
for i, poly in enumerate(polygons):
    for p in poly:
        dist1 = math.sqrt((p[0]-joint_1[0])**2 + (p[1]-joint_1[1])**2)
        dist2 = math.sqrt((p[0]-joint_2[0])**2 + (p[1]-joint_2[1])**2)
        if dist1 < LINK_1 or dist2 < LINK_2:
            polygons_to_remove.append(i)
            break

# Remove colliding polygons from the scene
for i in sorted(polygons_to_remove, reverse=True):
    polygons = np.delete(polygons, i, axis=0)

# Initialize Plot
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)

# Plot Polygons
for poly in polygons:
    polygon = patches.Polygon(poly, edgecolor='black', facecolor='none')
    ax.add_patch(polygon)

def draw_arm(joint_0, joint_1, joint_2):
    while len(ax.lines) > 0:
        ax.lines[0].remove()

    # Draw Arm
    plt.plot([joint_0[0], joint_1[0]], [joint_0[1], joint_1[1]], 'bo-')
    plt.plot([joint_1[0], joint_2[0]], [joint_1[1], joint_2[1]], 'bo-')


def on_press(event):
    print(f"Key pressed: {event.key}")
    global angle1, angle2
    prev_angle1 = angle1
    prev_angle2 = angle2

    if event.key == 'w':
        angle1 += 0.1
    elif event.key == 'z':
        angle1 -= 0.1
    elif event.key == 'a':
        angle2 -= 0.1
    elif event.key == 'd':
        angle2 += 0.1

    joint_1, joint_2 = update_joints()
    
    # Define rectangles representing each arm segment
    arm_segment_1 = np.array([joint_0, joint_1])
    arm_segment_2 = np.array([joint_1, joint_2])

    for poly in polygons:
        # Check collision of both segments with each polygon
        if collides(arm_segment_1, poly) or collides(arm_segment_2, poly):
            # Collision detected: revert angles
            angle1 = prev_angle1
            angle2 = prev_angle2
            update_joints()
            break

    # Update the arm drawing
    draw_arm(joint_0, joint_1, joint_2)
    fig.canvas.draw()


update_joints()
draw_arm(joint_0, joint_1, joint_2)
fig.canvas.mpl_connect('key_press_event', on_press)

def line_intersects_polygon(joint_1, joint_2, poly):
    def edges_intersect(edge1, edge2):
        def cross_product(v1, v2):
            return v1[0] * v2[1] - v1[1] * v2[0]

        def subtract_vectors(v1, v2):
            return (v1[0] - v2[0], v1[1] - v2[1])

        def vector(edge):
            return np.array([edge[1][0] - edge[0][0], edge[1][1] - edge[0][1]])

        if np.array_equal(edge1[0], edge1[1]) or np.array_equal(edge2[0], edge2[1]):
            return False

        d = cross_product(vector(edge1), subtract_vectors(edge2[0], edge1[0]))
        d1 = cross_product(vector(edge1), subtract_vectors(edge2[1], edge1[0]))
        d2 = cross_product(vector(edge2), subtract_vectors(edge1[0], edge2[0]))
        d3 = cross_product(vector(edge2), subtract_vectors(edge1[1], edge2[0]))

        if d * d1 <= 0 and d2 * d3 <= 0:
            return True

        return False
        
    # Check if arm intersects any of the polygon's edges
    for edge in zip(poly, np.roll(poly, -1, axis=0)):
        if edges_intersect((joint_1, joint_2), edge):
            return True
    
    # Check if any joint is inside the polygon
    if is_vertex_inside_polygon(joint_1, poly) or is_vertex_inside_polygon(joint_2, poly):
        return True

    return False

def arm_collides_with_polygons(joint_1, joint_2, polygons):
    for poly in polygons:
        if line_intersects_polygon(joint_1, joint_2, poly):
            return True
    return False


# Parameters
resolution = 100
config_space = np.zeros((resolution, resolution))

# Angle ranges
angles = np.linspace(-np.pi, np.pi, resolution)

# Iterate through all possible joint angles
for i, angle1 in enumerate(angles):
    for j, angle2 in enumerate(angles):
        joint_1, joint_2 = update_joints()  # Compute the positions of the joints
        if arm_collides_with_polygons(joint_1, joint_2, polygons):  # Check if there is a collision
            config_space[i, j] = 1  # Mark this configuration as in-collision

# Visualize Configuration Space
plt.imshow(config_space.T, origin='lower', extent=[-np.pi, np.pi, -np.pi, np.pi], cmap='hot', interpolation='nearest')
plt.xlabel("Joint 1 Angle")
plt.ylabel("Joint 2 Angle")
plt.title("Configuration Space")
plt.colorbar(label="Collision")
plt.savefig("config_space_scene5.png", dpi=300, bbox_inches='tight')
plt.show()





