import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull


def generate_convex_polygon(center, min_vertices, max_vertices, min_radius, max_radius):
    num_vertices = np.random.randint(min_vertices, max_vertices + 1)
    angles = np.deg2rad(np.random.uniform(0, 360, num_vertices))
    radii = np.random.uniform(min_radius, max_radius, num_vertices)

    vertices = np.column_stack(
        (center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles))
    )

    hull = ConvexHull(vertices)
    convex_vertices = vertices[hull.vertices]

    return convex_vertices


def generate_scene(num_polygons, m, min_vertices, max_vertices, min_radius, max_radius):
    scene = np.empty(num_polygons, dtype=object)
    for i in range(num_polygons):
        center = np.random.uniform(max_radius, m - max_radius, size=2)
        scene[i] = generate_convex_polygon(
            center, min_vertices, max_vertices, min_radius, max_radius
        )
    return scene


def visualize_scene(scene, resolution, m):
    fig, ax = plt.subplots(figsize=(resolution / 100, resolution / 100))
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)

    for polygon in scene:
        polygon = np.vstack((polygon, polygon[0]))
        ax.plot(polygon[:, 0], polygon[:, 1])

    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig("scene.png")
    plt.show()


def save_scene_to_file(scene, filename):
    np.save(filename, scene)


def load_scene_from_file(filename):
    return np.load(filename, allow_pickle=True)


def generate_A_and_T():
    scene = np.empty(6, dtype=object)
    # A
    scene[0] = np.array([[0.35, 1.5],
        [0.4, 1.4],
        [0.3, 1.4],
    ])
    scene[1] = np.array([[0.3, 1.4],
        [0.35, 1.4],
        [0.2, 1.1],
        [0.15, 1.1],
    ])
    scene[2] = np.array([[0.4, 1.4],
        [0.35, 1.4],
        [0.5, 1.1],
        [0.55, 1.1],
    ])
    scene[3] = np.array([[0.25, 1.3],
        [0.45, 1.3],
        [0.45, 1.25],
        [0.25, 1.25],
    ])
    # T
    scene[4] = np.array([[1.0, 1.5],
        [1.5, 1.5],
        [1.5, 1.4],
        [1.0, 1.4],
    ])
    scene[5] = np.array([[1.2, 1.4],
        [1.3, 1.4],
        [1.3, 1.0],
        [1.2, 1.0],
    ])
    return scene


if __name__ == "__main__":
    num_polygons = 32
    m = 2.0
    min_vertices = 3
    max_vertices = 8
    min_radius = 0.05
    max_radius = 0.3
    resolution = 800

    scene = generate_scene(
        num_polygons, m, min_vertices, max_vertices, min_radius, max_radius
    )
    scene = generate_A_and_T()
    visualize_scene(scene, resolution, m)

    save_scene_to_file(scene, "polygon_scene5.npy")
    loaded_scene = load_scene_from_file("polygon_scene5.npy")
    visualize_scene(loaded_scene, resolution, m)
    # print(np.info(loaded_scene))

    # Example: Accessing the first polygon's vertices
    # print("First Polygon Vertices:")
    # print(loaded_scene[0])
    # print("All Polygon Vertices:")
    # print(loaded_scene)
