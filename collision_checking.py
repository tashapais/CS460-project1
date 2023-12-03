import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

m = 2


def collides(poly1: np.ndarray, poly2: np.ndarray):
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

    for edge1 in zip(poly1, np.roll(poly1, -1, axis=0)):
        for edge2 in zip(poly2, np.roll(poly2, -1, axis=0)):
            if edges_intersect(edge1, edge2):
                return True

    for x in poly1:
        if is_vertex_inside_polygon(x, poly2):
            return True
    for x in poly2:
        if is_vertex_inside_polygon(x, poly1):
            return True

    return False


def is_vertex_inside_polygon(vertex, polygon):
    # Check if a vertex is inside a polygon using ray casting algorithm
    x, y = vertex
    n = len(polygon)
    inside = False
    for i in range(n):
        xi, yi = polygon[i]
        xi1, yi1 = polygon[(i + 1) % n]
        if ((yi > y) != (yi1 > y)) and (x < (xi1 - xi) * (y - yi) / (yi1 - yi) + xi):
            inside = not inside
    return inside


def plot(polys: np.ndarray, name="collision_plot.png"):
    fig, ax = plt.subplots(dpi=100)
    ax.set_aspect('equal')
    
    colliding_polygons = []
    non_colliding_polygons = []

    for i, poly1 in enumerate(polys):
        colliding = False
        for j, poly2 in enumerate(polys):
            if i != j and collides(poly1, poly2):
                colliding = True
                break
        
        if colliding:
            colliding_polygons.append(poly1)
        else:
            non_colliding_polygons.append(poly1)

    colliding_colors = ['gray'] * len(colliding_polygons)
    non_colliding_colors = ['white'] * len(non_colliding_polygons)

    colliding_patches = [Polygon(poly) for poly in colliding_polygons]
    non_colliding_patches = [Polygon(poly) for poly in non_colliding_polygons]

    colliding_collection = PatchCollection(colliding_patches, facecolors=colliding_colors, edgecolors="black", alpha=0.5)
    non_colliding_collection = PatchCollection(non_colliding_patches, facecolors=non_colliding_colors, edgecolors="black", alpha=0.5)

    
    ax.add_collection(colliding_collection)
    for polygon in colliding_polygons:
        ax.fill(polygon[:, 0], polygon[:, 1], 'black', alpha=0.5)
    ax.add_collection(non_colliding_collection)

    # ax.autoscale()
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)
    plt.savefig(name)
    plt.show()


if __name__ == "__main__":
    for x in range(1, 6):
        polygons = np.load(f"polygon_scene{x}.npy", allow_pickle=True)
        plot(polygons, f"collision_plot{x}.png")
    # # Load polygons from the file
    # polygons = np.load("collision_checking_polygons.npy", allow_pickle=True)

    # # Plot and save the collision detection result
    # plot(polygons)
