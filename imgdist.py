from heapq import heappush, heappop
import itertools
import math
import os
import sys
import networkx as nx
from PIL import Image


def load_image(file_path):
    image = Image.open(file_path)
    pixels = image.load()
    return image, pixels


def save_image(image, original_file_path):
    file_basename = os.path.basename(original_file_path).split('.')[0]
    image.save('results/{}-shortest-path.png'.format(file_basename))


def pixel(x, y):
    return '{},{}'.format(x, y)


def coords(p):
    return map(int, p.split(','))


def rgb(pixel_map, p):
    c, r = coords(p)
    return pixel_map[c, r]


def colordist(pixel_map, p1, p2):
    r1, g1, b1 = rgb(pixel_map, p1)
    r2, g2, b2 = rgb(pixel_map, p2)
    distance = math.sqrt((r2 - r1) ** 2 + (g2 - g1) ** 2 + (b2 - b1) ** 2)
    return distance


def generate_graph(image, pixels):
    image_width, image_length = image.size
    graph = nx.Graph()

    # Add each pixel to the graph as a node
    for col in range(image_width):
        for row in range(image_length):
            graph.add_node(pixel(col, row))

    # Add edges between adjacent pixels, weighted by their color distance
    for col in range(image_width):
        for row in range(image_length):
            curr = pixel(col, row)
            up = pixel(col, row - 1)
            down = pixel(col, row + 1)
            left = pixel(col - 1, row)
            right = pixel(col + 1, row)
            if row - 1 >= 0:
                graph.add_edge(curr, up, weight=colordist(pixels, curr, up))
            if row + 1 < image_length:
                graph.add_edge(curr, down, weight=colordist(pixels, curr, down))
            if col - 1 >= 0:
                graph.add_edge(curr, left, weight=colordist(pixels, curr, left))
            if col + 1 < image_width:
                graph.add_edge(curr, right, weight=colordist(pixels, curr, right))

    return graph


def draw_shortest_path(pixels, path):
    green = (0, 255, 0)
    for p in path:
        col, row = coords(p)
        pixels[col, row] = green


def visualize_graph(graph):
    pass


def decrease_key(heap, new_key, item):
    heap_size = len(heap)
    for i in range(heap_size):
        _, c, v = heap[i]
        if v == item:
            heap.pop(i)
            break
    new_tuple = (new_key, c, item)
    heappush(heap, new_tuple)


def dijkstra_dk(graph, source, target):
    dist = {source: 0}
    paths = {source: [source]}

    # heap tiebreaker - yields the shortest path with least edges
    c = itertools.count()

    heap = []

    for v in graph.nodes:
        if v != source:
            dist[v] = math.inf
        heappush(heap, (dist[v], next(c), v))

    while heap:
        _, _, u = heappop(heap)
        if u == target:
            break
        for v in nx.all_neighbors(graph=graph, node=u):
            e = graph.get_edge_data(u, v)
            alt = dist[u] + e['weight']
            if alt < dist[v]:
                dist[v] = alt
                paths[v] = paths[u] + [v]
                decrease_key(heap, dist[v], v)

    return dist, paths


def dijkstra(graph, source, target):
    dist = {}
    seen = {source: 0}
    paths = {source: [source]}

    # heap tiebreaker - yields the shortest path with least edges
    c = itertools.count()

    heap = []
    heappush(heap, (0, next(c), source))

    while heap:
        d, _, u = heappop(heap)
        if u in dist:
            continue
        dist[u] = d
        if u == target:
            break
        for v in nx.all_neighbors(graph=graph, node=u):
            e = graph.get_edge_data(u, v)
            alt = dist[u] + e['weight']
            if v not in seen or alt < seen[v]:
                seen[v] = alt
                heappush(heap, (alt, next(c), v))
                paths[v] = paths[u] + [v]

    return dist, paths


def main():
    try:
        file_path = os.path.join(os.getcwd(), sys.argv[1])
    except IndexError:
        print("Usage: python3 imgdist.py <image-filename>")
        return

    image, pixels = load_image(file_path)
    image_width, image_length = image.size

    graph = generate_graph(image, pixels)

    top_left = pixel(0, 0)
    bottom_right = pixel(image_width - 1, image_length - 1)

    source = top_left
    target = bottom_right
    distances, paths = dijkstra_dk(graph, source, target)
    distance, path = distances[target], paths[target]

    print("Shortest path distance:", distance)
    print("Shortest path length:", len(path))

    draw_shortest_path(pixels, path)
    save_image(image, file_path)
    image.show()


if __name__ == '__main__':
    main()
