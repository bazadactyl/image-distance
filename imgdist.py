from heapq import heappush, heappop
import itertools
import math
import os
import sys
from contexttimer import Timer
import matplotlib.pyplot as plt
import networkx as nx
import PIL


def load_image(file_path):
    image = PIL.Image.open(file_path)
    rgb_image = image.convert('RGB')
    pixels = rgb_image.load()
    return rgb_image, pixels


def save_image(image, original_file_path):
    file_basename = os.path.basename(original_file_path).split('.')[0]
    image.save('results/{}-shortest-path.png'.format(file_basename))


def colordist(pixels, p1, p2):
    g = 1  # constant
    col1, row1 = p1
    col2, row2 = p2
    r1, g1, b1 = pixels[col1, row1]
    r2, g2, b2 = pixels[col2, row2]
    l2_norm = math.sqrt((r2 - r1) ** 2 + (g2 - g1) ** 2 + (b2 - b1) ** 2)
    distance = math.sqrt(1 + g * l2_norm)
    return distance


def pixel_neighbours(pixel, image):
    col, row = pixel
    width, height = image.size

    up = (col, row - 1)
    down = (col, row + 1)
    left = (col - 1, row)
    right = (col + 1, row)

    neighbours = []
    if row - 1 >= 0:
        neighbours.append(up)
    if row + 1 < height:
        neighbours.append(down)
    if col - 1 >= 0:
        neighbours.append(left)
    if col + 1 < width:
        neighbours.append(right)

    return neighbours


def generate_graph(image, pixels):
    image_width, image_height = image.size
    graph = nx.Graph()

    # Add each pixel to the graph as a node
    for col in range(image_width):
        for row in range(image_height):
            p = (col, row)
            graph.add_node(p, color=pixels[col, row])

    # Add edges between adjacent pixels, weighted by their color distance
    for col in range(image_width):
        for row in range(image_height):
            curr = (col, row)
            up = (col, row - 1)
            down = (col, row + 1)
            left = (col - 1, row)
            right = (col + 1, row)
            if row - 1 >= 0:
                graph.add_edge(curr, up, weight=colordist(pixels, curr, up))
            if row + 1 < image_height:
                graph.add_edge(curr, down, weight=colordist(pixels, curr, down))
            if col - 1 >= 0:
                graph.add_edge(curr, left, weight=colordist(pixels, curr, left))
            if col + 1 < image_width:
                graph.add_edge(curr, right, weight=colordist(pixels, curr, right))

    return graph


def draw_shortest_path(pixels, path):
    green = (0, 255, 0)
    for p in path:
        col, row = p
        pixels[col, row] = green


def visualize_graph(graph):
    node_positions = {node: node for node in graph.nodes}
    nx.draw_networkx(graph, pos=node_positions, node_color='black', node_size=50, with_labels=False)
    plt.plot()
    plt.show()


def decrease_key(heap, new_key, identifier):
    c = 0
    for i in range(len(heap)):
        if heap[i][2] == identifier:
            _, c, _ = heap.pop(i)
            break
    heappush(heap, (new_key, c, identifier))


def standard_dijkstra(graph, source):
    """Dijkstra's algorithm implementation, textbook style."""
    dist = {source: 0}
    path = {source: [source]}

    # min-heap tiebreaker; causes output to be the shortest path with fewest edges
    c = itertools.count()

    heap = []

    for v in graph.nodes:
        if v != source:
            dist[v] = math.inf
        heappush(heap, (dist[v], next(c), v))

    while heap:
        _, _, u = heappop(heap)
        for v in nx.all_neighbors(graph=graph, node=u):
            e = graph.get_edge_data(u, v)
            alt = dist[u] + e['weight']
            if alt < dist[v]:
                dist[v] = alt
                path[v] = path[u] + [v]
                decrease_key(heap, dist[v], v)

    return dist, path


def heappush_dijkstra(graph, source):
    """Dijkstra's algorithm implementation that pushes new entries to the
    priority queue instead of performing decrease-key operations. Stale
    nodes are discarded when popped from the queue.
    """
    path = {source: [source]}
    graph.nodes[source]['dist'] = 0

    # min-heap tiebreaker; causes output to be the shortest path with fewest edges
    c = itertools.count()

    heap = []

    for node in graph.nodes:
        init_dist = 0 if node == source else math.inf
        graph.nodes[node]['dist'] = init_dist
        heappush(heap, (init_dist, next(c), node))

    while heap:
        u_dist, _, u = heappop(heap)
        if u_dist != graph.nodes[u]['dist']:
            continue
        for v in nx.all_neighbors(graph=graph, node=u):
            v_dist = graph.nodes[v]['dist']
            uv_edge = graph.get_edge_data(u, v)
            alt_dist = u_dist + uv_edge['weight']
            if alt_dist < v_dist:
                path[v] = path[u] + [v]
                graph.nodes[v]['dist'] = alt_dist
                heappush(heap, (alt_dist, next(c), v))

    dist = nx.get_node_attributes(graph, 'dist')
    return dist, path


def networkx_dijkstra(graph, source):
    """Dijkstra's algorithm implementation inspired by NetworkX source."""
    dist = {}
    path = {source: [source]}
    seen = {source: 0}

    # min-heap tiebreaker; causes output to be the shortest path with fewest edges
    c = itertools.count()

    heap = []
    heappush(heap, (0, next(c), source))

    while heap:
        d, _, u = heappop(heap)
        if u in dist:
            continue
        dist[u] = d
        for v in nx.all_neighbors(graph=graph, node=u):
            e = graph.get_edge_data(u, v)
            alt = dist[u] + e['weight']
            if v not in seen or alt < seen[v]:
                seen[v] = alt
                heappush(heap, (alt, next(c), v))
                path[v] = path[u] + [v]

    return dist, path


def test_correctness(graph, source, target):
    d1, p1 = standard_dijkstra(graph, source)
    d2, p2 = heappush_dijkstra(graph, source)
    d3, p3 = networkx_dijkstra(graph, source)

    print("standard_dijkstra:", d1[target], len(p1[target]))
    print("heappush_dijkstra:", d2[target], len(p2[target]))
    print("networkx_dijkstra:", d3[target], len(p3[target]))

    assert d1[target] == d2[target] == d3[target]


def test_runtime(functions, original_image, pixels):
    samples = 15
    step_size = 32
    results = {func: {'x': [], 'y': []} for func in functions}

    for func in results.keys():
        image = original_image.copy()
        for i in range(samples):
            width, height = image.size
            num_pixels = width * height
            new_size = (width - step_size, height - step_size)
            image.thumbnail(new_size, PIL.Image.ANTIALIAS)

            graph = generate_graph(image, pixels)
            source = (0, 0)
            target = (width - 1, height - 1)

            print("Running {} with image size {}x{} ({})".format(func.__name__, width, height, num_pixels))
            with Timer() as t:
                func(graph, source, target)
            print("\tTime:", t.elapsed)
            results[func]['x'].append(num_pixels)
            results[func]['y'].append(t.elapsed)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(results[standard_dijkstra]['x'], results[standard_dijkstra]['y'])
    ax.plot(results[heappush_dijkstra]['x'], results[heappush_dijkstra]['y'])
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlabel("Number of Pixels")
    plt.ylabel("Runtime (seconds)")
    ax.legend([
        'Dijkstra\'s (decrease-key)',
        'Dijkstra\'s (heap-push)',
    ], loc='upper left')
    plt.show()


def main():
    try:
        file_path = os.path.join(os.getcwd(), sys.argv[1])
    except IndexError:
        print("Usage: python3 imgdist.py <image-filename>")
        return

    image, pixels = load_image(file_path)
    width, height = image.size

    graph = generate_graph(image, pixels)

    top_left = (0, 0)
    bottom_right = (width - 1, height - 1)
    source = top_left
    target = bottom_right

    distance, path = heappush_dijkstra(graph, source)

    print("Shortest path distance:", distance[target])
    print("Shortest path length:", len(path[target]))

    draw_shortest_path(pixels, path[target])
    save_image(image, file_path)
    image.show()


if __name__ == '__main__':
    main()
