from heapq import heappush, heappop
from itertools import count
from math import sqrt
from sys import argv
import networkx as nx
from PIL import Image


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
    distance = sqrt((r2-r1)**2 + (g2-g1)**2 + (b2-b1)**2)
    return distance


def dijkstra(graph, source, target, cutoff=None):
    dist = {}
    seen = {}
    paths = {source: [source]}
    c = count()
    pq = []

    seen[source] = 0
    heappush(pq, (0, next(c), source))

    while pq:
        (d, _, v) = heappop(pq)
        if v in dist:
            continue
        dist[v] = d
        if v == target:
            break
        for u, e in graph._adj[v].items():
            cost = e['weight']
            if cost is None:
                continue
            vu_dist = dist[v] + cost
            if cutoff is not None:
                if vu_dist > cutoff:
                    continue
            if u in dist:
                if vu_dist < dist[u]:
                    raise ValueError('Contradictory paths found: negative weights?')
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                heappush(pq, (vu_dist, next(c), u))
                if paths is not None:
                    paths[u] = paths[v] + [u]

    return dist, paths[target]


def main():
    try:
        img_filename = argv[1]
    except IndexError:
        print("Usage: python imgdist.py <image-filename>")
        exit(1)

    img = Image.open(img_filename)
    img_width = img.size[0]
    img_length = img.size[1]
    pixels = img.load()

    g = nx.Graph()

    for i in range(img_width):
        for j in range(img_length):
            g.add_node(pixel(i, j))
            # print(pixels[i, j])

    for i in range(img_width):
        for j in range(img_length):
            curr = pixel(i, j)
            up = pixel(i, j-1)
            down = pixel(i, j+1)
            left = pixel(i-1, j)
            right = pixel(i+1, j)
            if j-1 >= 0:
                g.add_edge(curr, up, weight=colordist(pixels, curr, up))
            if j+1 < img_length:
                g.add_edge(curr, down, weight=colordist(pixels, curr, down))
            if i-1 >= 0:
                g.add_edge(curr, left, weight=colordist(pixels, curr, left))
            if i+1 < img_width:
                g.add_edge(curr, right, weight=colordist(pixels, curr, right))

    top_left = pixel(0, 0)
    bottom_right = pixel(img_width-1, img_length-1)
    # top_left = pixel(3, 3)
    # bottom_right = pixel(img_width-3, img_length-3)

    source = top_left
    target = bottom_right
    distance, path = dijkstra(g, source, target)

    print("Shortest path distance:", distance)
    print("Shortest path length:", len(path))

    for p in path:
        c, r = coords(p)
        pixels[c, r] = (0, 255, 0)

    img.show()
    img.save('{}-shortest-path.png'.format(img_filename.split('.')[0]))

if __name__ == '__main__':
    main()
