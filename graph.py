# Author: Karthik Reddy Pagilla

import sys
import math
import copy
import heapq
import numpy as np
import networkx as nx
from collections import deque

class Node:
    def __init__(self, vertex, dist=99999, prev=None):
        self.vertex = vertex
        self.dist = dist
        self.prev = prev

    def __lt__(self, other):
        return self.dist < other.dist
    
    def __eq__(self, other):
        return ((self.vertex == other.vertex) and (self.prev == other.prev))

def unvisited_neighbours(graph_node, graph):
    for i in range(graph.number_of_nodes()):
        if graph.has_edge(graph_node, i) and graph.nodes[i]['color'] == 'white':
            return i
    return -1

def get_all_neighbours(graph_node, graph):
    result = []
    for i in range(graph.number_of_nodes()):
        if graph.has_edge(graph_node, i) and graph.nodes[i]['color'] == 'white':
            result.append((i, graph[graph_node][i]['weight']))

    result.sort(key=lambda x:x[1])

    final = list(map(lambda x: x[0], result))
    return final

def get_neigbours(graph_node, graph):
    result = []
    for i in range(graph.number_of_nodes()):
        if graph.has_edge(graph_node, i):
            result.append(i)

    return result

def sort_edges(graph):
    result = []
    for g in graph.edges:
        u, v = g
        w = graph[u][v]['weight']
        result.append((u,v,w))

    result.sort(key=lambda x:x[2])

    return result

def DFS(graph, result):
    counter = 1
    stack = deque()
    for i in range(graph.number_of_nodes()):
        if graph.nodes[i]['color'] == 'white':
            graph.nodes[i]['color'] = 'gray'
            graph.nodes[i]['time'] = counter
            stack.append(i)
            result.append(i)
            break

    while len(stack) > 0:
        counter += 1
        x = stack[len(stack) - 1]
        y = unvisited_neighbours(x, graph)

        if y == -1:
            graph.nodes[x]['color'] = 'black'
            graph.nodes[x]['time'] = counter
            stack.pop()
        else:
            result.append(y)
            graph.nodes[y]['color'] = 'gray'
            graph.nodes[y]['time'] = counter
            stack.append(y)

    return result

def BFS(graph, result):
    counter = 1
    queue = deque()
    for i in range(graph.number_of_nodes()):
        if graph.nodes[i]['color'] == 'white':
            graph.nodes[i]['color'] = 'gray'
            graph.nodes[i]['time'] = counter
            queue.append(i)
            result.append(i)
            break

    while len(queue) > 0:
        x = queue.popleft()
        for y in get_all_neighbours(x, graph):
            result.append(y)
            counter += 1
            graph.nodes[y]['color'] = 'gray'
            graph.nodes[y]['time'] = counter
            queue.append(y)

        graph.nodes[x]['color'] = 'black'
        graph.nodes[x]['time'] = counter   

    return result

def graph_initializer(list_of_values, n):
    graph = nx.Graph()

    u = list(map(lambda x: x[0], list_of_values))
    v = list(map(lambda x: x[1], list_of_values))
    w = list(map(lambda x: x[2], list_of_values))

    graph.add_nodes_from(range(0, n), color='white', time=0)

    for i in range(len(u)):
        graph.add_edge(u[i], v[i], weight=w[i])

    return graph

def visited_node_checker(graph_node, graph, node_except):
    for i in range(graph.number_of_nodes()):
        if i != node_except and graph.has_edge(graph_node, i) and graph.nodes[i]['color'] != 'white':
            return i
    return -1

def cycle_checker(graph):
    counter = 1
    stack = deque()
    for i in range(graph.number_of_nodes()):
        if graph.nodes[i]['color'] == 'white':
            graph.nodes[i]['color'] = 'gray'
            graph.nodes[i]['time'] = counter
            stack.append(i)
            break

    while len(stack) > 0:
        counter += 1
        x = stack[len(stack) - 1]
        y = unvisited_neighbours(x, graph)

        if y == -1:
            graph.nodes[x]['color'] = 'black'
            graph.nodes[x]['time'] = counter
            stack.pop()
        else:
            if len(stack) > 1:
                if visited_node_checker(y, graph, x) != -1:
                    return True
            graph.nodes[y]['color'] = 'gray'
            graph.nodes[y]['time'] = counter
            stack.append(y)
    return False

def MST(graph):
    edge_set = []
    k = 0
    E = sort_edges(graph)

    while len(edge_set) < graph.number_of_nodes() - 1:
        copy_edges = copy.deepcopy(edge_set)
        copy_edges.append(E[k])
        if not cycle_checker(graph_initializer(copy_edges, graph.number_of_nodes())):
            edge_set.append(E[k])
        k += 1

    return edge_set

def shortest_path(graph, start_node, resultant):
    min_heap = []
    heapq.heapify(min_heap)

    for i in graph.nodes:
        node = Node(i)
        heapq.heappush(min_heap, node)
    
    heapq.heapreplace(min_heap, start_node)

    while len(min_heap) > 1:
        current_node = heapq.heappop(min_heap)
        vertices_in_heap = list(map(lambda x: x.vertex, list(min_heap)))

        for k in get_neigbours(current_node.vertex, graph):
            if k in vertices_in_heap:
                min_heap = list(min_heap)
                neighbour_node = None
                vertex = 0
                for j in range(0, len(min_heap)):
                    if min_heap[j].vertex == k:
                        neighbour_node = min_heap[j]
                        vertex = j
                        break
                if neighbour_node is not None and graph[current_node.vertex][k]['weight'] + current_node.dist < neighbour_node.dist:
                    neighbour_node.dist = graph[current_node.vertex][k]['weight'] + current_node.dist
                    neighbour_node.prev = current_node.vertex
                    min_heap[vertex] = neighbour_node
                    resultant.append(neighbour_node)
                heapq.heapify(min_heap)

    return resultant

def read_content():
    input_file = open('/content/input001.txt', 'r')
    n = int(input_file.readline().strip())
    g = nx.Graph()

    g.add_nodes_from(range(0, n), color='white', time=0)

    line = input_file.readline().split(" ")

    while len(line) != 0:
        if line[0] != '' and line[1] != '':
            g.add_edge(int(line[0]), int(line[1]), weight=float(line[2]))
            line = input_file.readline().split(" ")
        else:
            line = input_file.readline().split(" ")
            break
            
    return g

def unvisited_exists(graph):
    for i in range(graph.number_of_nodes()):
        if graph.nodes[i]['color'] == 'white':
            return True
    return False

print("Graph:")

for e in read_content().edges:
    u, v = e
    print("(" + str(u) + ", " + str(v) + ", " + str(read_content()[u][v]['weight']) + ")")

print()

print("Depth First Traversal (vertex visited order):")

g1 = read_content()
r1 = []
while unvisited_exists(g1):
    r1 = DFS(g1, r1)

print(r1)

print()

print("Breadth First Traversal (lowest-weight-next):")
g2 = read_content()
gtemp = read_content()
r2 = []
rtemp = []
while unvisited_exists(gtemp):
    r2 = BFS(g2, r2)
    rtemp = DFS(gtemp, rtemp)

print(r2)

print()

print("Minimum Spanning Tree:")
total_weight = 0

g3 = read_content()
r3 = []
rfinal = []
r3 = DFS(g3, r3)
text = "Full Spanning Tree"
l = [[-1]]
count = 0
while unvisited_exists(g3):
    l1 = []
    for i in range(g3.number_of_nodes()):
        if g3.nodes[i]['color'] != 'white' and i not in sum(l, []):
            l1.append(i)

        if len(l1) != 0 and l1 not in l:
            l.append(l1)
    count += 1
    r3 = DFS(g3, r3)

l1 = []
for i in range(g3.number_of_nodes()):
    if g3.nodes[i]['color'] != 'white' and i not in sum(l, []):
        l1.append(i)

    if len(l1) != 0 and l1 not in l:
        l.append(l1)

for i in range(1, len(l)):
    rfinal.append(MST(g3.subgraph(l[i])))

if (len(l) > 2):
    text = "Spanning Forest"

for i in sum(rfinal, []):
    print(str(i))
    u, v, w = i
    total_weight += w

print("")
print("Type: " + text)
print("Total Weight: " + str(total_weight))

resultant = []
g4 = read_content()
for i in range(g4.number_of_nodes()):
    resultant = shortest_path(g4, Node(i, 0), resultant)

mid_list = sorted(resultant, key=lambda x: x.dist)
new_list = sorted(resultant, key=lambda x: (x.prev, x.vertex))

print("Shortest Paths:")
for i in new_list:
    print(str(i.prev) + " -> " + str(i.vertex) + " = " + str(i.dist))