from collections import deque
import heapq

# Graph
GRAPH = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F"],
    "D": [],
    "E": ["G"],
    "F": ["G"],
    "G": []
}

# Heuristic values
HEURISTIC = {
    "A": 6,
    "B": 4,
    "C": 4,
    "D": 2,
    "E": 1,
    "F": 2,
    "G": 0
}

# Edge costs for A*
EDGE_COST = {
    ("A", "B"): 2,
    ("A", "C"): 3,
    ("B", "D"): 4,
    ("B", "E"): 2,
    ("C", "F"): 3,
    ("E", "G"): 2,
    ("F", "G"): 2
}


# 1. Breadth First Search (BFS)
def bfs(start, goal):
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        node, path = queue.popleft()

        if node == goal:
            return path

        for neighbor in GRAPH[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None


# 2. Depth First Search (DFS)
def dfs(start, goal):
    stack = [(start, [start])]
    visited = set()

    while stack:
        node, path = stack.pop()

        if node in visited:
            continue

        visited.add(node)

        if node == goal:
            return path

        for neighbor in reversed(GRAPH[node]):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))

    return None


# 3. Hill Climbing
def hill_climbing(start, goal):
    current = start
    path = [current]

    while current != goal:
        neighbors = GRAPH[current]

        if not neighbors:
            return None

        next_node = min(neighbors, key=lambda x: HEURISTIC[x])

        if HEURISTIC[next_node] >= HEURISTIC[current]:
            return None

        current = next_node
        path.append(current)

    return path


# 4. Best First Search
def best_first_search(start, goal):
    priority_queue = [(HEURISTIC[start], start, [start])]
    visited = set()

    while priority_queue:
        _, node, path = heapq.heappop(priority_queue)

        if node in visited:
            continue

        visited.add(node)

        if node == goal:
            return path

        for neighbor in GRAPH[node]:
            if neighbor not in visited:
                heapq.heappush(
                    priority_queue,
                    (HEURISTIC[neighbor], neighbor, path + [neighbor])
                )

    return None


# 5. A* Search
def a_star(start, goal):
    priority_queue = [(HEURISTIC[start], 0, start, [start])]
    best_cost = {start: 0}

    while priority_queue:
        _, g, node, path = heapq.heappop(priority_queue)

        if node == goal:
            return path, g

        for neighbor in GRAPH[node]:
            new_g = g + EDGE_COST[(node, neighbor)]

            if neighbor not in best_cost or new_g < best_cost[neighbor]:
                best_cost[neighbor] = new_g
                f = new_g + HEURISTIC[neighbor]

                heapq.heappush(
                    priority_queue,
                    (f, new_g, neighbor, path + [neighbor])
                )

    return None


# Main program
START = "A"
GOAL = "G"

print("BFS:", bfs(START, GOAL))
print("DFS:", dfs(START, GOAL))
print("Hill Climbing:", hill_climbing(START, GOAL))
print("Best First Search:", best_first_search(START, GOAL))
print("A*:", a_star(START, GOAL))
