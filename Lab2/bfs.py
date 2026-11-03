
from collections import deque


def bfs(graph, start, goal):
    """Find the shortest path in an unweighted graph using BFS."""
    if start not in graph or goal not in graph:
        raise ValueError("Start and goal must exist in the graph.")

    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        node, path = queue.popleft()

        if node == goal:
            return path

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None


if __name__ == "__main__":
    graph = {
        "A": ["B", "C"],
        "B": ["D", "E"],
        "C": ["F"],
        "D": [],
        "E": ["F"],
        "F": [],
    }

    start_node = "A"
    goal_node = "F"

    result = bfs(graph, start_node, goal_node)

    if result:
        print("Shortest path:", " -> ".join(result))
        print("Path length:", len(result) - 1)
    else:
        print("No path found.")
