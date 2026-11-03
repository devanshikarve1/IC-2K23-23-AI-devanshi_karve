
def dfs(graph, start, goal):
    """Find a path using Depth-First Search."""
    if start not in graph or goal not in graph:
        raise ValueError("Start and goal must exist in the graph.")

    stack = [(start, [start])]
    visited = {start}

    while stack:
        node, path = stack.pop()

        if node == goal:
            return path

        # Reverse to preserve the graph's listed neighbor order.
        for neighbor in reversed(graph[node]):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))

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

    result = dfs(graph, "A", "F")

    if result:
        print("Path found:", " -> ".join(result))
        print("Path length:", len(result) - 1)
    else:
        print("No path found.")
