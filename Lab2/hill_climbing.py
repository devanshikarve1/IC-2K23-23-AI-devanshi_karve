
def hill_climbing(graph, heuristic, start, goal):
    """Find a goal by repeatedly choosing the best neighbor."""
    current = start
    path = [current]
    visited = {current}

    while current != goal:
        neighbors = graph.get(current, [])

        # Choose the neighbor with the lowest heuristic value.
        candidates = [
            node for node in neighbors
            if node not in visited
        ]

        if not candidates:
            print("Search stopped: local optimum or dead end.")
            return path, False

        best_neighbor = min(
            candidates,
            key=lambda node: heuristic[node]
        )

        # Move only if the heuristic improves.
        if heuristic[best_neighbor] >= heuristic[current]:
            print("Search stopped: no better neighbor.")
            return path, False

        current = best_neighbor
        path.append(current)
        visited.add(current)

    return path, True


if __name__ == "__main__":
    graph = {
        "A": ["B", "C"],
        "B": ["D", "E"],
        "C": ["F"],
        "D": [],
        "E": ["F"],
        "F": [],
    }

    # Lower heuristic values indicate closeness to the goal.
    heuristic = {
        "A": 6,
        "B": 4,
        "C": 2,
        "D": 5,
        "E": 2,
        "F": 0,
    }

    path, success = hill_climbing(graph, heuristic, "A", "F")

    print("Path:", " -> ".join(path))
    print("Goal reached:", success)
