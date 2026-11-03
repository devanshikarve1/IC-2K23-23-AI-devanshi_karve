
import heapq


def best_first_search(graph, heuristic, start, goal):
    """Search for a goal by expanding the lowest-heuristic node."""
    if start not in graph or goal not in graph:
        raise ValueError("Start and goal must exist in the graph.")

    priority_queue = [(heuristic[start], start, [start])]
    visited = set()

    while priority_queue:
        _, node, path = heapq.heappop(priority_queue)

        if node in visited:
            continue

        visited.add(node)

        if node == goal:
            return path

        for neighbor in graph[node]:
            if neighbor not in visited:
                heapq.heappush(
                    priority_queue,
                    (heuristic[neighbor], neighbor, path + [neighbor])
                )

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

    heuristic = {
        "A": 6,
        "B": 4,
        "C": 2,
        "D": 5,
        "E": 2,
        "F": 0,
    }

    result = best_first_search(graph, heuristic, "A", "F")

    if result:
        print("Path found:", " -> ".join(result))
        print("Path length:", len(result) - 1)
    else:
        print("No path found.")
