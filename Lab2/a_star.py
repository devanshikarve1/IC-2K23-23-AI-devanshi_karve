
import heapq


def a_star(graph, heuristic, start, goal):
    """Find a lowest-cost path using the A* search algorithm."""
    if start not in graph or goal not in graph:
        raise ValueError("Start and goal must exist in the graph.")

    # Priority queue entries: (f_score, g_score, node, path)
    queue = [(heuristic[start], 0, start, [start])]
    best_cost = {start: 0}

    while queue:
        _, cost, node, path = heapq.heappop(queue)

        if cost != best_cost.get(node):
            continue

        if node == goal:
            return path, cost

        for neighbor, edge_cost in graph[node]:
            if edge_cost < 0:
                raise ValueError("Edge costs cannot be negative.")

            new_cost = cost + edge_cost

            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                priority = new_cost + heuristic[neighbor]

                heapq.heappush(
                    queue,
                    (priority, new_cost, neighbor, path + [neighbor])
                )

    return None, None


if __name__ == "__main__":
    graph = {
        "A": [("B", 1), ("C", 4)],
        "B": [("D", 2), ("E", 5)],
        "C": [("F", 3)],
        "D": [("F", 4)],
        "E": [("F", 1)],
        "F": [],
    }

    heuristic = {
        "A": 5,
        "B": 4,
        "C": 3,
        "D": 2,
        "E": 1,
        "F": 0,
    }

    path, cost = a_star(graph, heuristic, "A", "F")

    if path:
        print("Optimal path:", " -> ".join(path))
        print("Total path cost:", cost)
    else:
        print("No path found.")
