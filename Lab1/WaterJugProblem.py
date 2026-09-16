from collections import deque


def water_jug_problem(jug1_capacity, jug2_capacity, target):
    visited = set()
    queue = deque()

    initial_state = (0, 0)
    queue.append((initial_state, []))
    visited.add(initial_state)

    while queue:
        (jug1, jug2), path = queue.popleft()

        if jug1 == target or jug2 == target:
            return path + [(jug1, jug2)]

        states = [
            (jug1_capacity, jug2),
            (jug1, jug2_capacity),
            (0, jug2),
            (jug1, 0),
            (
                jug1 - min(jug1, jug2_capacity - jug2),
                jug2 + min(jug1, jug2_capacity - jug2)
            ),
            (
                jug1 + min(jug2, jug1_capacity - jug1),
                jug2 - min(jug2, jug1_capacity - jug1)
            )
        ]

        for state in states:
            if state not in visited:
                visited.add(state)
                queue.append((state, path + [(jug1, jug2)]))

    return None


# 4-liter and 3-liter jugs
jug1_capacity = 4
jug2_capacity = 3
target = 2

solution = water_jug_problem(jug1_capacity, jug2_capacity, target)

if solution:
    print("Solution:")
    for step, state in enumerate(solution):
        print(f"Step {step}: Jug 1 = {state[0]}L, Jug 2 = {state[1]}L")
else:
    print("No solution exists.")
