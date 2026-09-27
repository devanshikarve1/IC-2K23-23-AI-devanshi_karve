# Lab 4 - Knowledge Representation, Rule-Based System
# and Semantic Network

# -------------------------------
# 1. Knowledge Representation
# -------------------------------

knowledge_base = {
    "Dog": {
        "is_a": "Animal",
        "has": "Four legs",
        "can": "Bark"
    },
    "Cat": {
        "is_a": "Animal",
        "has": "Four legs",
        "can": "Meow"
    },
    "Bird": {
        "is_a": "Animal",
        "has": "Two legs",
        "can": "Fly"
    }
}

print("=== Knowledge Representation ===")

for entity, properties in knowledge_base.items():
    print("\nEntity:", entity)

    for property_name, value in properties.items():
        print(property_name, ":", value)


# -------------------------------
# 2. Rule-Based System
# -------------------------------

print("\n=== Rule-Based System ===")

temperature = 38

if temperature >= 38:
    print("Rule: If temperature >= 38, then Fever")
    print("Result: Fever detected.")
else:
    print("Result: Normal temperature.")


# Another rule
age = 20

if age >= 18:
    print("Rule: If age >= 18, then Adult")
    print("Result: Person is an Adult.")
else:
    print("Result: Person is a Minor.")


# -------------------------------
# 3. Semantic Network
# -------------------------------

print("\n=== Semantic Network ===")

semantic_network = {
    "Dog": {
        "is_a": "Animal",
        "has": "Tail",
        "can": "Bark"
    },
    "Animal": {
        "has": "Life",
        "needs": "Food"
    }
}

for node, relations in semantic_network.items():
    print("\nNode:", node)

    for relation, value in relations.items():
        print("  ", relation, "-->", value)

print("\nLab 4 completed successfully.")
