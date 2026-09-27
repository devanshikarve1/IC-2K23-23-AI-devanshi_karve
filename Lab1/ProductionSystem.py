# Simple Production Rule System

def production_system(facts):
    rules = [
        ({"fever", "cough"}, "Possible flu symptoms"),
        ({"fever", "rash"}, "Possible infection symptoms"),
        ({"headache", "fever"}, "Patient has common fever symptoms")
    ]

    conclusions = []

    for conditions, conclusion in rules:
        if conditions.issubset(facts):
            conclusions.append(conclusion)

    return conclusions


# Given facts
facts = {"fever", "cough"}

print("Facts:", facts)

print("\nConclusions:")

results = production_system(facts)

if results:
    for result in results:
        print("-", result)
else:
    print("No rule matched.")
