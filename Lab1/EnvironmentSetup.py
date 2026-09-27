import sys

print("AI Environment Setup")
print("=" * 30)

print("Python version:", sys.version)

libraries = ["numpy", "pandas", "matplotlib", "sklearn"]

print("\nChecking commonly used AI/data libraries:")

for library in libraries:
    try:
        module = __import__(library)
        version = getattr(module, "__version__", "installed")
        print(f"{library}: {version}")
    except ImportError:
        print(f"{library}: NOT INSTALLED")

print("\nEnvironment setup check completed.")
