"""Two isolated consumers share the integrated family and shared infrastructure."""
from consumer_a import run as run_a
from consumer_b import run as run_b

def main() raises:
    run_a()
    run_b()
