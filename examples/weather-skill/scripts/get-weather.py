import argparse

parser = argparse.ArgumentParser()
parser.add_argument("location", type=str)
args = parser.parse_args()

print(f"The current weather in {args.location} is 27 degrees celsius.")
print("The sky is clear and sunny.")
