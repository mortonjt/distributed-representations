#!/bin/env python
import fileinput
import sys

# Take a single argument as a path to a file that we should write a list
# of all nodes and their expected ranks.

outfile = sys.argv[1]
print(f"Writing to {outfile}")
with open(outfile, 'w+') as f:
    rank = 0
    for line in sys.stdin:
        nodes = line.strip().split()
        for node in nodes:
            f.write(f"{node} {rank}\n")
            rank += 1

