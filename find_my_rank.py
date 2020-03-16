#!/bin/env python
import socket
import sys

# Take the first (and only) argument as a path to the rank file
# and find my rank given hostname.

host = socket.gethostname()
path_to_rank_file = sys.argv[1]

rank = None
with open(path_to_rank_file, 'r') as f:
    for line in f.readlines():
        if line.startswith(host):
            rank = line.split()[1]

print(rank)
