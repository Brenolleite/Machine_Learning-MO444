from os import walk
import numpy as np

# Script used to find diference between number of files and ids

def intersetion():
    # Create array of files
    for (dirpath, dirnames, filenames) in walk('../documents/docs/'):
        ids = [line.rstrip('\n') for line in open('../documents/ids')]

        # Print files that intersect in ids and files
        print(len(set(ids) | set(filenames)))

# Find ids duplicated
def find_duplicates():
    # Run over file
    with open('../documents/ids') as f:
        # Create a set if seeb fukes
        seen = set()
        for line in f:
            # Transform line to lower case
            line_lower = line.lower()

            # Verify is it is duplicated
            if line.lower() in seen:
                print(line, end='')
            else:
                seen.add(line_lower)

find_duplicates()