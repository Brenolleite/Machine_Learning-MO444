from os import walk
import re

# Script used to understand the newsgroups

# Create vector of newgroups
newsgroups = {}

for (dirpath, dirnames, filenames) in walk('../documents/docs/'):
    # Run over all files
    for filename in filenames:
        f = open(dirpath + filename, 'r')
        filetext = f.read()
        f.close()

        # Use regex to find newsgroups for each text file
        matches = re.findall('Newsgroups:\s*\S+', filetext)

        # Go over all matches
        for match in matches:
            # Get all groups for that file
            groups = match.replace('Newsgroups: ', '').split(',')

            # Counting the occurrence of each group
            for item in groups:
                if item in newsgroups:
                    newsgroups[item] += 1
                else:
                    newsgroups[item] = 1

# Sort array
newgroups = sorted(newsgroups.items(), reverse=True, key=lambda x: x[1])

# Print groups and number of occurrences
for group in newgroups:
    print(group[0], group[1])
