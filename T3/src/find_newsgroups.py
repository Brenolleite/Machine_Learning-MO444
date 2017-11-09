from os import walk
import re

# Script used to understand the newsgroups

# Create vector of newgroups
newsgroups = {}

# Find just masters grouping
master_only = False

for (dirpath, dirnames, filenames) in walk('../documents/docs/'):
    # Run over all files
    for filename in filenames:
        f = open(dirpath + filename, 'r')
        filetext = f.read()
        f.close()

         # Use regex to find newsgroups for each text file
        # Fixing issues where a newsgroups is shown on the message
        matches = re.findall('Newsgroups: [\\/\w.,\d-]+', filetext)

        # Go over all matches
        for match in matches:
            # Get all groups for that file
            groups = match.replace('Newsgroups: ', '').split(',')

            # Counting the occurrence of each group
            for item in groups:
                if master_only:
                    item = re.sub('\..+', '', item)

                # Fix issue in the files
                # 'b04aa255198b5e2526cff7c76c7c6257ad70e49f'
                # '1cfd267dfba20241fac4126124d73c27840c27fa'
                # Where there is a comma but not another group
                if item != '' and item != 'm.h.a':
                    if item in newsgroups:
                        newsgroups[item] += 1
                    else:
                        newsgroups[item] = 1

# Sort array
newgroups = sorted(newsgroups.items(), reverse=True, key=lambda x: x[1])

# Print groups and number of occurrences
for group in newgroups:
    print(group[0], group[1])
