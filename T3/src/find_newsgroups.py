from os import walk
import re

# Create vector of newgroups
newsgroups = {}

counter = 0

for (dirpath, dirnames, filenames) in walk('../documents/docs/'):
    for filename in filenames:
        f = open(dirpath + filename, 'r')
        filetext = f.read()
        f.close()

        matches = re.findall('Newsgroups:\s*\S+', filetext)

        for match in matches:
            groups = match.replace('Newsgroups: ', '').split(',')
            for item in groups:
                if item in newsgroups:
                    newsgroups[item] += 1
                else:
                    newsgroups[item] = 1


newgroups = sorted(newsgroups.items(), reverse=True, key=lambda x: x[1])

for group in newgroups:
    print(group[0], group[1])
