import csv, itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Assemble training data
title = []
tag = []
with open('../training_data.csv', 'r') as data: # title, value, tag,
    data_reader = csv.reader(data, delimiter=",")
    next(data_reader) #skip header
    for row in data_reader:
        title.append(row[0]) 
        tag.append(row[2]) 
n = len(title)

# Open file to be tagged, process
with open('../raw_data.csv', 'r') as untagged_csv:# title, value,
    untagged_reader = csv.reader(untagged_csv, delimiter=",")
    next(untagged_reader) # skip header
    tagged_rows = [] # will add tags below
    for row in untagged_reader:
        title.append(row[0]) # add untagged titles to existing list of titles
        tagged_rows.append([row[0], row[1]])

# Tokenize titles
vectorizer = CountVectorizer()
T = vectorizer.fit_transform(title)

# Fit model
clf = LogisticRegression(C=1e5)
clf.fit(T[:n], tag) # fit model based on training data, n pairs of title, tag
tag_bar = clf.predict(T[n:]) # suggest tags for untagged titles

# Write tagged entries to new file
with open("tagged_data.csv", 'w') as tagged_csv:
    tagged_writer = csv.writer(tagged_csv)
    for i in range(len(tagged_rows)):
        tagged_rows[i].append(tag_bar[i])
        tagged_writer.writerow(tagged_rows[i])
