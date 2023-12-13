import nltk
from nltk.corpus import twitter_samples

nltk.download("twitter_samples")

import os

# Use os.system to run the shell command and save the result to a file
os.system('find /root -name "twitter_samples" > logfind.txt')

# Read the result from the file
with open("logfind.txt", "r") as file:
    twc_folder = file.read().strip()

print(twc_folder)

# List entries in the directory
entries = os.listdir(twc_folder)
print(entries)

# Find the path of the desired file
twc_file = os.path.join(twc_folder, [file for file in entries if 'tweets.20150430-223406.json' in file][0])
print(twc_file)
