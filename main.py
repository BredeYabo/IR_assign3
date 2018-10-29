import random
import codecs
import string
import gensim
from nltk.stem.porter import PorterStemmer
# random.seed(123)

# Part 1 of assignment
f = codecs.open("pg3300.txt", "r", "utf-8")
# Divide book into paragraphs
paragraphs = [[]]
paragraph_number = 0
past_line = ''
for line in f:
    if(line == '\r\n' and paragraph_number == 0):
        paragraph_number = paragraph_number + 1
        paragraphs.append([])
    elif(line == '\r\n' and past_line != '\r\n'):
        paragraph_number = paragraph_number + 1
        paragraphs.append([])
    else:
        paragraphs[paragraph_number].append(line)
    past_line = line

# Remove paragraphs containing 'Gutenberg'
paragraphs = [[word for word in (line for line in paragraph if line != '\r\n')] for paragraph in paragraphs]
# Combine lines into one paragraph
paragraphs = [[''.join(paragraph)] for paragraph in paragraphs]
# Split into words
paragraphs = [[word for word in (line.split(" ") for line in paragraph)] for paragraph in paragraphs]
# Remove punctuations, white characters and change to lowercase
paragraphs = [[word.strip().strip(string.punctuation).lower() for word in paragraph[0]] for paragraph in paragraphs]
# Remove whitecharacters (List is now unsorted)
for paragraph in paragraphs:
    for word in paragraph:
        newWord = word.split('\r\n')
        if len(newWord) > 1:
            paragraph.append(newWord[0])
            paragraph.append(newWord[1])
            paragraph.remove(word)

# Part 2 of assignment
stemmer = PorterStemmer()
dictionary = gensim.corpora.Dictionary()
stopword = ["a","able","about","across","after","all","almost","also","am","among","an","and","any","are","as","at","be","because","been","but","by","can","cannot","could","dear","did","do","does","either","else","ever","every","for","from","get","got","had","has","have","he","her","hers","him","his","how","however","i","if","in","into","is","it","its","just","least","let","like","likely","may","me","might","most","must","my","neither","no","nor","not","of","off","often","on","only","or","other","our","own","rather","said","say","says","she","should","since","so","some","than","that","the","their","them","then","there","these","they","this","tis","to","too","twas","us","wants","was","we","were","what","when","where","which","while","who","whom","why","will","with","would","yet","you","your"]

# Result
# Output so far
for i in range(8, 15):
    print(paragraphs[i])

f.close()
