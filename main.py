import codecs
import random;
import string

import gensim
from nltk.stem.porter import PorterStemmer

random.seed(123)

stemmer = PorterStemmer()

stopwords = ["a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and", "any", "are",
             "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could", "dear", "did", "do", "does",
             "either", "else", "ever", "every", "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers",
             "him", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", "like",
             "likely", "may", "me", "might", "most", "must", "my", "neither", "no", "nor", "not", "of", "off", "often",
             "on", "only", "or", "other", "our", "own", "rather", "said", "say", "says", "she", "should", "since", "so",
             "some", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "tis", "to",
             "too", "twas", "us", "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who",
             "whom",
             "why", "will", "with", "would", "yet", "you", "your"]


def print_five_lines(paragraph):
    i = 0
    for line in paragraph.split('\r\n'):
        if i < 5:
            print(line)
            i = i + 1
        else:
            break
    print('\n')


def preprocess(text):
    # Remove punctuations, white characters and change to lowercase
    text = text.strip().strip(string.punctuation).lower()
    text = [stemmer.stem(word) for word in text.split(" ")]
    # Remove whitecharacters (List is now unsorted)
    # text = [word.append(word.split('\r\n')[0]).append(word.split('\r\n')[1]).remove(word) if len(word.split('\r\n')>1) else word for word in text]
    for word in text:
        newWord = word.split('\r\n')
        if len(newWord) > 1:
            text.append(newWord[0])
            text.append(newWord[1])
            text.remove(word)
    text = [word for word in text if (word not in stopwords)]
    return text


def preprocess_book(paragraphs):
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
    paragraphs = [[stemmer.stem(word) for word in paragraph if (word not in stopwords)] for paragraph in paragraphs]
    return paragraphs


# Read file
f = codecs.open("pg3300.txt", "r", "utf-8")
paragraphs = [[]]
paragraph_number = 0
past_line = ''
# Split into paragraphs
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

# split words combined with '\r\n'
paragraphs = [[word for word in (line for line in paragraph if line != '\r\n')] for paragraph in paragraphs]

# Combine lines into one paragraph
paragraphs = [[''.join(paragraph)] for paragraph in paragraphs]

# Remove paragraphs containing 'Gutenberg'
paragraphs = [[word for word in paragraph] for paragraph in paragraphs if 'Gutenberg' not in paragraph[0]]

# Keep original paragraph
original_paragraphs = paragraphs[:]
paragraphs = preprocess_book(paragraphs)

print(len(paragraphs))

dictionary = gensim.corpora.Dictionary(paragraphs)
# Remove Stopwords from dictionary (Just in case)
stop_ids = [dictionary.token2id[stopword] for stopword in stopwords if stopword in dictionary.token2id]
dictionary.filter_tokens(stop_ids)
dictionary.compactify()

# Paragraphs to bag of words (Term frequency)
bow = [dictionary.doc2bow(paragraph) for paragraph in paragraphs]

# Term frequency-inverse Document Frequency (How important a term is in a document)
tfidf_model = gensim.models.TfidfModel(bow)
tfidf_corpus = tfidf_model[bow]
tfidf_index = gensim.similarities.MatrixSimilarity(tfidf_corpus)

# LSI model
lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
corpus_lsi = lsi_model[bow]
sim_matrix = gensim.similarities.MatrixSimilarity(corpus_lsi)

# Query
query = "What is the function of money?"
query = preprocess(query)
print(query)
query = dictionary.doc2bow(query)
query_tfidf = tfidf_model[query]
print(query_tfidf)
doc2similarity_tfidf = enumerate(tfidf_index[query_tfidf])

print("Query results (tdidf): ")
results_tfidf = sorted(doc2similarity_tfidf, key=lambda kv: - kv[1])[:3]
print('\n\n TF-IDF Query results: \n\n')
print(results_tfidf)

print_five_lines(original_paragraphs[results_tfidf[0][0]][0])
print_five_lines(original_paragraphs[results_tfidf[1][0]][0])
print_five_lines(original_paragraphs[results_tfidf[2][0]][0])

query_lsi = lsi_model[query_tfidf]
lsi_query_topics = sorted(query_lsi, key=lambda kv: - abs(kv[1]))[:3]
print("\n\n LSI topic results: \n\n")
print(lsi_query_topics)
print(lsi_model.show_topic(lsi_query_topics[0][0]))
print(lsi_model.show_topic(lsi_query_topics[1][0]))
print(lsi_model.show_topic(lsi_query_topics[2][0]))
lsi_index = gensim.similarities.MatrixSimilarity(corpus_lsi)
doc2similarity_lsi = enumerate(lsi_index[query_lsi])

results_lsi = sorted(doc2similarity_lsi, key=lambda kv: - kv[1])[:3]

print('\n\n LSI Query results: \n\n')

print(results_lsi)
print_five_lines(original_paragraphs[results_lsi[0][0]][0])
print_five_lines(original_paragraphs[results_lsi[1][0]][0])
print_five_lines(original_paragraphs[results_lsi[2][0]][0])

print("\nThese paragraphs are equal for both LSI and TF-IDF:\n")
for i in range(3):
    for k in range(3):
        if results_lsi[i][0] == results_tfidf[k][0]:
            print("Paragraph: ", results_lsi[i][0])
