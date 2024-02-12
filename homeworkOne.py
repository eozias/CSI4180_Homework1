import math
import re
import gensim
import nltk
import collections
import string

from gensim import corpora
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from datasets import load_dataset

news_dataset = load_dataset("ag_news", trust_remote_code=True)
label_names = ['World', 'Sports', 'Business', 'Science/Technology']


def preprocess_text(text):
    # Convert text to lower
    text = text.lower()

    # Remove Punctuation
    tokenizer = RegexpTokenizer(r'\w+\'?\w+|\w+')
    words = tokenizer.tokenize(text)
    words_without_punctuation = [''.join(c for c in word if c not in string.punctuation or c in ["'", "’"]) for
                                 word in words]
    text = ' '.join(words_without_punctuation)

    # Lemmatize text
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmaWords = [lemmatizer.lemmatize(token, pos="v") for token in tokens]
    text = ' '.join(lemmaWords)

    # Remove stop words
    stop_words_lower = set(word.lower() for word in stopwords.words('english'))
    stop_words_upper = set(word.title() for word in stopwords.words('english'))
    stop_words = stop_words_lower.union(stop_words_upper)
    tokens = word_tokenize(text)
    tokensNoSWs = [tok for tok in tokens if tok not in stop_words]
    processedText = ' '.join(tokensNoSWs)

    # Remove numbers
    processedText = re.sub(r'\d+', '', processedText)

    # Remove unimportant and common words, and singular letters from the text
    processedText = processedText.replace("reuters", "")
    processedText = processedText.replace("ap", "")
    processedText = processedText.replace("'s ", "")
    processedText = processedText.replace("washingtonpost", "")
    processedText = processedText.replace(" say ", "")
    processedText = processedText.replace(" new ", "")
    processedText = processedText.replace(" lt ", "")
    processedText = processedText.replace(" gt ", "")
    processedText = re.sub(r'\b\w\b', '', processedText)

    return processedText


def homeworkOne():
    texts = news_dataset['train']['text'][0:7000]
    labels = news_dataset['train']['label'][0:7000]
    processed_sentences = [];

    for text in texts:
        processed_text = preprocess_text(text)
        processed_sentences.append(processed_text)

    # Convert to Bag of Words Format
    vocab = set()
    bow_model = []
    for text in processed_sentences:
        word_counts = {}
        tokens = nltk.word_tokenize(text)
        vocab.update(tokens)
        for word in tokens:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        bow_model.append(word_counts)

    # Naive bayes beginning
    # Computes the count of each word in each label
    label_count = {0: collections.defaultdict(int), 1: collections.defaultdict(int), 2: collections.defaultdict(int),
                   3: collections.defaultdict(int)}
    not_label_count = {0: collections.defaultdict(int), 1: collections.defaultdict(int),
                       2: collections.defaultdict(int), 3: collections.defaultdict(int)}
    for i in range(len(bow_model)):
        word_counts = bow_model[i]
        label = labels[i]
        for word, count in word_counts.items():
            label_count[label][word] += count

    # Computes the count of each word in other labels than the current label
    for i in range(len(bow_model)):
        word_counts = bow_model[i]
        label = labels[i]
        for word, count in word_counts.items():
            if label == 1:
                count = label_count[0][word] + label_count[2][word] + label_count[3][word]
                not_label_count[label][word] += count
            elif label == 2:
                count = label_count[0][word] + label_count[1][word] + label_count[3][word]
                not_label_count[label][word] += count
            elif label == 3:
                count = label_count[0][word] + label_count[1][word] + label_count[2][word]
                not_label_count[label][word] += count
            else:
                count = label_count[1][word] + label_count[2][word] + label_count[3][word]
                not_label_count[label][word] += count

    # Computes the first term of the equation (log likelihood that a specific word is in a specific label)
    log_likelihoods = {}
    for label in label_count.keys():
        # each of these maps from word -> log-likelihood within the class 'label'
        log_likelihoods[label] = {}
        denominator = sum(label_count[label].values()) + len(vocab)
        for token in vocab:
            numerator = label_count[label][token] + 1
            likelihood = numerator / denominator
            log_likelihoods[label][token] = math.log(likelihood)

    # Computes the second term of the equation (log likelihood that a specific word is in any other label)
    log_likelihoodsSecondTerm = {}
    for label in label_count.keys():
        log_likelihoodsSecondTerm[label] = {}
        denominatorTwo = sum(not_label_count[label].values()) + len(vocab)
        for token in vocab:
            numeratorTwo = not_label_count[label][token] + 1
            likelihood = numeratorTwo / denominatorTwo
            log_likelihoodsSecondTerm[label][token] = math.log(likelihood)

    # Computes first term - second term
    log_likelihoodsFinal = {}
    for label in label_count.keys():
        log_likelihoodsFinal[label] = {}
        for token in vocab:
            log_likelihoodsFinal[label][token] = log_likelihoods[label][token] - log_likelihoodsSecondTerm[label][token]

    # Sort the log likelihoods in descending order for each label
    sorted_likelihoods = {}
    for label in label_count.keys():
        sorted_likelihoods[label] = sorted(log_likelihoodsFinal[label].items(), key=lambda x: x[1], reverse=True)

    # Print the top 10 words for each label using Naive Bayes
    for label in label_count.keys():
        print("Top 10 words for label " + str(label) + " (" + label_names[label] + "):")
        for i in range(10):
            print(sorted_likelihoods[label][i])
        print("\n")
    # End of Naive Bayes

    # Perform LDA analysis
    headlines = [x.split() for x in processed_sentences]
    id2word = corpora.Dictionary(headlines)
    corpus = [id2word.doc2bow(text) for text in headlines]
    lda_model = gensim.models.LdaMulticore(corpus=corpus, num_topics=20, id2word=id2word, passes=2, workers=2)
    topics_list = []
    for idx, topic in lda_model.print_topics(-1):
        topic_words = topic.split("+")  # Split the topic into individual words
        topic_words = [word.strip().split("*")[1] for word in topic_words]  # Extract the actual words
        topics_list.append(topic_words)  # Append the list of words to the topics_list
        print("Topic: {} \nWords: {}".format(idx, topic))
        print("\n")

    # Find the top 3 topics for each label (used code from chat.openai.com)
    # Calculate the dominant topic for each headline
    dominant_topics = []
    for bow in corpus:
        topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
        dominant_topic = max(topics, key=lambda x: x[1])[0]
        dominant_topics.append(dominant_topic)
    # Create a dictionary to store the topics for each label
    label_topics = {0: [], 1: [], 2: [], 3: []}
    # Populate the label_topics dictionary
    for i, topic in enumerate(dominant_topics):
        label = labels[i]
        label_topics[label].append(topic)
    # Count the occurrences of each topic for each label
    topic_counts = {label: collections.Counter(topics) for label, topics in label_topics.items()}
    # Get the top three topics for each label
    top_three_topics = {label: count.most_common(3) for label, count in topic_counts.items()}
    print("Top three topics for each label:")
    for label, topics in top_three_topics.items():
        label_name = label_names[label]
        topic_indices = [topic[0] for topic in topics]
        print(f"Label {label} ({label_name}): {', '.join(map(str, topic_indices))}")

    # Added in for the report
    # countlabel0 = 0;
    # countlabel1 = 0;
    # countlabel2 = 0;
    # countlabel3 = 0;
    # for i in range(len(labels)):
    #     if labels[i] == 0:
    #         countlabel0 += 1
    #     elif labels[i] == 1:
    #         countlabel1 += 1
    #     elif labels[i] == 2:
    #         countlabel2 += 1
    #     else:
    #         countlabel3 += 1
    # # Get the total word count for each of the labels
    # totalCountLabel0 = 0
    # totalCountLabel1 = 0
    # totalCountLabel2 = 0
    # totalCountLabel3 = 0
    # for label, word_counts in label_count.items():
    #     if label == 0:
    #         totalCountLabel0 = sum(word_counts.values())
    #     elif label == 1:
    #         totalCountLabel1 = sum(word_counts.values())
    #     elif label == 2:
    #         totalCountLabel2 = sum(word_counts.values())
    #     else:
    #         totalCountLabel3 = sum(word_counts.values())
    #
    # print("\n")
    # print("Total headlines in each category:")
    # print("Label 0 " + "(" + label_names[0] + ") count: " + str(countlabel0))
    # print("Label 1 " + "(" + label_names[1] + ") count: " + str(countlabel1))
    # print("Label 2 " + "(" + label_names[2] + ") count: " + str(countlabel2))
    # print("Label 3 " + "(" + label_names[3] + ") count: " + str(countlabel3))
    # print("\n")
    #
    # print("Total word count in each category:")
    # print("Label 0 " + "(" + label_names[0] + ") count: " + str(totalCountLabel0))
    # print("Label 1 " + "(" + label_names[1] + ") count: " + str(totalCountLabel1))
    # print("Label 2 " + "(" + label_names[2] + ") count: " + str(totalCountLabel2))
    # print("Label 3 " + "(" + label_names[3] + ") count: " + str(totalCountLabel3))
    # print("\n")
    #
    # print("Average word count in each headline by category:")
    # print("Label 0 " + "(" + label_names[0] + ") count: " + str(totalCountLabel0 / countlabel0))
    # print("Label 1 " + "(" + label_names[1] + ") count: " + str(totalCountLabel1 / countlabel1))
    # print("Label 2 " + "(" + label_names[2] + ") count: " + str(totalCountLabel2 / countlabel2))
    # print("Label 3 " + "(" + label_names[3] + ") count: " + str(totalCountLabel3 / countlabel3))
    # print("\n")
    # End of code added in for the report

if __name__ == "__main__":
    homeworkOne()
