import warnings
from youtube_transcript_api import YouTubeTranscriptApi as yta
from pytube import extract
from deepmultilingualpunctuation import PunctuationModel
import nltk
import re
import math
import operator
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
import spacy

warnings.filterwarnings("ignore", category=UserWarning, message="`grouped_entities` is deprecated and will be removed in version v5.0.0")

Stopwords = set(stopwords.words('english'))
wordlemmatizer = WordNetLemmatizer()
def lemmatize_words(words):
    lemmatized_words = []
    for word in words:
       lemmatized_words.append(wordlemmatizer.lemmatize(word))
    return lemmatized_words

def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\s]'
    text = re.sub(regex,'',text)
    return text

def pos_tagging(text):
    pos_tag = nltk.pos_tag(text.split())
    nouns_and_verbs = []
    for word,tag in pos_tag:
        if tag == "NN" or tag == "NNP" or tag == "NNS" or tag == "VB" or tag == "VBD" or tag == "VBG" or tag == "VBN" or tag == "VBP" or tag == "VBZ":
             nouns_and_verbs.append(word)
    return nouns_and_verbs #Returns list of words that contain only nouns and verbs

def tf_score(word,sentence): #No of times a word is repeated in a given sentence
    word_frequency_in_sentence = 0 #Count Variable
    len_sentence = len(sentence)
    words = sentence.split()
    for w in words:
        if w == word:
            word_frequency_in_sentence = word_frequency_in_sentence + 1

    tf =  word_frequency_in_sentence/ len_sentence #TF Formula
    return tf

def idf_score(word,sentences,no_of_sentences):
    no_of_sentence_containing_word = 0 # Count Variable
    for sentence in sentences:
        sentence = remove_special_characters(str(sentence))
        sentence = re.sub(r'\d+', '', sentence)

        words = sentence.split()
        words = [word.lower() for word in words]
        words = [word for word in words if word not in Stopwords] #Filtering words which are not stop words
        words = [wordlemmatizer.lemmatize(word) for word in words] #Converting each word into it's lemma form

        if word in words:
            no_of_sentence_containing_word = no_of_sentence_containing_word + 1

    idf = math.log10(no_of_sentences/no_of_sentence_containing_word) #IDF Formula

    return idf


def word_tfidf_score(word,sentence,sentences): #Final combined(TF-IDF) score for each word
    tf = tf_score(word,sentence)
    idf = idf_score(word,sentences,len(sentences))
    return tf*idf

def sentence_importance(sentence,sentences): #For ranking sentences in the input text
     sentence_score = 0
     sentence = remove_special_characters(str(sentence))
     sentence = re.sub(r'\d+', '', sentence)

     no_of_sentences = len(sentences)

     nouns_and_verbs = []
     nouns_and_verbs = pos_tagging(sentence)

     for word in nouns_and_verbs:
          if word.lower() not in Stopwords:
                word = word.lower()
                word = wordlemmatizer.lemmatize(word)
                sentence_score = sentence_score + word_tfidf_score(word,sentence,sentences)
     return sentence_score

video_id = extract.video_id('https://www.youtube.com/watch?v=t-pAO3qPwxs')

data = yta.get_transcript(video_id)

transcript = ''

for value in data:
    for key,val in value.items():
        if key=='text':
            transcript += val

l = transcript.splitlines()

final_transcript = " ".join(l)

model = PunctuationModel()

final_with_punct = model.restore_punctuation(final_transcript)

nlp = spacy.load('en_core_web_sm')

doc = nlp(final_with_punct)

tokenized_sentence = []

for sent in doc.sents:
  tokenized_sentence.append(sent.text)

no_of_sentences = int((0.3* len(tokenized_sentence)))

key_index = 0
sentence_with_importance = {}
for sent in tokenized_sentence:
    sentenceimp = sentence_importance(sent,tokenized_sentence)
    sentence_with_importance[key_index] = sentenceimp
    key_index = key_index + 1

sentence_with_importance = sorted(sentence_with_importance.items(), key=operator.itemgetter(1),reverse=True)

count = 0
sentence_no = [] #List of indices of all important sentences

for word_prob in sentence_with_importance:
    if count < no_of_sentences:
        sentence_no.append(word_prob[0])
        count = count+1
    else:
      break

summary = []
index = 0
for sentence in tokenized_sentence:
    if index in sentence_no:
       summary.append(sentence)
    index = index+1
summary = " ".join(summary)

print("Summary:")

result = nlp(summary)

for sent in result.sents:
  print("*",sent.text.capitalize())
