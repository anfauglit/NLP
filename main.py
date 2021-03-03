from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

with open('text.txt', 'r') as f:
	text = f.read()

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.3)

filtered_text = [word for word in word_tokenize(text) if word not in stop_words] 
lemmatized_text = list(map(lemmatizer.lemmatize, filtered_text))
lemmatized_text = [w for w in lemmatized_text if w.isalpha()]
fdist = nltk.FreqDist(lemmatized_text)

fdist.plot(20,cumulative=False)
plt.show()
