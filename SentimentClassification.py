# Import glob and os to read files
from sklearn.datasets import load_files
# Ntlk and lemmatizer for cleaning data
from nltk.stem import WordNetLemmatizer
# Count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Train test split
from sklearn.model_selection import train_test_split
# Import naive bayes model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import re
from sklearn.model_selection import cross_val_score
# import to serialize model
import pickle
# Treating 1 as positive sentiment and 0 as negative sentiment
# opening files for negative sentiments
movie_data = load_files(container_path='txt_sentoken')
review, label = movie_data.data, movie_data.target

lemmatizer = WordNetLemmatizer()


def clean_text(docs):
    """Function to clean text removing names and non letters"""
    documents = []
    for sen in range(0, len(docs)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(docs[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        document = [lemmatizer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)
    return documents


cleaned_review = clean_text(review)
print('cleaned review length', len(cleaned_review))
# Removing stop words, feature extraction and vectorizing reviews to use in naive bayes
cv = CountVectorizer(stop_words='english', max_features=1500, min_df=5, max_df=0.7)
cleaned_review_data = cv.fit_transform(cleaned_review).toarray()
print(cleaned_review_data)
# Cross validation
scores = cross_val_score(MultinomialNB(), cleaned_review_data, label, cv=5)
print('cross validation', scores)
# train, test split data for building model
x_train, x_test, y_train, y_test = train_test_split(cleaned_review_data, label, test_size=0.25, random_state=0)
# fiting testing and test data
# Naive bayes model initialization
clf = MultinomialNB(alpha=1, fit_prior=True)
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
accuracy = clf.score(x_test, y_test)
print('prediction', prediction)
print('Accuracy', accuracy * 100)
clas_report = classification_report(y_test, prediction)
print('Classification report', clas_report)
# Serializing model
pickle.dump(clf, open('model.pkl', 'wb'))
# ############################# testing model ###############################
rev = ["""Bad movie nothing good about it. what a waste of time"""]
cleaned_rev = clean_text(rev)
print(cleaned_rev)
x_term_tes = cv.transform(cleaned_rev)
prediction_test = clf.predict(x_term_tes)
if prediction_test == 1:
    print('This is positive sentiment')
else:
    print('This is negative Sentiment')
