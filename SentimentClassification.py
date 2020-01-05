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
import numpy as np
import re
from sklearn.model_selection import cross_val_score
# Treating 0 as positive sentiment and 1 as negative sentiment
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
print('prediction',prediction)
print('Accuracy', accuracy*100)
clas_report = classification_report(y_test, prediction )
print('Classification report', clas_report)

# ############################# testing model ###############################
rev = ["""every now and then a movie comes along from a suspect studio , with every indication that it will be a stinker , and to everybody's surprise ( perhaps even the studio ) the film becomes a critical darling . 
mtv films' _election , a high school comedy starring matthew broderick and reese witherspoon , is a current example . 
did anybody know this film existed a week before it opened ? 
the plot is deceptively simple . 
george washington carver high school is having student elections . 
tracy flick ( reese witherspoon ) is an over-achiever with her hand raised at nearly every question , way , way , high . 
mr . " m " ( matthew broderick ) , sick of the megalomaniac student , encourages paul , a popular-but-slow jock to run . 
and paul's nihilistic sister jumps in the race as well , for personal reasons . 
the dark side of such sleeper success is that , because expectations were so low going in , the fact that this was quality stuff made the reviews even more enthusiastic than they have any right to be . 
you can't help going in with the baggage of glowing reviews , which is in contrast to the negative baggage that the reviewers were likely to have . 
_election , a good film , does not live up to its hype . 
what makes _election_ so disappointing is that it contains significant plot details lifted directly from _rushmore_ , released a few months earlier . 
the similarities are staggering : 
tracy flick ( _election_ ) is the president of an extraordinary number of clubs , and is involved with the school play . 
max fischer ( _rushmore_ ) is the president of an extraordinary number of clubs , and is involved with the school play . 
the most significant tension of _election_ is the potential relationship between a teacher and his student . 
the most significant tension of _rushmore_ is the potential relationship between a teacher and his student . 
tracy flick is from a single parent home , which has contributed to her drive . 
max fischer is from a single parent home , which has contributed to his drive . 
the male bumbling adult in _election_ ( matthew broderick ) pursues an extramarital affair , gets caught , and his whole life is ruined . 
he even gets a bee sting . 
the male bumbling adult in _rushmore_ ( bill murray ) pursues an extramarital affair , gets caught , and his whole life is ruined . 
he gets several bee stings . 
and so on . 
what happened ? 
how is it that an individual screenplay ( _rushmore_ ) and a novel ( _election_ ) contain so many significant plot points , and yet both films were probably not even aware of each other , made from two different studios , from a genre ( the high school geeks revenge movie ) that hadn't been fully formed yet ? 
even so , the strengths of _election_ rely upon its fantastic performances from broderick , witherspoon , and newcomer jessica campbell , as paul's anti-social sister , tammy . 
broderick here is playing the mr . rooney role from _ferris bueller_ , and he seems to be having the most fun he's had since then . 
witherspoon is a revelation . 
it's early in the year , it's a comedy , and teenagers have little clout , but for my money , witherspoon deserves an oscar nomination . 
and once campbell's character gets going , like in her fantastic speech in the gymnasium , then you're won over . 
one thing that's been bothering me since i've seen it . 
there is an extraordinary amount of sexuality in this film . 
i suppose that , coming from mtv films , i should expect no less . . . 
but the film starts off light and airy , like a sitcom . 
as the screws tighten , and the tensions mount , alexander payne decides to add elements that , frankly , distract from the story . 
it is bad enough that mr . m doesn't like tracy's determination to win at all costs , but did they have to throw in the student/teacher relationship ? 
even so , there's no logical reason why mr . m has an affair when he does . 
there's a lot to like in _election_ , but the plot similarities to _rushmore_ , and the tonal nosedive it takes as it gets explicitly sex-driven , mark this as a disappointment . 

"""]

cleaned_rev = clean_text(rev)
x_term_tes = cv.transform(cleaned_rev)
prediction_test = clf.predict(x_term_tes)
if prediction_test == 1:
    print('This is positive sentiment')
else:
    print('This is negative Sentiment')
