# Import in flask
from flask import Flask, request, render_template
import pickle
# importing SentimentClassification.py for cleaning and processing data
import SentimentClassification as Sc
import numpy as np
# initializing flask
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [np.array(request.form['review_data'])]
    feature_1 = Sc.clean_text(features)
    print(feature_1)
    transformed_feature = Sc.cv.transform(feature_1)
    print(transformed_feature)
    prediction = Sc.clf.predict(transformed_feature)
    print(prediction[0], 2)
    review = 'No Sentiment yet'
    if prediction[0] == 1:
        review = 'Positive Sentiment'
    else:
        review = 'Negative Sentiment'
    return render_template('index.html', prediction_text=review)


if __name__ == "__main__":
    app.run(debug=True)
