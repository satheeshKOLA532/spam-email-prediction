from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the model and vectorizer
df = pd.read_csv(r'C:\Users\91939\Desktop\ANACONDA _PROJECTS\spam email prediction\spam.csv')
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
x_train, _, y_train, _ = train_test_split(df.Message, df.spam, test_size=0.25)
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values).toarray()
nb_model = MultinomialNB()
nb_model.fit(x_train_count, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        email_count = cv.transform([message])
        prediction = nb_model.predict(email_count)[0]
        result = 'Spam' if prediction == 1 else 'Not Spam....'
        return render_template('result.html', result=result, message=message)

if __name__ == '__main__':
    app.run(debug=True)
