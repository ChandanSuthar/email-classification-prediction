from flask import Flask, render_template, request, url_for
import joblib
import re

model = joblib.load('./models/multinomialnaivebayes.lb')
bow_obj = joblib.load('./models/countvectorizer.lb')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/get_result", methods=['GET','POST'])# type:ignore
def get_result():
    if request.method == "POST":
        emailContent = request.form.get('emailContent', '')
        
        emailContent = emailContent.lower()
        emailContent = re.sub(r'[^a-zA-Z ]', '', emailContent)
        
        emailContent_transformed = bow_obj.transform([emailContent])
        
        prediction = model.predict(emailContent_transformed)[0]
        
        labels = {'1': "SPAM", '0': "HAM"}
        result = labels.get(str(prediction))
        
        return render_template('output.html',output=result)

if __name__ == "__main__":
    app.run(debug=True)