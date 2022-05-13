from flask import Flask
from flask import render_template
from flask import url_for
import tensorflow as tf
# ?? import tensorflow_hub as hub

# ecport FLASK_APP=process_model.py
app = Flask(__name__, template_folder='./templates', static_folder='./static')

@app.route("/")
@app.route("/home")
def main_page():
    return render_template('home.html') # ,posts=posts

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/process')
def process(image):
    return render_template('process.html')


if __name__ == '__main__':
    app.run(debug=True)
