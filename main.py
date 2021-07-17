#import sys
#import os
#import shutil
import time
import traceback

from flask import Flask, request, jsonify, render_template, session
import pandas as pd
from joblib import load

from article_extraction import check_article
from text_preprocessing import preprocess
from text_classification import classify_sample
from web_searching import alternate_bias_search
from summarization import alternate_bias_summary

app = Flask(__name__)

#article_url = "https://www.nytimes.com/2021/06/21/us/supreme-court-ncaa-student-athletes.html"

model_directory = './TFIDF'

try:  
    print('Loading 1 vs All models...')
    centerclassifier = load('%s/centerclassifierarticle.pickle' % model_directory)
    leftclassifier = load('%s/leftclassifierarticle.pickle' % model_directory)
    rightclassifier = load('%s/rightclassifierarticle.pickle' % model_directory)

    print('Loading 1 vs 1 models...')
    leftcenterclassifier = load('%s/leftcenterclassifier.pickle' % model_directory)
    leftrightclassifier = load('%s/leftrightclassifier.pickle' % model_directory)
    rightcenterclassifier = load('%s/rightcenterclassifier.pickle' % model_directory)
    print('Models loaded.')

    classifiers = [centerclassifier, leftclassifier, rightclassifier]
    subclassifiers = [leftrightclassifier, rightcenterclassifier, leftcenterclassifier]

    print('Loading vectorizer...')
    vectorizer = load('%s/vectorizer.pickle' % model_directory)
    print('Vectorizer loaded.')

    print('Loading converter...')
    tfidfconverter = load('%s/tfidfconverter.pickle' % model_directory)
    print('Converter loaded.')
    
except Exception as e:
    print('Failed to load models with exception:')
    print(str(e))
    classifiers, subclassifiers, vectorizer, tfidfconverter = None, None, None, None

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    url = request.form.get("text")
    session["url"] = url
    try:
        # EXTRACT ARTICLE TEXT
        text, title, date = check_article(url)
        if text is None:
            return render_template("no_article.html", url=url)
        
        session["text"] = text
        session["title"] = title
        session["date"] = date
        
        # ARTICLE TEXT PROCESSING
        preprocessed_text = preprocess(text)
        
        # ARTICLE TEXT CLASSIFICATION
        predicted_class, scores, all_probs, least_expected = classify_sample(preprocessed_text, classifiers, subclassifiers, vectorizer, tfidfconverter)
        
        session["predicted_class"] = int(predicted_class)

        return render_template("predict.html", url=url)

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})

@app.route('/alternative-sources', methods=['POST'])
def alternative_sources():

    url = session.get("url", None)
    text = session.get("text", None)
    date = session.get("date", None)
    predicted_class = session.get("predicted_class", None)
    
    print(type(url))
    print(text)
    print(type(text))
    print(date)
    print(type(predicted_class))
          
    try:
        # WEB SEARCH FOR ALTERNATE BIAS
        alt_texts, alt_titles, alt_urls, alt_sources = alternate_bias_search(url, text, date, predicted_class)
        
        sources_found = len(alt_texts)
        
        session["alt_urls"] = alt_urls
        session["alt_titles"] = alt_titles
        session["alt_sources"] = alt_sources
        
        alt_classes = []
        for alt_txt in alt_texts:
            # ALTERNATE BIAS TEXT PROCESSING
            preprocessed_alt_txt = preprocess(alt_txt)
        
            # ALTERNATE BIAS CLASSIFICATION
            predicted_alt_class, scores, all_probs, least_expected = classify_sample(preprocessed_alt_txt, classifiers, subclassifiers, vectorizer, tfidfconverter)
            alt_classes.append(int(predicted_alt_class))
        
        session["alt_classes"] = alt_classes
        
        # ALTERNATE BIAS SUMMARY
        alt_summary = alternate_bias_summary(alt_texts)
        
        session["alt_summary"] = alt_summary

        return render_template("alternative_sources.html", url=url, sources_found=sources_found)

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})



#     start = time.time()
#     joblib.dump(clf, model_file_name)
#     message1 = 'Trained in %.5f seconds' % (time.time() - start)
#     message2 = 'Model training score: %s' % clf.score(x, y)
#     return_message = 'Success. \n{0}. \n{1}.'.format(message1, message2)


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)







