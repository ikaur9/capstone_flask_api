import sys
import os
#import shutil
import time
import traceback

from flask import Flask, request, jsonify, render_template
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
    uploaded_url = request.form.get("text")

    try:
        # EXTRACT ARTICLE TEXT
        raw_text, title, date = check_article(uploaded_url)
        if raw_text is None:
            return render_template("no_article.html", article_url=uploaded_url)

        # ARTICLE TEXT PROCESSING
        preprocessed_text = preprocess(raw_text)
        
        # ARTICLE TEXT CLASSIFICATION
        predicted_class, scores, all_probs, least_expected = classify_sample(preprocessed_text, classifiers, subclassifiers, vectorizer, tfidfconverter)
        
        # WEB SEARCH FOR ALTERNATE BIAS
        alt_texts, alt_titles, alt_urls, alt_sources = alternate_bias_search(uploaded_url, raw_text, date, predicted_class)
        
        alt_classifications = []
        for alt_txt in alt_texts:
            # ALTERNATE BIAS TEXT PROCESSING
            preprocessed_alt_txt = preprocess(alt_txt)
        
            # ALTERNATE BIAS CLASSIFICATION
            predicted_alt_class, alt_scores, all_alt_probs, alt_least_expected = classify_sample(preprocessed_alt_txt, classifiers, subclassifiers, vectorizer, tfidfconverter)
            alt_classifications.append((predicted_alt_class, alt_scores, all_alt_probs, alt_least_expected))
        
        # ALTERNATE BIAS SUMMARY
        alt_summary = alternate_bias_summary(alt_texts)

        return render_template("predict.html", predicted_class=predicted_class, url=uploaded_url, title=title, date=date, text=raw_text)

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})

@app.route('/alternative-sources', methods=['POST'])
def alternative_sources():

    try:
        # WEB SEARCH FOR ALTERNATE BIAS
        alt_texts, alt_titles, alt_urls, alt_sources = alternate_bias_search(uploaded_url, raw_text, date, predicted_class)
        
        alt_classifications = []
        for alt_txt in alt_texts:
            # ALTERNATE BIAS TEXT PROCESSING
            preprocessed_alt_txt = preprocess(alt_txt)
        
            # ALTERNATE BIAS CLASSIFICATION
            predicted_alt_class, alt_scores, all_alt_probs, alt_least_expected = classify_sample(preprocessed_alt_txt, classifiers, subclassifiers, vectorizer, tfidfconverter)
            alt_classifications.append((predicted_alt_class, alt_scores, all_alt_probs, alt_least_expected))
        
        # ALTERNATE BIAS SUMMARY
        alt_summary = alternate_bias_summary(alt_texts)

        return render_template("predict.html", predicted_class=predicted_class, url=uploaded_url, title=title, date=date, text=raw_text)

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})


# @app.route('/train', methods=['GET'])
# def train():
#     # using random forest as an example
#     # can do the training separately and just update the pickles
#     from sklearn.ensemble import RandomForestClassifier as rf

#     df = pd.read_csv(training_data)
#     df_ = df[include]

#     categoricals = []  # going to one-hot encode categorical variables

#     for col, col_type in df_.dtypes.items():
#         if col_type == 'O':
#             categoricals.append(col)
#         else:
#             df_[col].fillna(0, inplace=True)  # fill NA's with 0 for ints/floats, too generic

#     # get_dummies effectively creates one-hot encoded variables
#     df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

#     x = df_ohe[df_ohe.columns.difference([dependent_variable])]
#     y = df_ohe[dependent_variable]

#     # capture a list of columns that will be used for prediction
#     global model_columns
#     model_columns = list(x.columns)
#     joblib.dump(model_columns, model_columns_file_name)

#     global clf
#     clf = rf()
#     start = time.time()
#     clf.fit(x, y)

#     joblib.dump(clf, model_file_name)

#     message1 = 'Trained in %.5f seconds' % (time.time() - start)
#     message2 = 'Model training score: %s' % clf.score(x, y)
#     return_message = 'Success. \n{0}. \n{1}.'.format(message1, message2) 
#     return return_message


if __name__ == '__main__':
    
    app.run(debug=True)







