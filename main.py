import time
import traceback

from flask import Flask, request, jsonify, render_template, session

from article_extraction import check_article
from text_classification import classify
from web_searching import alternate_bias_search
from summarization import alternate_bias_summary

app = Flask(__name__)
app.secret_key = b'\xc4N\xcbk`k\xf88\na\x0c\xbf1\xd1\xf2\xe5'


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
        
        # ARTICLE TEXT CLASSIFICATION
        predicted_class, scores, all_probs, least_expected = classify(text)
        
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
          
    try:
        # WEB SEARCH FOR ALTERNATE BIAS
        alt_texts, alt_titles, alt_urls, alt_sources = alternate_bias_search(url, text, date, predicted_class)
        
        sources_found = len(alt_texts)
        
        session["alt_urls"] = alt_urls
        session["alt_titles"] = alt_titles
        session["alt_sources"] = alt_sources
        
        # ALTERNATE BIAS CLASSIFICATION
        alt_classes = []
        
        for alt_txt in alt_texts:
            predicted_alt_class, scores, all_probs, least_expected = classify_sample(alt_txt)
            alt_classes.append(int(predicted_alt_class))
        
        session["alt_classes"] = alt_classes
        
        # ALTERNATE BIAS SUMMARY
        alt_summary = alternate_bias_summary(alt_texts)
        
        session["alt_summary"] = alt_summary

        return render_template("alternative_sources.html", url=url, sources_found=sources_found)

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})


#     start = time.time()
#     message1 = 'Trained in %.5f seconds' % (time.time() - start)
#     message2 = 'Model training score: %s' % clf.score(x, y)
#     return_message = 'Success. \n{0}. \n{1}.'.format(message1, message2)


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)







