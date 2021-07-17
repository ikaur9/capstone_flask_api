import numpy as np
import pandas as pd
#import sklearn
from random import sample
#import re

# for summarization
#from transformers import TFXLNetForSequenceClassification, XLNetTokenizer, T5Tokenizer, TFT5ForConditionalGeneration, PegasusTokenizer, TFPegasusForConditionalGeneration
#import datetime
#import tensorflow as tf
#from newspaper import Article, Config
#from heapq import nlargest
from googlesearch import search
#from bs4 import BeautifulSoup
#import requests

from article_extraction import check_article
from summarization import short_summary

# for partial matching strings
from fuzzywuzzy import fuzz, process
#from fuzzywuzzy import process

# for document similarity 
from sklearn.feature_extraction.text import TfidfVectorizer

# put the news sources and biases in a dataframe: data
bias_data = pd.read_csv('./data/bias_df.csv')

def get_alternative_bias(article_bias_id):
    """
    Gets the other biases from the article bias 
    eg. get_opposite_bias('right') returns ['left', 'center']
    
    input: string, the bias of the article - options: left, center, right
    output: list, of the alternative biases       
    """
    biases = ['left', 'center', 'right']
    article_bias = biases[article_bias_id]
    try:
        biases.remove(article_bias)
        
    except ValueError:
        # no bias, return list of just center
        biases = ['center']
    return biases


def get_alternative_urls(original_url, title, date, alternative_bias):
    """
    Gets the related alternative articles url links through google search
    
    input: original_url - string, url of the article that we are trying to get alternative articles for
           title - string, short summary of article,
           date - string, month and year,
           alternative_bias - list of strings, the alternative sides of bias of the article,
           bias_data - dataframe - 2 columns, the source and the bias
    output: list of tuples, first element of the tuple is the url of the alternative bias covering the same topic
                            second element of the tuple is the source name
   """
    # using the googlesearch API      
    articles = []
    sources = []
    for bias in alternative_bias: 
        source_list = bias_data[bias_data.bias == bias].source.tolist()
        sample_sources = sample(source_list, 4)
        for source in sample_sources:
            query_list = [title, 'article', date, source]
            source_url = bias_data[bias_data.source == source].website.iloc[0]
            query = ' '.join(query_list)
            search_generator = search(query, num = 2, pause = 3)
            article_url = next(search_generator)
            if article_url in articles or fuzz.partial_ratio(original_url, article_url) > 80:
                article_url2 = next(search_generator)
                if article_url2 in articles or fuzz.partial_ratio(original_url, article_url2) > 80:
                    continue
                elif fuzz.partial_ratio(source_url, article_url2) > 80:
                    articles.append(article_url2)
                    sources.append(source)
            elif source_url not in article_url:
                if fuzz.partial_ratio(source_url, article_url) > 80:
                    articles.append(article_url)
                    sources.append(source)
                else:
                    article_url2 = next(search_generator)
                    if article_url2 in articles or fuzz.partial_ratio(original_url, article_url2) > 80:
                        continue
                    elif fuzz.partial_ratio(source_url, article_url2) > 80:
                        articles.append(article_url2)
                        sources.append(source)
                    else:
                        continue
            else:
                articles.append(article_url)
                sources.append(source)
    #zipped_list = list(zip(articles, sources))
    return articles, sources

def url_to_info(urls, sources):
    """
    Convert the urls and sources to the text and the titles of the related articles
    
    input: list, urls of related articles
    output: list of tuples, (article text, article title, source name)
    """
    article_texts = []
    article_titles = []
    article_urls = []
    article_sources = []
    #urls, sources = zip(*zipped_urls_sources)
    for index in range(len(urls)):
        
        alt_url = urls[index]
        alt_source = sources[index]
        alt_text, alt_title, _ = check_article(alt_url)
        
        # if there is less than 35 words in the article, it isn't included
        if alt_text is None or len(alt_text.split(' ')) < 35:
            continue
        else:
            article_texts.append(alt_text)
            article_titles.append(alt_title)
            article_urls.append(alt_url)
            article_sources.append(alt_source)
        
#         try:
#             article = Article(urls[index])
#             article.download()
#             article.parse()
#             txt = article.text
#             article.nlp()
#             # if there is no text in the article it isn't included
#             if txt:
        
#         except:
#             continue

    #zipped_articles = list(zip(article_texts, article_titles, article_sources, article_urls))
    return article_texts, article_titles, article_urls, article_sources

def similar_documents(texts, titles, urls, sources):
    """"
    function to get similar documents in order to ensure we have the articles that have the same topic, event or issue
    
    input: list of tuples, article texts, titles, sources, urls
    output: list of tuples, first element of the tuple are article texts that have high similarity to one another 
                            second element of the tuple are the titles of the articles
    """
    #texts, titles, sources, urls = zip(*articles)
    tfidf = TfidfVectorizer().fit_transform(texts)
    pairwise_similarity = tfidf * tfidf.T
    
    # for each document compute the average similarity score to the other documents
    # .53 is an arbitrary threshold
    # should be higher than .53 average to make sure that the documents talk about the same topic
    avg_similarity = np.average(pairwise_similarity.toarray(), axis = 1)
    bool_similarity = avg_similarity > 0.53
    
    # get the list of articles that fulfill the requirement of .53 avg similarity
    # if there are more than 4 articles that have greater than .53 similarities, only take the top 4 similarities 
    if sum(bool_similarity) > 4:
        top_indexes = avg_similarity.argsort()[-4:][::-1]
        updated_texts = [texts[i] for i in top_indexes]
        updated_titles = [titles[i] for i in top_indexes]
        updated_sources = [sources[i] for i in top_indexes]
        updated_urls = [urls[i] for i in top_indexes]
    elif sum(bool_similarity) == 0:
        #if there is no article that has a collective similarity score
        print("No similar articles")
        return np.nan
    else:
        updated_texts = list(np.array(texts)[bool_similarity])
        updated_titles = list(np.array(titles)[bool_similarity])
        updated_sources = list(np.array(sources)[bool_similarity])
        updated_urls = list(np.array(urls)[bool_similarity])
    #zipped_similar = list(zip(updated_texts, updated_titles, updated_sources, updated_urls))
    return updated_texts, updated_titles, updated_urls, updated_sources


def alternate_bias_search(orig_url, orig_text, orig_date, orig_bias):

    # summarize original article for web search
    search_summary = short_summary(orig_text)
    
    # web seach for alternative bias articles
    alt_bias = get_alternative_bias(orig_bias)
    urls, sources = get_alternative_urls(orig_url, search_summary, orig_date, alt_bias)

    # extract alternative article texts and titles
    alt_texts, alt_titles, alt_urls, alt_sources = url_to_info(urls, sources)
    
    # only keep relevant articles
    updated_texts, updated_titles, updated_urls, updated_sources = similar_documents(alt_texts, alt_titles, alt_urls, alt_sources)
    
    return updated_texts, updated_titles, updated_urls, updated_sources



