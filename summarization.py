import numpy as np
import pandas as pd
import sklearn
from random import sample
import re

# for summarization
from transformers import TFXLNetForSequenceClassification, XLNetTokenizer, T5Tokenizer, TFT5ForConditionalGeneration, PegasusTokenizer, TFPegasusForConditionalGeneration
import datetime
import tensorflow as tf
from newspaper import Article, Config
from heapq import nlargest
#from GoogleNews import GoogleNews
from googlesearch import search
from bs4 import BeautifulSoup
import requests

from article_extraction import check_article

# for partial matching strings
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# for document similarity 
from sklearn.feature_extraction.text import TfidfVectorizer


# pegasus model loaded
pegasus_model = TFPegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
pegasus_tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')


def short_summary(text):
    """
    Creates a short summary of the article for the search query for related articles
    
    input:  string, text of the article
    output: string, short summary of the article
    """
#     try:
#         article = Article(article_url)
#         article.download()
#         article.parse()
#         txt = article.text
#         try:
#             pub_date = article.publish_date
#             month_yr = pub_date.strftime("%B") + " " + str(pub_date.year)
#         except:
#             month_yr = ""
#             print('no published date')
    pegasus_input = pegasus_tokenizer([text], max_length=512, truncation=True, return_tensors='tf')
    # max_length is 20 because google search only takes up to 32 words in one search 
    pegasus_summary_id =  pegasus_model.generate(pegasus_input['input_ids'], 
                                no_repeat_ngram_size=5,
                                min_length=5,
                                max_length=29,
                                early_stopping=True)
    pegasus_summary_ = [pegasus_tokenizer.decode(g, skip_special_tokens=True, 
                       clean_up_tokenization_spaces=False) for g in pegasus_summary_id]
    return pegasus_summary_[0]


def alternate_bias_summary(similar_articles_text):
    """ 
    Summarize the article texts using pegasus model(abstractive) on each text and then combine the summaries into a string and
    put it into the t5 model (extractive)
    
    input: tuple of similar article text
    output: string of the summary of similar articles
    """
    # summarize each article using pegasus
    pegasus_input_list = [pegasus_tokenizer([text], max_length=512, truncation=True, return_tensors='tf')
                          for text in similar_articles_text]
    
    pegasus_summary_ids =  [pegasus_model.generate(i['input_ids'], 
                                    no_repeat_ngram_size=5,
                                    min_length=60,
                                    max_length=300,
                                    early_stopping=True) for i in pegasus_input_list]
    
    pegasus_summary_list = [[pegasus_tokenizer.decode(g, skip_special_tokens=True, 
                           clean_up_tokenization_spaces=False) for g in i] for i in pegasus_summary_ids]
    
    # combine the pegasus summaries into a string
    pegasus_summaries = " ".join([i[0] for i in pegasus_summary_list])
    
    # get final summary through t5 model
    total_input_list = t5_tokenizer(["summarize: " + pegasus_summaries], truncation = True, return_tensors = 'tf')
    t5_id =  t5_model.generate(total_input_list['input_ids'],
                                    num_beams=6,
                                    no_repeat_ngram_size=5,
                                    min_length=50,
                                    max_length=300)
    t5_summary = [t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in t5_id]
    return t5_summary[0]

















