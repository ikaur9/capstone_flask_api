import time
import numpy as np
import pandas as pd
from random import sample
from googlesearch import search

from article_extraction import check_article
from summarization import short_gensim_summary, short_pegasus_summary

# for partial matching strings
from fuzzywuzzy import fuzz, process

# for document similarity 
from sklearn.feature_extraction.text import TfidfVectorizer

# put the news sources and biases in a dataframe: data
bias_data = pd.read_csv('./data/bias_df.csv')

def get_alternative_bias(article_bias_id):
    """
    Get the alternate biases from the given article bias. 
    eg. get_opposite_bias(2) returns ['left', 'center']
    
    input: int, bias of given article (0: center, 1: left, 2: right)
    output: list of strings, alternative biases
    
    """
    biases = ['center', 'left', 'right']
    article_bias = biases[article_bias_id]
    
    try:
        biases.remove(article_bias)
    except ValueError:
        # no bias, return list of just center
        biases = ['center']
        
    return biases

def sample_alternative_bias_sources(biases, num_sources=5):
    """
    Get the alternate bias sources for each alternate biases. 
    
    input: list of strings, alternate biases
    output: list of strings, alternative biases sources
    
    """
    sources = []
    source_biases = []
    biases.sort()
    
    # search for articles from alternative bias sources
    for bias in biases:
        # sample num_sources alternative sources to search for
        source_list = bias_data[bias_data.bias == bias].source.tolist()
        sample_sources = sample(source_list, num_sources)
        
        # return the name and bias of each source to query from
        sources.extend(sample_sources)
        source_biases.extend([bias] * num_sources)
        
    return sources, source_biases


def get_alternative_urls(original_url, summary, date, source_bias, source_name):
    """
    Get the related alternative article urls and sources through Google search.
    Search for a particular source name and source bias.
    
    input: original_url - string, url of the article that we are trying to get alternative articles for
           summary - string, short summary of article
           date - string, month and year
           source_bias - string, the alternate bias that we are looking for
           source_name - string, alternative source to query
    output: articles - list of strings, alternate bias article URLS
            sources - list of strings, alternate bias article source names
            biases - list of strings, alternate article biases
            
    """
    # using the googlesearch API      
    articles = []
    sources = []
    biases = []
         
    # create Google search query
    query_list = [summary, 'article', date, source_name]
    source_url = bias_data[bias_data.source == source_name].website.iloc[0]
    query = ' '.join(query_list)

    # run Google search and get first result
    search_generator = search(query, num = 2, pause = 3)
    article_url = next(search_generator)

    # skip first result if already found or same as the original article
    if article_url in articles or fuzz.partial_ratio(original_url, article_url) > 80:
        article_url2 = next(search_generator)

        # skip second result if already found or same as the original article
        if article_url2 in articles or fuzz.partial_ratio(original_url, article_url2) > 80:
            return articles, sources, biases

        # keep result if it's from the target source
        elif fuzz.partial_ratio(source_url, article_url2) > 80:
            articles.append(article_url2)
            sources.append(source_name)
            biases.append(source_bias)

    elif source_url not in article_url:
        # keep result if it's from the target source
        if fuzz.partial_ratio(source_url, article_url) > 80:
            articles.append(article_url)
            sources.append(source_name)
            biases.append(source_bias)
        else:
            article_url2 = next(search_generator)

            # skip second result if already found or same as the original article
            if article_url2 in articles or fuzz.partial_ratio(original_url, article_url2) > 80:
                return articles, sources, biases

            # keep result if it's from the target source
            elif fuzz.partial_ratio(source_url, article_url2) > 80:
                articles.append(article_url2)
                sources.append(source_name)
                biases.append(source_bias)
            else:
                return articles, sources, biases
    else:
        articles.append(article_url)
        sources.append(source_name)
        biases.append(source_bias)
        
    return articles, sources, biases


def url_to_info(urls, sources, biases):
    """
    Extract the text and titles of related articles from their URLs and sources.
    
    input: urls - list, urls of related articles
           sources - list, sources of related articles
           biases - list, biases of related articles
    output: article_texts - list of strings, alternate bias article text
            artcle_titles - list of strings, alternate bias article titles
            article_urls - list of strings, alternate bias article URLS
            article_sources - list of strings, alternate bias article source names
            article_bias - list of strings, biases of alternate articles 
    
    """
    article_texts = []
    article_titles = []
    article_urls = []
    article_sources = []
    article_bias = []
    
    for index in range(len(urls)):
        
        alt_url = urls[index]
        alt_text, alt_title, _ = check_article(alt_url)
        
        # exclude articles less than 100 words long
        if alt_text is None or len(alt_text.split(' ')) < 100:
            continue
            
        else:
            article_texts.append(alt_text)
            article_titles.append(alt_title)
            article_urls.append(alt_url)
            article_sources.append(sources[index])
            article_bias.append(biases[index])
            
    return article_texts, article_titles, article_urls, article_sources, article_bias


def list_comprehension_indexed(lst, indexes):
    """
    helper function to subset list by indexes
    
    input: 2 list, 1 list is the list to subset, other list is the indexes to subset by
    output: list, subset of the list by the indexes
    
    """
    updated_list = [lst[i] for i in indexes]
    return updated_list


def boolean_indexed(lst, boolean):
    """
    helper function to sebset list by boolean list
    
    input: 2 list of the same length, 1 list is the list to subset, other list is the boolean list to subset other list
    output: list, subset of the list by the boolean list
    
    """
    updated_list = list(np.array(lst)[boolean])
    return updated_list

def similar_documents(texts, titles, urls, sources, biases):
    """"
    Get similar documents to ensure we have articles that cover the same topic, event, or issue.
    
    input: list of strings, article texts, titles, sources, urls, biases
    output: list of strings, updated article texts, titles, sources, urls, biases

    """
    tfidf = TfidfVectorizer().fit_transform(texts)
    pairwise_similarity = tfidf * tfidf.T
    
    # for each document compute the average similarity score to the other documents
    # keep documents with >.53 average similarity to make sure documents are about the same topic
    # (.53 is an arbitrary threshold)
    avg_similarity = np.average(pairwise_similarity.toarray(), axis = 1)
    bool_similarity = avg_similarity > 0.53

    #if there are more than 4 articles >.53 similarity, take the top 4  
    if sum(bool_similarity) > 4:
        top_indexes = avg_similarity.argsort()[-4:][::-1]
        updated_texts = list_comprehension_indexed(texts, top_indexes)
        updated_titles = list_comprehension_indexed(titles, top_indexes) 
        updated_sources = list_comprehension_indexed(sources, top_indexes) 
        updated_urls = list_comprehension_indexed(urls, top_indexes)
        updated_bias = list_comprehension_indexed(biases, top_indexes)
    
    # error if there are less than 2 articles that have a collective similarity score >.53
    elif sum(bool_similarity) <=1: 
        raise ValueError('No similar articles found')
        
    else:
        updated_texts = boolean_indexed(texts, bool_similarity)
        updated_titles = boolean_indexed(titles, bool_similarity)
        updated_sources = boolean_indexed(sources, bool_similarity)
        updated_urls = boolean_indexed(urls, bool_similarity)
        updated_bias = boolean_indexed(biases, bool_similarity)

    return updated_texts, updated_titles, updated_urls, updated_sources, updated_bias


def alternate_bias_search(orig_url, orig_text, orig_date, orig_bias):
    
    tic_0 = time.perf_counter()
    
    # summarize original article for web search
    search_gensim_summary = short_gensim_summary(orig_text)
    search_pegasus_summary = short_pegasus_summary(search_gensim_summary)
    
    tic_1 = time.perf_counter()
    print(f"Got the short summary in {tic_1 - tic_0:0.4f} seconds")
    
    # web seach for alternative bias articles
    alt_bias = get_alternative_bias(orig_bias)
    
    alt_sources, alt_source_biases = sample_alternative_bias_sources(alt_bias)
    print(alt_sources)
    print(alt_source_biases)
    
    tic_2 = time.perf_counter()
    print(f"Got the sample sources in {tic_2 - tic_1:0.4f} seconds")
    
    urls = []
    sources = []
    biases = []
    for idx in range(len(alt_sources)):
        source = alt_sources[idx]
        source_bias = alt_source_biases[idx]
        u, s, b = get_alternative_urls(orig_url, search_pegasus_summary, orig_date, source_bias, source)
        urls.extend(u)
        sources.extend(s)
        biases.extend(b)
        print(u)
        print(s)
        print(b)
        
    if len(urls) == 0:
        raise ValueError('No alternative articles found')
    
    tic_3 = time.perf_counter()
    print(f"Got the alternative URLs in {tic_3 - tic_2:0.4f} seconds")

    # extract alternative article texts and titles
    alt_texts, alt_titles, alt_urls, alt_sources, alt_biases = url_to_info(urls, sources, biases)
    
    tic_4 = time.perf_counter()
    print(f"Got the alternative texts in {tic_4 - tic_3:0.4f} seconds")
    
    # only keep relevant articles
    updated_texts, updated_titles, updated_urls, updated_sources, updated_bias = similar_documents(alt_texts, alt_titles, alt_urls, alt_sources, alt_biases)
    
    tic_5 = time.perf_counter()
    print(f"Got the similar docs in {tic_5 - tic_4:0.4f} seconds")
    
    print(f"Got the alternate bias search total in {tic_5 - tic_0:0.4f} seconds")
    
    return updated_texts, updated_titles, updated_urls, updated_sources, updated_bias



