# import statements
import time
from gensim.summarization.summarizer import summarize
from transformers import PegasusTokenizer, TFPegasusForConditionalGeneration

# load Pegasus model
pegasus_model = TFPegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
pegasus_tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')


def short_gensim_summary(text):
    """
    Creates a short summary of the given text using Gensim. 
    The output is used to obtain an abstractive Pegasus summary.
    
    input:  string, text of the article
    output: string, short Gensim summary of the article
    
    """
    tic_0 = time.perf_counter()
    try:    
        # if the text is less than 120 words raise an error
        if len(text.split(' ')) < 120:
            raise ValueError('Not enough text in document')
            
        tic = time.perf_counter()
        gensim_summary = summarize(text)
        
        tic_1 = time.perf_counter()
        print(f"Got the gensim short summary in {tic_1 - tic_0:0.4f} seconds")

        return gensim_summary

    except Exception as inst:
        raise inst

def short_pegasus_summary(gensim_summary):
    """
    Creates a short summary of the Gensim extracted sentences using Pegasus. 
    The output is used as the web search query to find related articles.
    
    input:  string, text of the article
    output: string, short Pegasus summary of the article
    
    """
    tic_0 = time.perf_counter()
    try:    
        pegasus_input = pegasus_tokenizer([gensim_summary], max_length=512, truncation=True, return_tensors='tf')
        
        # max_length is 20 because google search only takes up to 32 words in one search 
        pegasus_summary_id =  pegasus_model.generate(pegasus_input['input_ids'], 
                                    no_repeat_ngram_size=5,
                                    min_length=5,
                                    max_length=29,
                                    early_stopping=True)
        pegasus_summary_ = [pegasus_tokenizer.decode(g, skip_special_tokens=True, 
                           clean_up_tokenization_spaces=False) for g in pegasus_summary_id]
        
        tic_1 = time.perf_counter()
        print(f"Created the pegasus short summary in {tic_1 - tic_0:0.4f} seconds")

        return pegasus_summary_[0]
    
    except Exception as inst:
        raise inst

def short_summary(text):
    """
    Creates a short summary of the given text. 
    The output is used as the web search query to find related articles.
    
    input:  string, text of the article
    output: string, short summary of the article
    
    """
    gensim_summary = short_gensim_summary(text)
    pegasus_summary = short_pegasus_summary(gensim_summary)
    
    return pegasus_summary

def alternate_bias_summary(similar_articles_text):
    """ 
    Summarize each related article using Gensim, then combine the summaries into one text and use
    the Pegasus model to obtain an abstractive multi-document summary.
    
    input: list of strings, texts of similar articles
    output: string, summary of similar articles
    
    """
    tic_0 = time.perf_counter()
    
    # summarize the first 400 words from each article 
    texts = [summarize(" ".join(article.split(" ")[:400])) for article in similar_articles_text]
    
    tic_1 = time.perf_counter()
    print(f"Got the gensim summary for all articles in {tic_1 - tic_0:0.4f} seconds")
    
    # combine the summaries into one text
    combined_txt = " ".join(texts)
    
    # use the Pegasus model to obtain an abstractive multi-document summary 
    pegasus_input_list = pegasus_tokenizer([combined_txt], truncation=True, return_tensors='tf')
    pegasus_summary_ids = pegasus_model.generate(pegasus_input_list['input_ids'], 
                                    no_repeat_ngram_size=5,
                                    min_length=60,
                                    max_length=300,
                                    early_stopping=True)
    pegasus_summary_list = [pegasus_tokenizer.decode(g, skip_special_tokens=True, 
                           clean_up_tokenization_spaces=False) for g in pegasus_summary_ids]
    
    tic_2 = time.perf_counter()
    print(f"Got the pegasus summary in {tic_2 - tic_1:0.4f} seconds")
    
    return pegasus_summary_list











