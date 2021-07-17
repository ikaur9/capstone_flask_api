from transformers import TFXLNetForSequenceClassification, XLNetTokenizer, T5Tokenizer, TFT5ForConditionalGeneration, PegasusTokenizer, TFPegasusForConditionalGeneration

# load Pegasus model for first level summarization
pegasus_model = TFPegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
pegasus_tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')

# load T5 model for second level summarization
t5_model = TFT5ForConditionalGeneration.from_pretrained('t5-large')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-large')


def short_summary(text):
    """
    Creates a short summary of the article for the search query for related articles
    
    input:  string, text of the article
    output: string, short summary of the article
    """
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
    Summarize the article texts using Pegasus model (abstractive) on each text, then combine the summaries into a string and put it into the T5 model (extractive)
    
    input: list of strings, texts of similar articles
    output: string, summary of similar articles
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

















