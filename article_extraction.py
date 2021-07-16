from newspaper import Article

def check_article(url):
    
    # extract article text
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
    except:
        print('Article not found.')
        text = None
    
    # extract article title
    try:
        title = article.title
    except:
        print('Article title not found.')
        title = None
        
    # extract article publication date (format: Month Year)
    try:
        pub_date = article.publish_date
        month_yr = pub_date.strftime("%B") + " " + str(pub_date.year)
    except:
        print('Article publication date not found.')
        month_yr = ""
        
    return text, title, month_yr