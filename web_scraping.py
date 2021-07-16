# import statements
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd

# URL for AllSides media bias ratings data
bias_rating_url = "https://www.allsides.com/media-bias/media-bias-ratings"

if __name__ == "__main__":
    # webscrape the sources name and bias from AllSides
    bias_page = requests.get(bias_rating_url)
    bias_soup = BeautifulSoup(bias_page.content, 'html.parser')

    content = bias_soup.find_all(id = 'content')[0]
    table = content.find_all(class_ = 'views-table cols-4')[0]
    source_titles = table.tbody.find_all(class_ = 'views-field views-field-title source-title')
    source_biases = table.tbody.find_all(class_ = 'views-field views-field-field-bias-image')

    news_sources = []
    news_biases = []
    news_homepage = []
    for i in range(len(source_titles)):
        source_name = source_titles[i].a.text
        # common sources
        if (source_name == 'CBS News (Online)') or (
            source_name == "CNN (Opinion)") or (
            source_name == 'Fox News (Opinion)') or (
            source_name == 'AP Politics & Fact Check') or (
            source_name == 'New York Times (Opinion)') or (
            source_name == 'NPR (Opinion)') or (
            source_name == 'Wall Street Journal (Opinion)'):
            continue
        news_sources.append(source_name)
        news_url = next(search(source_titles[i].a.text))
        news_url =  re.findall(r"^.+\.com|^.+\.org|^.+\.uk", news_url)[0]
        news_homepage.append(news_url)
        news_biases.append(source_biases[i].a.get('href').strip('/media-bias/').split("-")[0])


    # put the news sources and biases in a dataframe: data
    data = pd.DataFrame(data = {'source': news_sources, 'website': news_homepage, 'bias': news_biases})
    data.to_csv('./data/bias_df.csv')