import streamlit as st
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Initializing variables
my_user = None
my_pass = None
search_item = None
all_tweets = []
driver = None

# Function to clean tweets
def tweetCleaning(tweet):
    cleanTweet = re.sub(r"@[a-zA-Z0-9]+", "", tweet)
    cleanTweet = re.sub(r"#[a-zA-Z0-9\s]+", "", cleanTweet)
    cleanTweet = ' '.join(word for word in cleanTweet.split() if word not in stopwords.words('english'))
    return cleanTweet

# Function to calculate polarity
def calPolarity(tweet):
    return TextBlob(tweet).sentiment.polarity

# Function to calculate subjectivity
def calSubjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity

# Function to determine segmentation
def segmentation(tweet):
    if tweet > 0:
        return "positive"
    elif tweet == 0:
        return "neutral"
    else:
        return "negative"

# Function to perform search
def performSearch():
    global all_tweets, driver

    # Initializing the Chrome driver
    driver = webdriver.Chrome(PATH)
    driver.maximize_window()

    # Openning the Twitter login page
    driver.get("https://twitter.com")
    
    # Logging in
    log_in = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, '//span[text()="Log in"]')))
    log_in.click()
    
    username_input = WebDriverWait(driver, 120).until(EC.visibility_of_element_located((By.XPATH, '//input[@name="text"]')))
    username_input.send_keys(my_user)
    username_input.send_keys(Keys.ENTER)
   
    password_input = WebDriverWait(driver, 120).until(EC.visibility_of_element_located((By.XPATH, '//input[@name="password"]')))
    password_input.send_keys(my_pass)
    password_input.send_keys(Keys.ENTER)
    
    search_box = WebDriverWait(driver, 240).until(EC.visibility_of_element_located((By.XPATH, '//input[@data-testid="SearchBox_Search_Input"]')))
    search_box.send_keys(search_item)
    search_box.send_keys(Keys.ENTER)
    
    # Getting the tweets
    tweets = driver.find_elements(By.XPATH,"//div[@data-testid='tweetText']")
    while True:
        for tweet in tweets:
            all_tweets.append(tweet.text)
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        sleep(3)
        tweets = driver.find_elements(By.XPATH,"//div[@data-testid='tweetText']")
        if len(all_tweets)>20:
            break

    # Closing the driver
    if driver:
        driver.quit()

# Setting up the Path
PATH = "C:\\Users\\levib\\Downloads\\chromedriver.exe"

# Main app
st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide")
st.title("Twitter Sentiment Analysis")
st.subheader("Collect tweets, analyze sentiment, and visualize the results")

# Customizing the sidebar
st.sidebar.title("Configure Twitter Search")
st.sidebar.subheader("Twitter Credentials")
my_user = st.sidebar.text_input("Enter your Twitter username")
my_pass = st.sidebar.text_input("Enter your Twitter password", type="password")
st.sidebar.subheader("Search Term")
search_item = st.sidebar.text_input("Enter the search term (e.g., product, brand, etc.)")
st.sidebar.info("Please note that providing your Twitter credentials is required to access the Twitter search API.")

if st.sidebar.button("Perform Search"):
    try:
        # Search and collect tweets
        performSearch()

        if len(all_tweets) > 0:
            # Cleaning the tweets
            cleaned_tweets = [tweetCleaning(tweet) for tweet in all_tweets]

            # Calculating polarity, subjectivity, and segmentation for each tweet
            polarities = [calPolarity(tweet) for tweet in cleaned_tweets]
            subjectivities = [calSubjectivity(tweet) for tweet in cleaned_tweets]
            segmentations = [segmentation(polarity) for polarity in polarities]

            # A DataFrame for the tweet data
            df = pd.DataFrame({
                "tweets": all_tweets,
                "tPolarity": polarities,
                "tSubjectivity": subjectivities,
                "segmentation": segmentations
            })

            # Customizing the interface
            st.subheader("Collected Tweets")
            st.dataframe(df.head().style.set_properties(**{'background-color': 'lightyellow', 'color': 'black'}))

            st.subheader("Count of Tweets by Segmentation")
            plt.figure(figsize=(8, 6))
            sns.countplot(data=df, x="segmentation", palette=["lightgreen", "lightblue", "lightcoral"])
            plt.xlabel("Segmentation")
            plt.ylabel("Count")
            plt.title("Count of Tweets by Segmentation")
            st.pyplot(plt)

            st.subheader("Top 3 Most Positive Tweets")
            st.dataframe(df.sort_values(by=["tPolarity"], ascending=False).head(3).style.set_properties(**{'background-color': 'lightgreen', 'color': 'black'}))

            st.subheader("Top 3 Most Negative Tweets")
            st.dataframe(df.sort_values(by=["tPolarity"], ascending=True).head(3).style.set_properties(**{'background-color': 'lightcoral', 'color': 'black'}))

            st.subheader("Neutral Tweets")
            st.dataframe(df[df["tPolarity"] == 0].style.set_properties(**{'background-color': 'lightblue', 'color': 'black'}))

            st.subheader("Count of Tweets by Segmentation (Pivot Table)")
            st.dataframe(df.pivot_table(index=["segmentation"], aggfunc={"segmentation": "count"}).style.set_properties(**{'background-color': 'lightyellow', 'color': 'black'}))
        else:
            st.info("No tweets found for the given search term.")
    except Exception as e:
        st.error(f"An error occurred during the search: {str(e)}")

