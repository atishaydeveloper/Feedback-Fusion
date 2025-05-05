import time
import re  # For regular expressions
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import googleapiclient.discovery
from pymongo import MongoClient
from urllib.parse import urlparse, parse_qs
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from wordcloud import WordCloud
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random

options = Options()
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument('--headless')

driver = webdriver.Chrome(options=options)

# NLTK dependencies
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize sentiment analyzer and other tools
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["feedback_fusion"]

def main_page():
     # Title of the page
    # Title of the page with custom color
    

    # Hero Section Title with color
    # Title with subtle emphasis
    st.markdown("<h1 style='color: #00FFFF;'>Feedback-Fusion: Empowering Smarter Decisions with Feedback & Research Intelligence</h1>", unsafe_allow_html=True)

    # Vision
    st.markdown("<h2 style='color: #00FFFF;'>🌟 Project Vision</h2>", unsafe_allow_html=True)
    st.write("""
    **Feedback-Fusion** is built to redefine how individuals and organizations leverage public opinion and market research. 

    By blending AI with natural language understanding, the platform delivers **actionable insights**, **concise summaries**, and **intelligent guidance** — for everything from **business strategies** to **personal tech purchases**.
    """)

    # Features
    st.markdown("<h2 style='color: #00FFFF;'>🚀 Key Features & Services</h2>", unsafe_allow_html=True)
    st.write("""
    **1. Feedback Intelligence Suite**
    - 🎥 **YouTube Feedback Analysis** – Understand audience sentiment from video comments.
    - 🛒 **Amazon Review Mining** – Uncover customer experiences and satisfaction trends.
    - 🏢 **Company Review Insights** – Analyze internal and external sentiment for businesses.
    - 🧠 **Sentiment Classification** – Automatically detect and tag feedback tone: Positive, Negative, or Neutral.
    - ☁️ **Word Cloud Generator** – Spot keywords and recurring themes visually.
    - ✂️ **AI-Powered Summarization** – Get instant summaries from large text data.
    - ➕➖ **Pros & Cons Extractor** – Break down reviews into strengths and weaknesses.
    - ❓ **FAQ Generator** – Generate smart FAQs based on real user concerns.

    **2. Smart Electronics Research Assistant**
    - 🔍 Structured product research & comparisons.
    - 📊 Feature breakdowns, reviews, and pricing insights.
    - 📚 Personalized buying guides for different user needs: budget, performance, premium.

    """)

    # Impact
    st.markdown("<h2 style='color: #00FFFF;'>🌐 Why Feedback-Fusion Matters</h2>", unsafe_allow_html=True)
    st.write("""
    **Feedback-Fusion** isn’t just a tool — it's a strategic assistant.

    - 📈 **Business Edge** – Make data-driven decisions based on real sentiment.
    - 🧑‍🤝‍🧑 **Improved Experience** – Decode what your users and employees *really* think.
    - 🧾 **Smarter Tech Buys** – Save hours of research and avoid buyer's remorse.
    - ⏳ **Time Optimization** – Skip the noise, focus on actionable insights.

    From raw feedback to refined intelligence — we help you close the loop and move with clarity.
    """)


    st.sidebar.markdown("<h2 style='color: #00FFFF;'>Explore Our Services", unsafe_allow_html=True)
    st.sidebar.write("Choose a service to begin analyzing feedback:")
    
    
    # Button for YouTube Video
    if st.sidebar.button("YouTube Video"):
        st.session_state.page = "youtube"
        st.rerun()


    # Button for Amazon Product
    if st.sidebar.button("Amazon Product Reviews"):
        st.session_state.page = "amazon"
        st.rerun()



    # Button for Company Review
    if st.sidebar.button("Company Review"):
        st.session_state.page = "company"
        st.rerun()
        
    if st.sidebar.button("Smart Electronics Research Assistant"):
        st.session_state.page = "electronics"
        st.rerun()
        
    if st.sidebar.button("Amazon Product Info"):
        st.session_state.page = "productInfo"
        st.rerun()


    

    # Project Contributors
    # Project Contributors with custom color
    st.markdown("<h2 style='color: #00FFFF;'>Contributors</h2>", unsafe_allow_html=True)
    st.write("""
    **Feedback-Fusion** is the result of the hard work and dedication of the following contributors from the **CSDS 3rd Year, Acropolis Institute of Technology and Research, Batch 2026**:
    - **Atishay Jain** (Project Lead)
    - **Sarthak Doshi**
    - **Om Chouksey**
    - **Shambhavi Bhadoria**
    
    We thank each contributor for their valuable input, collaboration, and commitment to making this project a success.
    """)
    
    # Contact Information with custom color
    st.markdown("<h2 style='color: #00FFFF;'>Contact Us</h2>", unsafe_allow_html=True)
    st.write("""
    For more information, feedback, or collaboration inquiries, please contact us at:
    - **Email**: atishayj288@gmail.com
    - **Phone**: +91 9522041334
    """)

    # Footer Section with custom color
    st.markdown("<h3 style='color: #FFFFFF;'>Feedback-Fusion | All rights reserved © 2024</h3>", unsafe_allow_html=True)
    
def youtube_page():
    # Define the YouTube API key here (predefined)
    api_key = "AIzaSyC_bHB2AEhn2nnmctEYpi_wKb8rofQyChU"  # Replace with your actual API key

    # Function to extract comments using YouTube Data API
    def extract_comments(video_id, api_key):
        youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
        comments = []

        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100
            )
            response = request.execute()

            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

            while "nextPageToken" in response:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=response["nextPageToken"],
                    maxResults=100
                )
                response = request.execute()

                for item in response.get("items", []):
                    comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    comments.append(comment)

        except Exception as e:
            print(f"Error extracting comments: {e}")
        
        return comments

    # Function to process comments and perform sentiment analysis
    def process_comments(comments):
        results = []
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for comment in comments:
            sentiment_score = sia.polarity_scores(comment)
            sentiment = (
                "Positive" if sentiment_score['compound'] > 0.05 else
                "Negative" if sentiment_score['compound'] < -0.05 else
                "Neutral"
            )
            results.append({
                "comment": comment,
                "sentiment": sentiment,
                "scores": sentiment_score
            })
            sentiment_counts[sentiment] += 1

        return results, sentiment_counts

    def preprocess_results(review):
        text = review.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        words = [word for word in words if word not in stop_words]
        words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(words)

    def generate_wordcloud(text, title="Word Cloud"):
        """
        Generates a word cloud from the given text.

        Args:
            text (str): The text to generate the word cloud from.
            title (str): The title of the word cloud.

        Returns:
            matplotlib.pyplot: The word cloud image.
        """
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(title)
        return plt


    # Function to store results in MongoDB (creates a new collection based on video ID)
    def store_results(results, video_id):
        collection_name = f"comments_{video_id}"  # Dynamic collection name based on video ID
        collection = db[collection_name]  # Create or access collection
        collection.insert_many(results)

    # Function to extract video ID from URL
    def extract_video_id(url):
        parsed_url = urlparse(url)
        video_id = parse_qs(parsed_url.query).get("v")
        return video_id[0] if video_id else None

    def summarize_comments(reviews):
        """
        Summarizes a list of reviews using the Gemini API.

        Args:
            reviews (list): List of strings containing user reviews.

        Returns:
            str: Summarized text of the reviews.
        """
        # Combine all reviews into a single text
        combined_text = " ".join(reviews)

        # Truncate text if it exceeds API limits
        if len(combined_text) > 3000:
            combined_text = combined_text[:3000]

        # Configure Gemini API with your API key
        genai.configure(api_key="AIzaSyCYAkTZfIJ2eUWBAHczQ4qaK5HbFCvzjUc")

        try:
            # Use Gemini to generate a summary
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(f"Summarize all the following reviews in a concise, single paragraph which summarizes over all reviews into one statement:\n{combined_text}")

            # Return the summary
            return response.text if response.text else "Error: No response from the model."
        
        except Exception as e:
            print(f"Error in summarization: {e}")
            return "Error in generating summary. Please check your API key and input format."
        

    #-----------------------------------------------------UI-------------------------------------------------------------

    st.markdown("<h2 style='color: #00FFFF;'>YouTube Video Feedback Analyzer",unsafe_allow_html=True)
    st.write("This page will allow you to analyze feedback for YouTube videos.")  

    st.sidebar.markdown("<h2 style='color: #00FFFF;'>Feedback-Fusion: AI-Powered Feedback Analysis and Sentiment Processing System",unsafe_allow_html=True)  # Sidebar header

    # Add widgets to the sidebar:

    video_url = st.sidebar.text_input("Enter YouTube Video URL:")

    selected_sentiment = st.sidebar.radio(
                "Select Sentiment:",
                options=["Positive", "Negative", "Neutral"],
                index=0,  # Default selection (e.g., "Positive")
            )

    if video_url:
        video_id = extract_video_id(video_url)
        
        if video_id:
            comments = extract_comments(video_id, api_key)
                
            if comments:
                processed_comments, sentiment_counts = process_comments(comments)
                store_results(processed_comments, video_id)  # Store results in MongoDB
                
                positive_reviews = [preprocess_results(item["comment"]) for item in processed_comments if item["sentiment"].lower() == "positive"]
                negative_reviews = [preprocess_results(item["comment"]) for item in processed_comments if item["sentiment"].lower() == "negative"]
                neutral_reviews = [preprocess_results(item["comment"]) for item in processed_comments if item["sentiment"].lower() == "neutral"]
                
                positive_text = " ".join(positive_reviews)
                negative_text = " ".join(negative_reviews)
                neutral_text = " ".join(neutral_reviews)
                
                total_comments = len(comments)
                st.sidebar.metric("Total Comments", total_comments)
                
                # Create sentiment distribution (Bar chart)
                st.markdown("<h2 style='color: #00FFFF;'>Sentiment Distribution Graph",unsafe_allow_html=True)
                sentiment_counts = pd.DataFrame(processed_comments)["sentiment"].value_counts()
                sentiment_fig = px.bar(
                    sentiment_counts, 
                    x=sentiment_counts.index, 
                    y=sentiment_counts.values, 
                    labels={"x": "Sentiment", "y": "Number of Comments"},
                    color=sentiment_counts.index
                )
                st.plotly_chart(sentiment_fig)
                
                if positive_text:
                    positive_wordcloud = generate_wordcloud(positive_text, title="Positive Word Cloud")
                    st.subheader("Word Cloud for Positive Reviews")
                    st.pyplot(positive_wordcloud)
                    
                else:
                    st.subheader("Word Cloud for Positive Reviews")
                    st.write("No positive reviews to generate a word cloud from.")

                if negative_text:
                    negative_wordcloud = generate_wordcloud(negative_text, title="Negative Word Cloud")
                    st.subheader("Word Cloud for Negative Reviews")
                    st.pyplot(negative_wordcloud)
                    
                else:
                    st.subheader("Word Cloud for Negative Reviews")
                    st.write("No negative reviews to generate a word cloud from.")

                if neutral_text:
                    neutral_wordcloud = generate_wordcloud(neutral_text, title="Neutral Word Cloud")
                    st.subheader("Word Cloud for Neutral Reviews")
                    st.pyplot(neutral_wordcloud)
                    
                else:
                    st.subheader("Word Cloud for Neutral Reviews")
                    st.write("No neutral reviews to generate a word cloud from.")
                
                filtered_reviews = [item for item in processed_comments if item["sentiment"].lower() == selected_sentiment.lower()]
            
                st.subheader(f"Comments ({selected_sentiment}):")
                if filtered_reviews:
                    df = pd.DataFrame(filtered_reviews)  # Convert list of dictionaries to a DataFrame
                    df.columns = ["Comments","sentiment","score","id"]  # Rename the first column to "Comment"
                    df.index = range(1, len(df) + 1)
                    st.dataframe(df["Comments"])  # Display as a DataFrame (interactive table)
                    # OR
                    # st.table(df) # If you want a static table
                else:
                    st.write("No reviews found for the selected sentiment.")
                    
                    
                st.subheader("Summary of Comments")
                
                
                summary = summarize_comments(comments)
                st.write(summary)
                
            else:
                st.warning("No comments found.")
        else:
            st.warning("Invalid YouTube video URL.")
            
    if st.sidebar.button("Return to Main Page"):
        st.session_state["page"] = "main"
        st.rerun()  
        
# Amazon Product Page
def amazon_page():
    def get_amazon_reviews_selenium(product_url):
        """
        Extracts reviews from an Amazon product page using Selenium.

        Args:
            product_url (str): The URL of the Amazon product page.

        Returns:
            tuple: (product_name, list of review texts)
        """
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Run in headless mode (no GUI)
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")  # Add a user agent
        driver = webdriver.Chrome(options=options)

        try:
            driver.get(product_url)
            product_name_element = driver.find_element(By.ID, "productTitle")
            product_name = product_name_element.text.strip()

            # Wait for the review list to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "cm-cr-dp-review-list"))
            )

            reviews = []
            review_elements = driver.find_elements(By.CSS_SELECTOR, "li.review.aok-relative")

            for review_element in review_elements:
                try:
                    # Extract review title
                    title_element = review_element.find_element(By.CSS_SELECTOR, "a[data-hook='review-title']")
                    review_title = title_element.text.strip()

                    # Extract review date
                    date_element = review_element.find_element(By.CSS_SELECTOR, "span[data-hook='review-date']")
                    review_date = date_element.text.strip()

                    # Extract review body (handle "Read More")
                    try:
                        # Check if "Read more" link exists and click it
                        read_more_link = review_element.find_element(By.CSS_SELECTOR, "a[data-hook='expand-collapse-read-more-less']")
                        driver.execute_script("arguments[0].click();", read_more_link)
                        time.sleep(0.5)  # Wait for content to expand
                    except:
                        pass #if link doesn't exist pass it

                    body_element = review_element.find_element(By.CSS_SELECTOR, "span[data-hook='review-body']")
                    review_text = body_element.text.strip()

                    # reviews.append({
                    #     "title": review_title,
                    #     "date": review_date,
                    #     "review": review_text
                    # })
                    
                    reviews.append(review_text)

                except Exception as e:
                    print(f"Error extracting review: {e}")

            return product_name, reviews

        except Exception as e:
            print(f"Error: {e}")
            return None, None

        finally:
            driver.quit()

    # Function to process reviews and perform sentiment analysis
    def process_reviews(reviews):
        results = []
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

        for review in reviews:
            sentiment_score = sia.polarity_scores(review)
            sentiment = (
                "Positive" if sentiment_score['compound'] > 0.05 else
                "Negative" if sentiment_score['compound'] < -0.05 else
                "Neutral"
            )
            results.append({
                "review": review,
                "sentiment": sentiment,
                "scores": sentiment_score
            })

            sentiment_counts[sentiment] += 1

        return results, sentiment_counts

    import re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk.download('stopwords')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_results(review):
        text = review.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        words = [word for word in words if word not in stop_words]
        words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(words)


    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    def generate_wordcloud(text, title="Word Cloud"):
        """
        Generates a word cloud from the given text.

        Args:
            text (str): The text to generate the word cloud from.
            title (str): The title of the word cloud.

        Returns:
            matplotlib.pyplot: The word cloud image.
        """
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(title)
        return plt

    def generate_tfidf_keywords(feedbacks):
        """
        Generates a DataFrame of top keywords based on TF-IDF analysis.

        Args:
            feedbacks (list): List of strings containing user reviews.

        Returns:
            pandas.DataFrame: DataFrame containing top keywords and their TF-IDF scores.
        """
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
        tfidf_matrix = vectorizer.fit_transform(feedbacks)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        keyword_df = pd.DataFrame(list(zip(feature_names, tfidf_scores)), columns=["Keyword", "TF-IDF Score"])
        keyword_df = keyword_df.sort_values(by="TF-IDF Score", ascending=False)
        return keyword_df


    # Function to store results in MongoDB (creates a new collection based on product name)
    def store_results(results, product_name):
        collection_name = f"reviews_{product_name.replace(' ', '_')}"  # Dynamic collection name based on product name
        collection = db[collection_name]  # Create or access collection
        collection.insert_many(results)
        



    def summarize_reviews_enhanced(reviews, num_pros=3, num_cons=3):
        """
        Summarizes reviews using Gemini API to identify key pros and cons.

        Args:
            reviews (list): List of review strings.
            num_pros (int): Number of pros to extract.
            num_cons (int): Number of cons to extract.

        Returns:
            dict: Dictionary containing summary, pros, and cons.
        """
        combined_text = " ".join(reviews)
        if len(combined_text) > 3000:
            combined_text = combined_text[:3000]

        genai.configure(api_key="AIzaSyCoB8kXfj4IPVxqYy57EW5RDOLWsI0BpXQ") #configgemini AI
        model = genai.GenerativeModel("gemini-2.0-flash")

        try:
            # Get Summary
            summary_prompt = f"Summarize the following reviews in a single paragraph:\n{combined_text}"
            summary_response = model.generate_content(summary_prompt)
            summary = summary_response.text if summary_response.text else "Summary generation failed."

            # Extract Pros and Cons
            pros_cons_prompt = f"""
            Based on the following reviews, identify the top {num_pros} pros and top {num_cons} cons.
            Present your response as a Markdown table with two columns: "Pros" and "Cons".
            {combined_text}

            Table Format:
            | Pros | Cons |
            |---|---|
            | - Pro 1 | - Con 1 |
            | - Pro 2 | - Con 2 |
            | - Pro 3 | - Con 3 |
            """
            
            pros_cons_response = model.generate_content(pros_cons_prompt)
            pros_cons_table = pros_cons_response.text if pros_cons_response.text else "Pros/Cons extraction failed."

            # Split into Pros and Cons
            # pros = []
            # cons = []
            # parts = pros_cons_text.split("Cons:")
            # if len(parts) == 2:
            #     pros_str = parts[0].replace("Pros:", "").strip()
            #     cons_str = parts[1].strip()

            #     pros = [p.strip() for p in pros_str.split("-") if p.strip()]
            #     cons = [c.strip() for c in cons_str.split("-") if c.strip()]

            return {"summary": summary, "table": pros_cons_table}

        except Exception as e:
            return {"summary": f"Error: {e}", "table": "Error generating table."}


    # Function to summarize reviews using Hugging Face summarization
    # def summarize_reviews(reviews):
    #     """
    #     Summarizes a list of reviews using the Gemini API.

    #     Args:
    #         reviews (list): List of strings containing user reviews.

    #     Returns:
    #         str: Summarized text of the reviews.
    #     """
    #     # Combine all reviews into a single text
    #     combined_text = " ".join(reviews)

    #     # Truncate text if it exceeds API limits
    #     if len(combined_text) > 3000:
    #         combined_text = combined_text[:3000]

    #     # Configure Gemini API with your API key
    #     genai.configure(api_key="AIzaSyCYAkTZfIJ2eUWBAHczQ4qaK5HbFCvzjUc")

    #     try:
    #         # Use Gemini to generate a summary
    #         model = genai.GenerativeModel("gemini-pro")
    #         response = model.generate_content(f"Summarize all the following reviews in a concise, single paragraph which summarizes over all reviews into one statement:\n{combined_text}")

    #         # Return the summary
    #         return response.text if response.text else "Error: No response from the model."
        
    #     except Exception as e:
    #         print(f"Error in summarization: {e}")
    #         return "Error in generating summary. Please check your API key and input format."
        
        
    def generate_ideal_for_recommendation(sentiment_counts, tfidf_keywords, processed_reviews, reviews, product_name="This Product"):
        """
        Generates an "ideal for" recommendation using the Gemini API based on feedback analysis.
        """

        positive_percentage = (sentiment_counts['Positive'] / sum(sentiment_counts.values())) * 100
        negative_percentage = (sentiment_counts['Negative'] / sum(sentiment_counts.values())) * 100

        # Get snippets of representative positive and negative reviews
        positive_reviews_snippets = [item['review'] for item in processed_reviews if item['sentiment'] == 'Positive'][:2]
        negative_reviews_snippets = [item['review'] for item in processed_reviews if item['sentiment'] == 'Negative'][:2]

        # Format the top keywords for the prompt
        top_keywords_str = ", ".join(tfidf_keywords['Keyword'].tolist()[:5])

        prompt = f"""
        You are an expert product recommender. Given the following customer reviews for {product_name}, identify the ideal user for this product:

        Overall Sentiment:
        - Positive: {positive_percentage:.2f}%
        - Negative: {negative_percentage:.2f}%

        Key Topics Discussed: {top_keywords_str}

        Representative Positive Reviews:
        {chr(10).join(positive_reviews_snippets) or "No positive reviews."}

        Representative Negative Reviews:
        {chr(10).join(negative_reviews_snippets) or "No negative reviews."}

        Task: Based on this information, describe the type of person who would be most satisfied with this product.  Consider their needs, priorities, and potential drawbacks of the product. The format should be. "This product is ideal for". Explain why.

        """

        # Configure Gemini API with your API key
        genai.configure(api_key="AIzaSyCoB8kXfj4IPVxqYy57EW5RDOLWsI0BpXQ")

        try:
            # Use Gemini to generate a recommendation
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            return response.text if response.text else "I am unable to give you the information"

        except Exception as e:
            return f"Error generating buy recommendation: {e}"
        
        
        
    def generate_buy_recommendation(sentiment_counts, tfidf_keywords, processed_reviews, reviews, product_name="This Product"):
        """
        Generates a buy recommendation using the Gemini API based on feedback analysis.

        Args:
            sentiment_counts (dict): Dictionary of sentiment counts (Positive, Negative, Neutral).
            tfidf_keywords (pandas.DataFrame): DataFrame of top keywords from TF-IDF analysis.
            processed_reviews (list): List of processed reviews (review, sentiment, scores).
            reviews (list): List of original review texts.
            product_name (str): The name of the product being reviewed.

        Returns:
            str: The generated buy recommendation and reasoning from Gemini API.
        """

        positive_percentage = (sentiment_counts['Positive'] / sum(sentiment_counts.values())) * 100
        negative_percentage = (sentiment_counts['Negative'] / sum(sentiment_counts.values())) * 100
        neutral_percentage = (sentiment_counts['Neutral'] / sum(sentiment_counts.values())) * 100

        # Get snippets of representative positive and negative reviews
        positive_reviews_snippets = [item['review'] for item in processed_reviews if item['sentiment'] == 'Positive'][:2]
        negative_reviews_snippets = [item['review'] for item in processed_reviews if item['sentiment'] == 'Negative'][:2]

        # Format the top keywords for the prompt
        top_keywords_str = ", ".join(tfidf_keywords['Keyword'].tolist()[:5]) #limiting the size of the keyword list

        # Construct the prompt
        prompt = f"""You are an expert product reviewer. Analyze customer reviews of {product_name} and provide a clear, concise recommendation.
        Overall Sentiment: Positive: {positive_percentage:.2f}%, Negative: {negative_percentage:.2f}%, Neutral: {neutral_percentage:.2f}%
        Key Topics: {top_keywords_str}
        Representative Positive Reviews: {positive_reviews_snippets}
        Representative Negative Reviews: {negative_reviews_snippets}
        Task: Based on this, should someone buy this product? Answer with 'Yes' or 'No' followed by a short explanation."""

        # Configure Gemini API with your API key
        genai.configure(api_key="AIzaSyCoB8kXfj4IPVxqYy57EW5RDOLWsI0BpXQ")

        try:
            # Use Gemini to generate a recommendation
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            return response.text if response.text else "I am unable to give you the information"

        except Exception as e:
            return f"Error generating buy recommendation: {e}"
        
        
    def identify_faq_questions(reviews, num_questions=5):
        """
        Identifies common questions asked or implied in the reviews using the Gemini API.

        Args:
            reviews (list): List of review strings.
            num_questions (int): The number of questions to identify.

        Returns:
            list: A list of identified questions.
        """
        combined_text = " ".join(reviews)
        if len(combined_text) > 3000:
            combined_text = combined_text[:3000]

        genai.configure(api_key="AIzaSyCoB8kXfj4IPVxqYy57EW5RDOLWsI0BpXQ")
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""
        Identify the {num_questions} most common questions that customers are asking or implying in the following product reviews.
        Return each question on a new line. Focus on questions about product features, performance, usability, or compatibility.
        Reviews:
        {combined_text}
        """

        try:
            response = model.generate_content(prompt)
            questions = [q.strip() for q in response.text.split('\n') if q.strip()]
            return questions

        except Exception as e:
            print(f"Error identifying questions: {e}")
            return []
        
        
        

    def generate_faq_answers(question, reviews):
        """
        Generates an answer to a question based on the product reviews using the Gemini API.

        Args:
            question (str): The question to answer.
            reviews (list): List of review strings.

        Returns:
            str: The generated answer.
        """
        combined_text = " ".join(reviews)
        if len(combined_text) > 3000:
            combined_text = combined_text[:3000]

        genai.configure(api_key="AIzaSyCoB8kXfj4IPVxqYy57EW5RDOLWsI0BpXQ")
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""
        Answer the following question based on the information in the following product reviews. Be concise and factual.
        Question: {question}
        Reviews: {combined_text}
        """

        try:
            response = model.generate_content(prompt)
            answer = response.text.strip() if response.text else "No answer found in the reviews."
            return answer

        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Error generating answer."

        
    #-----------------------------------------------------UI-------------------------------------------------------------
        
    st.markdown("<h2 style='color: #00FFFF;'>Amazon Product Feedback Analyzer",unsafe_allow_html=True)
    st.write("This page will allow you to analyze reviews for Amazon products.")    


    st.sidebar.markdown("<h2 style='color: #00FFFF;'>Feedback-Fusion: AI-Powered Feedback Analysis and Sentiment Processing System",unsafe_allow_html=True)  # Sidebar header

    # Add widgets to the sidebar:
    st.sidebar.subheader("Enter the Amazon product URL below to analyze reviews:")
    product_url = st.sidebar.text_input("Product URL")

    selected_sentiment = st.sidebar.radio(
                "Select Sentiment:",
                options=["Positive", "Negative", "Neutral"],
                index=0,  # Default selection (e.g., "Positive")
            )

    # Main content area:
    if product_url:
        st.write("Fetching reviews... Please wait.")
        product_name, reviews = get_amazon_reviews_selenium(product_url)

        if reviews:
            # Process the reviews
            processed_reviews, sentiment_counts = process_reviews(reviews)

            # Store the results in MongoDB
            store_results(processed_reviews, product_name)
            
            positive_reviews = [preprocess_results(item["review"]) for item in processed_reviews if item["sentiment"].lower() == "positive"]
            negative_reviews = [preprocess_results(item["review"]) for item in processed_reviews if item["sentiment"].lower() == "negative"]
            neutral_reviews = [preprocess_results(item["review"]) for item in processed_reviews if item["sentiment"].lower() == "neutral"]
            
            positive_text = " ".join(positive_reviews)
            negative_text = " ".join(negative_reviews)
            neutral_text = " ".join(neutral_reviews)
            
            

            # Display results
            st.subheader(f"Sentiment Analysis Results for {product_name}")
            
            
            st.sidebar.metric("Total Comments", len(reviews))
            # Sentiment Distribution Visualization (Bar chart)
            
            
            
            st.subheader("Sentiment Distribution")
            sentiment_df = pd.DataFrame(list(sentiment_counts.items()), columns=['Sentiment', 'Count'])
            fig = px.bar(sentiment_df, x='Sentiment', y='Count', title="Sentiment Distribution of Reviews", color='Sentiment')
            st.plotly_chart(fig)
            
            if positive_text:
                positive_wordcloud = generate_wordcloud(positive_text, title="Positive Word Cloud")
                st.subheader("Word Cloud for Positive Reviews")
                st.pyplot(positive_wordcloud)
                
            else:
                st.subheader("Word Cloud for Positive Reviews")
                st.write("No positive reviews to generate a word cloud from.")

            if negative_text:
                negative_wordcloud = generate_wordcloud(negative_text, title="Negative Word Cloud")
                st.subheader("Word Cloud for Negative Reviews")
                st.pyplot(negative_wordcloud)
                
            else:
                st.subheader("Word Cloud for Negative Reviews")
                st.write("No negative reviews to generate a word cloud from.")

            if neutral_text:
                neutral_wordcloud = generate_wordcloud(neutral_text, title="Neutral Word Cloud")
                st.subheader("Word Cloud for Neutral Reviews")
                st.pyplot(neutral_wordcloud)
                
            else:
                st.subheader("Word Cloud for Neutral Reviews")
                st.write("No neutral reviews to generate a word cloud from.")
            
            
            # Generate a word cloud
            # st.subheader("Word Cloud for Reviews")
            # wordcloud = generate_word_cloud(reviews)
            # st.image(wordcloud.to_array(), use_container_width=True)

            # Show first 10 reviews in an organized table
            
            
            
            filtered_reviews = [item["review"] for item in processed_reviews if item["sentiment"].lower() == selected_sentiment.lower()]
            
            st.subheader(f"Reviews ({selected_sentiment}):")
            if filtered_reviews:
                df = pd.DataFrame(filtered_reviews)  # Convert list of dictionaries to a DataFrame
                df.columns = ["Review"]
                df.index = range(1, len(df) + 1)
                st.dataframe(df)  # Display as a DataFrame (interactive table)
                # OR
                # st.table(df) # If you want a static table
            else:
                st.write("No reviews found for the selected sentiment.")
            
            
            revs = 0
            if len(reviews) > 10:
                revs = 10
            else:
                revs = len(reviews)
                
            # st.subheader(f"First {revs} Reviews")
            # reviews_df = pd.DataFrame(processed_reviews[:10])  # Display first 10 reviews
            # st.write(reviews_df[['review', 'sentiment']])
            enhanced_summary = summarize_reviews_enhanced(reviews)

            st.subheader("Enhanced Summary")
            st.write(f"Summary: {enhanced_summary['summary']}")

            st.subheader("Pros and Cons")
            st.markdown(enhanced_summary["table"])

            # Summarize the reviews
            # st.subheader("Summary of Reviews")
            # summary = summarize_reviews(reviews)
            # st.write(summary)
            
            tfidf_keywords = generate_tfidf_keywords(reviews)
            
            ideal_for = generate_ideal_for_recommendation(sentiment_counts, tfidf_keywords, processed_reviews, reviews)
            st.subheader("Ideal For:")
            st.write(ideal_for)
            
            buy_recommendation = generate_buy_recommendation(sentiment_counts, tfidf_keywords, processed_reviews, reviews)
            st.subheader("Buy Recommendation")
            st.write(buy_recommendation)
            
            # Identify FAQ Questions
            faq_questions = identify_faq_questions(reviews)

            st.subheader("Frequently Asked Questions")
            if faq_questions:
                for i, question in enumerate(faq_questions):
                    answer = generate_faq_answers(question, reviews)

                    with st.expander(f"Q{i+1}: {question}"):
                        st.write(f"**A:** {answer}")
            else:
                st.write("No common questions could be identified.")

        

        else:
            st.write("No reviews found for this product.")
            
    if st.sidebar.button("Return to Main Page"):
        st.session_state["page"] = "main"
        st.rerun()
            
def company_page():
    # Utility functions
    def load_feedback(file):
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.json'):
            df = pd.read_json(file)
        else:
            raise ValueError("File format not supported. Please upload a CSV or JSON file.")

        return df

    # Function to process feedback and perform sentiment analysis
    def process_feedback(feedbacks):
        results = []
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

        for feedback in feedbacks:
            processed_feedback = preprocess_results(feedback) #preprocess feedback using the preprocess_results function
            sentiment_score = sia.polarity_scores(processed_feedback) # perform sentiment analysis on preprocessed data
            sentiment = (
                "Positive" if sentiment_score['compound'] > 0.05 else
                "Negative" if sentiment_score['compound'] < -0.05 else
                "Neutral"
            )
            results.append({
                "feedback": feedback,
                "sentiment": sentiment,
                "scores": sentiment_score
            })

            sentiment_counts[sentiment] += 1

        return results, sentiment_counts

    def preprocess_results(review):
        text = review.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        words = [word for word in words if word not in stop_words]
        words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(words)

    def generate_wordcloud(text, title="Word Cloud"):
        """
        Generates a word cloud from the given text and displays it using Streamlit.

        Args:
            text (str): The text to generate the word cloud from.
            title (str): The title of the word cloud.

        Returns:
            None (displays the word cloud in Streamlit)
        """
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(title)

        # Display the word cloud in Streamlit
        st.pyplot(plt)

    # def store_results(results, product_name):
    #     collection_name = f"feedbacks_{product_name.replace(' ', '_')}"  # Dynamic collection name based on product name
    #     collection = db[collection_name]  # Create or access collection
    #     collection.insert_many(results)

    # Function to generate a TF-IDF visualization for keyword extraction
    def generate_tfidf_keywords(feedbacks):
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
        tfidf_matrix = vectorizer.fit_transform(feedbacks)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        keyword_df = pd.DataFrame(list(zip(feature_names, tfidf_scores)), columns=["Keyword", "TF-IDF Score"])
        keyword_df = keyword_df.sort_values(by="TF-IDF Score", ascending=False)
        return keyword_df

    def summarize_comments(reviews):
        """
        Summarizes a list of reviews using the Gemini API.

        Args:
            reviews (list): List of strings containing user reviews.

        Returns:
            str: Summarized text of the reviews.
        """
        # Combine all reviews into a single text
        combined_text = " ".join(reviews)

        # Truncate text if it exceeds API limits
        if len(combined_text) > 3000:
            combined_text = combined_text[:3000]

        # Configure Gemini API with your API key
        genai.configure(api_key="AIzaSyCYAkTZfIJ2eUWBAHczQ4qaK5HbFCvzjUc")  # Replace with your actual API key

        try:
            # Use Gemini to generate a summary
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(f"Summarize all the following reviews in a concise, single paragraph which summarizes over all reviews into one statement:\n{combined_text}")

            # Return the summary
            return response.text if response.text else "Error: No response from the model."
        
        except Exception as e:
            print(f"Error in summarization: {e}")
            return "Error in generating summary. Please check your API key and input format."
    #-----------------------------------------------------UI-------------------------------------------------------------
    # Streamlit UI

    st.markdown("<h2 style='color: #00FFFF;'>Company Review Analyzer",unsafe_allow_html=True)
    st.write("This page will allow you to analyze company reviews from employees and customers.")

    st.sidebar.markdown("<h2 style='color: #00FFFF;'>Feedback-Fusion: AI-Powered Feedback Analysis and Sentiment Processing System",unsafe_allow_html=True)

    st.subheader("Instructions for Uploading Feedback Files")
    st.write("""
        Please upload a CSV or JSON file containing feedback from both customers and employees.
        
        **File Format Requirements**:
        1. The file should have at least two columns:
            - **Feedback Column**: Contains the actual feedback text. You can name it `feedback`, `review`, or similar.
            - **Category Column**: Specifies whether the feedback is from a `Customer` or an `Employee`. You can name it `category`, `type`, or similar.
        
        2. Optional Columns:
            - **Rating**: A numeric rating (e.g., 1 to 5 or 1 to 10).
            - **Date**: The date when the feedback was given.
            - **Department**: For employee feedback, you may include a `department` column to specify which department the feedback is related to (e.g., `Sales`, `HR`).
        
        **Example CSV Format**:
        | feedback                                | category   | rating | date       |
        |-----------------------------------------|------------|--------|------------|
        | "Great place to work, very supportive." | Employee   | 5      | 2024-12-01 |
        | "Customer service was slow, not happy." | Customer   | 2      | 2024-12-02 |
        | "I love the products and the team."     | Employee   | 4      | 2024-12-03 |
        | "The product quality is excellent."     | Customer   | 5      | 2024-12-04 |
        | "Need better management."               | Employee   | 3      | 2024-12-05 |
        
        **Example JSON Format**:
        ```json
        [
        {"feedback": "Great place to work, very supportive.", "category": "Employee", "rating": 5, "date": "2024-12-01"},
        {"feedback": "Customer service was slow, not happy.", "category": "Customer", "rating": 2, "date": "2024-12-02"},
        {"feedback": "I love the products and the team."     | Employee   | 4      | 2024-12-03"},
        {"feedback": "The product quality is excellent."     | Customer   | 5      | 2024-12-04"},
        {"feedback": "Need better management."               | Employee   | 3      | 2024-12-05"}
        ]
        ```
    """)
    st.sidebar.subheader("Upload your feedback data (CSV or JSON file):")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or JSON file", type=["csv", "json"])

    if uploaded_file is not None:
        try:
            # Load the uploaded feedback file
            feedback_df = load_feedback(uploaded_file)
            
            # Display the first few rows of the dataset
            st.subheader("Uploaded Feedback Data")
            st.write(feedback_df.head())

            # Process the feedbacks (assuming the column containing feedback text is named 'feedback')
            feedbacks = feedback_df['feedback'].astype(str).tolist()

            # Process the feedback for sentiment analysis
            processed_feedback, sentiment_counts = process_feedback(feedbacks)
            
            total_feeds = len(processed_feedback)   
            st.sidebar.metric("Total Feeds", total_feeds)
            

            # Display sentiment analysis results
            st.markdown("<h2 style='color: #00FFFF;'>Sentiment Analysis Results",unsafe_allow_html=True)
            st.write(f"Positive: {sentiment_counts['Positive']}")
            st.write(f"Negative: {sentiment_counts['Negative']}")
            st.write(f"Neutral: {sentiment_counts['Neutral']}")

            # Generate a word cloud
            st.markdown("<h2 style='color: #00FFFF;'>Word Cloud for Feedback",unsafe_allow_html=True)
            all_feedbacks_text = " ".join(feedbacks)  # Combine all feedback into one string
            generate_wordcloud(all_feedbacks_text)

            # # Display TF-IDF keyword analysis
            # st.markdown("<h2 style='color: #00FFFF;'>Top Keywords Based on TF-IDF",unsafe_allow_html=True)
            # tfidf_keywords = generate_tfidf_keywords(feedbacks)
            # st.write(tfidf_keywords.head(10))  # Show top 10 keywords

            # Visualize sentiment distribution
            sentiment_df = pd.DataFrame(list(sentiment_counts.items()), columns=['Sentiment', 'Count'])
            fig = px.bar(sentiment_df, x='Sentiment', y='Count', title="Sentiment Distribution of Feedback", color='Sentiment')
            st.plotly_chart(fig)

            # Optionally: Separate feedback into Customer and Employee (assuming the file contains this information)
            if 'category' in feedback_df.columns:
                st.markdown("<h2 style='color: #00FFFF;'>Sentiment Analysis by Category",unsafe_allow_html=True)
                categories = feedback_df['category'].unique()
                for category in categories:
                    category_feedbacks = feedback_df[feedback_df['category'] == category]['feedback'].astype(str).tolist()
                    category_results, category_counts = process_feedback(category_feedbacks)
                    st.write(f"Sentiment Analysis for {category} Feedback")
                    st.write(f"Positive: {category_counts['Positive']}")
                    st.write(f"Negative: {category_counts['Negative']}")
                    st.write(f"Neutral: {category_counts['Neutral']}")
                    category_df = pd.DataFrame(list(category_counts.items()), columns=['Sentiment', 'Count'])
                    category_fig = px.bar(category_df, x='Sentiment', y='Count', title=f"Sentiment Distribution for {category}", color='Sentiment')
                    st.plotly_chart(category_fig)
            
            st.subheader("Summary of Comments")
                
                
            summary = summarize_comments(feedbacks)
            st.write(summary)
                
        except Exception as e:
            st.error(f"Error: {e}")
            

    if st.sidebar.button("Return to Main Page"):
        st.session_state["page"] = "main"
        st.rerun()
        
        
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv
import os        
        
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
serp_api_key = os.getenv("SERPAPI_API_KEY")


def electronics_page():
    # Utility functions
    def simple_calculator(x: str) -> str:
        try:
            result = eval(x)
            return str(result)
        except Exception as e:
            return str(e)

    # Tool: Web search using SerpAPI
    def search_google(query: str) -> str:
        serp = SerpAPIWrapper()
        return serp.run(query)

    # Define Tools
    calculator = Tool(
        name="Calculator",
        func=simple_calculator,
        description="A calculator for basic operations. Input: '2 + 2'."
    )

    web_search = Tool(
        name="Web Search",
        func=search_google,
        description="Search the web using Google. Input: 'Best smart TV in India 2024'."
    )

    tools = [calculator, web_search]
    
    # Agent for Researching
    class ResearcherAgent:
        def __init__(self):
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            self.agent = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False,
                handle_parsing_errors=True
            )

        def research_topic(self, topic):
            prompt = f"""Role:
                You are a dedicated research agent responsible solely for gathering accurate, up-to-date, and structured information about electronic appliances or devices. Your task is to perform focused research only on the device(s) or appliance(s) mentioned by the user.

                User Input Handling Rules:

                If the user specifies a particular product or brand (e.g., “Samsung QLED TV”), retrieve:

                Key specifications

                Unique features

                Customer reviews summary

                Pros & cons

                Pricing details from major marketplaces

                Known issues or limitations

                If the user mentions only a general category (e.g., “smart TV” or “Bluetooth headphones”) without naming a brand:

                Fetch a comparison of the top 3-5 latest and most popular devices in that category.

                Include a structured comparison table with:

                Brand & Model

                Key specifications

                Price range

                Unique selling points

                User rating (e.g., from Amazon, Flipkart, or other reliable sources)

                If the user input is ambiguous, request clarification or offer a set of relevant device categories to choose from.

                ⚠️ Scope Restriction:
                You are not allowed to write descriptive summaries or recommendations. Your only job is to collect and structure the research data. This data will be passed to another agent that will handle explanation, content generation, and final comparison writing.

                ✅ Final Output Format:

                Device/Category Name

                Top Models (if generic category)

                Structured Specifications

                Feature Highlights

                Comparative Table (if applicable)

                User Review Summary (key points only)

                Source Links (if required)
                
                Query entered by the user is {topic}. Reseach for this topic. 

    """
            return self.agent.run(prompt)
        
    # Agent for Writing Summary
    class WriterAgent:
        def __init__(self):
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            self.agent = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False,
                handle_parsing_errors=True
            )

        def write_article(self, research):
            prompt = f"""Role:
                        You are a technical writing agent specializing in crafting consumer-friendly, informative, and persuasive purchase guides for electronic appliances and gadgets. You are provided with structured research data from a Research Agent — your job is to analyze, summarize, and write a helpful comparison to guide the user's purchase decision.

                        Input Provided:

                        {research}

                        Research data containing specifications, comparisons, user review summaries, pricing info, and top models

                        Your Output Should Include:

                        Introductory Summary

                        Rephrase the user’s intent and explain what’s being compared or reviewed.

                        If a general category was asked (e.g., "best smart TVs"), briefly describe what to consider when buying that type of product.

                        Model Breakdown (if multiple options)

                        Provide clear, concise overviews of each top model.

                        Highlight pros, cons, and standout features for each.

                        Comparison Table (recommended)

                        Reuse or enhance the research agent’s comparison table for clarity.

                        Recommendation Section

                        Suggest the best options for different needs: budget buyers, performance seekers, premium users, etc.

                        Avoid personal bias — base everything on research and real reviews.

                        Verdict

                        Summarize with a short paragraph stating which device offers the best value or balance and why.

                        🧠 Key Guidelines:

                        Keep language clear, neutral, and informative.

                        Mention model names, prices, and features accurately.

                        Reference real user feedback when pointing out issues or praise.

                        Do not invent data or assume preferences not given by the user.

                        Avoid unnecessary fluff — the goal is to save user time and enable smart decision-making.
                        
            """
            return self.agent.run(prompt)
        
    #-----------------------------------------------------UI-------------------------------------------------------------
    # Streamlit UI
    st.title("🔍 Smart Electronics Research Assistant")

    user_query = st.text_input("Enter the device name or category you'd like to research:")

    if st.button("Research and Generate Guide"):
        if user_query:
            with st.spinner("🔍 Conducting research..."):
                researcher = ResearcherAgent()
                research_data = researcher.research_topic(user_query)
                st.subheader("📚 Research Data")
                st.text_area("Raw Research Output", research_data, height=250)

            with st.spinner("✍️ Writing user guide..."):
                writer = WriterAgent()
                article = writer.write_article(research_data)
                st.subheader("📝 Final Guide")
                st.write(article)
        else:
            st.warning("Please enter a topic before starting.")

    
            

    if st.sidebar.button("Return to Main Page"):
        st.session_state["page"] = "main"
        st.rerun()
        

def info():


    import streamlit as st
    import json
    import os
    from bs4 import BeautifulSoup
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain.vectorstores import FAISS
    from crawl4ai import AsyncWebCrawler
    from dotenv import load_dotenv
    from langchain.embeddings import HuggingFaceEmbeddings
    import asyncio
    import sys

    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())



    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    # Asynchronous scraping function
    async def scrap(url):
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url)
            return result.html

    # Parse & clean HTML content
    def beauty(scrapped_text):
        fixed_html = scrapped_text.encode().decode('unicode_escape')
        soup = BeautifulSoup(fixed_html, "html.parser")
        target_ids = ["ppd", "prodDetails"]

        for div_id in target_ids:
            div = soup.find("div", {"id": div_id})
            if div:
                content = div.get_text(separator="\n", strip=True)
                return preprocess(content)
        return "No relevant product information found."

    # Split text into chunks
    def preprocess(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(text)
        return chunks

    # Generate embeddings & search
    def embeds(chunks, query, product_url):
        if product_url in st.session_state.embedding_cache:
            db = st.session_state.embedding_cache[product_url]
        else:
            # embedder = GoogleGenerativeAIEmbeddings(
            #     model="models/text-embedding-004",
            #     google_api_key=GOOGLE_API_KEY
            # )
            db = FAISS.from_texts(chunks, embedding_model)
            st.session_state.embedding_cache[product_url] = db

        docs = db.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        return gemini_resp(context, query)

    # Store vectors and perform semantic search
    def vector_database(chunks, embedder, query):
        db = FAISS.from_texts(chunks, embedder)
        docs = db.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        return gemini_resp(context, query)

    # Generate answer using Gemini
    def gemini_resp(context, query):
        prompt = f"""Answer the following based on product information below:

    {context}

    Question: {query}
    Answer:"""

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY
        )
        result = llm.invoke(prompt)
        return result.content

    # Streamlit app UI
    if 'embedding_cache' not in st.session_state:
        st.session_state.embedding_cache = {}

    st.title("Amazon Product Info Extractor 🔍")
    product_url = st.text_input("Enter Amazon Product URL:")
    query = st.text_input("Ask something about the product:")

    if st.button("Extract Info"):
        if product_url:
            with st.spinner("Scraping and analyzing product..."):
                scrapped_html = asyncio.run(scrap(product_url))
                chunks = beauty(scrapped_html)
                if isinstance(chunks, list):
                    answer = embeds(chunks, query, product_url)

                    st.success("✅ Product Info Extracted:")
                    st.write(answer)
                else:
                    st.warning(chunks)
        else:
            st.error("Please enter a valid product URL.")

    if st.sidebar.button("Return to Main Page"):
        st.session_state["page"] = "main"
        st.rerun()

        
# Main execution block
if 'page' not in st.session_state:
    st.session_state.page = "main"

if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "youtube":
    youtube_page()
elif st.session_state.page == "amazon":
    amazon_page()
elif st.session_state.page == "company":
    company_page()
elif st.session_state.page == "electronics":
    electronics_page()
elif st.session_state.page == "productInfo":
    info()
