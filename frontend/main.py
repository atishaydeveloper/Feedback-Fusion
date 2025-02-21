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
    

    st.markdown("<h1 style='color: #00FFFF;'>Feedback-Fusion: Empowering Better Decisions with Feedback Analysis</h1>", unsafe_allow_html=True)
    
    # Vision of the Project with custom color
    st.markdown("<h2 style='color: #00FFFF;'>Project Vision</h2>", unsafe_allow_html=True)
    st.write("""
    The vision of **Feedback-Fusion** is to revolutionize how organizations and businesses understand and utilize feedback from their customers, employees, and other stakeholders. 
    By harnessing the power of sentiment analysis and feedback processing, our goal is to help companies gain deep insights into the mindset of their audiences, enabling them to make informed, data-driven decisions for continuous improvement.
    """)
    
    # Services provided by the project with custom color
    st.markdown("<h2 style='color: #00FFFF;'>Services We Provide</h2>", unsafe_allow_html=True)
    st.write("""
    **Feedback-Fusion** offers a range of services designed to analyze feedback in various forms, whether it's customer reviews, employee surveys, or product feedback:
    - **YouTube Video Feedback Analysis**: Analyze YouTube video comments to gain insights into viewer sentiment.
    - **Amazon Product Review Analysis**: Analyze product reviews from Amazon to gauge customer satisfaction and product performance.
    - **Company Review Analysis**: Analyze feedback from both employees and customers within your company, enabling you to improve workplace satisfaction and customer experience.
    - **Sentiment Analysis**: Automatically classify feedback as positive, negative, or neutral and generate insightful reports.
    - **Word Cloud Visualization**: Generate word clouds to visualize the most common terms in the feedback, helping to highlight key themes.
    - **Summarization**: Provide summaries of large sets of feedback for quick and easy understanding.
    """)
    
    # Impact of the Project with custom color
    st.markdown("<h2 style='color: #00FFFF;'>Impact of Feedback-Fusion</h2>", unsafe_allow_html=True)
    st.write("""
    The **Feedback-Fusion** platform is designed to have a transformative impact on businesses, organizations, and individuals:
    - **Improved Decision-Making**: By understanding the sentiments of customers, employees, and stakeholders, organizations can make more informed decisions.
    - **Better Customer Experience**: Analyzing customer feedback helps identify areas of improvement, allowing companies to enhance their offerings and services.
    - **Employee Satisfaction**: Analyzing employee feedback ensures that companies can maintain a positive work environment and improve employee engagement.
    - **Data-Driven Insights**: Feedback-Fusion empowers organizations to move from gut-feeling decisions to data-backed strategies, ultimately leading to better outcomes and growth.
    """)

    st.sidebar.markdown("<h2 style='color: #00FFFF;'>Explore Our Services", unsafe_allow_html=True)
    st.sidebar.write("Choose a service to begin analyzing feedback:")
    
    
    # Button for YouTube Video
    if st.sidebar.button("YouTube Video"):
        st.session_state.page = "youtube"
        st.rerun()


    # Button for Amazon Product
    if st.sidebar.button("Amazon Product"):
        st.session_state.page = "amazon"
        st.rerun()


    # Button for Company Review
    if st.sidebar.button("Company Review"):
        st.session_state.page = "company"
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
    # Set up Selenium WebDriver
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        
        # Path to chromedriver (make sure it's correct)
        service = Service(ChromeDriverManager().install())  # Update with your path
        
        # Initialize WebDriver
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get('https://www.amazon.com')
        try:
            # Open the product URL
            driver.get(product_url)
            time.sleep(5)  # Wait for dynamic content to load
            
            # Extract product name
            try:
                product_name = driver.find_element(By.ID, "productTitle").text.strip()
            except Exception:
                product_name = "Product title not found"
            
            # Extract reviews
            reviews = []
            while True:
                review_elements = driver.find_elements(By.XPATH, "//span[@data-hook='review-body']")
                reviews.extend([review.text.strip() for review in review_elements])

                # Scroll down to load more reviews
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)

                # Check if there's a "Next" button to load more reviews
                next_button = driver.find_elements(By.XPATH, "//li[@class='a-last']/a")
                if next_button:
                    next_button[0].click()
                    time.sleep(3)
                else:
                    break

            return product_name, reviews
        
        finally:
            # Quit the driver after scraping
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

    # Function to summarize reviews using Hugging Face summarization
    def summarize_reviews(reviews):
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
        genai.configure(api_key="AIzaSyCYAkTZfIJ2eUWBAHczQ4qaK5HbFCvzjUc")

        try:
            # Use Gemini to generate a recommendation
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            return response.text if response.text else "I am unable to give you the information"

        except Exception as e:
            return f"Error generating buy recommendation: {e}"
        
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
            
            
            st.sidebar.metric("Total Reviews", len(reviews))
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
            

            # Summarize the reviews
            st.subheader("Summary of Reviews")
            summary = summarize_reviews(reviews)
            st.write(summary)
            
            tfidf_keywords = generate_tfidf_keywords(reviews)
            
            buy_recommendation = generate_buy_recommendation(sentiment_counts, tfidf_keywords, processed_reviews, reviews)
            st.subheader("Buy Recommendation")
            st.write(buy_recommendation)

        

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