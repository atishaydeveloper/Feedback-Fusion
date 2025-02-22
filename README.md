# Feedback-Fusion: AI-Powered Feedback Analysis and Sentiment Processing System

## Project Vision

Feedback-Fusion aims to revolutionize how organizations and businesses understand and utilize feedback from their customers, employees, and other stakeholders. By harnessing the power of sentiment analysis, natural language processing, and AI-driven insights, we provide a platform for gaining deep insights into audience sentiment and making informed, data-driven decisions for continuous improvement.

## Overview

Feedback-Fusion is a Streamlit-based application that analyzes feedback from various sources, including:

*   Amazon product reviews
*   YouTube video comments
*   Company reviews (customer and employee)

The application provides the following key features:

*   **Data Ingestion:** Supports uploading feedback data from CSV and JSON files.
*   **Sentiment Analysis:** Automatically classifies feedback as positive, negative, or neutral using the VADER sentiment analysis library.
*   **Data Preprocessing:** Cleans and prepares text data for analysis using techniques like lowercasing, punctuation removal, stop word removal, and lemmatization.
*   **Word Cloud Visualization:** Generates word clouds to visualize the most common terms in the feedback.
*   **TF-IDF Keyword Extraction:** Identifies key topics and themes in the feedback using TF-IDF analysis.
*   **AI-Powered Summarization:** Summarizes large sets of feedback using the Gemini Pro API.
*   **AI-Driven Recommendations:** Leverages the Gemini Pro API to generate actionable insights, such as "buy recommendations" for products or "ideal for" recommendations for specific user profiles.
*   **FAQ Generation:** Automatically identifies common questions from the feedback and generates answers using the Gemini Pro API.
*   **Interactive Visualizations:** Presents sentiment distributions using interactive bar charts.
*   **Clear Results Display:** Organizes feedback and analysis results in a user-friendly Streamlit interface with expanders for detailed information.
*   **Web Scraping (Amazon):** Uses Selenium to scrape reviews directly from Amazon product pages.
*   **Customization:** Users can adjust various parameters, such as the number of questions in the FAQ and sentiment filters.

## Key Features and Services

*   **YouTube Video Feedback Analysis:** Analyze YouTube video comments to gain insights into viewer sentiment.
*   **Amazon Product Review Analysis:** Analyze product reviews from Amazon to gauge customer satisfaction and product performance.
*   **Company Review Analysis:** Analyze feedback from both employees and customers within your company, enabling you to improve workplace satisfaction and customer experience.
*   **Sentiment Analysis:** Automatically classify feedback as positive, negative, or neutral and generate insightful reports.
*   **Word Cloud Visualization:** Generate word clouds to visualize the most common terms in the feedback, helping to highlight key themes.
*   **TF-IDF Keyword Extraction:** Extracts the top keywords using scikit-learn's TF-IDF Vectorizer.
*   **Summarization:** Provides summaries of large sets of feedback for quick and easy understanding using Gemini API.
*   **AI-Driven Recommendations:** Leverages the Gemini Pro API to generate actionable insights, such as "buy recommendations" for products or "ideal for" recommendations for specific user profiles.
*   **FAQ Generation:** Automatically identifies common questions from the feedback and generates answers using the Gemini Pro API.

## Impact of Feedback-Fusion

The Feedback-Fusion platform is designed to have a transformative impact on businesses, organizations, and individuals:

*   **Improved Decision-Making:** By understanding the sentiments of customers, employees, and stakeholders, organizations can make more informed decisions.
*   **Better Customer Experience:** Analyzing customer feedback helps identify areas of improvement, allowing companies to enhance their offerings and services.
*   **Employee Satisfaction:** Analyzing employee feedback ensures that companies can maintain a positive work environment and improve employee engagement.
*   **Data-Driven Insights:** Feedback-Fusion empowers organizations to move from gut-feeling decisions to data-backed strategies, ultimately leading to better outcomes and growth.

## Technologies Used

*   **Streamlit:** For creating the user interface.
*   **Pandas:** For data manipulation and analysis.
*   **Plotly:** For interactive visualizations.
*   **NLTK:** For natural language processing tasks (sentiment analysis, stop word removal, lemmatization).
*   **scikit-learn:** For TF-IDF keyword extraction.
*   **Selenium:** For web scraping Amazon reviews.
*   **Google Gemini API:** For AI-powered summarization, recommendation generation, and FAQ generation.
*   **WordCloud:** for the generation of the different wordclouds.

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone [your_repository_url]
    cd feedback-fusion
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API keys:**

    *   Obtain a Google Gemini API key from [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey).
    *   Set the API key as an environment variable:

        ```bash
        export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"  # On Linux/macOS
        set GEMINI_API_KEY="YOUR_GEMINI_API_KEY"  # On Windows
        ```

        Or, you can directly insert your API key in the code (Not recommended):

        ```python
        genai.configure(api_key="YOUR_GEMINI_API_KEY")
        ```

5.  **Run the Streamlit app:**

    ```bash
    streamlit run your_app_file.py  # Replace your_app_file.py with the name of your main script
    ```

## Usage

1.  **Amazon Product Review Analysis:**

    *   Enter the Amazon product URL in the provided text box.
    *   Click the "Analyze" button.
    *   The app will scrape the reviews, perform sentiment analysis, generate visualizations, and provide AI-powered insights.

2.  **Company Review Analysis and YouTube Analysis**

    *   Upload a CSV or JSON file containing feedback data or the link to your youtube video.
    *   Follow the specified file format requirements (see below).
    *   The app will process the data and display the results.

### File Format Requirements (CSV/JSON)

The CSV or JSON file should have at least two columns:

*   **Feedback Column:** Contains the actual feedback text. You can name it `feedback`, `review`, or similar.
*   **Category Column:** Specifies whether the feedback is from a `Customer` or an `Employee` (for company reviews). You can name it `category`, `type`, or similar.

Optional Columns:

*   `rating`: A numeric rating (e.g., 1 to 5 or 1 to 10).
*   `date`: The date when the feedback was given.
*   `department`: For employee feedback, you may include a `department` column to specify which department the feedback is related to (e.g., `Sales`, `HR`).

### File Format Requirements (Youtube Link)
*   Insert the youtube video link and the app will automatically perform its processing.

## Web Scraping Disclaimer

This project includes web scraping functionality to retrieve Amazon product reviews. Please be aware of the following:

*   **Terms of Service:** Web scraping should be conducted in compliance with Amazon's Terms of Service.
*   **Respectful Scraping:** Implement appropriate delays and rate limiting to avoid overloading Amazon's servers.
*   **Ethical Considerations:** Use the scraped data responsibly and ethically.
*   **Changes to Website Structure:** Amazon's website structure may change, which could break the scraping functionality. You may need to update the scraping code periodically.
*   **Legal Compliance:** Comply with all applicable laws and regulations related to web scraping.


## Contributing

Contributions to Feedback-Fusion are welcome! To contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with clear and concise commit messages.
4.  Submit a pull request.

## Contributors

*   Atishay Jain (Project Lead)
*   Sarthak Doshi
*   Om Chouksey
*   Shambhavi Bhadoria

We thank each contributor for their valuable input, collaboration, and commitment to making this project a success.

## License

This project is licensed under the [Specify License Type] License. See the `LICENSE` file for more information.

## Contact

For more information, feedback, or collaboration inquiries, please contact us at:

*   Email: atishayj288@gmail.com
*   Phone: +91 9522041334

Feedback-Fusion | All rights reserved © 2024