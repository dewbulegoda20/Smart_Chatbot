# Smart Chatbot for Health-related Queries

## Project Overview
This project is a smart chatbot designed to answer health-related questions by analyzing live content from the web. It uses Natural Language Processing (NLP) to extract, clean, and understand health information from online articles, enabling intelligent and relevant responses.

## Features
- Extracts real-time health information using web scraping.
- Processes and tokenizes content with NLP techniques for understanding queries.
- Matches user queries to the most relevant content using similarity-based logic.

## Technologies
- Python
- NLTK
- newspaper3k
- scikit-learn
- NumPy

## How It Works
1. Scrapes recent health articles from the web using `newspaper3k`.
2. Cleans and processes the text with `NLTK`.
3. Vectorizes and compares the user’s input with the extracted content using `scikit-learn`.
4. Returns the most relevant response based on similarity score.

---

*This project is aimed at exploring practical applications of NLP in real-world healthcare communication.*
