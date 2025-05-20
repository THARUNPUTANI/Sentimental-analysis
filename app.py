import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def main():
    # ---------- Sample Dataset ----------
    data = {
        'text': [
            'Great quality and fast delivery!',
            'The item arrived broken.',
            'Excellent value for money.',
            'Very bad customer service.',
            'Totally worth it!',
            'Product does not match the description.',
            'I love this product!',
            'This is the worst I’ve ever used.',
            'Superb experience overall!',
            'Terrible quality, not recommended.'
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
    }
    df = pd.DataFrame(data)

    # ---------- Build and Train Model ----------
    model = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])
    X = df['text']
    y = df['label']
    model.fit(X, y)

    # ---------- User Input ----------
    user_input = input("Enter a review to analyze sentiment: ")

    if user_input.strip() == "":
        print("Please enter some text to analyze.")
    else:
        prediction = model.predict([user_input])[0]
        if prediction == 1:
            print("Sentiment: Positive 😊")
        else:
            print("Sentiment: Negative 😡")

    # ---------- Sentiment Distribution Chart ----------
    sentiment_counts = df['label'].map({0: "Negative", 1: "Positive"}).value_counts()
    plt.figure(figsize=(6,4))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="pastel")
    plt.ylabel("Number of Reviews")
    plt.title("Sentiment Distribution in Dataset")
    plt.show()

    # ---------- Word Cloud ----------
    all_text = " ".join(df['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud of All Reviews")
    plt.show()

if __name__ == "__main__":
    main()
