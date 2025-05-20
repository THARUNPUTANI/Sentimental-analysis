import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

# ---------- Preprocessing Class ----------
class TextPreprocessor(TransformerMixin):
    def transform(self, X, **transform_params):
        return [self._clean_text(text) for text in X]
    
    def fit(self, X, y=None, **fit_params):
        return self

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

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
        'Terrible quality, not recommended.',
        'Absolutely fantastic!',
        'Disappointed with the product.',
        'Best thing I bought all year!',
        'Not good at all.',
        'Really amazing experience.'
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# ---------- Train Model ----------
model = Pipeline([
    ('preprocessor', TextPreprocessor()),
    ('vectorizer', TfidfVectorizer(ngram_range=(1,2), stop_words='english')),
    ('classifier', MultinomialNB())
])
model.fit(df['text'], df['label'])

# ---------- Streamlit UI ----------
st.title("💬 Sentiment Analysis App")
st.markdown("Enter a review below to predict the sentiment.")

# User input
user_input = st.text_area("📝 Your Review:", "")

if st.button("📈 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        prediction = model.predict([user_input])[0]
        if prediction == 1:
            st.success("✅ Sentiment: Positive 😊")
        else:
            st.error("❌ Sentiment: Negative 😡")

# ---------- Visualizations ----------
st.subheader("📊 Sentiment Distribution")
sentiment_counts = df['label'].map({0: "Negative", 1: "Positive"}).value_counts()
fig, ax = plt.subplots()
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="pastel", ax=ax)
ax.set_ylabel("Number of Reviews")
st.pyplot(fig)

st.subheader("☁️ Word Cloud")
all_text = " ".join(df['text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
ax_wc.imshow(wordcloud, interpolation='bilinear')
ax_wc.axis('off')
st.pyplot(fig_wc)

st.markdown("---")
st.markdown("Made with ❤️ by **Putani Tharun**")
