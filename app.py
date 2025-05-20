import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ---------- Title ----------
st.title("💬 Sentiment Analysis App")
st.markdown("Enter a sentence and get sentiment prediction. Visualizations included!")

# ---------- Sample Real-World-like Dataset ----------
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
user_input = st.text_area("📝 Enter a review:", "")

if st.button("📈 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        prediction = model.predict([user_input])[0]
        if prediction == 1:
            st.success("✅ Sentiment: Positive 😊")
        else:
            st.error("❌ Sentiment: Negative 😡")

# ---------- Sentiment Distribution Chart ----------
st.subheader("📊 Sentiment Distribution in Dataset")
sentiment_counts = df['label'].value_counts().rename({0: "Negative", 1: "Positive"})
fig, ax = plt.subplots()
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="pastel", ax=ax)
ax.set_ylabel("Number of Reviews")
st.pyplot(fig)

# ---------- Word Cloud ----------
st.subheader("☁️ Word Cloud of All Reviews")
all_text = " ".join(df['text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
ax_wc.imshow(wordcloud, interpolation='bilinear')
ax_wc.axis('off')
st.pyplot(fig_wc)

# ---------- Footer ----------
st.markdown("---")
st.markdown("Made with ❤️ by **Putani Tharun**")
