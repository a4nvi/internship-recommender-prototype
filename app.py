import streamlit as st
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet')

st.title("AI Internship Recommendation Prototype")

level = st.selectbox("Select AI Proficiency Level", ["Level 1", "Level 2", "Level 3", "Level 4"])

resume_text = st.text_area("Paste Resume Text Here")
internship_text = st.text_area("Paste Internship Description Here")

if st.button("Get Match Score"):
    if level == "Level 1":
        # Simple word overlap
        resume_words = set(resume_text.lower().split())
        internship_words = set(internship_text.lower().split())
        match_score = len(resume_words & internship_words) / len(internship_words) * 100
        st.write(f"Level 1 Match Score: {match_score:.2f}%")
    
    elif level == "Level 2":
        # Synonym matching example
        resume_words = set(resume_text.lower().split())
        internship_words = set(internship_text.lower().split())
        expanded_resume_words = set(resume_words)
        for word in resume_words:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    expanded_resume_words.add(lemma.name().lower())
        match_score = len(expanded_resume_words & internship_words) / len(internship_words) * 100
        st.write(f"Level 2 Match Score: {match_score:.2f}%")

    elif level == "Level 3":
        # Semantic similarity
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb1 = model.encode(resume_text, convert_to_tensor=True)
        emb2 = model.encode(internship_text, convert_to_tensor=True)
        score = util.pytorch_cos_sim(emb1, emb2).item() * 100
        st.write(f"Level 3 Semantic Similarity Score: {score:.2f}%")

    else:  # Level 4
        from transformers import pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(resume_text, max_length=60, min_length=30, do_sample=False)[0]['summary_text']
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb1 = model.encode(summary, convert_to_tensor=True)
        emb2 = model.encode(internship_text, convert_to_tensor=True)
        score = util.pytorch_cos_sim(emb1, emb2).item() * 100
        st.write(f"Level 4 Summary Match Score: {score:.2f}%")

