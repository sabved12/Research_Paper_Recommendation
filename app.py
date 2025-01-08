import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
import pickle


# load saved files
embeddings = pickle.load(open('embeddings.pkl','rb'))
sentences = pickle.load(open('sentences.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

def recommendation(input_paper):
    # Calculate cosine similarity scores between the embeddings of input_paper and all papers in the dataset.
    cosine_scores = util.cos_sim(embeddings, model.encode(input_paper))
    
    # Get the indices of the top-k most similar papers based on cosine similarity.
    top_similar_papers = torch.topk(cosine_scores, dim=0, k=20, sorted=True)
                                 
    # Retrieve the titles of the top similar papers.
    papers_set = set()
    papers_list = []
    for i in top_similar_papers.indices:
        title = sentences[i.item()]
        if title not in papers_set:
            papers_list.append(title)
            papers_set.add(title)
        # if len(papers_list) == 5:  # Stop once we have 5 unique papers
        #     break
    return papers_list

#  
st.title("Research Papers Recommendation System and Subject Area Prediction App")
input_paper=st.text_input("Enter Paper title")

if st.button("Recommend"):
    recommend_paper=recommendation(input_paper)
    st.subheader('Recommended papers')
    st.write(recommend_paper)
   
    