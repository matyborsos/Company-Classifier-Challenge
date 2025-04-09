from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

class InsuranceClassifier:
    def __init__(self, taxonomy_labels, model_name="paraphrase-MiniLM-L6-v2", similarity_threshold=0.3):
        self.model = SentenceTransformer(model_name)
        self.labels = taxonomy_labels
        self.similarity_threshold = similarity_threshold
        self.label_embeddings = self.model.encode(taxonomy_labels)

    def combine_fields(self, row):
        fields = ['description', 'business_tags', 'sector', 'category', 'niche']
        return ' | '.join(str(row[field]) for field in fields if pd.notnull(row[field]))

    def classify_companies(self, company_df):
        # apply combine_fields to each row in the DataFrame to combine the fields
        texts = company_df.apply(self.combine_fields, axis=1)
        # get embeddings for the company descriptions (or combined fields)
        company_embeddings = self.model.encode(texts.tolist())
        
        # calculate cosine similarity between the company embeddings and label embeddings
        sims = cosine_similarity(company_embeddings, self.label_embeddings)
        
        # cet top N labels based on cosine similarity
        top_labels = []
        for sim in sims:
            # get labels that exceed the similarity threshold
            relevant_labels = [self.labels[i] for i in range(len(sim)) if sim[i] >= self.similarity_threshold]
            
            # if no relevant labels exceed the threshold, add the label with the highest similarity
            if not relevant_labels:
                max_sim_index = sim.argmax()
                relevant_labels = [self.labels[max_sim_index]]
            
            top_labels.append(relevant_labels)

        # add the predicted labels to the DataFrame
        company_df['insurance_labels'] = top_labels
        return company_df

