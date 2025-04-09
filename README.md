# Insurance Classifier Solution 

## Introduction

In this project, I have built an insurance classifier capable of categorizing companies into relevant insurance categories based on their business descriptions and associated metadata. This solution leverages Natural Language Processing (NLP) and machine learning techniques to compute semantic similarities between company data and predefined insurance taxonomy labels. The goal is to automatically label companies with appropriate insurance categories based on their business characteristics, such as sector, niche, and product/service description.

## Problem Definition

Given a dataset of companies, the challenge was to map each company to a specific or set of relevant insurance labels. The information provided for each company includes a description, business tags, sector, category, and niche, all of which are intended to help categorize the company into one or more insurance types.

This is a classical case of **text classification** and **similarity analysis** where we need to:
1. Understand the content of a company’s business description.
2. Compare it to predefined categories in an insurance taxonomy.
3. Assign one or more labels to the company based on the best matches.

### Approach

To solve this problem, I followed these major steps:

1. **Data Understanding**: 
   - I reviewed the dataset structure to understand the fields provided. These include `description`, `business_tags`, `sector`, `category`, and `niche`, which are textual data points describing the company’s focus. 
   - I also created a static insurance taxonomy, which is a set of predefined labels representing the insurance categories.

2. **Text Preprocessing and Combination**:
   - Instead of relying solely on the `description` field, I chose to combine multiple fields (description, tags, sector, category, and niche) to create a more comprehensive representation of the company.
   - The combined fields were processed into a single string, forming a more robust "business profile" that could be used for classification.

3. **Embedding with Sentence Transformers**:
   - I used a **Sentence Transformer** model, which is specifically designed for generating dense vector embeddings for sentences or textual data.
   - This model transforms the company descriptions into vector representations that can be compared in a high-dimensional space using cosine similarity. The choice of a sentence transformer model was motivated by its ability to generate meaningful embeddings that capture semantic relationships between text data.

4. **Similarity Computation**:
   - After embedding both the company profiles and the insurance labels, I calculated the **cosine similarity** between the two sets of vectors. This gave me a numerical measure of how similar each company was to the various insurance labels.
   - Based on this similarity score, I selected the most relevant labels for each company.

5. **Multiple Label Assignment**:
   - The classifier was designed to support multiple labels per company. This was done by selecting all labels that met a certain similarity threshold, or if none do the onw with the best similarity, to adjust the recall.

6. **Evaluation**:
   - Once the classifier assigned labels to the companies, I evaluated the results manually to ensure the labels made sense.
   - The results were not perfect, and the system had some inconsistencies, particularly with edge cases where the company descriptions were vague or too broad. However, the system worked reasonably well for the majority of the companies, correctly classifying them into related categories.

### Challenges and Decision Points

1. **Selecting a Pre-trained Model**:
   - One of the key decisions was choosing the **Sentence Transformer** model. While there are many NLP models available, Sentence Transformers are designed specifically for tasks that involve comparing textual data. This made it an ideal choice for measuring the similarity between company profiles and insurance categories.

2. **Handling Multiple Labels**:
   - Initially, I thought about assigning a single label per company, but upon further reflection, I realized that companies often operate in multiple sectors or have multiple relevant insurance categories. This led to the decision to support multiple labels per company.

3. **Similarity Threshold**:
   - Another important aspect of the classifier is the similarity threshold. This threshold controls how strict the classifier is when assigning labels. I experimented with different thresholds to find a balance between precision and recall.
   - A low threshold might result in too many irrelevant labels, while a high threshold might miss relevant labels. This trade-off required fine-tuning for optimal performance.

4. **Fallback Mechanism**:
   - When no label meets the similarity threshold, it’s tempting to leave the label field empty or mark it as "no suitable label". However, I chose to assign the label with the highest similarity score, ensuring that every company gets a label even when it’s not a perfect match.                      |

### Future Improvements

1. **Fine-tuning the Model**:
   - While the pre-trained model performs adequately, there’s room to improve performance by fine-tuning the model on a dataset specifically related to insurance labels. This would make the embeddings even more relevant to the task.

2. **Handling Ambiguities**:
   - Some companies had descriptions that were too general, leading to ambiguous labels. Further work on pre-processing or using more specific input features could improve the classifier's accuracy.

3. **Parallel Processing**:
   - For large datasets, the classification process can be slow. Implementing parallel processing or using GPU-based models would improve speed and scalability.

### Conclusion

This solution demonstrates how modern NLP techniques, specifically transformer models, can be used to classify companies into predefined categories based on textual data. While it’s not perfect, it’s a strong foundation that can be iteratively improved. By combining company fields, generating vector embeddings, and comparing them with predefined labels, the classifier can automate insurance categorization tasks—saving both time and effort for insurance companies, brokers, and analysts.
