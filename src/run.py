import pandas as pd
from classifier import InsuranceClassifier

# load data
company_df = pd.read_csv("../csv/company_list.csv")

# load taxonomy 
with open("../csv/insurance_taxonomy.csv", "r") as f:
    taxonomy_labels = [line.strip() for line in f.readlines() if line.strip()]

# initialize and run classifier
classifier = InsuranceClassifier(taxonomy_labels)
annotated_df = classifier.classify_companies(company_df)

# save result
annotated_df.to_csv("../csv/annotated_company_list.csv", index=False)
print("Classification complete. Results saved to annotated_company_list.csv.")
