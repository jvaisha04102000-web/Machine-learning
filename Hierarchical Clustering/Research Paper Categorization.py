import os
import PyPDF2
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# -----------------------------
# Extract Text from PDFs
# -----------------------------

pdf_folder = "research_papers"

documents = []
file_names = []

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        path = os.path.join(pdf_folder, file)

        text = ""

        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)

            for page in reader.pages:
                text += page.extract_text()

        documents.append(text)
        file_names.append(file)

# -----------------------------
# NLP TF-IDF
# -----------------------------

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(documents)

# -----------------------------
# Hierarchical Clustering
# -----------------------------

linkage_matrix = linkage(X.toarray(), method='ward')

# -----------------------------
# Dendrogram
# -----------------------------

plt.figure(figsize=(12, 6))

dendrogram(
    linkage_matrix,
    labels=file_names,
    leaf_rotation=90
)

plt.title("Research Paper Clustering")
plt.xlabel("Research Papers")
plt.ylabel("Distance")

plt.show()