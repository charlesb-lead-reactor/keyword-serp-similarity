import streamlit as st
import hashlib
import pickle
import os
import pandas as pd
import difflib
import seaborn as sns
from googleapiclient.discovery import build

def generate_keywords_hash(keywords):
    keywords_string = ' '.join(sorted(keywords)).encode()
    return hashlib.md5(keywords_string).hexdigest()

def save_serp_results(hash_key, serp_results, directory='serp_cache'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, f"{hash_key}.pkl")
    with open(filepath, 'wb') as file:
        pickle.dump(serp_results, file)

def load_serp_results(hash_key, directory='serp_cache'):
    filepath = os.path.join(directory, f"{hash_key}.pkl")
    if os.path.exists(filepath):
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    return None

def calculate_serp_similarity(serp_results):
    num_keywords = len(serp_results)    
    similarity_matrix = [[0] * num_keywords for _ in range(num_keywords)]
    for i in range(num_keywords):
        for j in range(i, num_keywords):
            if i == j:
                similarity_matrix[i][j] = 100
            else:
                sequence_matcher = difflib.SequenceMatcher(None, serp_results[i], serp_results[j])
                ratio = sequence_matcher.ratio()
                ratio100 = ratio * 100
                ratio100rounded = round(ratio100)
                similarityTest = int(ratio100rounded)
                similarity = int(round(sequence_matcher.ratio() * 100))
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
    return similarity_matrix

def fetch_google_results(api_key, cse_id, query, num_results=10):
    service = build("customsearch", "v1", developerKey=api_key)
    try:
        result = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
        return result.get('items', [])
    except Exception as e:
        st.error(f"Error fetching results for '{query}': {str(e)}")
        return None

def extract_urls_from_google(results):
    return [item['link'] for item in results if 'link' in item]

'''
def cluster_and_sort_keywords(similarity_matrix, keywords, threshold=20):
    n = len(keywords)
    unclustered = set(range(n))
    clusters = []
    sorted_keywords = []
    while unclustered:
        main_keyword_index = max(unclustered, key=lambda i: sum(similarity_matrix[i][j] for j in unclustered))
        cluster = [main_keyword_index]
        for j in unclustered:
            if j != main_keyword_index and similarity_matrix[main_keyword_index][j] >= threshold:
                cluster.append(j)
        cluster.sort(key=lambda i: similarity_matrix[main_keyword_index][i], reverse=True)
        clusters.append([keywords[i] for i in cluster])
        sorted_keywords.extend([keywords[i] for i in cluster])
        unclustered -= set(cluster)
    return sorted_keywords, clusters
'''

def cluster_and_sort_keywords(similarity_matrix, keywords, threshold=20):
    n = len(keywords)
    unclustered = set(range(n))
    clusters = []
    sorted_keywords = []
    
    while unclustered:
        main_keyword_index = max(unclustered, key=lambda i: sum(similarity_matrix[i][j] for j in unclustered))
        cluster = [main_keyword_index]
        
        for j in unclustered:
            if j != main_keyword_index and similarity_matrix[main_keyword_index][j] >= threshold:
                cluster.append(j)
        
        # Filtrer les mots-clés ayant une similarité de 0 avec au moins un autre mot-clé du cluster
        filtered_cluster = []
        for i in cluster:
            if all(similarity_matrix[i][j] > 0 for j in cluster if i != j):
                filtered_cluster.append(i)
        
        # Trier le cluster filtré en fonction de la similarité avec le mot-clé principal
        filtered_cluster.sort(key=lambda i: similarity_matrix[main_keyword_index][i], reverse=True)
        
        if filtered_cluster:  # Ajouter le cluster seulement s'il n'est pas vide
            clusters.append([keywords[i] for i in filtered_cluster])
            sorted_keywords.extend([keywords[i] for i in filtered_cluster])
        
        unclustered -= set(cluster)  # Retirer tous les mots-clés considérés, même ceux filtrés
    
    return sorted_keywords, clusters

# Streamlit UI
st.sidebar.title("Parameters")
api_key = st.sidebar.text_input("Google API Key", "AIzaSyCkebdoOTN3dkKS3Kpp3ghyNnU4uUFBqkk")
cse_id = st.sidebar.text_input("Custom Search Engine ID", "75392133b2c1841ce")
threshold = st.sidebar.number_input("Similarity Threshold", value=20)

st.title("Google Custom Search Analysis Tool")
query = st.text_area("Keywords (one per line)", "International Business Machines Corporation\nIBM\nbig blue\nInternational Business Machines\nWatson")

if st.button("Fetch Google Search Results"):
    if not api_key or not cse_id:
        st.warning("Please provide both the Google API key and Custom Search Engine ID in the sidebar.")
    else:
        keywords = [kw.strip() for kw in query.split('\n') if kw.strip()]
        keywords_hash = generate_keywords_hash(keywords)
        
        serp_results = load_serp_results(keywords_hash)
        
        if serp_results is None:
            serp_results = {}
            for keyword in keywords:
                results = fetch_google_results(api_key, cse_id, keyword)
                if results:
                    serp_results[keyword] = extract_urls_from_google(results)
                else:
                    st.write(f"Failed to fetch results for keyword: {keyword}")
            
            save_serp_results(keywords_hash, serp_results)
        
        if serp_results:
            max_urls = max(len(urls) for urls in serp_results.values())
            df_urls = pd.DataFrame({k: urls + [None]*(max_urls - len(urls)) for k, urls in serp_results.items()})
            
            st.write("Google Search Results (URLs):")
            st.dataframe(df_urls)
            
            serp_comp_list = list(serp_results.values())
            similarity_matrix = calculate_serp_similarity(serp_comp_list)
            
            # Affichage de la matrice de similarité non traitée
            st.write("Raw Similarity Matrix:")
            df_raw_similarity = pd.DataFrame(similarity_matrix, index=keywords, columns=keywords)
            st.dataframe(df_raw_similarity)
            
            # Le reste du code pour le clustering et l'affichage de la matrice triée
            sorted_keywords, clusters = cluster_and_sort_keywords(similarity_matrix, keywords, threshold)
            
            sorted_matrix = [[similarity_matrix[keywords.index(k1)][keywords.index(k2)] for k2 in sorted_keywords] for k1 in sorted_keywords]
            
            df_similarity = pd.DataFrame(sorted_matrix, index=sorted_keywords, columns=sorted_keywords)
            cm = sns.light_palette("green", as_cmap=True)
            df_styled = df_similarity.style.background_gradient(cmap=cm)
            
            st.write("Keyword Similarity Matrix (clustered and sorted):")
            st.dataframe(df_styled)
            
            st.write("Keyword Clusters:")
            for i, cluster in enumerate(clusters, 1):
                st.write(f"Cluster {i}: {', '.join(cluster)}")
        else:
            st.write("Failed to fetch any Google search results.")


        # if serp_results:
        #     max_urls = max(len(urls) for urls in serp_results.values())
        #     df_urls = pd.DataFrame({k: urls + [None]*(max_urls - len(urls)) for k, urls in serp_results.items()})
            
        #     st.write("Google Search Results (URLs):")
        #     st.dataframe(df_urls)
            
        #     serp_comp_list = list(serp_results.values())
        #     similarity_matrix = calculate_serp_similarity(serp_comp_list)
            
        #     sorted_keywords, clusters = cluster_and_sort_keywords(similarity_matrix, keywords, threshold)
            
        #     sorted_matrix = [[similarity_matrix[keywords.index(k1)][keywords.index(k2)] for k2 in sorted_keywords] for k1 in sorted_keywords]
            
        #     df_similarity = pd.DataFrame(sorted_matrix, index=sorted_keywords, columns=sorted_keywords)
        #     cm = sns.light_palette("green", as_cmap=True)
        #     df_styled = df_similarity.style.background_gradient(cmap=cm)
            
        #     st.write("Keyword Similarity Matrix (clustered and sorted):")
        #     st.dataframe(df_styled)
            
        #     st.write("Keyword Clusters:")
        #     for i, cluster in enumerate(clusters, 1):
        #         st.write(f"Cluster {i}: {', '.join(cluster)}")
        # else:
        #     st.write("Failed to fetch any Google search results.")
