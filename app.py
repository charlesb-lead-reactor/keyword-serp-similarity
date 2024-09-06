import streamlit as st
import hashlib
import pickle
import os
import pandas as pd
from serpapi import GoogleSearch
import difflib
import tldextract
import logging
import seaborn as sns

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
        for j in range(i, num_keywords):  # Changement ici
            if i == j:
                similarity_matrix[i][j] = 100  # Similarité maximale avec soi-même
            else:
                # Utilise difflib.SequenceMatcher pour calculer la similarité
                sequence_matcher = difflib.SequenceMatcher(None, serp_results[i], serp_results[j])
                similarity = int(round(sequence_matcher.ratio() * 100))
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity  # Symétrie assurée
    
    return similarity_matrix

def cluster_keywords(similarity_matrix, keywords, threshold=40):
    clusters = []
    used_keywords = set()

    for i, keyword in enumerate(keywords):
        if keyword in used_keywords:
            continue
        
        cluster = [keyword]
        used_keywords.add(keyword)

        for j, other_keyword in enumerate(keywords):
            if other_keyword != keyword and other_keyword not in used_keywords:
                if similarity_matrix[i][j] > threshold:
                    cluster.append(other_keyword)
                    used_keywords.add(other_keyword)
        
        clusters.append(cluster)

    return clusters


def extract_urls_from_serp(results):
    return [result.get("link") for result in results.get("organic_results", []) if result.get("link")]

def fetch_serp_results(api_key, query, location="United States", language="en", country="us", 
                       google_domain="google.com", device="desktop", num_results=15, **additional_params):
    """
    Appelle l'API SERP pour obtenir les résultats de recherche pour un mot-clé donné.

    Parameters:
    api_key (str): La clé API pour authentifier la requête.
    query (str): Le mot-clé pour lequel effectuer la recherche.
    location (str): La localisation géographique des résultats de recherche (par défaut: "United States").
    language (str): La langue des résultats de recherche (par défaut: "en").
    country (str): Le paramètre de localisation de Google (par défaut: "us").
    google_domain (str): Le domaine de Google à utiliser pour la recherche (par défaut: "google.com").
    device (str): Le type de dispositif pour la recherche (par défaut: "desktop").
    num_results (int): Le nombre de résultats de recherche organiques à récupérer (par défaut: 9).
    **additional_params: Paramètres supplémentaires à passer à l'API.

    Returns:
    dict: Les résultats de recherche au format JSON, ou None en cas d'erreur.
    """
    # Définition des paramètres de recherche
    params = {
        "q": query,
        "location": location,
        "hl": language,
        "gl": country,
        "google_domain": google_domain,
        "device": device,
        "num": str(num_results),
        "api_key": api_key
    }

    # Ajout des paramètres supplémentaires
    params.update(additional_params)

    try:
        # Création de l'objet de recherche et récupération des résultats
        search = GoogleSearch(params)
        results = search.get_dict()
        return results
    except Exception as e:
        # Gestion des erreurs et journalisation
        logging.error(f"Erreur lors de l'appel à l'API SERP pour la requête '{query}': {e}")
        return None
    
def cluster_and_sort_keywords(similarity_matrix, keywords, threshold=20):
    n = len(keywords)
    unclustered = set(range(n))
    clusters = []
    sorted_keywords = []
    
    while unclustered:
        # Trouver le mot-clé avec la plus grande somme de similarité parmi les non clusterisés
        main_keyword_index = max(unclustered, key=lambda i: sum(similarity_matrix[i][j] for j in unclustered))
        
        # Créer un nouveau cluster
        cluster = [main_keyword_index]
        for j in unclustered:
            if j != main_keyword_index and similarity_matrix[main_keyword_index][j] >= threshold:
                cluster.append(j)
        
        # Trier le cluster par similarité décroissante avec le mot-clé principal
        cluster.sort(key=lambda i: similarity_matrix[main_keyword_index][i], reverse=True)
        
        # Ajouter le cluster à la liste des clusters
        clusters.append([keywords[i] for i in cluster])
        
        # Mettre à jour la liste des mots-clés triés
        sorted_keywords.extend([keywords[i] for i in cluster])
        
        # Retirer les mots-clés clusterisés de l'ensemble des non clusterisés
        unclustered -= set(cluster)
    
    return sorted_keywords, clusters

import streamlit as st

# Barre latérale pour les paramètres
st.sidebar.title("Parameters")
api_key = st.sidebar.text_input("API Key (SERPAPI)")
location = st.sidebar.selectbox("Location", ["France", "United States", "United Kingdom", "Spain", "Italy"])
language = st.sidebar.selectbox("Language", ["fr", "en", "uk", "es", "it"])
google_domain = st.sidebar.selectbox("Google Domain", ["google.fr", "google.com", "google.co.uk", "google.es", "google.it"])
device = st.sidebar.selectbox("Device", ["desktop", "mobile"])

# Partie principale pour les mots-clés
st.title("SERP Analysis Tool")
query = st.text_area("Keywords (one per line)", "International Business Machines Corporation\nIBM\nbig blue\nInternational Business Machines\nWatson")

if st.button("Fetch SERP Results"):
    if not api_key:
        st.warning("Please provide the SERP API key in the sidebar to fetch results.")
    else:
        country_mapping = {
            "United States": "us",
            "France": "fr",
            "United Kingdom": "uk",
            "Spain": "es",
            "Italy": "it"
        }
        country = country_mapping[location]
        keywords = [kw.strip() for kw in query.split('\n') if kw.strip()]
        keywords_hash = generate_keywords_hash(keywords)
        
        serp_results = load_serp_results(keywords_hash)
        
        if serp_results is None:
            serp_results = {}
            for keyword in keywords:
                results = fetch_serp_results(api_key, keyword, location=location, language=language, country=country, google_domain=google_domain, device=device)
                if results:
                    serp_results[keyword] = extract_urls_from_serp(results)
                else:
                    st.write(f"Failed to fetch results for keyword: {keyword}")
            
            save_serp_results(keywords_hash, serp_results)
        
        if serp_results:
            # Créer le DataFrame des URLs
            max_urls = max(len(urls) for urls in serp_results.values())
            df_urls = pd.DataFrame({k: urls + [None]*(max_urls - len(urls)) for k, urls in serp_results.items()})
            
            # Afficher le DataFrame des URLs
            st.write("SERP Results (URLs):")
            st.dataframe(df_urls)
            
            # Calculer et afficher la matrice de similarité
            serp_comp_list = list(serp_results.values())
            similarity_matrix = calculate_serp_similarity(serp_comp_list)
            
            # Clusteriser et trier les mots-clés
            sorted_keywords, clusters = cluster_and_sort_keywords(similarity_matrix, keywords)
            
            # Réorganiser la matrice de similarité
            sorted_matrix = [[similarity_matrix[keywords.index(k1)][keywords.index(k2)] for k2 in sorted_keywords] for k1 in sorted_keywords]
            
            # Créer le DataFrame trié
            df_similarity = pd.DataFrame(sorted_matrix, index=sorted_keywords, columns=sorted_keywords)
            cm = sns.light_palette("green", as_cmap=True)
            df_styled = df_similarity.style.background_gradient(cmap=cm)
            
            st.write("Keyword Similarity Matrix (clustered and sorted):")
            st.dataframe(df_styled)
            
            # Afficher les clusters
            st.write("Clusters de mots-clés :")
            for i, cluster in enumerate(clusters, 1):
                st.write(f"Cluster {i}: {', '.join(cluster)}")
        else:
            st.write("Failed to fetch any SERP results.")
