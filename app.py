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
    """
    Calcule la similarité des listes de résultats SERP pour chaque mot-clé.

    Parameters:
    serp_results (list of list of str): Liste des listes de domaines/URLs des résultats SERP pour chaque mot-clé.

    Returns:
    list of list of int: Matrice de pourcentages de similarité pour chaque paire de mots-clés.
    """
    num_keywords = len(serp_results)
    similarity_matrix = [[0] * num_keywords for _ in range(num_keywords)]

    for i, current_serp in enumerate(serp_results):
        for j, other_serp in enumerate(serp_results):
            if i != j:
                # Utilise difflib.SequenceMatcher pour calculer la similarité
                sequence_matcher = difflib.SequenceMatcher(None, current_serp, other_serp)
                similarity_matrix[i][j] = int(round(sequence_matcher.ratio() * 100))
            else:
                similarity_matrix[i][j] = 100  # Similarité maximale avec soi-même

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

def extract_domains_from_serp(results, include_path=False):
    """
    Extrait les domaines ou les URLs complètes des résultats de recherche organiques.

    Parameters:
    results (dict): JSON des résultats de recherche contenant les résultats organiques.
    include_path (bool): Si True, inclut le chemin complet de l'URL dans l'extraction.

    Returns:
    list of str: Liste des domaines ou URLs extraits des résultats de recherche.
    """
    extracted_items = []

    for result in results.get("organic_results", []):
        link = result.get("link")
        
        if link:
            try:
                ext = tldextract.extract(link)
                if include_path:
                    extracted_items.append(link)
                else:
                    domain = f"{ext.domain}.{ext.suffix}"
                    extracted_items.append(domain)
            except Exception as e:
                print(f"Erreur lors de l'extraction du lien {link}: {e}")
                continue

    return extracted_items

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
        st.write("keywords hash : " + keywords_hash)

        serp_comp_list = load_serp_results(keywords_hash)
        if serp_comp_list is None:
            serp_comp_list = []
            for keyword in keywords:
                results = fetch_serp_results(api_key, keyword, location=location, language=language, country=country, google_domain=google_domain, device=device)
                if not results:
                    st.write(f"Failed to fetch results for keyword: {keyword}")
                else:
                    serp_comp = extract_domains_from_serp(results)
                    serp_comp_list.append(serp_comp)

            save_serp_results(keywords_hash, serp_comp_list)

        if not serp_comp_list:
            st.write("Failed to fetch any SERP results.")
        
        else:
            similarity_matrix = calculate_serp_similarity(serp_comp_list)
            
            df = pd.DataFrame(similarity_matrix, index=keywords, columns=keywords)

            cm = sns.light_palette("green", as_cmap=True)
            df_styled = df.style.background_gradient(cmap=cm)

            st.write("Keyword Similarity Matrix:")
            st.dataframe(df_styled)

            # Clusterisation hiérarchique
            clusters = cluster_keywords(similarity_matrix, keywords)

            st.write("Clusters de mots-clés :")
            for i, cluster in enumerate(clusters, 1):
                st.write(f"Cluster {i}: {', '.join(cluster)}")            
