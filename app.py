import streamlit as st
import hashlib
import pickle
import os
import pandas as pd
import difflib
import seaborn as sns
from googleapiclient.discovery import build
from python_semrush.semrush import SemrushClient
import time


def check_file_format(df):
    required_columns = ['query', 'url']
    return all(col in df.columns for col in required_columns)


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
                similarity = int(round(sequence_matcher.ratio() * 100))
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
    return similarity_matrix


def fetch_google_results(api_key, cse_id, query, num_results=10):
    service = build("customsearch", "v1", developerKey=api_key)
    try:
        result = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
        time.sleep(0.7)
        return result.get('items', [])
    except Exception as e:
        st.error(f"Error fetching results for '{query}': {str(e)}")
        return None


def extract_urls_from_google(results):
    return [item['link'] for item in results if 'link' in item]


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

        filtered_cluster = []
        for i in cluster:
            if all(similarity_matrix[i][j] > 0 for j in cluster if i != j):
                filtered_cluster.append(i)

        filtered_cluster.sort(key=lambda i: similarity_matrix[main_keyword_index][i], reverse=True)

        if filtered_cluster:
            clusters.append([keywords[i] for i in filtered_cluster])
            sorted_keywords.extend([keywords[i] for i in filtered_cluster])

        unclustered -= set(cluster)

    return sorted_keywords, clusters


def display_clusters_with_different_urls(clusters, query_url_dict):
    for i, cluster in enumerate(clusters, 1):
        cluster_urls = set()
        for keyword in cluster:
            urls = query_url_dict.get(keyword, [])
            cluster_urls.update(urls)
        
        if len(cluster_urls) > 1:  # Afficher seulement si le cluster a plus d'une URL unique
            st.write(f"## Cluster {i}")
            st.write(f"**Mots-clés :** {', '.join(cluster)}")
            st.write("**URLs uniques dans ce cluster :**")
            for url in cluster_urls:
                st.write(f"- {url}")
            st.write("---")


def main():
    STREAMLIT_ENV = os.environ.get("STREAMLIT_ENV", "dev")

    st.sidebar.title("Paramètres")
    if STREAMLIT_ENV == 'dev':
        api_key = st.secrets['api_key']
        cse_id = st.secrets['cse_id']
        #semrush_api_key = st.secrets['semrush_api_key']
    else:
        api_key = st.sidebar.text_input("Clé API Google")
        cse_id = st.sidebar.text_input("ID du moteur de recherche personnalisé")
        #semrush_api_key = st.sidebar.text_input("Clé API Smerush")
    threshold = st.sidebar.number_input("Seuil de similarité", value=20)

    st.title("Google Custom Search Analysis Tool")

    input_method = st.radio("Choisissez la méthode d'entrée des mots-clés :", 
                            ("Télécharger un fichier (queries / urls)", "Saisir manuellement (queries)"))

    keywords = []
    #client = SemrushClient(key=semrush_api_key)

    if input_method == "Télécharger un fichier (queries / urls)":
        uploaded_file = st.file_uploader("Téléchargez votre fichier XLSX ou CSV", type=['xlsx', 'csv'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)
                
                if check_file_format(df):
                    st.success("Fichier téléchargé avec succès !")
                    # st.write("Contenu du fichier :")
                    # st.dataframe(df[['query', 'url']])
                    
                    keywords = df['query'].unique().tolist()
                    query_url_dict = {query: df[df['query'] == query]['url'].tolist() for query in keywords}
                else:
                    st.error("Le fichier doit contenir les colonnes 'query' et 'url'.")
            except Exception as e:
                st.error(f"Une erreur s'est produite lors de la lecture du fichier : {str(e)}")
    else:
        query = st.text_area("Keywords (one per line)",
                     "bachelor rse\nformation responsable rse\nformation rse reconversion\nformation rse\nmaster rse\nmaster rse alternance\nmaster rse à distance\necole rse\nformation développement durable\necole developpement durable\nmaster développement durable\nmaster développement durable alternance\nmaster developpement durable\nresponsable rse\nformation rse certifiante\nmaster 2 rse\nécole environnement lyon\necole environnement\nrse ecole de commerce\nformation continue rse\nformation rse certifiante en ligne\nformation rse lyon\nformation consultant rse\nformation rse à distance\nformation rse bordeaux\nformation rse diplômante\nformation rse nantes\nformation rse toulouse\nformation ecologie\nformation environnement\nmaster qhse\nmaster qse\nmaster environnement\nmaster ecologie\nmaster finance durable\nbachelor développement durable\nbachelor environnement et développement durable\nformation environnement developpement durable\nbachelor qse\nmaster qse alternance\nbachelor responsable qse\nmaster qse lyon\nétudes en environnement\nmaster gestion de l environnement\nbachelor environnement\nmaster environnement alternance\nformation environnement pour adulte\nbachelor en environnement\nformation environnement à distance\nbachelor qhse\nformation transition écologique\nformation en écologie\nformation continue développement durable\nformation rse cpf\nformation certifiante rse\nformation en environnement et développement durable\nmaster en environnement\nmaster transition écologique\nécole développement durable\nécole environnement\nmaster en alternance environnement\nmaster 1 rse\nmaster 2 développement durable\nmaster développement durable paris\nmaster rse paris\nécole rse\nmaster écologie alternance\nmaster rse en alternance\nécole transition écologique\nécole développement durable paris\nmaster environnement et développement durable\nreconversion rse\nreconversion développement durable\nreconversion ecologie\nreconversion metier ecologie\nreconversion professionnelle ecologie\nmétier rse\nmétiers de la transition écologique\nmétier écologie\nmétier développement durable\nresponsable rse fiche métier\nmaster rse lyon\nresponsable rse formation\nformation rse en ligne\nfiche de poste responsable rse\nrse métier\nmétiers rse\nmetiers rse\nmanager rse\nmetier rse\ndirecteur rse\naudit rse\ndirectrice rse\nstage rse\nstage rse paris\nrse en anglais")
        keywords = [kw.strip() for kw in query.split('\n') if kw.strip()]

    if st.button("Récupérer les résultats de recherche Google"):
        # Affichez les résultats
        st.write(resultat)
    
        if not api_key or not cse_id:
            st.warning("Veuillez fournir à la fois la clé API Google et l'ID du moteur de recherche personnalisé dans la barre latérale.")
        elif not keywords:
            st.warning("Veuillez entrer des mots-clés ou télécharger un fichier valide.")
        else:
            keywords_hash = generate_keywords_hash(keywords)
            serp_results = load_serp_results(keywords_hash)

            if serp_results is None:
                serp_results = {}
                total_keywords = len(keywords)
                
                progress_bar = st.progress(0)
                for i, keyword in enumerate(keywords):
                    try:
                        results = fetch_google_results(api_key, cse_id, keyword)
                        if results:
                            serp_results[keyword] = extract_urls_from_google(results)
                        else:
                            st.write(f"Aucun résultat trouvé pour le mot-clé : {keyword}")
                    except Exception as e:
                        st.write(f"Erreur lors de la récupération des résultats pour le mot-clé '{keyword}': {str(e)}")
                    
                    progress_bar.progress((i + 1) / total_keywords)

                save_serp_results(keywords_hash, serp_results)

            if serp_results:
                max_urls = max(len(urls) for urls in serp_results.values())
                df_urls = pd.DataFrame({k: urls + [None]*(max_urls - len(urls)) for k, urls in serp_results.items()})

                st.write("Google Search Results (URLs):")
                st.dataframe(df_urls)

                serp_comp_list = list(serp_results.values())
                similarity_matrix = calculate_serp_similarity(serp_comp_list)

                sorted_keywords, clusters = cluster_and_sort_keywords(similarity_matrix, keywords, threshold)

                sorted_matrix = [[similarity_matrix[keywords.index(k1)][keywords.index(
                    k2)] for k2 in sorted_keywords] for k1 in sorted_keywords]

                df_similarity = pd.DataFrame(sorted_matrix, index=sorted_keywords, columns=sorted_keywords)
                cm = sns.light_palette("green", as_cmap=True)
                df_styled = df_similarity.style.background_gradient(cmap=cm)

                st.write("Keyword Similarity Matrix (clustered and sorted):")
                st.dataframe(df_styled)

                st.write("Keyword Clusters:")
                for i, cluster in enumerate(clusters, 1):
                    st.write(f"Cluster {i}: {', '.join(cluster)}")
            else:
                st.write("Échec de la récupération des résultats de recherche Google.")

            
            st.write("## Clusters avec des URLs différentes")
            display_clusters_with_different_urls(clusters, query_url_dict)

if __name__ == "__main__":
    main()
