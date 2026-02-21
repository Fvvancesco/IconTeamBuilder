import csv
from typing import List, Dict, Callable

import numpy as np
from scipy.spatial.distance import pdist

from csp.csp import greedy_ascent
from supervised_learning.ranking import ottieni_ranking

def crea_funzione_calcolo_punteggio(dizionario_pokemon: Dict[str, Dict]) -> Callable[[List[str]], float]:
    """
    Crea una funzione (closure) che calcola il punteggio avendo già i dati in RAM.
    Questo evita di passare il dizionario a greedy_ascent mantenendola una "black box".
    """

    def calcola_punteggio(squadra: List[str]) -> float:
        score = 0
        for p_id in squadra:
            score += dizionario_pokemon[p_id]['score']
        return score

    return calcola_punteggio

def crea_valutatore_squadra(dict_forza: Dict, dict_ruoli: Dict, alpha: float = 1.0, beta: float = 0.5) -> Callable:
    """
    dict_forza: Dizionario { 'id': score_lgbm }
    dict_ruoli: Dizionario { 'id': {'cluster': int, 'vector': np.array} }
    alpha: Peso della forza bruta
    beta: Peso della diversità di squadra
    """

    def calcola_punteggio_sinergico(squadra_id: List[str]) -> float:
        forza_totale = 0.0
        vettori_squadra = []
        cluster_visti = set()
        penalita_ruoli_doppi = 0.0

        for p_id in squadra_id:
            # 1. Somma la forza predetta da LightGBM
            forza_totale += dict_forza[p_id]["score"]

            # Raccogli i vettori per il calcolo delle distanze
            vettori_squadra.append(dict_ruoli[p_id]['vector'])

            # Opzionale: Penalizza se due Pokémon hanno lo stesso cluster (ruolo)
            ruolo = dict_ruoli[p_id]['cluster']
            if ruolo in cluster_visti:
                penalita_ruoli_doppi += 1.0  # Valore da bilanciare empiricamente
            cluster_visti.add(ruolo)

        # 2. Calcola la diversità usando la distanza coseno media tra tutti i membri
        # pdist calcola le distanze a coppie (15 coppie per 6 vettori)
        # La distanza coseno va da 0 (identici) a 2 (opposti).
        distanze = pdist(vettori_squadra, metric='cosine')
        diversita_media = np.mean(distanze)

        # 3. Score finale
        score_finale = (alpha * forza_totale) + (beta * diversita_media) - penalita_ruoli_doppi

        return score_finale

    return calcola_punteggio_sinergico

def carica_ruoli_da_csv(file_ruoli: str) -> Dict[str, Dict]:
    """
    Legge il CSV esportato dal Role Analyzer e restituisce un dizionario
    con i cluster e i vettori delle feature normalizzate per il calcolo delle distanze.
    """
    dict_ruoli = {}
    try:
        with open(file_ruoli, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)

            # Identifichiamo dinamicamente le colonne
            # L'ID dovrebbe essere la prima colonna (indice 0)
            try:
                idx_cluster = header.index('cluster_label')
                # Tutto ciò che viene DOPO cluster_label sono le feature normalizzate (il vettore)
                idx_vector_start = idx_cluster + 1
            except ValueError:
                print("Errore: la colonna 'cluster_label' non esiste nel file esportato.")
                return {}

            for riga in reader:
                if not riga:
                    continue
                p_id = riga[0].strip()
                cluster_id = int(riga[idx_cluster])
                # Convertiamo le feature rimanenti in un array numpy per pdist
                vettore = np.array([float(v) for v in riga[idx_vector_start:]], dtype=np.float32)

                dict_ruoli[p_id] = {
                    'cluster': cluster_id,
                    'vector': vettore
                }
    except FileNotFoundError:
        print(f"Errore: Il file {file_ruoli} non è stato trovato.")

    return dict_ruoli

# --- CONFIGURAZIONE VINCOLI ---
TEAM_PARZIALE = ["6", "25", "149"] #Insieme di pokémon obbligatori, può essere cambiato dall'utente per partire da un team diverso

if __name__ == "__main__":
    # --- 1. SETUP DEI FILE ---
    # I file in ingresso che devono essere già stati generati dagli altri script
    FILE_NOMI = 'dataset/combat_pokemon_db.csv'
    FILE_DATI = 'dataset/pokemon_strength_scores.csv'
    FILE_RUOLI = 'unsupervised_learning/pokemon_clusters.csv'  # O 'pokemon_roles.csv' se lo rinomini

    # --- 2. CARICAMENTO DATI IN RAM ---
    print("Caricamento ranking forza...")
    db_pokemon = ottieni_ranking(FILE_DATI, FILE_NOMI)

    print("Caricamento ruoli precalcolati...")
    dizionario_ruoli = carica_ruoli_da_csv(FILE_RUOLI)

    if not db_pokemon or not dizionario_ruoli:
        print("Errore nel caricamento dei dati. Controlla i file CSV.")
        exit(1)

    # --- 3. PREPARAZIONE RICERCA ---
    # Prendiamo solo gli ID che esistono in ENTRAMBI i dataset per evitare KeyError
    lista_id = [p_id for p_id in db_pokemon.keys() if p_id in dizionario_ruoli]

    print(f"Pokémon validi pronti per la selezione: {len(lista_id)}")

    evaluator = crea_valutatore_squadra(db_pokemon, dizionario_ruoli, alpha=1.0, beta=50.0)

    # --- 4. AVVIO RICERCA ---
    print("\nInizio la ricerca con Greedy Ascent...")
    team_id, punti_con_distanza = greedy_ascent(lista_id, evaluator, TEAM_PARZIALE)
    true_eval = crea_valutatore_squadra(db_pokemon, dizionario_ruoli, alpha=1.0, beta=0.0)
    punti = true_eval(team_id)

    # --- 5. RISULTATI ---
    print("\n" + "=" * 50)
    print("=== SQUADRA OTTIMA (LOCALE) TROVATA ===")
    print("=" * 50)
    print(f"Punteggio Totale (Forza + Sinergia): {punti:.2f}\n")

    for i, p_id in enumerate(team_id, 1):
        nome = db_pokemon[p_id]['nome']
        score = db_pokemon[p_id]['score']
        cluster = dizionario_ruoli[p_id]['cluster']
        print(f"{i}. {nome:<20} (ID: {p_id:>3}) | Score: {score:7.3f} | Ruolo: Cluster {cluster}")