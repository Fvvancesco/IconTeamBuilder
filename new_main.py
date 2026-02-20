from typing import List, Dict, Callable

import numpy as np
from scipy.spatial.distance import pdist
import os

from csp.csp import greedy_ascent
import unsupervised_learning.role_analyzer
from supervised_learning.ranking import ottieni_ranking
from unsupervised_learning.role_analyzer import PokemonRoleAnalyzer


def crea_valutatore_squadra(dict_forza: Dict, dict_ruoli: Dict, alpha: float = 1.0, beta: float = 0.5) -> Callable[
    [List[str]], float]:
    """
    dict_forza: Dizionario { 'id': {'nome': str, 'score': float} }
    dict_ruoli: Dizionario { 'id': {'name': str, 'cluster': int, 'vector': np.array} }
    alpha: Moltiplicatore della forza bruta predetta
    beta: Moltiplicatore della diversità della squadra
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

            # Penalizza fortemente se due Pokémon hanno lo stesso ruolo (stesso cluster)
            ruolo = dict_ruoli[p_id]['cluster']
            if ruolo in cluster_visti:
                penalita_ruoli_doppi += 1000.0  # Valore da bilanciare empiricamente
            cluster_visti.add(ruolo)

        # 2. Calcola la diversità usando la distanza coseno media tra tutti i membri
        # La distanza coseno va da 0 (identici) a 2 (opposti)
        distanze = pdist(vettori_squadra, metric='cosine')
        diversita_media = np.mean(distanze)

        # 3. Score finale
        score_finale = (alpha * forza_totale) + (beta * diversita_media * 10) - penalita_ruoli_doppi

        return score_finale

    return calcola_punteggio_sinergico


if __name__ == "__main__":
    # 1. Configurazione Percorsi File
    FILE_NOMI = "dataset/combat_pokemon_db.csv"  # Sostituisci con il path reale se diverso
    FILE_DATI = "supervised_learning/pokemon_strength_scores.csv"  # Generato da gradient_tree.py

    if not os.path.exists(FILE_NOMI) or not os.path.exists(FILE_DATI):
        print(f"Errore: File mancanti. Assicurati che '{FILE_NOMI}' e '{FILE_DATI}' siano presenti.")
        print("Ricorda di eseguire prima 'gradient_tree.py' per generare i punteggi.")
        exit(1)

    # 2. Caricamento Dati
    print("Caricamento ranking forza...")
    db_pokemon = ottieni_ranking(FILE_DATI, FILE_NOMI)

    print("Inizializzazione e analisi ruoli (Clustering)...")
    pk = PokemonRoleAnalyzer(FILE_NOMI, verbose=True)
    pk.load_data()

    # Identifichiamo dinamicamente le colonne delle statistiche base presenti
    statistiche_base = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    statistiche_base_alt = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']

    colonne_da_scalare = [c for c in statistiche_base if c in pk.df.columns]
    if not colonne_da_scalare:
        colonne_da_scalare = [c for c in statistiche_base_alt if c in pk.df.columns]

    # Usiamo lo zscore per normalizzare le statistiche per il KMeans
    pk.scale_features(method="zscore", features=colonne_da_scalare)

    # Estraiamo 6 cluster (ruoli ottimali in un team) e salviamo un log
    dizionario_ruoli = pk.fit_and_export(k=6, output_csv="pokemon_roles_output.csv")

    # 3. Intersezione Sicura degli ID
    # Garantiamo che l'algoritmo processi solo Pokémon presenti in *entrambi* i dizionari
    lista_id = [p_id for p_id in db_pokemon.keys() if p_id in dizionario_ruoli]

    if len(lista_id) < 6:
        print("Errore: Non ci sono abbastanza Pokémon validi in comune tra i dataset.")
        exit(1)

    print(f"\nPokémon validi pronti per la selezione: {len(lista_id)}")

    # 4. Creazione Valutatore e Ricerca
    evaluator = crea_valutatore_squadra(db_pokemon, dizionario_ruoli, alpha=1.0, beta=20.0)

    print("\nInizio la ricerca con Greedy Ascent (Hill Climbing)...")
    team_id, punti = greedy_ascent(lista_id, evaluator)

    # 5. Stampa Risultati
    print("\n" + "=" * 50)
    print("=== SQUADRA OTTIMALE (OTTIMO LOCALE) TROVATA ===")
    print("=" * 50)
    print(f"Punteggio Sinergico Totale: {punti:.2f}\n")

    for i, p_id in enumerate(team_id, 1):
        nome = db_pokemon[p_id]['nome']
        score = db_pokemon[p_id]['score']
        ruolo_id = dizionario_ruoli[p_id]['cluster']
        print(f"{i}. {nome:<20} | ID: {p_id:>3} | Score Forza: {score:7.3f} | Cluster Ruolo: {ruolo_id}")