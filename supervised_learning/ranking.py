import csv
from typing import Dict


def ottieni_ranking(file_dati: str, file_nomi: str) -> Dict[str, Dict]:
    """
    Legge i CSV e restituisce un dizionario per un accesso O(1).
    Formato: { 'id_pokemon': {'nome': 'Pikachu', 'score': 85.5} }
    """
    nomi_mappa = {}
    try:
        with open(file_nomi, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # next(reader) # Scommenta se c'è un header in file_nomi
            for riga in reader:
                if len(riga) >= 2:
                    nomi_mappa[riga[0].strip()] = riga[1].strip()
    except FileNotFoundError:
        print(f"Errore: Il file {file_nomi} non è stato trovato.")
        return {}

    ranking_dict = {}
    try:
        with open(file_dati, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Salta l'header
            for riga in reader:
                if len(riga) >= 2:
                    id_entita = riga[0].strip()
                    score = float(riga[1].strip())
                    nome = nomi_mappa.get(id_entita, f"ID Sconosciuto ({id_entita})")

                    ranking_dict[id_entita] = {'nome': nome, 'score': score}
    except FileNotFoundError:
        print(f"Errore: Il file {file_dati} non è stato trovato.")
        return {}

    return ranking_dict


def stampa_ranking(ranking_dict: Dict[str, Dict]):
    """Stampa il ranking partendo dal dizionario già generato."""
    if not ranking_dict:
        return

    # Ordiniamo il dizionario per punteggio decrescente per la stampa
    ranking_ordinato = sorted(ranking_dict.items(), key=lambda x: x[1]['score'], reverse=True)

    print(f"{'POS':<5} | {'ID':<4} | {'NOME':<25} | {'STRENGTH SCORE'}")
    print("-" * 60)
    for i, (id_entita, dati) in enumerate(ranking_ordinato, 1):
        print(f"{i:<5} | {id_entita:<4} | {dati['nome']:<25} | {dati['score']:.4f}")