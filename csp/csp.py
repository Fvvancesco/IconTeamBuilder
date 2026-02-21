import random
from typing import List, Callable, Tuple, Dict, Any

def genera_team_casuale(lista_id: List[str], n: int = 6) -> List[str]:
    """Sceglie n ID univoci casuali dalla lista dei Pokémon disponibili."""
    # random.sample garantisce che non ci siano duplicati iniziali
    return random.sample(lista_id, n)

def greedy_ascent(
        lista_id_pokemon: List[str],
        black_box_score: Callable[[List[str]], float],
        fixed_ids=[],
        num_restarts: int = 1,
        max_iterations: int = 1000
) -> Tuple[List[str], float]:
    """
    Esegue l'algoritmo Hill Climbing (Greedy Ascent) con supporto opzionale ai Random Restarts.
    """
    miglior_team_assoluto = None
    miglior_punteggio_assoluto = float('-inf')

    # Ciclo per i Riavvi Casuali (Random Restarts)
    for run in range(num_restarts):
        # 1. Inizializzazione casuale per la run corrente

        # Filtriamo gli ID già bloccati per evitare duplicati
        pool_disponibili = [pid for pid in lista_id_pokemon if pid not in fixed_ids]
        team_corrente = fixed_ids + genera_team_casuale(pool_disponibili, 6 - len(fixed_ids))
        punteggio_corrente = black_box_score(team_corrente)

        iterazione = 0
        # 2. Ricerca dell'ottimo locale (Hill Climbing)
        while iterazione < max_iterations:
            iterazione += 1
            miglior_vicino = None
            miglior_punteggio_vicino = punteggio_corrente

            # Genera e valuta i vicini (cambiando 1 Pokemon alla volta per ogni slot)
            for i in range(len(fixed_ids), 6):
                for id_pokemon in lista_id_pokemon:
                    if id_pokemon not in team_corrente:
                        # Crea un nuovo team scambiando il Pokemon nello slot i
                        team_vicino = team_corrente.copy()
                        team_vicino[i] = id_pokemon

                        punteggio_vicino = black_box_score(team_vicino)

                        if punteggio_vicino > miglior_punteggio_vicino:
                            miglior_vicino = team_vicino
                            miglior_punteggio_vicino = punteggio_vicino

            # Condizione di stop: nessun vicino migliora il punteggio (Ottimo locale raggiunto)
            if miglior_vicino is None:
                break

                # Aggiorna e continua la salita
            team_corrente = miglior_vicino
            punteggio_corrente = miglior_punteggio_vicino

        # Alla fine della run, verifica se è il miglior risultato globale mai trovato
        if punteggio_corrente > miglior_punteggio_assoluto:
            miglior_punteggio_assoluto = punteggio_corrente
            miglior_team_assoluto = team_corrente

    return miglior_team_assoluto, miglior_punteggio_assoluto