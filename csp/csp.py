import random
from typing import List, Callable, Tuple

def genera_team_casuale_di_6(lista_id: List[str]) -> List[str]:
    """Sceglie 6 ID univoci casuali dalla lista dei Pokémon disponibili."""
    # random.sample garantisce che non ci siano duplicati iniziali
    return random.sample(lista_id, 6)

def greedy_ascent(lista_id_pokemon: List[str], black_box_score: Callable[[List[str]], float]) -> Tuple[List[str], float]:
    """
    Esegue l'algoritmo Hill Climbing (Greedy Ascent) per trovare un ottimo locale.
    """
    # 1. Inizializzazione casuale
    team_corrente = genera_team_casuale_di_6(lista_id_pokemon)
    punteggio_corrente = black_box_score(team_corrente)


    while True:
        miglior_vicino = None
        miglior_punteggio_vicino = punteggio_corrente

        # 2. Genera e valuta i vicini (cambiando 1 Pokemon alla volta)
        for i in range(6):  # Per ogni slot del team
            for id_pokemon in lista_id_pokemon:
                if id_pokemon not in team_corrente:
                    # Crea un nuovo team scambiando il Pokemon nello slot i
                    team_vicino = team_corrente.copy()
                    team_vicino[i] = id_pokemon

                    # 3. Valuta con la black-box
                    punteggio_vicino = black_box_score(team_vicino)

                    if punteggio_vicino > miglior_punteggio_vicino:
                        miglior_vicino = team_vicino
                        miglior_punteggio_vicino = punteggio_vicino

        # 4. Condizione di stop (Ottimo locale)
        if miglior_vicino is None:
            break  # Nessun vicino migliora il punteggio

        # Aggiorna e continua
        team_corrente = miglior_vicino
        punteggio_corrente = miglior_punteggio_vicino
        # Opzionale: print per vedere la salita: print(f"Nuovo ottimo trovato: {punteggio_corrente}")

    return team_corrente, punteggio_corrente