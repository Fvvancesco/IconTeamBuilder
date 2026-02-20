import pandas as pd
from joblib import Parallel, delayed

from role_analyzer import PokemonRoleAnalyzer
import grid_search
"""
esperimenti_db = [
{
        {
            "nome": "7. Offesa Totale vs Difesa Totale (MinMax)",
            "aggregazioni": {
                'total_offense': {'cols': ['attack', 'sp_attack', 'speed'], 'method': 'sum'},
                'total_defense': {'cols': ['hp', 'defense', 'sp_defense'], 'method': 'sum'}
            },
            "feature": ['total_offense', 'total_defense'],
            "scaling": "minmax",
            "k_test": [3, 4, 5, 6]
        },
        {
            "nome": "8. No HP - Solo Attive (Ratio L2)",
            "aggregazioni": None,
            "feature": ['attack', 'defense', 'sp_attack', 'sp_defense', 'speed'],
            "scaling": "l2",
            "k_test": [4, 5, 6, 7]
        },
        {
            "nome": "9. Preservazione Zeri (MaxAbs Scaler)",
            "aggregazioni": None,
            "feature": ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed'],
            "scaling": "maxabs",
            "k_test": [4, 5, 6, 7]
        },
        {
            "nome": "10. Mixed Tank e Sweeper (RobustScaler Aggregato)",
            "aggregazioni": {
                'mixed_offense': {'cols': ['attack', 'sp_attack'], 'method': 'mean'},
                'mixed_defense': {'cols': ['defense', 'sp_defense'], 'method': 'mean'}
            },
            "feature": ['hp', 'mixed_offense', 'mixed_defense', 'speed'],
            "scaling": "robust",
            "k_test": [4, 5, 6]
        },
        {
            "nome": "11. Analisi Ruoli Competitivi (Ratio + Bias)",
            "aggregazioni": {
                "phys_bias": {'cols': ['attack', 'defense'], 'method': 'mean'},
                "spec_bias": {'cols': ['sp_attack', 'sp_defense'], 'method': 'mean'},
                "speed_impact": {'cols': ['speed'], 'method': 'sum'}
            },
            "feature": ['phys_bias', 'spec_bias', 'speed_impact', 'hp'],
            "scaling": "l2",
            "k_test": [4, 5, 6, 7]
        },
        {
            "nome": "12. Strategia Team Variegato (Separazione Fisico/Speciale)",
            "aggregazioni": {
                "atk_bias": {'cols': ['attack', 'sp_attack'], 'method': 'diff'},
                # Positivo = Fisico, Negativo = Speciale
                "def_bias": {'cols': ['defense', 'sp_defense'], 'method': 'diff'},
                "offense_total": {'cols': ['attack', 'sp_attack'], 'method': 'mean'},
                "bulk_total": {'cols': ['hp', 'defense', 'sp_defense'], 'method': 'mean'}
            },
            "feature": ['atk_bias', 'def_bias', 'offense_total', 'speed', 'bulk_total'],
            "scaling": "zscore",  # Z-score aiuta a enfatizzare le differenze dai valori medi
            "k_test": [2, 3, 4, 5, 6, 7, 8, 9, 10]
        },
        { #Soluzione particolarmente buona, funge come da selettore di statistiche nei videogiochi, i pokémon si specializzano
            "nome": "13. Il Triangolo dei Ruoli (Ternary Ratio)",
            "aggregazioni": {
                "offense_pool": {'cols': ['attack', 'sp_attack'], 'method': 'sum'},
                "bulk_pool": {'cols': ['hp', 'defense', 'sp_defense'], 'method': 'sum'}
                # La velocità la prendiamo pura senza aggregarla
            },
            # Usiamo solo 3 macro-contenitori: Potenza, Resistenza, Velocità
            "feature": ['offense_pool', 'bulk_pool', 'speed'],
            # 'ratio' assicura che Offense + Bulk + Speed = 100%
            "scaling": "ratio",
            "k_test": [3,4,5,6]
        },
        {
            "nome": "14. Il Triangolo dei Ruoli (Ternary L2)",
            "aggregazioni": {
                "offense_pool": {'cols': ['attack', 'sp_attack'], 'method': 'sum'},
                "bulk_pool": {'cols': ['hp', 'defense', 'sp_defense'], 'method': 'sum'}
                # La velocità la prendiamo pura senza aggregarla
            },
            # Usiamo solo 3 macro-contenitori: Potenza, Resistenza, Velocità
            "feature": ['offense_pool', 'bulk_pool', 'speed'],
            # IL SEGRETO È QUI: 'ratio' assicura che Offense + Bulk + Speed = 1 (o 100%)
            "scaling": "l2",
            "k_test": [3, 4, 5, 6]
        },
        {
            "nome": "15. Il Triangolo dei Ruoli (Robust)",
            "aggregazioni": {
                "offense_pool": {'cols': ['attack', 'sp_attack'], 'method': 'sum'},
                "bulk_pool": {'cols': ['hp', 'defense', 'sp_defense'], 'method': 'sum'}
                # La velocità la prendiamo pura senza aggregarla
            },
            # Usiamo solo 3 macro-contenitori: Potenza, Resistenza, Velocità
            "feature": ['offense_pool', 'bulk_pool', 'speed'],
            # IL SEGRETO È QUI: 'ratio' assicura che Offense + Bulk + Speed = 1 (o 100%)
            "scaling": "robust",
            "k_test": [3, 4, 5, 6]
        },
        #Attaccanti
        {
            "nome": "16. Separazione Sweeper (Fisico Veloce vs Speciale Veloce)",
            "aggregazioni": {
                "fast_phys": {'cols': ['attack', 'speed'], 'method': 'mean'},
                "fast_spec": {'cols': ['sp_attack', 'speed'], 'method': 'mean'},
                "bulk_totale": {'cols': ['hp', 'defense', 'sp_defense'], 'method': 'mean'}
            },
            "feature": ['fast_phys', 'fast_spec', 'bulk_totale'],
            "scaling": "ratio",  # Ratio per vedere la percentuale di focus
            "k_test": [5, 6, 7]
        },
        {
            "nome": "17. Indice Glass Cannon (Puro Danno e Velocità vs Resto)",
            "aggregazioni": {
                "glass_cannon_idx": {'cols': ['attack', 'sp_attack', 'speed'], 'method': 'sum'},
                "sponge_idx": {'cols': ['hp', 'defense', 'sp_defense'], 'method': 'sum'}
            },
            "feature": ['glass_cannon_idx', 'sponge_idx'],
            "scaling": "minmax",
            "k_test": [4, 5, 6]
        },
        {
            "nome": "18. Wallbreaker Supremacy (Massimo Danno Ibrido)",
            "aggregazioni": {
                "best_offense": {'cols': ['attack', 'sp_attack'], 'method': 'max'},  # Prende la stat offensiva più alta
                "bulk_medio": {'cols': ['hp', 'defense', 'sp_defense'], 'method': 'mean'}
            },
            "feature": ['best_offense', 'bulk_medio', 'speed'],
            "scaling": "zscore",
            "k_test": [5, 6]
        },
        {
            "nome": "19. Momentum Offensivo (Velocità come moltiplicatore)",
            "aggregazioni": {
                "momentum_fisico": {'cols': ['attack', 'speed'], 'method': 'sum'},
                "momentum_speciale": {'cols': ['sp_attack', 'speed'], 'method': 'sum'},
                "pura_difesa": {'cols': ['defense', 'sp_defense'], 'method': 'sum'}
            },
            "feature": ['momentum_fisico', 'momentum_speciale', 'pura_difesa', 'hp'],
            "scaling": "l2",
            "k_test": [6, 7]
        },
        {
            "nome": "20. L'Attaccante Misto (Mixed Sweeper Radar)",
            "aggregazioni": {
                "mixed_offense": {'cols': ['attack', 'sp_attack'], 'method': 'mean'},
                "mixed_bulk": {'cols': ['defense', 'sp_defense'], 'method': 'mean'}
            },
            "feature": ['mixed_offense', 'mixed_bulk', 'speed', 'hp'],
            "scaling": "robust",
            "k_test": [5, 6, 7]
        },

        #Difensori
        {
            "nome": "21. Il Bruiser e il Mago (Cores Fisici vs Speciali)",
            "aggregazioni": {
                "bruiser_core": {'cols': ['attack', 'defense'], 'method': 'sum'},
                "mage_core": {'cols': ['sp_attack', 'sp_defense'], 'method': 'sum'}
            },
            "feature": ['bruiser_core', 'mage_core', 'hp', 'speed'],
            "scaling": "ratio",
            "k_test": [5, 6]
        },
        {
            "nome": "22. Specializzazione Mura (Wall Fisico vs Speciale)",
            "aggregazioni": {
                "phys_wall": {'cols': ['hp', 'defense'], 'method': 'sum'},
                "spec_wall": {'cols': ['hp', 'sp_defense'], 'method': 'sum'},
                "best_attack": {'cols': ['attack', 'sp_attack'], 'method': 'max'}
            },
            "feature": ['phys_wall', 'spec_wall', 'best_attack', 'speed'],
            "scaling": "l2",
            "k_test": [6]
        },
        {
            "nome": "23. Il Tank Agile (Sopravvivenza + Velocità)",
            "aggregazioni": {
                "agile_bulk": {'cols': ['defense', 'sp_defense', 'speed'], 'method': 'mean'},
                "potenza_fuoco": {'cols': ['attack', 'sp_attack'], 'method': 'max'}
            },
            "feature": ['agile_bulk', 'potenza_fuoco', 'hp'],
            "scaling": "zscore",
            "k_test": [4, 5, 6]
        },
        {
            "nome": "24. HP come Risorsa Suprema",
            "aggregazioni": {
                "avg_offense": {'cols': ['attack', 'sp_attack'], 'method': 'mean'},
                "avg_defense": {'cols': ['defense', 'sp_defense'], 'method': 'mean'}
            },
            "feature": ['hp', 'avg_offense', 'avg_defense', 'speed'],
            "scaling": "minmax",
            "k_test": [5, 6, 7]
        },
        {
            "nome": "25. Profilo da Stall/Evasion (Logoramento)",
            "aggregazioni": {
                "stall_core": {'cols': ['defense', 'sp_defense', 'hp'], 'method': 'mean'},
                "evasion_core": {'cols': ['speed', 'hp'], 'method': 'mean'},
                "danno_diretto": {'cols': ['attack', 'sp_attack'], 'method': 'max'}
            },
            "feature": ['stall_core', 'evasion_core', 'danno_diretto'],
            "scaling": "ratio",
            "k_test": [6]
        },

        #Support
        {
            "nome": "26. Pivot Lento vs Pivot Veloce",
            "aggregazioni": {
                "slow_pivot_bulk": {'cols': ['hp', 'defense', 'sp_defense'], 'method': 'sum'},
                "fast_pivot_util": {'cols': ['hp', 'speed'], 'method': 'sum'},
                "minaccia_offensiva": {'cols': ['attack', 'sp_attack'], 'method': 'max'}
            },
            "feature": ['slow_pivot_bulk', 'fast_pivot_util', 'minaccia_offensiva'],
            "scaling": "ratio",
            "k_test": [5, 6, 7]
        },
        {
            "nome": "27. Bilanciamento per il Doppio",
            "aggregazioni": {
                "vgc_survivability": {'cols': ['hp', 'defense', 'sp_defense', 'speed'], 'method': 'mean'},
                "vgc_offense": {'cols': ['attack', 'sp_attack'], 'method': 'max'}
            },
            "feature": ['vgc_survivability', 'vgc_offense'],
            "scaling": "zscore",
            "k_test": [4, 5, 6]
        },
        {
            "nome": "28. Suicide Lead",
            "aggregazioni": {
                "suicide_stats": {'cols': ['attack', 'sp_attack', 'speed'], 'method': 'sum'},
                "useless_bulk": {'cols': ['defense', 'sp_defense'], 'method': 'mean'}
            },
            "feature": ['suicide_stats', 'useless_bulk', 'hp'],
            "scaling": "l2",
            "k_test": [5, 6]
        },
        {
            "nome": "29. Asimmetria Pura (Uso della Differenza)",
            "aggregazioni": {
                "atk_diff": {'cols': ['attack', 'sp_attack'], 'method': 'diff'},
                "def_diff": {'cols': ['defense', 'sp_defense'], 'method': 'diff'}
            },
            "feature": ['atk_diff', 'def_diff', 'speed', 'hp'],
            "scaling": "robust",  # Ottimo per gestire valori negativi generati da 'diff'
            "k_test": [6]
        },
        {
            "nome": "30. Speed Tiering (La Velocità domina)",
            "aggregazioni": {
                "all_other_stats": {'cols': ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense'], 'method': 'mean'}
            },
            "feature": ['speed', 'all_other_stats'],
            "scaling": "minmax",
            "k_test": [4, 5, 6]
        },

        #Ibrido
        {
            "nome": "31. Il Triathlon (Fisico, Speciale, Utility)",
            "aggregazioni": {
                "ramo_fisico": {'cols': ['attack', 'defense'], 'method': 'sum'},
                "ramo_speciale": {'cols': ['sp_attack', 'sp_defense'], 'method': 'sum'},
                "ramo_utility": {'cols': ['hp', 'speed'], 'method': 'sum'}
            },
            "feature": ['ramo_fisico', 'ramo_speciale', 'ramo_utility'],
            "scaling": "ratio",
            "k_test": [6]
        },
        {
            "nome": "32. Estremi Assoluti (Max vs Max)",
            "aggregazioni": {
                "punta_offensiva": {'cols': ['attack', 'sp_attack'], 'method': 'max'},
                "punta_difensiva": {'cols': ['defense', 'sp_defense'], 'method': 'max'},
                "punta_utility": {'cols': ['hp', 'speed'], 'method': 'max'}
            },
            "feature": ['punta_offensiva', 'punta_difensiva', 'punta_utility'],
            "scaling": "l2",
            "k_test": [5, 6, 7]
        },
        {
            "nome": "33. Pressione vs Stabilità (Approccio Macroscopico)",
            "aggregazioni": {
                "pressione_totale": {'cols': ['attack', 'sp_attack', 'speed'], 'method': 'sum'},
                "stabilita_totale": {'cols': ['hp', 'defense', 'sp_defense'], 'method': 'sum'}
            },
            "feature": ['pressione_totale', 'stabilita_totale'],
            "scaling": "ratio",
            "k_test": [4, 5, 6]
        },
        {
            "nome": "34. Il Quad-Core (4 Dimensioni Bilanciate)",
            "aggregazioni": {
                "phys_core": {'cols': ['attack', 'defense'], 'method': 'sum'},
                "spec_core": {'cols': ['sp_attack', 'sp_defense'], 'method': 'sum'}
            },
            "feature": ['phys_core', 'spec_core', 'hp', 'speed'],
            "scaling": "minmax",
            "k_test": [6]
        },
        {
            "nome": "35. Il Nuovo Esagono (Triangolo modificato con Diff)",
            "aggregazioni": {
                "offense_pool": {'cols': ['attack', 'sp_attack'], 'method': 'sum'},
                "bulk_pool": {'cols': ['hp', 'defense', 'sp_defense'], 'method': 'sum'},
                "atk_bias": {'cols': ['attack', 'sp_attack'], 'method': 'diff'}
            },
            # Qui mischiamo il triangolo puro con una spinta direzionale (bias)
            "feature": ['offense_pool', 'bulk_pool', 'speed', 'atk_bias'],
            "scaling": "zscore",
            "k_test": [6]
        },
            #Migliore sia come score che per explainability
        {
            "nome": "36. Attacchi e Difese",
            "aggregazioni": {
                "offense_pool": {'cols': ['attack', 'sp_attack'], 'method': 'max'},
                "bulk_pool": {'cols': ['defense', 'sp_defense'], 'method': 'max'},
                "final_attack": {'cols': ["offense_pool", "speed"], 'method': 'sum'},
                "final_defense": {'cols': ["bulk_pool", "hp"], 'method': 'sum'},
            },
            "feature": ['final_attack', 'final_defense'],
            "scaling": "ratio",
            "k_test": [2,3,4,5, 6,7,8]
        },
    }
]
"""
esperimenti_manuali = [
    {
        "name": "36. Attacchi e Difese",
        "aggregations": {
            "offense_pool": {'cols': ['attack', 'sp_attack'], 'method': 'max'},
            "bulk_pool": {'cols': ['defense', 'sp_defense'], 'method': 'max'},
            "final_attack": {'cols': ["offense_pool", "speed"], 'method': 'sum'},
            "final_defense": {'cols': ["bulk_pool", "hp"], 'method': 'sum'},
        },
        "features": ['final_attack', 'final_defense'],
        "scaling": "ratio",
        "k_test": [2, 3, 4, 5, 6, 7, 8]
    },
]


# -----------------------------
# 2) Esecuzione Lista Predefinita
# -----------------------------
def run_experiments(data_path: str, experiments: list, base_filter: str = "total_points > 500") -> pd.DataFrame:
    analyzer_base = PokemonRoleAnalyzer(data_path)
    analyzer_base.load_data()
    if base_filter:
        analyzer_base.filter_data(base_filter)

    print(f"\nEsecuzione in parallelo di {len(experiments)} esperimenti manuali...")

    results_nested = Parallel(n_jobs=-1)(
        delayed(grid_search.run_single_experiment)(
            exp,
            data_path,
            analyzer_base.df,
            analyzer_base.id_col,
        )
        for exp in experiments
    )

    results = [r for sub in results_nested for r in sub]
    df = pd.DataFrame(results)

    if df.empty:
        print("Nessun risultato valido dagli esperimenti.")
        return df

    df = df.sort_values("Silhouette", ascending=False).reset_index(drop=True)

    print("\n RISULTATI ESPERIMENTI MANUALI:")
    print("-" * 140)
    print(df.to_string(index=False))
    print("-" * 140)
    return df



# Esecuzione Grid Search Completa
def run_grid_search(data_path: str, base_filter: str = "total_points > 500", limit: int | None = 300) -> pd.DataFrame:
    analyzer_base = PokemonRoleAnalyzer(data_path)
    analyzer_base.load_data()
    if base_filter:
        analyzer_base.filter_data(base_filter)

    base_stats = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]

    experiments = grid_search.generate_grid_search_total(
        base_stats=base_stats,
        aggregation_methods=["mean", "max", "min", "sum", "diff"],
        scalers=["ratio", "zscore", "l2", "robust"],
        k_test=[4, 5, 6],
    )

    if limit is not None:
        experiments = experiments[:limit]

    print(f"\nGRID SEARCH in parallelo: {len(experiments)} esperimenti...")

    results_nested = Parallel(n_jobs=-1)(
        delayed(grid_search.run_single_experiment)(
            exp,
            data_path,
            analyzer_base.df,
            analyzer_base.id_col,
        )
        for exp in experiments
    )

    results = [r for sub in results_nested for r in sub]
    df = pd.DataFrame(results)

    if df.empty:
        print("Nessun risultato valido dalla grid search.")
        return df

    df = df.sort_values("Silhouette", ascending=False).reset_index(drop=True)

    print("\nTOP 10 GRID SEARCH:")
    print("-" * 140)
    print(df.head(10).to_string(index=False))
    print("-" * 140)
    return df



if __name__ == "__main__":
    DATA_PATH = "../Dataset/pokemon_cleaned.csv"

    print("Quale task vuoi eseguire?")
    print("1) Esegui lista di esperimenti manuali")
    print("2) Esegui Grid Search completa")
    scelta = input("Seleziona (1 o 2): ")

    if scelta == '1':
        run_experiments(data_path=DATA_PATH, experiments=esperimenti_manuali)
    elif scelta == '2':
        run_grid_search(data_path=DATA_PATH, limit=None)
    else:
        print("Scelta non valida, uscita in corso.")