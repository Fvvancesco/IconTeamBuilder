from __future__ import annotations

import itertools
from typing import Any, Dict, List, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from role_analyzer import PokemonRoleAnalyzer

def partitions_of_set(items: List[str]) -> Iterable[List[List[str]]]:
    """Genera tutte le partizioni di un insieme (Numeri di Bell)."""
    if not items:
        yield []
        return

    first = items[0]
    for part in partitions_of_set(items[1:]):
        # inserisci in un gruppo esistente
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]
        # crea nuovo gruppo
        yield [[first]] + part


def generate_grid_search_total(
        base_stats: List[str],
        aggregation_methods: List[str] = ["mean", "max"],
        scalers: List[str] = ["ratio", "zscore"],
        k_test: List[int] = [4, 5, 6],
) -> List[Dict[str, Any]]:
    experiments: List[Dict[str, Any]] = []
    parts = list(partitions_of_set(base_stats))
    counter = 1

    for part in parts:
        for methods in itertools.product(aggregation_methods, repeat=len(part)):
            for scaler in scalers:
                aggregations: Dict[str, Dict[str, Any]] = {}
                features: List[str] = []

                for idx, (group, method) in enumerate(zip(part, methods), start=1):
                    feat = f"macro_feat_{idx}"
                    features.append(feat)
                    aggregations[feat] = {"cols": group, "method": method}

                experiments.append(
                    {
                        "name": f"{counter}. [GS] groups:{len(part)} scaler:{scaler}",
                        "aggregations": aggregations,
                        "features": features,
                        "scaling": scaler,
                        "k_test": k_test,
                    }
                )
                counter += 1

    return experiments


def run_single_experiment(
        exp: Dict[str, Any],
        data_path: str,
        df_base: pd.DataFrame,
        id_col: str,
        random_state: int = 42,
        n_init: int = 10,
) -> List[Dict[str, Any]]:
    """
    Worker robusto: ritorna lista risultati per k.
    Se qualcosa fallisce, stampa l'errore e ritorna [].
    """
    analyzer = PokemonRoleAnalyzer(data_path)
    analyzer.df = df_base.copy()
    analyzer.id_col = id_col

    # 1. Aggregazioni
    try:
        if exp.get("aggregations"):
            analyzer.aggregate_features(exp["aggregations"])
    except Exception as e:
        #print(f"Errore aggregazione in '{exp.get('name', 'Sconosciuto')}': {e}")
        return []

    # 2. Scaling (Rimosso fillna=None per compatibilità con role_analyzer immutabile)
    try:
        analyzer.scale_features(exp["features"], method=exp["scaling"])
    except Exception as e:
        #print(f"Errore scaling in '{exp.get('name', 'Sconosciuto')}': {e}")
        return []

    results: List[Dict[str, Any]] = []
    X = analyzer.X_scaled

    if X is None or len(X) == 0:
        return []

    # Descrizione struttura per i log
    structure = " | ".join(
        [f"{'+'.join(v['cols'])}({v['method']})" for v in exp.get("aggregations", {}).values()]
    )

    # 3. Valutazione Clustering
    for k in exp.get("k_test", []):
        if k < 2 or k >= len(X):
            continue
        try:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
            labels = km.fit_predict(X)
            score = float(silhouette_score(X, labels))
            results.append(
                {
                    "K": k,
                    "Scaler": exp["scaling"],
                    "N_Groups": len(exp.get("aggregations", {})),
                    "Structure": structure,
                    "Silhouette": score,
                    "Experiment": exp["name"],
                }
            )
        except Exception:
            continue

    return results