# train_strength_model.py
import warnings

warnings.filterwarnings("ignore")

from lightgbm import LGBMRanker
from sklearn.model_selection import GroupKFold, ParameterSampler
import pandas as pd
import numpy as np
import joblib
import warnings

from ranking import ottieni_ranking, stampa_ranking

from gradient_tree import build_pokemon_feature_table, POKEMON_PATH, MODEL_OUT, SCORES_OUT

def analizza_feature_importance(modello, feature_names):
    print("\n" + "=" * 40)
    print("FEATURE IMPORTANCE (Split)")
    print("=" * 40)

    importance = modello.booster_.feature_importance(importance_type='split')

    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importanza': importance
    }).sort_values('Importanza', ascending=False)

    for _, row in feat_imp.iterrows():
        print(f"- {row['Feature']:<15}: {row['Importanza']}")


def simula_scontro(modello, pokemon_df, id_p1, id_p2):
    print("\n" + "=" * 40)
    print(f"SIMULAZIONE SCONTRO: ID {id_p1} vs ID {id_p2}")
    print("=" * 40)

    features = build_pokemon_feature_table(pokemon_df)

    if id_p1 not in features.index or id_p2 not in features.index:
        print("Errore: ID Pokémon non validi.")
        return

    p1_feats = features.loc[id_p1].copy()
    p2_feats = features.loc[id_p2].copy()

    X_s1 = pd.DataFrame([
        list(p1_feats) + [1.0],
        list(p2_feats) + [0.0]
    ], columns=list(features.columns) + ['is_first']).astype(np.float32)

    X_s2 = pd.DataFrame([
        list(p1_feats) + [0.0],
        list(p2_feats) + [1.0]
    ], columns=list(features.columns) + ['is_first']).astype(np.float32)

    pred_s1 = modello.predict(X_s1)
    pred_s2 = modello.predict(X_s2)

    score_p1 = (pred_s1[0] + pred_s2[0]) / 2
    score_p2 = (pred_s1[1] + pred_s2[1]) / 2

    print(f"Punteggio in questo scontro - P1 (ID {id_p1}): {score_p1:.4f}")
    print(f"Punteggio in questo scontro - P2 (ID {id_p2}): {score_p2:.4f}")

    vincitore = id_p1 if score_p1 > score_p2 else id_p2
    print(f"Il modello prevede che VINCERÀ IL POKÉMON CON ID: {vincitore}")


def main():
    try:
        modello = joblib.load(MODEL_OUT)
        pokemon_df = pd.read_csv(POKEMON_PATH)
        features = build_pokemon_feature_table(pokemon_df)
        feature_names = features.columns.tolist() + ["is_first"]

        analizza_feature_importance(modello, feature_names)

        print("\n" + "=" * 40)
        print("ANALISI CLASSIFICA GENERALE")
        print("=" * 40)

        ranking = ottieni_ranking(SCORES_OUT, POKEMON_PATH)

        if ranking:
            print("\nI 5 Pokémon più forti in assoluto:")
            ranking_ord = sorted(ranking.items(), key=lambda x: x[1]['score'], reverse=True)
            top_5 = {k: v for k, v in ranking_ord[:5]}
            stampa_ranking(top_5)

            print("\nI 5 Pokémon più deboli in assoluto:")
            flop_5 = {k: v for k, v in ranking_ord[-5:]}
            stampa_ranking(flop_5)

        simula_scontro(modello, pokemon_df, id_p1=1, id_p2=150)

    except FileNotFoundError as e:
        print(f"Errore: File mancante. Esegui prima gradient_tree.py! Dettagli: {e}")


if __name__ == "__main__":
    main()