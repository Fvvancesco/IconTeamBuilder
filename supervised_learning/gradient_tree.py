# train_strength_model.py
import warnings

warnings.filterwarnings("ignore")

from lightgbm import LGBMRanker
from sklearn.model_selection import GroupKFold, ParameterSampler
import pandas as pd
import numpy as np
import joblib
import warnings



warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURAZIONI GLOBALI
# ==========================================
POKEMON_PATH = "../dataset/combat_pokemon_db.csv"
COMBATS_PATH = "../dataset/combats.csv"
MODEL_OUT = "strength_ranker_lgbm.pkl"
SCORES_OUT = "pokemon_strength_scores.csv"

ID_NAME = "#"
USE_GPU = False  # Ho riscontrato che rallenta l'allenamento, pare che non supporti l'accelerazione hw


def build_pokemon_feature_table(pokemon_df: pd.DataFrame) -> pd.DataFrame:
    """Pulisce le feature e codifica le categorie per LightGBM."""
    df = pokemon_df.copy()

    if ID_NAME not in df.columns:
        raise ValueError(f"Colonna '{ID_NAME}' non trovata in {POKEMON_PATH}")

    df = df.drop_duplicates(subset=[ID_NAME], keep="first")

    # Rimuoviamo il nome, non utile ai fini predittivi
    df = df.drop(columns=["Name"], errors="ignore")

    # Mappiamo i tipi a interi condividendo lo stesso dizionario (0 = Nessun tipo)
    all_types = pd.concat([df["Type 1"], df["Type 2"]]).dropna().unique()
    type_map = {t: i for i, t in enumerate(all_types, start=1)}

    df["Type 1"] = df["Type 1"].map(type_map).fillna(0).astype(int)
    df["Type 2"] = df["Type 2"].map(type_map).fillna(0).astype(int)

    # Assicuriamoci che Legendary sia numerico (0/1)
    if "Legendary" in df.columns:
        df["Legendary"] = df["Legendary"].astype(int)

    # Teniamo solo le colonne numeriche (ora includono anche i tipi codificati)
    keep = [ID_NAME] + [c for c in df.columns if c != ID_NAME and pd.api.types.is_numeric_dtype(df[c])]
    df = df[keep].fillna(0)

    df = df.set_index(ID_NAME).sort_index()
    return df.astype(np.float32)


def build_rank_dataset(features: pd.DataFrame, combats: pd.DataFrame):
    """Vettorizza la creazione del dataset per il ranking in O(N)."""
    valid_mask = combats["First_pokemon"].isin(features.index) & combats["Second_pokemon"].isin(features.index)
    combats_valid = combats[valid_mask]

    p1_ids = combats_valid["First_pokemon"].values
    p2_ids = combats_valid["Second_pokemon"].values
    winners = combats_valid["Winner"].values

    n_fights = len(p1_ids)

    # Reindexing veloce
    max_id = max(features.index.max(), p1_ids.max(), p2_ids.max())
    feat_arr = features.reindex(range(max_id + 1)).fillna(0).values

    p1_feats = feat_arr[p1_ids]
    p2_feats = feat_arr[p2_ids]

    n_features = p1_feats.shape[1]
    X_interleaved = np.empty((n_fights * 2, n_features + 1), dtype=np.float32)

    X_interleaved[0::2, :-1] = p1_feats
    X_interleaved[0::2, -1] = 1.0  # is_first per il Pokemon 1
    X_interleaved[1::2, :-1] = p2_feats
    X_interleaved[1::2, -1] = 0.0  # is_first per il Pokemon 2

    y_interleaved = np.empty(n_fights * 2, dtype=np.int8)
    y_interleaved[0::2] = (winners == p1_ids).astype(np.int8)
    y_interleaved[1::2] = (winners == p2_ids).astype(np.int8)

    groups = np.full(n_fights, 2, dtype=np.int32)
    fight_id = np.repeat(np.arange(n_fights), 2)

    feats_cols = features.columns.tolist() + ["is_first"]
    X = pd.DataFrame(X_interleaved, columns=feats_cols)

    return X, y_interleaved, groups, fight_id


def pairwise_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcola l'accuratezza confrontando lo score dei 2 Pokémon nello scontro."""
    pred_diff = y_pred[0::2] - y_pred[1::2]
    true_diff = y_true[0::2] - y_true[1::2]
    correct_predictions = (np.sign(pred_diff) == np.sign(true_diff)).sum()
    return correct_predictions / max(1, len(pred_diff))



def main():
    print("Caricamento dataset...")
    pokemon_df = pd.read_csv(POKEMON_PATH)
    # Assumo che combats.csv esista localmente
    combats_df = pd.read_csv(COMBATS_PATH)

    features = build_pokemon_feature_table(pokemon_df)
    X, y, groups, fight_id = build_rank_dataset(features, combats_df)

    print(f"dataset preparato: {len(X)} righe ({len(groups)} scontri validi).")

    gkf = GroupKFold(n_splits=5)

    param_dist = {
        "n_estimators": [300, 600, 1000],
        "learning_rate": [0.01, 0.03, 0.05],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [20, 40],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "max_depth": [-1, 6, 10],
    }

    rng = np.random.RandomState(42)
    param_list = list(ParameterSampler(param_dist, n_iter=10, random_state=rng))

    best_score = -1.0
    best_params = None

    # Diciamo a LightGBM quali sono le categorie
    cat_features = ["Type 1", "Type 2"]

    print("\nInizio ricerca iperparametri (Manual CV)...")
    for idx, params in enumerate(param_list, start=1):
        fold_scores = []

        for train_idx, val_idx in gkf.split(X, y, groups=fight_id):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_va, y_va = X.iloc[val_idx], y[val_idx]

            g_tr = np.full(len(y_tr) // 2, 2, dtype=np.int32)

            model = LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                random_state=42,
                n_jobs=-1,
                device_type="gpu" if USE_GPU else "cpu",
                **params
            )
            # Passiamo le feature categoriche al fit
            model.fit(X_tr, y_tr, group=g_tr, categorical_feature=cat_features)

            y_pred = model.predict(X_va)
            fold_scores.append(pairwise_accuracy(y_va, y_pred))

        mean_score = float(np.mean(fold_scores))
        print(f"[{idx:02d}/10] Accuracy CV: {mean_score:.4f} | Parametri: {params}")

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    print("\nMigliori Parametri trovati:", best_params)

    print("\nAddestramento modello finale su tutti i dati...")
    final_model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        random_state=42,
        n_jobs=-1,
        device_type="gpu" if USE_GPU else "cpu",
        **best_params
    )
    final_model.fit(X, y, group=groups, categorical_feature=cat_features)

    joblib.dump(final_model, MODEL_OUT)
    print(f"Modello salvato in: {MODEL_OUT}")

    print("\nCalcolo punteggio intrinseco di forza...")
    X_first = features.copy()
    X_first["is_first"] = 1.0

    X_second = features.copy()
    X_second["is_first"] = 0.0

    s_first = final_model.predict(X_first)
    s_second = final_model.predict(X_second)
    strength_score = (s_first + s_second) / 2.0

    out = pd.DataFrame({
        ID_NAME: features.index.values,
        "strength_score": strength_score
    }).sort_values("strength_score", ascending=False)

    out.to_csv(SCORES_OUT, index=False)
    print(f"Punteggi salvati in: {SCORES_OUT}")
