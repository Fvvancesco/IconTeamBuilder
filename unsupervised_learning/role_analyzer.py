import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from typing import Dict, Any, Optional, List

class PokemonRoleAnalyzer:
    """
    Classe modulare per l'analisi, l'aggregazione, il clustering, l'esportazione
    e la visualizzazione dei ruoli dei Pokémon.
    """

    def __init__(self, data_path: str, verbose: bool = False):
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self.X_scaled: Optional[np.ndarray] = None
        self.scaled_features: List[str] = []
        self.verbose = verbose
        self.id_col: str = 'pokedex_number'

    def load_data(self) -> None:
        if self.verbose:
            print("Caricamento dati...")
        self.df = pd.read_csv(self.data_path)

        if 'pokedex_number' not in self.df.columns and '#' in self.df.columns:
            self.id_col = '#'

        base_stats = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
        if set(base_stats).issubset(self.df.columns) and 'total_points' not in self.df.columns:
            self.df['total_points'] = self.df[base_stats].sum(axis=1)

    def filter_data(self, query_str: str) -> None:
        if self.verbose:
            print(f"Applicazione filtro: '{query_str}'...")
        try:
            self.df = self.df.query(query_str).copy()
            if self.verbose:
                print(f"Pokémon rimanenti dopo il filtro: {len(self.df)}")
        except Exception as e:
            raise ValueError(f"Errore nella query di filtraggio: {e}")

    def aggregate_features(self, aggregations: Dict[str, Dict[str, Any]]) -> None:
        """
        Crea nuove feature aggregando quelle esistenti con metodi indipendenti.
        Formato: {'nome_nuova_col': {'cols': ['col1', 'col2'], 'method': 'sum'}}
        """
        if self.verbose:
            print(f"Aggregazione feature dinamica...")
        for new_col_name, config in aggregations.items():
            cols_to_combine = config['cols']
            method = config['method']

            missing = [c for c in cols_to_combine if c not in self.df.columns]
            if missing:
                raise KeyError(f"Colonne mancanti per l'aggregazione '{new_col_name}': {missing}")

            if method == 'sum':
                self.df[new_col_name] = self.df[cols_to_combine].sum(axis=1)
            elif method == 'mean':
                self.df[new_col_name] = self.df[cols_to_combine].mean(axis=1)
            elif method == 'diff':
                # Sottrazione tra la prima e la seconda colonna specificata
                self.df[new_col_name] = self.df[cols_to_combine[0]] - self.df[cols_to_combine[1]]
            elif method == 'max':
                self.df[new_col_name] = self.df[cols_to_combine].max(axis=1)
            elif method == 'min':
                self.df[new_col_name] = self.df[cols_to_combine].min(axis=1)
            else:
                raise ValueError(f"Metodo di aggregazione '{method}' non supportato.")

    def scale_features(self, features: List[str], method: str = 'zscore', verbose: bool = False) -> None:
        if verbose:
            print(f"Applicazione scaling ({method}) sulle feature: {features}...")
        self.scaled_features = features
        X_raw = self.df[self.scaled_features]

        if method == 'ratio':
            denominators = X_raw.sum(axis=1)
            self.X_scaled = X_raw.div(denominators, axis=0).values
        elif method == 'l2':
            denominators = np.linalg.norm(X_raw, axis=1)
            self.X_scaled = X_raw.div(denominators, axis=0).values
        elif method == 'zscore':
            scaler = StandardScaler()
            self.X_scaled = scaler.fit_transform(X_raw)
        elif method == 'minmax':
            scaler = MinMaxScaler()
            self.X_scaled = scaler.fit_transform(X_raw)
        elif method == 'robust':
            scaler = RobustScaler()
            self.X_scaled = scaler.fit_transform(X_raw)
        elif method == 'maxabs':
            scaler = MaxAbsScaler()
            self.X_scaled = scaler.fit_transform(X_raw)
        elif method == 'none':
            self.X_scaled = X_raw.values
        else:
            raise ValueError(f"Metodo di scaling '{method}' non riconosciuto.")

        for i, col in enumerate(self.scaled_features):
            self.df[f'norm_{col}'] = self.X_scaled[:, i]

    def fit_and_export(self, k: int, output_csv: str | None) -> Dict[str, Dict[str, Any]]:
        if self.verbose:
            print(f"Esecuzione clustering finale con k={k}...")

        kmeans_final = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.df['cluster_label'] = kmeans_final.fit_predict(self.X_scaled)

        norm_cols = [f'norm_{f}' for f in self.scaled_features]
        colonne_export = [self.id_col, 'name', 'cluster_label'] + norm_cols

        df_export = self.df[colonne_export]
        if output_csv is not None:
            df_export.to_csv(output_csv, index=False)
            if self.verbose:
                print(f"Dati esportati con successo in {output_csv}")

        dict_ruoli = {}
        for _, row in df_export.iterrows():
            p_id = str(row[self.id_col])
            vettore_norm = np.array([row[c] for c in norm_cols], dtype=np.float32)
            dict_ruoli[p_id] = {
                'name': row['name'],
                'cluster': int(row['cluster_label']),
                'vector': vettore_norm
            }

        return dict_ruoli

    def analyze_centroids(self) -> None:
        if 'cluster_label' not in self.df.columns:
            raise ValueError("Devi prima eseguire fit_and_export() per generare i cluster.")

        if self.verbose:
            print("\n--- PROFILI DEI CLUSTER (CENTROIDI) ---")
        norm_cols = [f'norm_{f}' for f in self.scaled_features]

        centroids = self.df.groupby('cluster_label')[norm_cols].mean()
        counts = self.df['cluster_label'].value_counts()

        for cluster_id, row in centroids.iterrows():
            n_pokemon = counts.get(cluster_id, 0)

            top_stats = row.nlargest(2)
            stat_names = [idx.replace('norm_', '') for idx in top_stats.index]
            if self.verbose:
                print(f"\n🔹 CLUSTER {cluster_id} ({n_pokemon} Pokémon):")
                print(f"   Punti di forza: {stat_names[0].upper()} e {stat_names[1].upper()}")

            for col in norm_cols:
                stat_name = col.replace('norm_', '').ljust(20)
                print(f"   - {stat_name}: {row[col]:.3f}")

    def plot_clustering_metrics(self, max_k: int = 10, save_path: Optional[str] = None, show: bool = True):
        """Unisce Gomito e Silhouette in un unico grafico a doppio asse."""
        if self.X_scaled is None:
            raise ValueError("Esegui scale_features() prima di valutare i cluster.")

        if self.verbose:
            print("Calcolo Metodo del Gomito e Silhouette...")
        inertias, silhouettes = [], []
        K_range = range(2, max_k + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_scaled)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(self.X_scaled, labels))

        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Asse Y sinistro (Inerzia)
        ax1.plot(K_range, inertias, 'bx-', label='Inerzia (Gomito)')
        ax1.set_xlabel('Numero di Cluster (K)')
        ax1.set_ylabel('Inerzia', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Asse Y destro (Silhouette)
        ax2 = ax1.twinx()
        ax2.plot(K_range, silhouettes, 'rx-', label='Silhouette Score')
        ax2.set_ylabel('Silhouette Score', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        plt.title('Valutazione Cluster: Metodo del Gomito vs Silhouette Score')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            if self.verbose:
                print(f"   [Grafico Metriche salvato in '{save_path}']")
        if show:
            plt.show()
        plt.close()

    def plot_2d_scatter(self, x_col: str, y_col: str, color_col: str, title: str,
                        xlabel: str = None, ylabel: str = None,
                        axhline_y: float = None, axvline_x: float = None,
                        save_path: Optional[str] = None, show: bool = True):
        """Generatore universale di scatter plot 2D per il dataframe analizzato."""
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(self.df[x_col], self.df[y_col], c=self.df[color_col], cmap='Set1', alpha=0.7,
                              edgecolors='k')

        if axhline_y is not None:
            plt.axhline(axhline_y, color='grey', linestyle='--', alpha=0.8)
        if axvline_x is not None:
            plt.axvline(axvline_x, color='grey', linestyle='--', alpha=0.8)

        plt.title(title, fontsize=14)
        plt.xlabel(xlabel if xlabel else x_col)
        plt.ylabel(ylabel if ylabel else y_col)
        plt.legend(*scatter.legend_elements(), title="Cluster", loc="best")

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            if self.verbose:
                print(f"   [Grafico 2D salvato in '{save_path}']")
        if show:
            plt.show()
        plt.close()

    def plot_3d_scatter(self, x_col: str, y_col: str, z_col: str, color_col: str, title: str,
                        xlabel: str = None, ylabel: str = None, zlabel: str = None,
                        save_path: Optional[str] = None, show: bool = True):
        """Generatore universale di scatter plot 3D."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(self.df[x_col], self.df[y_col], self.df[z_col],
                             c=self.df[color_col], cmap='Set1', alpha=0.7, edgecolors='k', s=40)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel if xlabel else x_col)
        ax.set_ylabel(ylabel if ylabel else y_col)
        ax.set_zlabel(zlabel if zlabel else z_col)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            if self.verbose:
                print(f"   [Grafico 3D salvato in '{save_path}']")
        if show:
            plt.show()
        plt.close()

    def plot_pca_projection(self, cluster_col: str = 'cluster_label', title: str = 'Proiezione PCA 2D dei Cluster',
                            save_path: Optional[str] = None, show: bool = True):
        """Applica la PCA on-the-fly per proiettare i dati in 2D e li colora in base al cluster specificato."""
        if cluster_col not in self.df.columns and "cluster_label" not in self.df.columns:
            raise ValueError(f"Colonna '{cluster_col}' non trovata nel DataFrame. Effettua prima il clustering.")

        print("\nCalcolo PCA per la visualizzazione 2D...")
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(self.X_scaled)
        var_1, var_2 = pca.explained_variance_ratio_ * 100

        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1],
                              c=self.df[cluster_col], cmap='viridis', alpha=0.8, edgecolors='k', s=50)

        plt.title(title, fontsize=14)
        plt.xlabel(f'PC 1 ({var_1:.1f}% varianza)')
        plt.ylabel(f'PC 2 ({var_2:.1f}% varianza)')
        plt.colorbar(scatter, label="ID Cluster")
        plt.grid(True, linestyle='--', alpha=0.5)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            if self.verbose:
                print(f"   [Proiezione PCA salvata in '{save_path}']")
        if show:
            plt.show()
        plt.close()


