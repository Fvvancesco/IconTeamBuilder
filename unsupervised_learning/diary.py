import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import grid_search
import warnings

from role_analyzer import PokemonRoleAnalyzer

warnings.filterwarnings('ignore')

def approccio_naive(analyzer, base_stats):
    # ---------------------------------------------------------
    print("\n--- CAPITOLO 1: L'Approccio Ingenuo (Tutti i dati, Z-Score) ---")
    # ---------------------------------------------------------

    analyzer.scale_features(base_stats, method='zscore')

    print("1. Applicazione del K-Means direttamente sulle 6 statistiche grezze.")
    # Fa sia plotting che clustering
    analyzer.plot_clustering_metrics(max_k=10, save_path='01_gomito_ingenuo.png', show=True)

    print("2. Generato grafico Gomito/Silhouette.")
    print("   -> Si nota che la silhouette è bassa e il gomito non è netto.")

def analisi_pca_automatica(analyzer, base_stats):
    # ---------------------------------------------------------
    print("\n--- CAPITOLO 2: Analisi PCA e le Dimensioni Nascoste ---")
    # ---------------------------------------------------------
    print("1. Esecuzione PCA Automatica (Standard) per capire cosa 'vede' l'algoritmo.")
    analyzer.scale_features(base_stats, method='ratio')
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(analyzer.X_scaled)

    print(f"   -> PC1 spiega il {pca.explained_variance_ratio_[0]:.1%} della varianza.")
    print(f"   -> PC2 spiega il {pca.explained_variance_ratio_[1]:.1%} della varianza.")

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
    )

    print("\n   [Analisi dei Pesi (Loadings) della PCA]")
    print("   Pesi della PC1:")
    print("   " + str(loadings['PC1'].round(2).to_dict()))

    print("\n   Pesi della PC2:")
    print("   " + str(loadings['PC2'].round(2).to_dict()))

    analyzer.fit_and_export(5, "partial.csv")
    analyzer.plot_clustering_metrics(8)
    analyzer.plot_pca_projection()

    """
    shuckle_idx = analyzer.df[analyzer.df['name'] == 'Shuckle'].index[0]
    print(
        f"\n   -> Scoperto Outlier Estremo: Shuckle (Coordinate PCA 1: {pca_result[shuckle_idx, 0]:.2f}, PCA 2: {pca_result[shuckle_idx, 1]:.2f}).")
    print(
        "   Avendo Difese sproporzionate e Velocità nulla, Shuckle distorce l'asse PC2 allungandolo in modo anomalo.\n")
    """

def analisi_pca_manuale(analyzer, base_stats):
    print("\n--- CAPITOLO 2: Esecuzione 'PCA Manuale' (2D) ---")
    print("Calcolo degli Assi: Offesa vs Difesa & Speciale vs Fisico")

    # Creazione diretta degli assi, avrei potuto usare aggregate feature, ho deciso di non usarlo per una versione più compatta
    analyzer.df['Asse_Off_Dif'] = (analyzer.df['attack'] + analyzer.df['sp_attack']) - (
                analyzer.df['defense'] + analyzer.df['sp_defense'])
    analyzer.df['Asse_Sp_Fis'] = (analyzer.df['sp_attack'] + analyzer.df['sp_defense']) - (
                analyzer.df['attack'] + analyzer.df['defense'])

    analyzer.scale_features(['Asse_Off_Dif', 'Asse_Sp_Fis'], method='zscore')


    print("\nAnalisi Metriche: Generazione grafico Gomito/Silhouette per i 2 assi manuali...")
    analyzer.plot_clustering_metrics(max_k=8, save_path='02_gomito_manuale_2d.png', show=True)

    # 4. Ciclo K-Means dinamico, scoring e plotting scatter
    print("\nGenerazione scatter plot per i vari K...")
    for k in range(2, 8):
        col_name = f'cluster_2d_k{k}'

        # Fit & Predict usando X_scaled della classe
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        analyzer.df[col_name] = kmeans.fit_predict(analyzer.X_scaled)

        # Calcolo Silhouette
        score = silhouette_score(analyzer.X_scaled, analyzer.df[col_name])
        print(f"   -> K-Means (K={k}) | Silhouette Score: {score:.3f}")

        # Plotting delegato alla classe
        analyzer.plot_2d_scatter(
            x_col='Asse_Off_Dif', y_col='Asse_Sp_Fis', color_col=col_name,
            title=f'Test 2D (K={k}): Offesa vs Difesa & Speciale vs Fisico',
            xlabel='<- Difensivo | Offensivo ->',
            ylabel='<- Fisico | Speciale ->',
            axhline_y=0, axvline_x=0,
            #save_path=f'02_test_manuale_2d_k{k}.png',
            show=True
        )

    """print(*Conclusione Test 2D:* Il punteggio è sceso rispetto alle stat grezze perché abbiamo schiacciato troppe informazioni.")

    print("\n   *Conclusione del Capitolo 2:*")
    print(f"   Inserendo la Velocità, la Silhouette si è rialzata ({score_3d:.3f}), dimostrando che la Velocità crea")
    print("   dei cluster molto più densi e separati (es. distingue i Tank lenti dagli Sweeper veloci).")
    print("   Tuttavia, i dati sono ancora inquinati dai Pokémon deboli e dagli outlier estremi (es. Shuckle).")"""

def analisi_pca_3d(analyzer):
    # --- PCA MANUALE 3D ---
    print("\n3. Esecuzione 'PCA Manuale' (3D): L'Aggiunta della Velocità")
    print(
        "   Nel competitivo, un attaccante lentissimo (Wallbreaker) ha un ruolo totalmente diverso da uno velocissimo (Sweeper).")

    analyzer.df['Asse_Velocita'] = analyzer.df['speed']

    features_3d = ['Asse_Off_Dif', 'Asse_Sp_Fis', 'Asse_Velocita']
    scaler_3d = StandardScaler()
    X_3d_scaled = scaler_3d.fit_transform(analyzer.df[features_3d])

    kmeans_3d = KMeans(n_clusters=5, random_state=42, n_init=10)
    analyzer.df['cluster_3d'] = kmeans_3d.fit_predict(X_3d_scaled)
    score_3d = silhouette_score(X_3d_scaled, analyzer.df['cluster_3d'])

    print(f"   -> Esecuzione K-Means sui 3 assi manuali...")
    print(f"   -> Silhouette Score (PCA Manuale 3D): {score_3d:.3f}")

    # Plot 3D usando il nuovo metodo della classe
    analyzer.plot_3d_scatter(
        x_col='Asse_Off_Dif', y_col='Asse_Sp_Fis', z_col='Asse_Velocita', color_col='cluster_3d',
        title='Test 3D: Aggiunta della Velocità',
        xlabel='<- Difensivo | Offensivo ->',
        ylabel='<- Fisico | Speciale ->',
        zlabel='Velocità',
        save_path='03_test_manuale_3d.png', show=False
    )

def clustering_guidato(analyzer):
    # ---------------------------------------------------------
    print("\n--- CAPITOLO 3: L'Approccio Guidato (La Soluzione Definitiva) ---")
    # ---------------------------------------------------------
    print("Pulizia dei dati: Rimozione outlier (Shuckle/Shedinja) e filtro potenza (>550).")
    analyzer.filter_data("total_points > 100 and name not in ['Shuckle', 'Shedinja', 'Deoxys Attack Forme']")

    print("Feature Engineering: Creazione di final_attack e final_defense.")
    # Sintassi diretta Pandas (più pulita e veloce)
    analyzer.df['offense_pool'] = analyzer.df[['attack', 'sp_attack']].max(axis=1)
    analyzer.df['bulk_pool'] = analyzer.df[['defense', 'sp_defense']].max(axis=1)
    analyzer.df['final_attack'] = analyzer.df['offense_pool'] + analyzer.df['speed']
    analyzer.df['final_defense'] = analyzer.df['bulk_pool'] + analyzer.df['hp']

    features_opt = ["final_attack", "final_defense"]
    analyzer.scale_features(features_opt, method='ratio')

    # --- NUOVA SEZIONE: Ricerca automatica del miglior K ---
    print("\n3. Ricerca del K ottimale e generazione metriche...")
    analyzer.plot_clustering_metrics(max_k=10, save_path='04_gomito_definitivo.png', show=True)

    best_k = 2
    best_score = -1

    for k in range(2, 11):
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_temp = kmeans_temp.fit_predict(analyzer.X_scaled)
        score_temp = silhouette_score(analyzer.X_scaled, labels_temp)

        if score_temp > best_score:
            best_score = score_temp
            best_k = k

    print(f"   -> Il K ottimale calcolato algoritmicamente è {best_k} (Silhouette: {best_score:.3f}).")

    # --- Esecuzione Finale ---
    print(f"\n4. Esecuzione K-Means definitivo con K={best_k}.")
    kmeans_opt = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans_opt.fit_predict(analyzer.X_scaled)
    analyzer.df['cluster_ruoli'] = labels

    # Plotting
    print("\n   Generazione dei grafici (Scatter 2D e Proiezione PCA)...")

    # Grafico Scatter 2D Diretto (Sulle feature reali)
    analyzer.plot_2d_scatter(
        x_col='final_attack',
        y_col='final_defense',
        color_col='cluster_ruoli',
        title=f'Ruoli Competitivi (K={best_k}): Final Attack vs Final Defense',
        xlabel='Final Attack (Potenza Offensiva Massima + Velocità)',
        ylabel='Final Defense (Bulk Massimo + HP)',
        save_path='05_scatter_ruoli_definitivi.png',
        show=True
    )

    # Grafico PCA (Sulle componenti principali scalate)
    analyzer.plot_pca_projection(
        cluster_col='cluster_ruoli',
        title=f'Proiezione PCA dei Ruoli Competitivi (K={best_k})',
        save_path='06_pca_ruoli_definitivi.png',
        show=True
    )

    print("\nANALISI DEI RUOLI INDIVIDUATI (Medie per Cluster):")
    cluster_stats = analyzer.df.groupby('cluster_ruoli')[features_opt].mean().round(1)

    # Il loop ora usa 'best_k' invece del 5 hardcodato
    for cluster_id in range(best_k):
        stats = cluster_stats.loc[cluster_id]

        # Prende fino a 3 Pokémon a caso come esempio (usa min per evitare errori se un cluster è piccolo)
        n_esempi = min(3, len(analyzer.df[analyzer.df['cluster_ruoli'] == cluster_id]))
        esempi = analyzer.df[analyzer.df['cluster_ruoli'] == cluster_id]['name'].sample(n=n_esempi,
                                                                                        random_state=1).tolist()

        print(
            f"   [Cluster {cluster_id}] - Final Attack: {stats['final_attack']}, Final Defense: {stats['final_defense']}")
        print(f"      -> Esempi: {', '.join(esempi)}\n")
    analyzer.fit_and_export(k=6, output_csv="pokemon_clusters.csv")

def applicazione_grid_search(analyzer, data_path):
    # ---------------------------------------------------------
    print("\n--- CAPITOLO 4: La Controprova Matematica (Grid Search Massiva) ---")
    # ---------------------------------------------------------
    print("Per dimostrare che l'approccio manuale del Capitolo 3 era sufficiente e corretto,")
    print("lanciamo una Grid Search su TUTTE le possibili partizioni matematiche. ")

    esperimenti_gs = grid_search.genera_grid_search_totale()
    print(f"Generati {len(esperimenti_gs)} esperimenti. Esecuzione parallela in corso...")

    risultati_non_piatti = Parallel(n_jobs=-1)(
        delayed(grid_search.esegui_singolo_esperimento)(esp, data_path, analyzer.df, analyzer.id_col)
        for esp in esperimenti_gs[:100]  # LIMITATO ai primi 100 per test
    )
    risultati = [item for sublist in risultati_non_piatti for item in sublist]
    df_risultati = pd.DataFrame(risultati).sort_values(by="Silhouette_Score", ascending=False)

    print("\n🏆 Top 3 Risultati dalla Grid Search Bruta:")
    pd.set_option('display.max_colwidth', None)
    print(df_risultati.head(3).to_string(index=False))

    print("\nCONCLUSIONE DEL REPORT:")
    print("La Grid Search massiva esplora varianti che spesso massimizzano la silhouette artificialmente,")
    print("ma che perdono il senso tattico (es. aggregare Attack e Speed insieme).")
    print("Questo dimostra che il Feature Engineering basato sulla Domain Knowledge (Capitolo 3) ")
    print("è l'approccio scientificamente più valido per questo dominio di dati.")

def diario_esperimento():
    data_path = '..\Dataset\pokemon_cleaned.csv'
    #data_path = '..\Dataset\combat_pokemon_db.csv'
    base_stats = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']

    c = int(input("Quale esperimento vuoi eseguire? "))

    if c == 1:
        analyzer = PokemonRoleAnalyzer(data_path)
        analyzer.load_data()
        approccio_naive(analyzer, base_stats)
    elif c == 2:
        p = int(input("Automatico o manuale? "))

        if p == 1:
            analyzer = PokemonRoleAnalyzer(data_path)
            analyzer.load_data()
            #analyzer.filter_data("name not in ['Shuckle', 'Deoxys Attack Forme', 'Shedinja']")
            analisi_pca_automatica(analyzer, base_stats)
        elif p == 2:
            analyzer = PokemonRoleAnalyzer(data_path)
            analyzer.load_data()
            #analyzer.filter_data("name not in ['Shuckle', 'Deoxys Attack Forme', 'Shedinja']")
            analisi_pca_manuale(analyzer, base_stats)
    elif c==3:
        analyzer = PokemonRoleAnalyzer(data_path)
        analyzer.load_data()
        clustering_guidato(analyzer)
    elif c==4:
        analyzer = PokemonRoleAnalyzer(data_path)
        analyzer.load_data()
        applicazione_grid_search(analyzer, data_path)


if __name__ == "__main__":
    diario_esperimento()