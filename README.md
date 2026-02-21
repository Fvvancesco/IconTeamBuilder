Ciao! Questo Teambuilder per pokèmon è stato realizzato per l'esame di Ingegneria della Conoscenza. L'idea è semplice: aiutare chi ha appena iniziato a giocare a creare una squadra forte ed equilibrata aiutandolo a navigare tra migliaia di combinazioni.

Il sistema si articola in tre fasi fondamentali:

Scoperta dei Ruoli (Clustering):
  Utilizza l'algoritmo K-Means per identificare matematicamente 6 ruoli competitivi (es. Sweeper, Wall, Tank) basandosi sulle statistiche base dei Pokémon.
  Include analisi tramite PCA per la riduzione della dimensionalità e valutazioni di qualità tramite il metodo del gomito e lo silhouette score.

Valutazione della Forza (Supervised Learning):
  Impiega LightGBM (LGBMRanker) con funzione obiettivo lambdarank per assegnare un punteggio di forza individuale a ogni Pokémon.
  L'analisi ha confermato che la Velocità (Speed) è il fattore più critico nelle lotte 1v1, dominando i criteri di split del modello.

Costruzione del Team (Ottimizzazione CSP):
Il teambuilding è modellato come un Constraint Satisfaction Problem (CSP).
Utilizza un algoritmo di Hill Climbing (Greedy Ascent) con Random Restarts per esplorare lo spazio delle soluzioni ed evitare ottimi locali.

La funzione obiettivo bilancia la potenza pura, la sinergia dei ruoli (diversità dei cluster) e la copertura strategica.
