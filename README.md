# Lymphoma-classification

Confronto di differenti approcci per la diagnosi di Linfoma attraverso lo studio di un dataset genomico (pubblico).

## Analisi dei metodi
I metodi confrontati sono:
- *Principal component*
- *AutoEncoder*
per la riduzione della dimensionalità. 

Per la classificazione sono state utilizzate tre differenti reti neurali utilizzando `keras`, una chiamata *Baseline*, una *Larger* e una *Smaller*. Ogni rete è stata girata con 100 epoche.

### *Baseline network*
Nella rete *baseline* sono stati stimati due strati, uno con 77 nodi (il numero delle nostre osservazioni), uno con 1. Gli attivatori erano rispettivamente `relu` e `sigmoid`. La *loss* era la `binary cross entropy` con ottimizzatore `adam` per ogni rete.

### *Larger network*
Nella rete *larger* sono stati stimati tre strati, uno con 77 nodi, uno con 35 ed uno con 1. Gli attivatori erano rispettivamente `relu` per i primi due e `sigmoid`. 

### *Smaller network*
Nella rete *smaller* sono stati stimati due strati, uno con 35 nodi ed uno con 1. Gli attivatori erano rispettivamente `relu` e `sigmoid`. 

## Risultati
Le reti che hanno fornito risultati migliori sono state quelle successive alla riduzione della dimensionalità tramite *AutoEncoder*.\
Possiamo vedere più nel dettaglio in tabella:
  Metodo  Accuratezza 
  *PCA + baseline*  69.05% (9.86%)
