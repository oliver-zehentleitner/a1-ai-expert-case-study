# Clustering – K-Means, Silhouette, UMAP

## K-Means Algorithmus

1. Wähle K zufällige Startpunkte (Centroids) im Vektorraum
2. Weise jeden Punkt dem nächsten Centroid zu
3. Berechne neuen Centroid als Mittelwert aller zugewiesenen Punkte
4. Wiederhole 2-3 bis sich nichts mehr ändert

**K muss im Voraus gewählt werden** — das ist die große Frage.

## Wie wählen wir K? – Silhouette Score

```python
silhouette_score = mean( (b - a) / max(a, b) ) für alle Punkte

a = mittlere Distanz zum eigenen Cluster (Kohäsion)
b = mittlere Distanz zum nächsten fremden Cluster (Separation)
```

- Wert von -1 bis 1
- **~0.5+** = gute Struktur
- Wir testen K=3 bis K=15 und wählen das K mit dem höchsten Score

## Elbow Method (visuell)

Plottet Inertia (Summe der Abstände aller Punkte zu ihrem Centroid) gegen K.  
Suche nach dem "Ellenbogen" — wo die Kurve abflacht. Mathematisch weniger präzise als Silhouette, aber visuell überzeugend für Präsentationen.

## UMAP – Visualisierung in 2D

1024- oder 1536-dimensionale Embeddings kann man nicht plotten. UMAP reduziert auf 2D:

- **UMAP** (Uniform Manifold Approximation and Projection) — preserviert lokale Struktur besser als t-SNE
- Cluster die im hochdimensionalen Raum nahe sind, bleiben auch im 2D-Plot nahe
- **Wichtig:** Die 2D-Achsen haben keine direkte Bedeutung — nur die relative Position zählt

**Fallback:** Wenn umap-learn nicht installiert ist, verwenden wir PCA (schneller, aber verliert mehr Struktur).

### Wie UMAP intern arbeitet — die wichtigen Konzepte

#### k-NN-Graph (k-Nearest-Neighbor-Graph)
Als erstes baut UMAP einen Graphen aus den hochdimensionalen Punkten: jeder Punkt wird mit seinen `k` nächsten Nachbarn verbunden (typisch k=15). Das Ergebnis ist ein Netz — Punkte sind Knoten, Verbindungen sind Kanten. Dieser Graph kodiert die lokale Struktur der Daten: wer ist nah bei wem?

#### Laplacian-Matrix
Aus dem k-NN-Graphen wird eine Matrix berechnet: die **Laplacian-Matrix** `L = D - A`.  
- `A` = Adjazenzmatrix: `A[i][j] = 1` wenn Punkt i und j verbunden sind, sonst 0  
- `D` = Diagonalmatrix: `D[i][i]` = Anzahl der Verbindungen von Punkt i (sein "Grad")  

Die Laplacian-Matrix kodiert die Konnektivität des gesamten Graphen.

#### Eigenvektoren und Eigenwerte
Für eine Matrix `M` gilt: `M · v = λ · v`  
- `v` = **Eigenvektor**: ein spezieller Vektor, den die Matrix nur streckt, nicht dreht  
- `λ` = **Eigenwert**: der Streckungsfaktor  

Die Eigenvektoren der Laplacian-Matrix sind besonders: sie repräsentieren **globale Strukturen** im Graphen. Der erste Eigenvektor zeigt die stärkste Trennlinie im Graph (die Grenze zwischen den am stärksten getrennten Gruppen), der zweite die zweitstärkste usw.

#### Eigengap
Der **Eigengap** ist der Unterschied zwischen aufeinanderfolgenden Eigenwerten: `λ₂ - λ₁`, `λ₃ - λ₂`, usw.

- **Großer Eigengap** → die Eigenwerte sind klar voneinander getrennt → der Graph hat klare globale Struktur → der Eigenvektorsolver konvergiert zuverlässig
- **Kleiner Eigengap** → die Eigenwerte liegen sehr nahe beieinander → keine klare globale Trennlinie → der Solver kann nicht entscheiden welcher Eigenvektor "der erste" ist → er schlägt fehl

Bei 5574 SMS mit 1024 Dimensionen und vielen kleinen Themengruppen ist der Eigengap klein. Der Solver scheitert, UMAP gibt die Warnung aus.

#### Warum `init="spectral"` auf großen Datasets versagt
UMAP berechnet mit Spectral-Initialisierung die ersten 2 Eigenvektoren des Laplacian als **Startpositionen** für die 2D-Einbettung. Die Idee: ein "warmer Start" der der finalen Struktur schon ähnelt, damit die Optimierung schneller und stabiler konvergiert.

Das Problem: mit wachsender Datenmenge wächst der Graph exponentiell, der Eigengap schrumpft, und der numerische Solver kommt an seine Grenzen.

#### Unsere Lösung: `init="pca"`
Statt Eigenvektoren des Laplacian verwenden wir die **ersten 2 Hauptkomponenten** (Principal Components) der Originaldaten als Startpositionen.

- **Deterministisch**: PCA hat keine numerischen Stabilitätsprobleme bei großen Datasets
- **Datengetrieben**: die Startpositionen spiegeln die stärksten Varianzrichtungen in den Embeddings wider — das ist ein sinnvoller Startpunkt, nicht Rauschen
- **Skaliert**: funktioniert gleich gut bei 500 und bei 5574 Punkten
- `random_state=42` kontrolliert dann nur noch die stochastische Feinoptimierung

#### Warum nicht "Noise oder Jitter hinzufügen"?
UMAP's eigene Warnung schlägt vor: *"Consider adding some noise or jitter to your data."*

Das ist keine gute Idee aus einem grundsätzlichen Grund: **wir würden korrekte Daten absichtlich verschlechtern um einen numerischen Fehler zu umgehen.**

Die Embeddings die wir berechnet haben sind richtig — sie kodieren präzise die semantische Bedeutung der SMS. Rauschen hinzuzufügen würde diese Information absichtlich degradieren, damit ein Solver stabil läuft. Das ist Symptombehandlung, nicht Problemlösung. Es löst den eigentlichen Sachverhalt (falscher Init-Algorithmus für diese Datenmenge) nicht, sondern verschleiert ihn.

## Was du im Hearing sagen kannst

> "Ich verwende K-Means mit automatischer K-Auswahl via Silhouette Score. K-Means minimiert die Inertia — die Summe der quadratischen Abstände aller Punkte zu ihrem Cluster-Centroid. Für die Visualisierung reduziere ich mit UMAP auf 2D, weil UMAP lokale Nachbarschaftsstrukturen besser erhält als PCA oder t-SNE bei diesen Dimensionen."

> "K-Means hat eine Schwäche: es nimmt sphärische Cluster an und reagiert empfindlich auf Ausreißer. Für einen Produktionseinsatz würde ich auch HDBSCAN evaluieren, das keine fixe Clusterzahl braucht und Ausreißer explizit modelliert."

## Cluster-Interpretation

Nach dem Clustering: Was ist in jedem Cluster? Zufallssample von 3-5 SMS aus jedem Cluster zeigen → semantische Gemeinsamkeiten sichtbar. Das ist die Qualitätskontrolle.

---

## Provider-Vergleich (echte Zahlen, 500 SMS, gleicher Seed)

| Provider | Modell | k | Silhouette | Spam-Verteilung |
|----------|--------|---|-----------|-----------------|
| Ollama (lokal) | nomic-embed-text | 4 | 0.022 | C3: 55% Spam |
| OpenAI | text-embedding-3-small | 4 | 0.042 | C1: 51% Spam |
| Azure AI Foundry | Cohere-embed-v3-multilingual | 3 | 0.044 | C2: 50%, C0/C1: <3% |

**Was die Zahlen zeigen:**

- Cloud-Embeddings (OpenAI, Azure/Cohere) liefern deutlich höhere Silhouette-Scores als lokales Ollama (0.042–0.044 vs. 0.022) — bessere geometrische Cluster-Trennung.
- Cohere multilingual findet die sauberste Struktur: k=3 statt k=4, ~91% aller Spam-Nachrichten in einem einzigen Cluster, die anderen zwei Cluster fast rein HAM.
- OpenAI und Ollama verteilen Spam auf mehr Cluster — die Spam-Templates sind zwar erkennbar, aber weniger konzentriert.

**Warum Cohere multilingual trotz gleicher Silhouette besser strukturiert ist:**

Silhouette misst geometrische Kompaktheit, nicht fachliche Relevanz. Cohere findet eine binärere Struktur (HAM vs. SPAM), die direkt der Klassifikationsaufgabe entspricht. OpenAI und Ollama fragmentieren HAM feiner (Singlish-Cluster, kurze Antworten, etc.) — interessant für die Analyse, aber für Spam-Detection nicht entscheidend.

**Was du im Hearing sagen kannst:**

> "Notebook 5 läuft alle verfügbaren Provider automatisch auf dem gleichen Sample durch und vergleicht sie direkt. Das zeigt, dass die Provider-Abstraktion echte Alternativen unterstützt — nicht dasselbe Modell dreimal. Das Cohere-Modell auf Azure AI Foundry ist als einziges multilingual, was für A1 direkt relevant ist: es kodiert deutsche Wörter statt sie als Out-of-Vocabulary zu ignorieren."

---

## Laufzeit-Warnungen die kein Problem sind

### `UserWarning: Spectral initialisation failed! … Falling back to random initialisation!`

**Was passiert:** UMAP hat versucht die Eigenvektoren des Laplacian zu berechnen (siehe oben), aber der Eigengap war zu klein für den numerischen Solver.

**Warum wir das nicht mehr sehen:** Wir setzen `init="pca"` explizit (statt Default `"spectral"`). Damit entfällt der Versuch der Spectral-Berechnung komplett.

---

### `UserWarning: n_jobs value 1 overridden to 1 by setting random_state`

**Was passiert:** UMAP kann Berechnungen auf mehrere CPU-Cores parallelisieren (`n_jobs`).  
**Warum Parallelisierung + Reproduzierbarkeit sich ausschließen:** Bei echter Multi-Thread-Ausführung laufen Zufallszahlen-Sequenzen auf mehreren Threads gleichzeitig — die Reihenfolge ist nicht deterministisch, das Ergebnis wäre bei jedem Lauf leicht anders.  
**Lösung:** Wir setzen `n_jobs=1` explizit im UMAP-Konstruktor. Damit gibt es nichts zu "overridden" — UMAP sieht dass Single-Thread bereits gewünscht ist und warnt nicht mehr. `random_state=42` für Reproduzierbarkeit ist der bewusste Tradeoff.

**Im Hearing:** *"Wir setzen `n_jobs=1` und `random_state=42` explizit — Single-Thread ist gewollt, weil Parallelisierung und reproduzierbare Ergebnisse sich technisch ausschließen."*

---

### `Numba: Attempted to fork from a non-main thread, the TBB library may be in an invalid state`

**Was UMAP intern macht:** UMAP nutzt Numba (Python JIT-Compiler) für schnelle Vektorberechnungen. Numba nutzt Intel TBB (Threading Building Blocks) für sein eigenes Thread-Management.

**Warum die Meldung kommt:** Wenn `sms-analyze all` mehrere Tasks sequenziell aus dem gleichen Prozess startet, initialisiert Numba's Thread-System in einem Kontext, den TBB als "Nicht-Hauptthread" wahrnimmt. TBB gibt eine präventive Warnung aus: *Falls* in diesem Zustand ein echter Prozess-Fork (z.B. `multiprocessing`) passieren würde, könnte der Child-Prozess in einem inkonsistenten Thread-State starten.

**Warum es trotzdem kein Problem ist:** Es passiert kein Fork. Wir rufen UMAP direkt auf, kein `multiprocessing`, kein `fork()`. Die Warnung beschreibt eine *hypothetische* Gefahr die in unserem Ausführungskontext nicht eintritt. Das Ergebnis ist bit-identisch mit einem Lauf ohne die Warnung.

**Die präzise Erklärung:** `sms-analyze all` dispatched mehrere Tasks und nutzt dafür Threads — aber diese Threads laufen für völlig unabhängige Aufgaben (Frequency, Classification, …). UMAP selbst läuft single-threaded (`n_jobs=1`). Es gibt keine geteilten Daten zwischen den Task-Threads und UMAP, also keine Race Conditions. Die TBB-Detection ist zu aggressiv: sie sieht "mehrere Threads im Prozess" und warnt, obwohl diese Threads nichts mit UMAP zu tun haben. Es ist eine Falschwarnung im technischen Sinne.

**Im Hearing:** *"Das ist eine Falschwarnung. TBB erkennt, dass der Prozess in einem Kontext mit mehreren Threads läuft, und warnt präventiv vor einem möglichen Fork. Es passiert aber kein Fork, kein geteilter State, keine Race Condition. UMAP selbst läuft single-threaded (`n_jobs=1`), das Ergebnis ist korrekt und reproduzierbar."*
