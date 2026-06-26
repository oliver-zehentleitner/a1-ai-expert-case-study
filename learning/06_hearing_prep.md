# Hearing-Vorbereitung – Erwartete Fragen & Antworten

15 Minuten, Fachdiskussion + Code-Challenge. Zeig Tiefe, keine Auswendiglernerei.

---

## Zum Datensatz

**"Woher kommen die Daten?"**
> "Ein Kompositum aus vier Quellen: Grumbletext (UK-Forum, manuell extrahierte SPAM-Messages), NUS SMS Corpus (Singapur-Studierende, HAM), Caroline Tags PhD-Arbeit und ein weiterer Spam-Corpus. Das ist wichtig, weil die Quellen unterschiedlich verarbeitet wurden — die NUS-Messages sind anonymisiert, die Spam-Messages nicht."

**"Warum 5.574 und nicht 5.572 wie in manchen Quellen steht?"**
> "Das war ein Parsing-Bug. Das Standard-CSV-Parsing von pandas interpretiert Anführungszeichen als Feldbegrenzer. Einige SMS-Texte enthalten Anführungszeichen — dadurch wurden 2 Rows verschluckt. Mit `quoting=csv.QUOTE_NONE` lädt man alle 5.574 Zeilen korrekt. Die Zahl 5.572 ist die fehlerhafte historische Angabe aus der Zeit vor dem Fix."

**"Was sind diese `&lt;#&gt;` und `&lt;URL&gt;` Token in den Daten?"**
> "Das habe ich systematisch untersucht. Ich habe alle HTML-kodierten Token gezählt und geprüft in welcher Klasse sie vorkommen. Ergebnis: alle Token — &lt;#&gt;, &lt;DECIMAL&gt;, &lt;TIME&gt;, &lt;URL&gt;, &lt;EMAIL&gt; — kommen ausschließlich in HAM-Nachrichten vor, kein einziger Treffer in SPAM. Das schließt zwei naheliegende Hypothesen aus: es sind weder unbefüllte Spam-Vorlagen noch Scraping-Artefakte vom Grumbletext-Forum. Es sind Anonymisierungsplatzhalter die das NUS-Team eingefügt hat, bevor der Datensatz veröffentlicht wurde."

**"Welche Biases hat der Datensatz?"**
> "Geografisch: SPAM stammt von einem UK-Forum, HAM großteils von Singapur-Studierenden. Das Modell lernt UK-Premium-SMS-Betrug und Singlish-HAM — auf österreichischem oder deutschem Traffic würde es schlechter performen. Zeitlich: Daten aus ~2011, Spam-Muster haben sich seither verändert. Als konkretes Beispiel für Normalisierung: das Pfund-Zeichen `£` erscheint fast nur in SPAM — ohne Normalisierung würde das Modell `£` als Spam-Signal lernen, das auf Euro-Traffic nicht überträgt. Deshalb normalisiere ich alle Währungssymbole, ISO-Codes (`EUR`, `USD`, `GBP`, …) und ausgeschriebene Wörter (`pounds`, `dollars`, `euros`, …) zu `TOKEN_CURRENCY` — damit `£1000` und `1000 EUR` dasselbe Signal liefern."

**"Macht dieser englische Datensatz für A1 überhaupt Sinn?"**
> "Für eine Produktivlösung nicht. Der Datensatz stammt aus UK und Singapur, die Sprache ist Englisch, die Spam-Muster sind britische Premium-Rate-SMS aus ~2011. A1-Kunden kommunizieren auf Deutsch; österreichischer Spam hat andere Muster: Paketbenachrichtigungs-Phishing, 0900-Nummern, Kleinanzeigen-Betrug auf Deutsch. Das Modell würde auf echtem A1-Traffic schlechter performen. Für die Fallstudie taugt der Datensatz als Methodikdemonstration. In Produktion: gleiches Package, gleiche Architektur — anderes Trainingskorpus."

**"Wie verhält sich das Modell auf deutschen Nachrichten konkret?"**
> "Das haben wir mit einer deutschen Testserie in Notebook 4 direkt gemessen — das Ergebnis ist differenzierter als 'funktioniert nicht'. Deutschen SPAM erkennt das Modell zuverlässig: eine deutsche Gewinnspiel-SMS ('Sie haben GEWONNEN! … rufen Sie jetzt unter 0800 an!') wird mit 98 % Konfidenz als SPAM klassifiziert — sogar höher als die englische Entsprechung mit 74 %. Der Grund: strukturelle Spam-Signale — `TOKEN_NUM` für die 0800-Nummer, Ausrufezeichen, Dringlichkeit — übertragen sich sprachunabhängig.
>
> Das Problem liegt bei legitimem deutschen Traffic mit spam-ähnlicher Struktur. Eine Termin-Erinnerung mit URL ('Termin am Di, 30.06.2026 … Ändern/Absagen: https://praxis.at/t') wird mit 74 % als SPAM klassifiziert, die englische Entsprechung nur mit 45 % (knapp HAM). Eine TAN-Bestätigung der Bank ('45,50 EUR an Muster GmbH. Ihre TAN: 823 914') landet bei 89 % SPAM, englisch bei 59 %. Die Ursache: englische Wörter wie 'appointment', 'reschedule', 'transaction' haben im Modell minimales HAM-Gewicht aus dem Trainingskorpus; ihre deutschen Entsprechungen ('Termin', 'Absagen', 'Freigabe') sind dem Modell unbekannt und tragen null Gewicht — die Spam-Struktur dominiert ungebremst.
>
> Das ist exakt das Falsch-Positiv-Risiko das ich in Notebook 4 diskutiere: für einen Carrier wie A1 ist ein blockierter Arzt-Termin schlimmer als eine durchgelassene Spam-SMS. Auf deutschem Traffic wäre dieses Risiko höher als auf englischem — ein weiterer Grund warum ein deutschsprachiges Trainingskorpus notwendig wäre."

---

## Zur Textverarbeitung und N-Grammen

**"Was macht eure Normalisierungspipeline?"**
> "Erkannte Entitäten werden durch semantische Platzhalter ersetzt: URLs → `TOKEN_URL`, Nummern → `TOKEN_NUM`, Währungssymbole → `TOKEN_CURRENCY`, Mentions → `TOKEN_USER`. NUS-Anonymisierungs-Token wie `&lt;#&gt;` oder `&lt;URL&gt;` werden auf dieselben Tokens gemappt wie echter Nutzerinhalt — die *Präsenz* einer URL oder Nummer ist das Signal, nicht ihr konkreter Wert. Für den konkreten Wert braucht es ein separates System: Blacklists, IoC-Feeds. Die Rohdaten bleiben unverändert; Normalisierung passiert on demand. Und: dieselbe Funktion ist der erste Schritt der sklearn-Pipeline — kein Train-Serve-Skew möglich. Formatierte Zahlen wie `£100,000` oder Datumsangaben wie `02/09/03` werden zu einem einzigen `TOKEN_NUM` normalisiert — sonst entstehen nach Punctuation-Stripping Artefakte wie `token_numtoken_num` oder `943` im Training."

**"Was sind N-Gramme und warum verwendest du Bigrams?"**
> "N-Gramme sind Folgen von N aufeinanderfolgenden Wörtern. 'Free' alleine ist mehrdeutig — 'free time' ist HAM, 'free prize' ist SPAM. Das Bigramm 'free prize' ist eindeutig. Mit `ngram_range=(1,2)` hat das Modell sowohl Einzelwörter als auch Wortpaare als Features — es kann Phrasen lernen, nicht nur Vokabular."

**"Was hast du in den N-Grammen Interessantes gefunden?"**
> "Drei Dinge. Erstens ein Bug: vor der Normalisierung dominierten `lt gt` die HAM-Bigrams mit 270 Vorkommen — der CountVectorizer hat auf rohem Text gearbeitet und `&lt;` als `lt` tokenisiert. Nach dem Fix auf normalisiertem Text: weg. Zweitens der by-class Vergleich: SPAM-Bigrams sind fast ausschließlich strukturell — `token_currency token_num` kommt 273 Mal in 747 SPAM-Nachrichten vor, `call token_num` 229 Mal. HAM-Bigrams dagegen sind echter Dialog: `call later`, `let know`, `take care`. Das Muster ist trennschärfer als einzelne Wörter. Drittens: in den SPAM-Trigrammen sieht man Kampagnen-Fingerprints — `guaranteed call token_num`, `prize guaranteed call` und `token_num prize guaranteed` beschreiben denselben Spam-Sender aus verschiedenen Blickwinkeln."

**"Warum eigene Stopword-Liste statt einer fertigen?"**
> "Zwei Gründe. sklearn's `ENGLISH_STOP_WORDS` ist zu aggressiv — sie filtert `call` und `free` raus, beides starke Spam-Marker. NLTK hat eine bessere Liste, lädt sie aber live aus dem Internet — kein Netzwerk-Zugriff in CI/CD und kein deterministisches Deployment. Deshalb: Top-80 der eigenen Daten nach Häufigkeit sortiert, jeden Eintrag manuell geprüft, echte Funktionswörter auf die Liste. Ergebnis ist eine editierbare Textdatei, kein Code-Change nötig wenn man anpassen will."

---

## Zu deinem Code

**"Warum hast du das als Package strukturiert statt als einzelnes Notebook?"**  
> "Ein Package mit klaren Interfaces ist wartbar und erweiterbar. Die Notebooks importieren die Bibliothek — das bedeutet jede Analyse ist reproduzierbar und die Logik ist testbar. Notebooks alleine sind schwer zu testen und versionieren."

**"Was würdest du in Produktion anders machen?"**  
> "API-Rate-Limiting für die Embedding-Requests, Caching der Embeddings (die 5574 SMS einmal embedden, abspeichern, nicht bei jedem Run neu berechnen), MLflow oder ähnliches für Experiment-Tracking. Für den Classifier: ein A/B-Test Framework und Monitoring auf Drift (wenn die SMS-Sprache sich ändert, degradiert das Modell)."

**"Warum nicht einfach GPT-4 fragen ob eine SMS Spam ist?"**  
> "Für Batch-Klassifikation von 5000 SMS wäre das teuer, langsam und nicht reproduzierbar. Das klassische Modell ist deterministisch, kostet nichts und läuft lokal. LLMs für Klassifikation sind sinnvoll wenn man Zero-Shot auf neue Kategorien muss oder die Erklärung genauso wichtig ist wie die Klassifikation."

---

## Zur Methodik

**"Warum `solver='liblinear'`?"**
> "TF-IDF erzeugt eine sehr sparse Matrix — 10.000 Features, davon die meisten 0. `liblinear` verwendet Coordinate Descent und iteriert direkt über die nicht-null-Einträge. Das ist auf sparse Matrizen deutlich effizienter als `lbfgs` (der sklearn-Default), der mit vollen Gradienten rechnet und für dense Features optimiert ist. Für binäre Klassifikation auf sparse TF-IDF-Daten ist `liblinear` der Standardchoice."

**"Warum Logistic Regression statt Random Forest oder XGBoost?"**  
> "Für sparse high-dimensional TF-IDF Features funktionieren lineare Modelle sehr gut. Random Forest und Boosting sind eher für tabellarische Daten mit niedrigerer Dimensionalität optimiert. LR ist interpretierbar, schnell und hat hier State-of-the-art Performance."

**"Was ist der Unterschied zwischen Precision und Recall für Spam — und welchen Trade-off hast du gewählt?"**  
> "Precision: Wie viel Prozent der als Spam markierten SMS sind wirklich Spam — schlechte Precision bedeutet HAM landet im Spam-Ordner. Recall: Wie viel Prozent aller echten Spams werden erkannt — schlechter Recall lässt Spam durch. Das sind zwei verschiedene Fehlertypen mit verschiedenen Kosten.  
> Für A1 als Carrier ist ein **False Positive teurer** als ein False Negative: wenn eine Transaktionsbestätigung oder ein Arzttermin-Reminder geblockt wird, ist das ein konkreter Schaden für den Kunden. Spam der durchkommt ist ärgerlich, aber nicht sicherheitskritisch.  
> Deshalb habe ich keinen `class_weight='balanced'` gesetzt — das würde Recall auf Kosten von Precision erhöhen. Das Ergebnis: **Precision=0.935, Recall=0.763**. Von den als Spam geflaggten Messages sind 93,5% wirklich Spam (niedrige FP-Rate). 23,7% des echten Spams schlüpft durch (erhöhte FN-Rate) — das ist der bewusste Trade-off.  
> Der Threshold ist einstellbar: wer höheren Recall braucht kann `predict_proba` verwenden und die Schwelle unter 0.5 absenken. Das ist eine Produktentscheidung, kein Modell-Fix."

**"Was sind eure finalen Modell-Metriken?"**
> "Auf dem deduplizierten Datensatz (5.171 unique Messages): F1-SPAM=0.840, Precision=0.935, Recall=0.763, ROC-AUC=0.991. Ich betone 'dedupliziert' bewusst — der Rohdatensatz hat 403 Duplikate. Ohne Deduplication vor dem Split wären die Zahlen besser ausgefallen (F1=0.906), weil identische Nachrichten in Training und Test landen können. Das ist Data Leakage. Die ehrlichen Zahlen sind niedriger aber sauber."

**"Warum F1-SPAM als CV-Metrik statt F1-macro?"**
> "F1-macro mittelt über beide Klassen und gibt ihnen gleiches Gewicht — bei 86/14-Imbalance würde das HAM und SPAM gleich behandeln, obwohl HAM 6× häufiger ist. F1-SPAM misst nur die Minderheitsklasse die uns tatsächlich interessiert. Die Zahlen unterscheiden sich signifikant: F1-macro wäre bei ~0.92, F1-SPAM bei ~0.84. Wenn man F1-macro als Metrik nimmt, sieht das Modell besser aus als es auf SPAM tatsächlich ist — das ist irreführend."

**"Warum hast du keine Hyperparameter-Optimierung gemacht?"**
> "C=1.0 und max_features=10.000 sind Defaults — und für TF-IDF-Textklassifikation gut etablierte Startpunkte. Bei 5.000 Samples würde Grid Search drei Risiken mitbringen: erstens Overfitting auf den Validierungssplit (zu wenig Daten für zuverlässige CV-Ergebnisse über einem Grid), zweitens marginale Verbesserungen (typisch unter 0.01 F1), drittens Mehraufwand der in keinem Verhältnis zum Nutzen steht.  
> C reguliert die Stärke der L2-Regularisierung — für sparse TF-IDF ist der Default robust. max_features=10.000 cappt das Vokabular und vermeidet Noise durch seltene Tokens.  
> In Produktion mit mehr Daten: `GridSearchCV` über `{C: [0.01, 0.1, 1, 10], max_features: [5000, 10000, 20000]}` wäre der nächste Schritt."

**"Warum dedupliziert ihr auf rohem Text und nicht auf normalisiertem?"**
> "Weil die beiden Fälle semantisch verschieden sind. Raw-Duplikate — dieselbe Zeichenkette erscheint zweimal — das ist echter Data Leakage: das Modell sieht dieselbe Nachricht im Training und im Test. Normalized-Duplikate sind eine andere Sache: 'Call 0800-123456 now' und 'Call 0800-999999 now' werden beide zu 'call token_num now' — aber das sind zwei verschiedene echte Nachrichten, die Normalisierung hat nur die irrelevanten Details abstrahiert. Das ist genau der Zweck der Normalisierung, kein Fehler. Für den Evaluierungszweck ist Raw-Dedup die richtige Wahl."

**"Warum ist dein Datensatz imbalanced und was hast du dagegen getan?"**  
> "86% HAM / 14% SPAM ist realistisch — Spam ist in der Minderheit. Ich verwende stratifizierte Train/Test-Splits und stratifiziertes Cross-Validation, damit jeder Fold die echte Verteilung erhält. `class_weight='balanced'` habe ich bewusst nicht gesetzt — Begründung siehe Precision/Recall-Trade-off oben."

---

## Zur Architektur

**"Warum drei Embedding-Provider — und warum nicht einfach OpenAI dreimal?"**  
> "Zwei Gründe. Erstens das Strategy Pattern: die Clustering-Logik hängt nur vom abstrakten Interface `EmbeddingProvider` ab. Lokale Entwicklung läuft mit Ollama ohne API-Kosten, Produktion switcht durch eine Konfigurationsänderung. Zweitens echte Diversität statt Varietät: alle drei Provider haben unterschiedliche Modell-Familien und Deployment-Modelle — `nomic-embed-text` (Open Source, lokal), `text-embedding-3-small` (OpenAI), `Cohere-embed-v3-multilingual` (Cohere auf Azure). Das Cohere-Modell ist als einziges multilingual — direkt relevant für A1's Anwendungsfall mit deutschen SMS. Dasselbe Modell auf drei Infrastrukturen würde zeigen dass die Abstraktion funktioniert, aber nicht dass sie echten Mehrwert liefert."

**"Warum Azure AI Foundry und nicht classic Azure OpenAI Service?"**  
> "Azure AI Foundry ist die aktuelle Microsoft-Plattform — Azure OpenAI Service ist darin aufgegangen. Der technische Unterschied: classic Azure OpenAI benötigt ein explizites Deployment pro Modell im Portal; Azure AI Foundry bietet Serverless-Catalog-Modelle an, die ohne Deployment direkt verfügbar sind. Das Cohere-Modell ist auf dem resource als Serverless-Modell verfügbar, `text-embedding-3-small` würde ein Deployment brauchen — und wäre dann inhaltlich dasselbe wie der OpenAI-Provider. Der `/openai/v1` Compatibility-Endpoint erlaubt den Zugriff mit dem normalen OpenAI Python Client, nur mit anderer `base_url`. Das macht den Provider-Swap besonders sauber: identisches Interface, unterschiedliche Infrastruktur und Modell-Family."

**"Wie würdest du das skalieren wenn du 10 Millionen SMS hast?"**  
> "Embeddings parallelisieren: Batch-Processing mit mehreren API-Calls parallel. K-Means skaliert schlecht auf sehr viele Datenpunkte — ich würde Mini-Batch K-Means verwenden oder auf Approximate Nearest Neighbor Methoden wechseln. Für die Persistenz der Embeddings: ein Vector Store wie Qdrant oder pgvector. Das Setup von Docker/Kubernetes das ich hier angelegt habe wäre der natürliche nächste Schritt."

---

## Zu GenAI / LLM allgemein

**"Was ist der Unterschied zwischen Embeddings und dem LLM-Output?"**  
> "Embeddings sind latente Repräsentationen — dichte Zahlenvektoren die semantische Bedeutung kodieren. Sie sind Input für downstream Tasks wie Clustering oder Similarity Search. LLM-Output ist generierter Text. Beides kommt aus demselben Modell, aber an verschiedenen Stellen: Embeddings aus dem letzten Hidden State vor dem Output-Layer, generierter Text nach dem Sampling aus der Output-Distribution."

**"Was haben deine Cluster inhaltlich gefunden?"**
> "SPAM konzentriert sich auf 1–2 Cluster, HAM verteilt sich auf 8+. Das ist eigentlich die interessanteste Beobachtung: Spam ist semantisch repetitiv — es gibt eine endliche Menge von Manipulationsvorlagen. HAM ist semantisch divers — unbegrenzte Themen und Register.  
> Konkret: ein Cluster fängt UK-Premium-Rate-Betrug ein ('Hard LIVE 121 chat, 60p/min', Shortcodes, per-Minute-Preise) — strukturell fast identische Nachrichten von derselben Kampagne. Ein anderer fängt Gewinnspiel-Spam ('won a £1000 prize', 'GUARANTEED', 'reply WIN').  
> HAM-Cluster sind inhaltlich: kurze Bestätigungen ('Ok', 'K'), Logistik-Koordination, persönliche/emotionale Nachrichten, University-Singlish ('lor', 'lah'). TF-IDF würde diese Cluster nie finden — 'Ok lor' und 'K thanks' haben keine Wortüberlappung aber landen semantisch im selben Cluster.  
> Die Silhouette-Score ist niedrig (~0.02) weil HAM so divers ist — Gespräche sind inhärent schwer zu clustern. Das ist kein Modell-Fehler, sondern eine Eigenschaft der Daten."

**"Warum ist Silhouette-Score so niedrig?"**
> "Silhouette misst wie gut ein Punkt zu seinem eigenen Cluster passt im Vergleich zum nächsten anderen Cluster. HAM-Nachrichten sind thematisch extrem divers — 'I love you' und 'Can you pick up milk?' landen beide in HAM und sind semantisch weit voneinander entfernt. Ein guter Silhouette-Score bei SMS-HAM würde bedeuten, dass HAM-Nachrichten inhaltlich ähnlich sind — das wäre gerade kein gutes Zeichen. Der niedrige Score für HAM ist eine Eigenschaft der Daten, nicht des Algorithmus."

**"Was wären Use Cases für Agentic AI bei A1?"**  
> "Customer Service Agent der Tickets klassifiziert, weiterleitet und Standardanfragen automatisch beantwortet. Network Operations: Agent der Monitoring-Alerts analysiert, korreliert und eine Diagnose vorschlägt. Fraud Detection: Agent der ungewöhnliche Verhaltensmuster erkennt und eskaliert. Kern-Vorteil gegenüber klassischem ML: der Agent kann mit unsicherem Output umgehen, nachfragen und mehrere Tools kombinieren."

---

## Zu Tests und Code-Qualität

**"Was testest du, was testest du nicht?"**
> "Ich teste: Preprocessing (18 Tests — alle Token-Transformationen, Edge Cases wie leerer String, Whitespace-only, Idempotenz), den Dataset-Loader (Parsen, Klassen-Balance, Fehlerfall bei fehlendem File), das Classification-Pipeline End-to-End (fit→predict, Serialisierung und Deserialisierung mit joblib, predict_proba-Shape, evaluate-Metriken, CV-Keys). Insgesamt 56 Tests.  
> Ich teste nicht: die Visualisierungsfunktionen (Plots sind zu schwer sinnvoll zu assertieren), das CLI (Integration-Tests würden subprocess benötigen), die Embedding-Provider (externe APIs — würden Mocks oder echte Keys brauchen), und das Clustering (nicht-deterministisch durch K-Means+UMAP).  
> Was ich bewusst nicht gemacht habe: die Datenbank oder externe APIs mocken. Tests die nur Mocks testen, testen nichts Wertvolles."

**"Warum hast du `spam_idx` nicht hardcoded?"**
> "Die ursprüngliche Version hatte `spam_idx = 1` mit einem Kommentar 'assumes classes sorted alphabetically: [ham, spam]'. Das ist korrekt — sklearn sortiert Klassen alphabetisch, also ist Index 1 immer SPAM. Aber es ist ein Kommentar der beweist, dass eine Annahme im Code steckt. Die aktuelle Version macht es explizit: `spam_idx = sorted(set(y_true)).index('spam')` — das leitet den Index aus den Daten ab statt ihn zu erraten. Wenn sich die Klassen-Strings jemals ändern würden, würde das sofort mit einem klaren ValueError fehlschlagen statt still falsche Zahlen zu produzieren."
