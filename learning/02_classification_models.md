# Klassifikationsmodelle – Was wir verwenden und warum

## Naive Bayes (MultinomialNB)

**Idee:** Bayes'scher Wahrscheinlichkeits-Spam-Filter. Berechnet:

```
P(SPAM | Wörter) ∝ P(SPAM) × P(w1|SPAM) × P(w2|SPAM) × ...
```

"Naive" = nimmt an dass Wörter unabhängig voneinander sind (das stimmt nicht, funktioniert aber trotzdem gut).

**Stärken:** Sehr schnell. Funktioniert gut bei kleinen Datensätzen. Historisch *der* Spam-Filter.  
**Schwäche:** Ignoriert Wortreihenfolge komplett. Schlechter bei komplexen sprachlichen Mustern.

## Logistic Regression (unser Hauptmodell)

**Idee:** Lernt ein Gewicht `w` für jedes Wort. Positives Gewicht = Spam-Signal, negatives = Ham-Signal.

```
P(SPAM) = sigmoid( w₁×tfidf("free") + w₂×tfidf("call") + ... + bias )
```

**Stärken:**  
- **Interpretierbar**: Du kannst die Gewichte direkt ansehen: "Das Wort 'FREE' hat Gewicht +2.1 → starkes Spam-Signal"  
- Funktioniert sehr gut mit TF-IDF  
- Robust gegen overfitting mit dem Regularisierungsparameter `C`

**Der `C`-Parameter:**  
- Kleines C → starke Regularisierung → Modell ist vorsichtiger, generalisiert besser  
- Großes C → schwache Regularisierung → Modell passt sich stärker an Trainingsdaten an  
- Wir verwenden `C=1.0` (sklearn-Default, guter Ausgangspunkt)

## Solver-Wahl: liblinear

sklearn's `LogisticRegression` hat mehrere Solver (Optimierungsalgorithmen). Wir setzen explizit `solver="liblinear"`:

| Solver | Geeignet für | Nicht geeignet für |
|--------|-------------|-------------------|
| `liblinear` | **Sparse, high-dimensional** (TF-IDF), binäre Klassifikation | Multiclass, sehr große Datensätze |
| `lbfgs` (sklearn-Default) | Dense Features, multiclass | Große sparse Matrizen — kann langsam werden |
| `saga` | Große Datensätze, L1+L2 | Kleine Datensätze |

TF-IDF erzeugt eine sehr sparse Matrix (10.000 Features, die meisten 0). `liblinear` verwendet Coordinate Descent — iteriert direkt über die nicht-null-Einträge, sehr effizient. `lbfgs` rechnet mit vollen Gradienten und ist für dense Matrizen optimiert. Auf einem TF-IDF-Feature-Space kann `lbfgs` ohne explizites Solver-Setting deutlich langsamer werden oder Konvergenzprobleme bekommen.

## Was du im Hearing sagen kannst

> "Ich habe Logistic Regression als Hauptmodell gewählt, weil die Koeffizienten direkt interpretierbar sind — man sieht welche Wörter das Modell in Richtung SPAM oder HAM schieben. Naive Bayes als Baseline zeigt dass LR besser ist, obwohl beide für diesen Datensatz sehr gut funktionieren. Ich verwende `solver='liblinear'` explizit — das ist der klassische Solver für binäre Klassifikation auf sparse TF-IDF-Matrizen: Coordinate Descent über nicht-null-Einträge, deutlich effizienter als der sklearn-Default `lbfgs` auf diesem Feature-Typ."

## Warum nicht SVM oder Random Forest?

- **SVM** wäre auch sehr gut für Text, aber schwerer zu interpretieren  
- **Random Forest** ist eher für tabellarische Daten geeignet, weniger für sparse TF-IDF-Matrizen  
- **Deep Learning / BERT**: Overkill für 5500 SMS, keine echte Verbesserung rechtfertigbar

Die Wahl *muss* zum Problem und zur Datenmenge passen — das ist auch ein Qualitätsmerkmal.
