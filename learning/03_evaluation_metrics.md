# Evaluierungsmetriken – Das Wichtigste für dein Hearing

## Das Imbalance-Problem

Unser Datensatz: **86% HAM, 14% SPAM**.

Ein Modell das einfach IMMER "HAM" vorhersagt hätte **86% Accuracy** — ohne irgendetwas gelernt zu haben.  
→ **Accuracy alleine ist hier wertlos.**

## Confusion Matrix

```
                  Predicted HAM    Predicted SPAM
Actual HAM    [  True Negative   |  False Positive  ]
Actual SPAM   [  False Negative  |  True Positive   ]
```

- **False Positive**: Ham fälschlich als Spam erkannt → Nutzer verpasst echte SMS (ärgerlich)  
- **False Negative**: Spam nicht erkannt → landet im Posteingang (das eigentliche Problem)

## Precision, Recall, F1

```
Precision (SPAM) = TP / (TP + FP)
  = "Von allen als SPAM erkannten: wievielen % sind wirklich SPAM?"
  Schlechte Precision → viele Ham-SMS landen im Spam-Ordner

Recall (SPAM) = TP / (TP + FN)  
  = "Von allen echten SPAMs: wievielen % haben wir gefunden?"
  Schlechter Recall → Spam kommt durch

F1 = 2 × (Precision × Recall) / (Precision + Recall)
  = Harmonisches Mittel – ausgewogene Bewertung beider
```

**Trade-off:** Du kannst nicht gleichzeitig Precision und Recall maximieren. Für Spam-Filter ist Recall wichtiger (lieber etwas Ham fälschlicherweise filtern als Spam durchlassen).

## ROC-AUC

Der Classifier gibt Wahrscheinlichkeiten aus (z.B. 0.87 Spam). Der Threshold bestimmt die Entscheidung.

- **ROC-Kurve**: Zeigt Recall vs. False Positive Rate für alle möglichen Thresholds
- **AUC** (Area Under Curve): 0.5 = Zufallsraten, 1.0 = perfekt, ≥0.99 erwarten wir hier

## Was du im Hearing sagen kannst

> "Accuracy ist bei diesem Datensatz kein sinnvolles Maß wegen der 86/14-Imbalance. Ich schaue primär auf F1-SPAM und ROC-AUC. Nach Deduplication (5.171 unique Messages, 403 Duplikate entfernt) erreicht Logistic Regression F1-SPAM=0.840, Precision=0.935, Recall=0.763, ROC-AUC=0.991. Das ist bewusst auf dem sauberen deduplizierten Set — ohne Deduplication waren die Zahlen künstlich besser (F1=0.913), weil dieselben Nachrichten in Train und Test landen konnten. Die echten Zahlen sind konservativer und ehrlicher."

**Wichtige Erkenntnis:** Die Differenz zwischen F1=0.913 (mit Duplikaten) und F1=0.840 (ohne) ist ein konkretes Beispiel für Data Leakage: wenn die gleiche SMS im Trainings- und Testset ist, "kennt" das Modell sie bereits und Recall ist künstlich hoch. Das ist genau der Grund warum Deduplication vor dem Split gemacht werden muss.

## Stratified Cross-Validation

Wir verwenden StratifiedKFold: Jeder Fold enthält dieselbe HAM/SPAM-Verteilung wie der Gesamtdatensatz. Wichtig bei imbalanced Data, sonst kann ein Fold zufällig zu wenig SPAM-Samples haben.
