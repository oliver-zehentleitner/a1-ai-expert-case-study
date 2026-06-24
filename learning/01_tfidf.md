# TF-IDF – Text in Zahlen verwandeln

## Das Problem

ML-Modelle brauchen Zahlen, kein Text. Die naivste Idee: zähl einfach wie oft jedes Wort vorkommt (Bag of Words). Problem: "the", "a", "I" kommen überall vor und sind sinnlos.

## Die Lösung: TF-IDF

**TF** (Term Frequency): Wie oft kommt das Wort in *dieser* SMS vor?  
**IDF** (Inverse Document Frequency): Wie selten ist das Wort über *alle* SMS?

```
TF-IDF(word, doc) = TF(word, doc) × IDF(word)

IDF(word) = log( N / df(word) )
  N       = Anzahl aller Dokumente (5574 SMS)
  df(word) = Anzahl der Dokumente die das Wort enthalten
```

**Beispiel:**

| Wort      | Kommt in 1 SMS vor | In wievielen SMS? | IDF     | TF-IDF |
|-----------|--------------------|-------------------|---------|--------|
| "free"    | 3×                 | 220 von 5.574     | 3.23    | 9.69   |
| "the"     | 5×                 | 4.800 von 5.574   | 0.15    | 0.75   |

"free" bekommt einen hohen Score weil es in dieser SMS oft steht **und** in den meisten anderen SMS nicht vorkommt → starkes Signal für Spam.

"the" hat TF-IDF nahe 0 obwohl es oft vorkommt — weil es überall ist, ist es kein Unterscheidungsmerkmal.

## sublinear_tf=True

Wir verwenden `log(1 + tf)` statt rohem `tf`. Warum? Eine Spam-SMS mit "FREE" 10× ist nicht 10× spammiger als eine mit "FREE" 1×. Der Log dämpft diesen Effekt.

## Was du im Hearing sagen kannst

> "Ich verwende TF-IDF mit `sublinear_tf` und Bigrams. TF-IDF gewichtet Wörter die in einer Klasse charakteristisch sind hoch, aber ignoriert Stoppwörter automatisch durch den IDF-Term. Bigrams erlauben dem Modell Phrasen wie 'free entry' oder 'win prize' als Einheit zu lernen."
