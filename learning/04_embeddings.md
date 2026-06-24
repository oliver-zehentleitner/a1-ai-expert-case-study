# Embeddings – Semantische Vektoren verstehen

## Was sind Embeddings?

TF-IDF-Vektoren sind riesig (10.000 Dimensionen) und wissen nichts über Bedeutung:  
"Car" und "Automobile" hätten komplett unterschiedliche TF-IDF-Vektoren, obwohl sie dasselbe bedeuten.

**Embeddings** sind dichte, niedrigdimensionale Vektoren (768 oder 1536 Zahlen) die von einem großen Sprachmodell (LLM) gelernt wurden. Das Modell hat auf Milliarden Texten trainiert und dabei gelernt welche Wörter und Sätze ähnliche Bedeutungen haben.

## Das geometrische Prinzip

Semantisch ähnliche Texte → ähnliche Vektoren → geringer Abstand im Vektorraum.

```
embed("Win a free iPhone!")    ≈  embed("Claim your prize now!")
embed("See you tomorrow")      ≈  embed("Talk to you later")

cosine_similarity(embed("Win a free iPhone!"), embed("Claim your prize now!")) ≈ 0.89
cosine_similarity(embed("Win a free iPhone!"), embed("See you tomorrow"))      ≈ 0.21
```

## Cosine Similarity vs. Euklidischer Abstand

Für hochdimensionale Vektoren ist **Cosine Similarity** besser als euklidischer Abstand:

```
cosine_sim(a, b) = (a · b) / (|a| × |b|)
```

Misst den Winkel zwischen Vektoren, nicht die absolute Distanz. 1 = identisch, 0 = unähnlich, -1 = entgegengesetzt.

## Unser Modell: nomic-embed-text (Ollama)

- Lokales Open-Source-Modell, 768 Dimensionen
- Gut für englische Texte
- Läuft auf CPU, kein API-Key nötig

## Für das Video: text-embedding-3-small (Azure OpenAI)

- 1536 Dimensionen
- State-of-the-art Qualität
- Batch-Processing (256 Texte pro Request)
- Über Azure: enterprise-grade, DSGVO-konform

## Was du im Hearing sagen kannst

> "Embeddings kodieren die semantische Bedeutung eines Textes als Vektor. Ähnliche Texte liegen nahe beieinander im Vektorraum. Das erlaubt uns SMS zu clustern nicht nach Wortwahl (TF-IDF), sondern nach *Bedeutung* — auch wenn zwei Spam-Nachrichten vollständig unterschiedliche Wörter verwenden, landen sie im selben Cluster. Das ist der fundamentale Unterschied zur klassischen ML-Analyse in Teil 1."
