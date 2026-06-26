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

## Cloud-Provider: drei bewusst diverse Ansätze

Die Entscheidung war, drei echte Alternativen zu zeigen, nicht dasselbe Modell dreimal zu hosten:

| Provider | Modell | Dim | Fokus |
|----------|--------|-----|-------|
| Ollama (lokal) | nomic-embed-text | 768 | Englisch, Privacy-first, kein API-Key |
| OpenAI | text-embedding-3-small | 1536 | Englisch, State-of-the-art |
| Azure AI Foundry | Cohere-embed-v3-multilingual | 1024 | **Multilingual** — relevant für deutsche SMS |

### Warum nicht text-embedding-3-small über Azure?

`text-embedding-3-small` wäre auf Azure verfügbar — aber nur mit explizitem Deployment im Azure Portal konfigurieren. Inhaltlich wäre das OpenAI ein zweites Mal, nur auf Azure-Infrastruktur. Cohere multilingual ist ein anderes Modell-Family, ein anderer Vendor, und direkt als Serverless-Modell verfügbar ohne Deployment-Setup.

### Azure AI Foundry vs. classic Azure OpenAI

Azure AI Foundry ist die aktuelle Microsoft-Plattform (Azure OpenAI Service ist darin aufgegangen). Der Zugriff läuft über `/openai/v1` Compatibility-Endpoint mit direkten Model-IDs statt Deployment-Namen — identisches Interface zu OpenAI, andere Infrastruktur und andere Modell-Auswahl.

## Was du im Hearing sagen kannst

> "Embeddings kodieren die semantische Bedeutung eines Textes als Vektor. Ähnliche Texte liegen nahe beieinander im Vektorraum. Das erlaubt uns SMS zu clustern nicht nach Wortwahl (TF-IDF), sondern nach *Bedeutung* — auch wenn zwei Spam-Nachrichten vollständig unterschiedliche Wörter verwenden, landen sie im selben Cluster. Das ist der fundamentale Unterschied zur klassischen ML-Analyse in Teil 1."
