# Der Datensatz – Herkunft, Untersuchung, Befunde, Konsequenzen

## Woher kommen die Daten?

Der SMS Spam Collection v.1 Datensatz ist ein Kompositum aus vier unabhängigen Quellen:

| Quelle | Typ | Menge |
|--------|-----|------:|
| **Grumbletext** (UK-Forum, manuell extrahiert) | SPAM | ~425 |
| **NUS SMS Corpus** (National University of Singapore) | HAM | ~3.375 |
| **Caroline Tag's PhD-Arbeit** | HAM | ~450 |
| **SMS Spam Corpus v.0.1 Big** | HAM + SPAM | ~1.324 |

Das ist keine zufällige Stichprobe — es ist ein zusammengesetzter Datensatz aus Texten die unterschiedlich gesammelt, anonymisiert und verarbeitet wurden. Das hat direkte Konsequenzen für das Modell.

**Gesamtzahl:** 5.574 Messages (4.827 HAM / 747 SPAM = 86% / 14%)

---

## Was wir in den Rohdaten gefunden haben

Beim Durchsehen der Outputs fielen Artefakte auf:
- In den HAM-Bigrams dominierte `lt gt` mit 270 Vorkommen — weit mehr als jedes sinnvolle Wortpaar
- Trigrams wie `lt decimal gt`, `gt lt gt` — kein echter Sprachinhalt

Das war ein Signal: irgendwas stimmt mit den Rohdaten nicht.

### Die Untersuchung

Wir haben systematisch geprüft: Welche HTML-kodierten Token (`&lt;URL&gt;`, `&lt;#&gt;`, etc.) kommen vor, und in welcher Klasse?

```
&lt;#&gt;       214 Treffer — ausschließlich HAM
&lt;DECIMAL&gt;   21 Treffer — ausschließlich HAM
&lt;TIME&gt;       7 Treffer — ausschließlich HAM
&lt;URL&gt;        3 Treffer — ausschließlich HAM
&lt;EMAIL&gt;      1 Treffer — ausschließlich HAM
Kein einziger SPAM-Treffer bei allen Token.
```

### Die Antwort

Das sind **Anonymisierungsplatzhalter des NUS-Teams**. Die Singapur-Studenten haben echte SMS beigetragen, das NUS-Team hat vor der Veröffentlichung Nummern, Beträge, URLs, Zeiten und E-Mail-Adressen durch getypte Platzhalter ersetzt — aus Datenschutzgründen.

Die SPAM-Nachrichten (Grumbletext) wurden nicht gleich behandelt. Dort stehen echte Telefonnummern, Preise und Links drin.

**Warum `lt gt` in den Bigrams?** Der CountVectorizer in unserer N-Gramm-Analyse hat auf rohem Text gearbeitet und `&lt;` als `lt` und `&gt;` als `gt` tokenisiert — weil `&` und `;` als Trennzeichen behandelt wurden. Das war ein Bug, der durch die Untersuchung aufgefallen ist und behoben wurde: N-Gramm-Analyse läuft jetzt auf normalisiertem Text.

### Was das für das Modell bedeutet

NUS-Platzhalter (`&lt;#&gt;`, `&lt;URL&gt;`, …) werden auf **dieselben Tokens** gemappt wie getippter Nutzerinhalt (`TOKEN_NUM`, `TOKEN_URL`, …). Die spezifische Nummer oder URL interessiert das Modell nicht — es lernt die *Präsenz* solcher Elemente als Signal. In SPAM sind Nummern und URLs dicht gesät (Telefonnummern, Shortcodes, Preise, Links), in HAM tauchen sie seltener und vereinzelter auf.

Das Modell lernt: "dichte Häufung von Zahlen/URLs = Spam". Das ist das richtige Signal.

Für die spezifische Nummer oder URL — ob sie auf einer Blacklist steht, ob sie ein bekannter IoC ist — braucht es ein separates System (Blacklists, Threat-Intel-Feeds). Das ist nicht die Aufgabe des Textklassifizierers.

---

## Zwei Parsing-Details die man kennen sollte

### quoting=QUOTE_NONE

Die Datei ist Tab-separated. Das Standard-CSV-Parsing von pandas interpretiert Anführungszeichen als Feldbegrenzer. Einige SMS-Texte enthalten Anführungszeichen — das hat dazu geführt, dass 2 HAM-Nachrichten vom Parser verschluckt wurden (5.572 statt 5.574 Rows). Mit `quoting=csv.QUOTE_NONE` behandelt pandas jedes Tab als Feldtrenner und ignoriert Anführungszeichen. Ergebnis: alle 5.574 Nachrichten werden korrekt geladen.

Das ist kein akademisches Detail — in produktionsnahem Code ist falsches Parsing ein echter Bug.

### Die 5.574 / 5.572 Verwirrung

Das readme des Datensatzes nennt 5.572. Die Rohdatei hat 5.574 Zeilen. Erklärt sich durch das Parsing: ohne `QUOTE_NONE` wurden früher 2 Zeilen verschluckt, und diese fehlerhafte Zahl wurde dokumentiert. Die korrekte Zahl ist 5.574.

### Duplikate im Datensatz

403 der 5.574 Rows sind exakte Duplikate (309 HAM / 94 SPAM). 5.171 unique Texte bleiben nach Deduplication. Kein Duplikat-Paar hat widersprüchliche Labels.

**Warum das für die Evaluation wichtig ist:** Wenn man `train_test_split` auf dem Rohdatensatz macht, kann dieselbe SMS-Nachricht im Training- und im Test-Set landen. Das Modell "kennt" sie dann bereits — Recall und F1 werden künstlich besser. Das ist **Data Leakage**.

Konkret: Ohne Deduplication → F1-SPAM=0.913. Mit Deduplication → F1-SPAM=0.840. Die ehrlichen Zahlen sind niedriger.

**Warum wir auf rohem Text deduplizieren (nicht auf normalisiertem):**

Man könnte auch auf normalisiertem Text deduplizieren. Der Unterschied:

```
Raw:        "Call 0800-123456 now!"   ≠  "Call 0800-999999 now!"
Normalized: "call token_num now"        =  "call token_num now"
```

Das wären *Normalized-Duplikate* — strukturell gleich, aber inhaltlich verschiedene Nachrichten. Für die Evaluierung des Classifiers ist das kein Problem: der Classifier trainiert auf rohem Text und normalisiert intern in der Pipeline. Was wir verhindern wollen ist, dass dieselbe *Zeichenkette* in Train und Test ist. Normalized-Duplikate sind verschiedene echte Nachrichten die zufällig auf dasselbe Muster mappen — das ist genau was Normalisierung leisten soll, kein Fehler.

→ Raw-Dedup ist die richtige Wahl für Evaluierungszwecke.

---

## Biases im Datensatz

### Geografisch und sprachlich

**HAM:** Dominiert vom NUS-Corpus — Singapur-Studierende in Singlinenglisch. Erkennbar an Wörtern wie "lor", "lah", "da", "cos" (Diskurspartikeln im Singapore English). Das Schreibmuster ist spezifisch für diesen Kontext.

**SPAM:** UK-Premium-SMS-Betrug aus dem Jahr ~2011. Erkennbar an: "BT national rate", "150p per msg", "await collection", britischen Postadressen (London W1J Postleitzahlen).

Ein Modell das auf diesem Datensatz trainiert wird, ist ein guter Detektor für UK/Singapur-Spam aus 2011. Auf österreichischem, deutschem oder brasilianischem Traffic von 2026 würde seine Treffsicherheit sinken — weil die Sprachmuster, die Spam-Typen und die kulturellen Referenzen andere sind.

### Währungssymbole als konkretes Beispiel

In diesem Datensatz erscheint `£` fast ausschließlich in SPAM. Ohne Normalisierung würde das Modell "£ = Spam-Signal" lernen — was für UK-Daten stimmt, aber für Euro- oder Dollar-Traffic wertlos wäre.

Deshalb normalisieren wir alle Währungssymbole (`£`, `$`, `€`, `¥`, `₹`, `₽`, `₩`, `₿`), ISO-Codes (`EUR`, `USD`, `GBP`, …) und ausgeschriebene Währungswörter (`pounds`, `dollars`, `euros`, …) zu `TOKEN_CURRENCY`. Das Modell lernt "Währungsreferenz vor/nach Betrag = Spam-Signal" — locale-unabhängig. Ohne diese Normalisierung würde `"Win £1000 now!"` korrekt als Spam erkannt, `"Win 1000 EUR now!"` aber nicht.

**Zahlen mit Trennzeichen:** Beträge wie `£100,000` oder `1.50 per msg` enthalten Komma oder Punkt als Tausender- oder Dezimaltrennzeichen. Ohne explizite Normalisierung würde `_STANDALONE_NR` die Gruppen einzeln ersetzen und `tokenize()` das Trennzeichen danach löschen — aus `1,000` wird `token_numtoken_num` (kontextloser Rauschwert im Training). Fix: `_FORMATTED_NR` matcht die gesamte formatierte Zahl vor `_STANDALONE_NR` und ersetzt sie durch ein einzelnes `TOKEN_NUM`.

**Slash/Dash-Sequenzen:** Datumsangaben wie `02/09/03` und Telefonnummern wie `07-815-296` werden von `_STANDALONE_NR` in einzelne Gruppen geteilt. Nach Punctuation-Stripping im Tokenizer (`,` und `/` werden entfernt) entstehen Artefakte wie `token_numtoken_numtoken_num` oder noch schlimmer `943` (aus `9/4/3` ohne Normalisierung). Fix: `_SEP_NR` normalisiert die gesamte Sequenz zu einem einzelnen `TOKEN_NUM` bevor der Tokenizer greift.

### Normalisierung inspizieren: normalized_texts.tsv

Die Normalisierungspipeline lässt sich direkt inspizieren:

```bash
sms-analyze normalized   # oder: sms-analyze all
```

Das schreibt `outputs/normalized_texts.tsv` — eine **Tab-Separated Values**-Datei mit drei Spalten:

| label | original | normalized |
|-------|----------|------------|
| ham   | "I'll be there at 8" | "i'll be there at TOKEN_NUM" |
| spam  | "WIN £1000 cash — call 08001234567!" | "WIN TOKEN_CURRENCY TOKEN_NUM cash call TOKEN_NUM" |

**Warum TSV und nicht CSV?** SMS-Texte enthalten selbst häufig Kommas und Anführungszeichen. Ein kommagetrennte Datei würde durch Standard-CSV-Parser falsch gesplittet. Tabs kommen in SMS-Texten praktisch nie vor — daher ist Tab als Trennzeichen sicher.

Die Datei öffnet sich in Excel, LibreOffice Calc oder direkt in Python:

```python
import pandas as pd
df = pd.read_csv("outputs/normalized_texts.tsv", sep="\t")
```

Damit kann man konkret nachvollziehen was die Pipeline aus einzelnen Nachrichten macht — vor allem bei Grenzfällen wie `£100,000`, `07-815-296` oder `&lt;#&gt;` Platzhaltern.

### Zeitlicher Bias

Der Datensatz stammt aus ~2011. Spam-Muster ändern sich. Aktuelle Phishing-SMS sehen anders aus als Premium-Rate-SMS von vor 15 Jahren. Das Modell würde auf heutigem Spam-Traffic in der Praxis regelmäßig neu trainiert werden müssen.

---

## Spurious Correlations und Adversarial Robustness

Im Datensatz gibt es ~50 HAM-Nachrichten wo jemand `Ü` als SMS-Kurzform für "you" verwendet — wahrscheinlich Nokia-Tastatur mit European Language Pack, wo die U-Taste bei Mehrfachdruck auf `Ü` sprang. Ergebnis: `Ü` kommt 50× in HAM vor, 0× in SPAM.

Ohne Normalisierung würde das Modell lernen: `ü` → HAM. Das ist eine **Spurious Correlation** — kein echter Inhalt, nur Tastatur-Artefakt einer einzelnen Person. Gleichzeitig ein **Adversarial Attack Surface**: ein Spammer der diesen Bias kennt, hängt ein `Ü` an seine Nachricht und kippt die Klassifikation Richtung HAM.

Fix: `Ü` → `u` in Preprocessing. Semantisch korrekt (beides = "you"), schließt den Angriffsvektor.

**Das ist der Punkt der im Hearing zählt:** Ein Modell mit 99% AUC kann stille, exploitierbare Schwachstellen tragen, die nur jemand findet der die Daten wirklich versteht — nicht nur die Metriken liest.

## Was du im Hearing sagen kannst

> "Ich habe die Rohdaten nicht einfach genommen wie sie sind. Ich habe mir die HTML-kodierten Token angeschaut, die im Datensatz auftauchen — und systematisch geprüft ob sie in HAM oder SPAM vorkommen. Das Ergebnis: alle diese Token sind ausschließlich in HAM-Nachrichten. Das bedeutet sie kommen nicht von Spammern die Vorlagen nicht befüllt haben, und sie sind auch keine Scraping-Artefakte. Es sind Anonymisierungsplatzhalter die das NUS-Team aus Datenschutzgründen eingefügt hat, bevor sie den Datensatz veröffentlicht haben."

> "Auf Basis dieser Untersuchung habe ich die Normalisierungspipeline entsprechend aufgebaut — und ich habe auch erkannt dass der Datensatz geografische und sprachliche Biases hat die man explizit benennen muss wenn man über die Grenzen des Modells spricht."
