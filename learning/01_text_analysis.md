# Textanalyse – Tasks 1, 2, 3

## Task 1: Wortfrequenzanalyse

### Was passiert

Jeder Text wird tokenisiert (Kleinschreibung, Satzzeichen entfernt, Stoppwörter entfernt), dann werden alle Tokens gezählt. Top-20 nach Häufigkeit.

### Was wir sehen (echte Zahlen aus dem Datensatz)

Top-20 über alle SMS nach Normalisierung:

| Token | Count | Bedeutung |
|-------|------:|-----------|
| `__num__` | 1.671 | Normalisierte Zahlen — in beiden Klassen, aber SPAM ist dichter |
| `u` | 1.301 | SMS-Kurzform für "you" — fast ausschließlich HAM |
| `your` | 677 | "your prize", "your mobile" — leicht SPAM-lastig |
| `call` | 578 | Ambig: "call me later" (HAM) vs. "call now to claim" (SPAM) |
| `__currency__` | 315 | Währungssymbol + Betrag — strukturelles SPAM-Signal |
| `free` | 275 | Starkes SPAM-Signal ("free prize", "free entry") |

Die aggregierte Ansicht vermischt die Klassen. `call` sieht wichtig aus, ist aber für sich allein nicht trennscharf. Erst der Klassenvergleich und die N-Gramme zeigen wo `call` tatsächlich hingehört.

### Stoppwörter

Stoppwörter ("the", "a", "is", "to") werden entfernt weil sie in beiden Klassen gleich häufig sind und kein Unterscheidungsmerkmal liefern.

**Wie wir zur Liste gekommen sind:** Zunächst ohne Filterung gelaufen, Top-80 nach Häufigkeit sortiert, jeden Eintrag manuell geprüft — ist das ein reines Funktionswort ohne diskriminativen Wert, oder trägt es Signal? Das Ergebnis ist `src/sms_analysis/analysis/stopwords.txt` — eine editierbare Datei mit einer Wort pro Zeile, `#`-Kommentare für Kategorien.

**Bewusst auf der Liste behalten** (Signal-Words die bleiben): `call`, `free`, `u`, `ur`, `2`, `4`, `ok`, `just`, `know`, `got`, `go`, `like`, `dont`, `ill`, `im` — das sind echte Spam- oder Ham-Signale.

**Kein NLTK, kein Netzwerk-Download** — die sklearn-eigene `ENGLISH_STOP_WORDS` ist zu aggressiv (filtert `call` und `free` raus, beides wichtige Spam-Marker). NLTK lädt Wortlisten live aus dem Internet was in produktionsnahem Code ein No-Go ist. Eine eigene, aus den echten Daten abgeleitete Liste ist die sauberste Lösung.

### Zwei Stopword-Ebenen — warum das kein Widerspruch ist

Es gibt im Code zwei verschiedene Stellen wo Stoppwörter relevant sind:

| Kontext | Stopwords | Warum |
|---------|-----------|-------|
| Frequenzanalyse (`frequency.py`, `ngrams.py`) | `stopwords.txt` (eigene Liste) | Für explorative Analyse wollen wir die echten Signalwörter sehen — `call`, `free` müssen erscheinen |
| TF-IDF im Classifier (`pipeline.py`) | `stop_words="english"` (sklearn) | Hier macht der IDF-Term Stoppwörter ohnehin bedeutungslos (IDF nahe 0 weil sie überall vorkommen). `call` und `free` haben hohen IDF-Wert und bleiben als Features erhalten — die Vorabfilterung betrifft nur reine Funktionswörter |

Kurz: Die eigene `stopwords.txt` ist für die Analyse da. Im Classifier erledigt TF-IDF das automatisch — kein manuelles Filtering nötig, weil der IDF-Term Stoppwörter selbst auf nahe 0 gewichtet.

---

## Task 2: HAM vs. SPAM – Log-Odds-Ratio

### Was ist Log-Odds?

Statt rohe Häufigkeiten zu vergleichen (HAM hat 4.827 Messages, SPAM nur 747 — ein direkter Vergleich wäre verzerrt), berechnen wir wie stark ein Wort zur einen oder anderen Klasse neigt.

```
p_spam(word) = (count_in_spam + 1) / (total_spam_words + vocab_size)
p_ham(word)  = (count_in_ham + 1)  / (total_ham_words + vocab_size)

log_odds = log( p_spam / p_ham )

positiv → Wort neigt zu SPAM
negativ → Wort neigt zu HAM
```

Das `+1` ist Laplace-Smoothing: verhindert Division durch null wenn ein Wort in einer Klasse nicht vorkommt.

### Was wir sehen

Starke SPAM-Signale (tatsächliche Top-Wörter aus dem Datensatz): `claim`, `token_currency`, `prize`, `guaranteed`, `awarded`, `150p`, `150ppm`, `tcs` (Terms & Conditions), `ringtone`, `sae` (Stamped Addressed Envelope — britisches Postverfahren), `token_url`

Hinweis: `£` taucht hier *nicht* direkt auf — es wird von `preprocess()` zu `token_currency` umgewandelt, bevor die Analyse rechnet. Das Feature im Modell ist `token_currency`, nicht das rohe Symbol.

Starke HAM-Signale: `u`, `lor` (Singlish), `later`, `da`, `ill`, `lol`, `morning`, `sorry`

Das ist exakt der geografische Bias sichtbar gemacht: SPAM klingt nach britischem Gewinnspiel-Betrug, HAM klingt nach Singapur-Studierenden.

---

## Task 3: N-Gramme

### Was ist ein N-Gramm?

Statt einzelner Wörter (Unigramme) schauen wir uns Wortpaare (Bigramme) oder Worttripel (Trigramme) an.

```
Text: "win a free prize now"

Unigramme:  win, free, prize, now          (nach Stoppwörtern: win, free, prize)
Bigramme:   win a, a free, free prize, prize now
Trigramme:  win a free, a free prize, free prize now
```

### Warum sind N-Gramme nützlich?

"Free" alleine ist mehrdeutig: "Free time" (HAM) vs. "Free prize" (SPAM).
Das Bigramm "free prize" ist eindeutig SPAM.

N-Gramme fangen Phrasen auf die einzelne Wörter nicht erfassen können. Das TF-IDF-Modell in Task 4 nutzt genau deshalb `ngram_range=(1,2)`.

### Was wir in den Daten sehen — echte Zahlen

**HAM-Bigramme (Top-20):**

| Bigramm | Count | Bedeutung |
|---------|------:|-----------|
| `call later` | 50 | Konversation über Rückruf |
| `ll call` | 44 | Ellipse: "I'll call" |
| `let know` | 40 | "let me know" |
| `sorry ll` | 39 | "sorry I'll" — Entschuldigungen |
| `take care` | 38 | Verabschiedung |
| `right now` | 35 | Zeitreferenz |
| `good morning` | 32 | Begrüßung |
| `ok lor` | 22 | Singlish-Diskurspartikel |
| `happy new` | 19 | Teil von "happy new year" |
| `wat time` | 18 | Singlish: "what time" |

→ Konversation, Zeitplanung, Grüße. Das sind Alltagssätze. Singlish erkennbar (`ok lor`, `wat time`).

**SPAM-Bigramme (Top-20):**

| Bigramm | Count | Bedeutung |
|---------|------:|-----------|
| `__currency__ __num__` | 273 | Geldbeträge — 273 Vorkommen in 747 SPAM-Messages! |
| `call __num__` | 229 | "Call 0800..." — Handlungsaufforderung + Telefonnummer |
| `__num__ __num__` | 154 | Zwei Nummern hintereinander (Shortcodes, Preise) |
| `__num__ now` | 87 | Urgency: "text 84484 now" |
| `__num__ cash` | 54 | Gewinnversprechen |
| `__num__ prize` | 54 | Gewinnversprechen |
| `your mobile` | 50 | "to your mobile" |
| `please call` | 46 | Handlungsaufforderung |
| `won __currency__` | 33 | "You won £..." |
| `nokia __num__` | 24 | Nokia-Gewinnspiel-Kampagne (konkreter Kampagnen-Fingerprint) |

→ Fast ausschließlich strukturell: Geldbeträge, Telefonnummern, Urgency-Phrasen. Kein echter Gesprächsinhalt.

**SPAM-Trigramme zeigen Template-Struktur:**

| Trigramm | Count | Bedeutung |
|----------|------:|-----------|
| `__currency__ __num__ prize` | 49 | "£500 prize" — Grundstruktur jedes Gewinnspiels |
| `__num__ __num__ __num__` | 49 | Drei Nummern hintereinander (Nummer, Preis, Shortcode) |
| `won __currency__ __num__` | 32 | "won £1000" |
| `guaranteed call __num__` | 23 | "Guaranteed. Call 08006..." |
| `prize guaranteed call` | 21 | Gleiche Kampagne, leicht anders |
| `call now __num__` | 18 | Urgency + Shortcode |
| `urgent your mobile` | 18 | Klassischer Spam-Opener |

→ Das sind keine Sätze — das sind Templates. Mehrere Trigramme (`guaranteed call __num__`, `prize guaranteed call`, `__num__ prize guaranteed`) beschreiben dieselbe Spam-Kampagne aus verschiedenen Blickwinkeln.

**HAM-Trigramme:**

| Trigramm | Count | Bedeutung |
|----------|------:|-----------|
| `ll call later` | 41 | "I'll call later" — echter Dialog |
| `sorry ll call` | 38 | "Sorry, I'll call" |
| `happy new year` | 18 | Neujahrsgrüße |
| `cant pick phone right now pls` | 12 | Echter Kontext |
| `hi hi hi` | 11 | Playful repetition — echte Person |
| `birthday wishes nimya` | 6 | Konkreter Name, persönliche Nachricht |
| `kiss across sea` | 6 | Fernbeziehung — sehr persönlich |

→ Ellipsen, Umgangssprache, konkrete Namen und Orte. Das ist echte menschliche Kommunikation.

**Was die Gegenüberstellung zeigt:** SPAM-N-Gramme sind strukturell und number-heavy. HAM-N-Gramme sind sprachlich und kontextuell. Dieses Muster ist robuster als einzelne Wörter — ein Spammer der das Wort "free" vermeidet, kann die Grundstruktur `__currency__ __num__ prize` nicht vermeiden.

### Der lt/gt-Bug und warum er wichtig war

Vor dem Fix dominierte `lt gt` die HAM-Bigramme mit 270 Vorkommen. Ursache: der CountVectorizer hat auf rohem Text gearbeitet, `&lt;` wurde zu `lt` und `&gt;` zu `gt` tokenisiert. Nach dem Fix (Normalisierung vor der N-Gramm-Analyse) ist `lt gt` komplett verschwunden. Das zeigt: Preprocessing-Fehler verfälschen die Analyse fundamental — man würde aus den "Top Bigrammen" falsche Schlüsse ziehen.

### Was du im Hearing sagen kannst

> "N-Gramme fangen kontextuelle Muster auf die Unigramme nicht sehen. 'Free' alleine ist mehrdeutig — im by-class Vergleich sieht man sofort: SPAM-Bigramme sind fast ausschließlich strukturell (`__currency__ __num__` 273 Mal, `call __num__` 229 Mal), während HAM-Bigramme echter Dialog sind (`call later`, `let know`, `take care`). Im TF-IDF-Modell nutze ich `ngram_range=(1,2)` — das Modell lernt genau diese Phrasenstruktur als Feature. Und in den Trigrammen sieht man ganze Kampagnen: `guaranteed call __num__`, `prize guaranteed call` und `__num__ prize guaranteed` beschreiben denselben Spam-Sender aus drei verschiedenen Winkeln."
