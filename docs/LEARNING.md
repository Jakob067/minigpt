# Lernnotizen und Interviewfragen

## 1 · Grundgerüst

Das installierbare Python-Paket heißt `minigpt`.
Der Quellcode liegt in `src/`, getrennt von den Tests.
`pyproject.toml` definiert Abhängigkeiten und CLI-Einstiegspunkte.
Makefile und CI führen dieselben Qualitätsprüfungen aus.
Kurze Feature-Branches und atomare Commits halten Änderungen nachvollziehbar.

1. **Warum ein src-Layout?** Tests verwenden das installierte Paket und entdecken Packaging-Probleme früher.
2. **Warum eine eigene virtuelle Umgebung?** Die Projektabhängigkeiten bleiben von anderen Projekten getrennt.
3. **Was prüft CI?** Ruff und CPU-Tests; ein tatsächlicher GitHub-Lauf steht bis zum ersten Push aus.

## 2 · Daten und Reproduzierbarkeit

Jedes Zeichen bekommt eine ID aus einer sortierten Vokabelliste.
Training und Validierung werden zusammenhängend 90/10 geteilt.
Ein Batch enthält Eingaben und um ein Zeichen verschobene Targets.
Konfigurationsklassen prüfen Größen, Lernraten und Architekturbedingungen.
Seeds und ein Datenfingerabdruck machen die Eingaben eines Laufs nachvollziehbar.

1. **Warum verschobene Targets?** Das Modell lernt, aus dem bisherigen Kontext das nächste Zeichen vorherzusagen.
2. **Warum den Wortschatz speichern?** Dieselbe ID muss nach dem Laden wieder dasselbe Zeichen meinen.
3. **Welche Information sieht der Tokenizer aus der Validierung?** Nur das Zeicheninventar; Trainingsstatistiken und Gewichte nutzen keine Validierungstargets.

## 3 · Transformer

Embedding und optionale gelernte Positionen ergeben den Residualstrom.
Pre-LayerNorm normalisiert die Eingabe jeder Attention- und MLP-Verzweigung.
Attention berechnet Q, K, V, skalierte Scores, Kausalmaske und Softmax explizit.
Das GELU-MLP erweitert die Kanalbreite um Faktor vier und projiziert zurück.
Embedding und Ausgabekopf teilen eine Gewichtsmatrix; Residualprojektionen werden kleiner initialisiert.

1. **Warum √Head-Dimension im Nenner?** Das begrenzt die Größenordnung der Skalarprodukte und verhindert zu früh gesättigte Softmax-Verteilungen.
2. **Wie beweist der Test Kausalität?** Er ändert spätere Tokens und prüft unveränderte frühere Logits.
3. **Wozu die Residualverbindung?** Sie transportiert Information und Gradienten direkt durch den Block.

## 4 · Training und Checkpoints

AdamW verwendet die vorgegebenen Betas und Weight Decay nur für Matrizen.
Die Lernrate steigt linear und fällt anschließend kosinusförmig.
Gradient Clipping begrenzt die gemeinsame Gradientennorm.
Evaluation mittelt mehrere feste Batches und verwendet einen eigenen Zufallsgenerator.
Checkpoints speichern Modell, Optimizer, RNG, Schritt, Konfiguration und Git-Commit.

1. **Warum reicht ein state_dict des Modells beim Resume nicht?** Optimizer-Momente, Sampling und Dropout müssten sonst von einem anderen Zustand fortsetzen.
2. **Warum bleibt max_iters beim Resume gleich?** Es bestimmt den gesamten Verlauf des LR-Schedules; `--stop-after` unterbricht ihn ohne Änderung.
3. **Was misst der Benchmark?** Synchronisierte Batch-Erzeugung, Forward, Backward, Clipping und AdamW nach Warmup; Kaltstart und Checkpoint-Schreiben sind separat.

## 5 · Generierung und Auswertung

Generierung füttert die bisherige Sequenz wiederholt ins Modell zurück.
Das Kontextfenster wird auf die letzten block_size Tokens gekürzt.
Temperatur verändert die Schärfe; Top-k und Top-p filtern Kandidaten.
Die vollständige Evaluation zählt jedes Validierungstarget genau einmal.
Perplexity ist exp(Loss), Bits pro Zeichen sind Loss/ln(2).

1. **Warum ist Temperatur null greedy?** Sie verwendet direkt argmax, ohne Division durch null.
2. **Warum jeden Loss mit der Tokenanzahl gewichten?** Ein kurzer Restblock darf nicht so viel Gewicht erhalten wie ein voller Batch.
3. **Warum sind Trainingsevaluation und volle Evaluation leicht verschieden?** Zufällige Fenster und nicht überlappende Vollauswertung haben andere Kontextanfänge und Targetmengen.

## 6 · Ablationen

Sechs eindeutige Varianten decken drei Faktoren ab; die Baseline wird wiederverwendet.
Jede Variante wird mit zwei Seeds und gleicher Anzahl Trainingstokens trainiert.
Pro Vergleich wird genau ein Modellparameter verändert.
Bei kürzerem Kontext führen feste Sequenz-Batchgrößen zu mehr Optimizer-Schritten.
Mittelwert und Stichprobenstandardabweichung beschreiben Ergebnisse, ohne statistische Sicherheit vorzutäuschen.

1. **Warum nicht die beste einzelne Seed-Zahl berichten?** Das würde günstige Zufallsergebnisse bevorzugen.
2. **Beweist das Ergebnis, dass kurze Kontexte besser sind?** Nein; hier sind Budget, Updatezahl und kontextabhängiges Evaluationsprotokoll entscheidende Einschränkungen.
3. **Warum den letzten Checkpoint vergleichen?** Jede Ablation verwendet das ganze gleiche Tokenbudget und dieselbe Auswahlregel.

## 7 · Dokumentation und Release

Alle Zahlen der README stammen aus gespeicherten, abgeschlossenen Läufen.
JSON-Berichte enthalten Konfiguration, Git-Commit, Hardwareumgebung und Rohwerte.
Beispieltexte werden mit festen Seeds erzeugt und unverändert gespeichert.
Das kleine Tiny-Modell liegt bei; das größere Small-Modell bleibt lokal und ist reproduzierbar.
Der lokale Release-Tag dokumentiert die geprüfte Python-Version des Projekts.

1. **Warum keine erfundene grüne CI?** Ohne Push wurde GitHub Actions nicht ausgeführt; lokale Prüfungen und Remote-CI sind unterschiedliche Nachweise.
2. **Warum Softwareversionen festhalten?** Unterschiedliche Kernel und Bibliotheken können Rundung, Laufzeit und Ergebnisse verändern.
3. **Was wäre ein sinnvoller nächster Schritt?** Zuerst mehr Seeds und längere, kontrollierte Ablationen; danach separat freigegebene Stretch Goals.
