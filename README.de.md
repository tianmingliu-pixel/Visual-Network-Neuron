# NeuroCore · Visual Network Neuron

[中文](README.md) | [English](README.en.md) | [한국어](README.ko.md) | **Deutsch**

Eine erweiterbare PyTorch-Bibliothek für neuronale Netze – mit einer visuellen Trainingsoberfläche, in der man einem Netz beim Lernen zusehen kann.

Phase 1 implementiert vier grundlegende Netze. Über eine **Registry + einen Erweiterungsordner** lassen sich später Netze aus jedem IT-Bereich (GNN, Audio, Zeitreihen, Empfehlungssysteme, Sicherheit, Reinforcement Learning …) ergänzen, ohne den Kern anzufassen.

| Modul | Datei | Kernidee |
|---|---|---|
| **Transformer** | `neurocore/models/transformer.py` | Encoder–Decoder nur aus Attention: `softmax(QKᵀ/√d)V`, kausale Maske + Cross-Attention |
| **ViT** | `neurocore/models/vit.py` | Ein Bild wird in Patches zerlegt → Tokens, [CLS] + Positionskodierung → Transformer-Encoder |
| **U-Net** | `neurocore/models/unet.py` | Down-/Up-Sampling mit Skip-Verbindungen; optional Zeitschritt-/Klassenbedingung für Diffusion |
| **DiT** | `neurocore/models/dit.py` | Transformer als Diffusions-Entrauscher, Zeitschritt und Klasse per adaLN-Zero |
| DDPM | `neurocore/diffusion/ddpm.py` | Vorwärts-Verrauschen / ε-Vorhersage / Ancestral Sampling + CFG, für U-Net und DiT |
| MLP | `neurocore/models/mlp.py` | Das einfachste vollvernetzte Netz – jedes Neuron und jedes Gewicht ist im Graphen sichtbar |
| Gemeinsame Bausteine | `neurocore/layers/` | Multi-Head-Attention, FFN, Pre-LN-Block, Sinus-/Zeitschritt-/Patch-Embeddings |

## Online-Seite

**<https://visual-network-neuron.vercel.app>** – öffnen und loslegen. Die Seite verbindet sich automatisch mit einem Backend: läuft `.\scripts\run.ps1 -Task ui` auf Ihrem Rechner, wird dort trainiert, sonst wird das Backend des Autors im Zuschau-Modus verwendet. Passwort, Tunnel und Fehlerbehebung: [DEPLOY.de.md](DEPLOY.de.md).

👉 **Zum ersten Mal hier?** Folgen Sie der [Einsteiger-Anleitung GETTING_STARTED.de.md](GETTING_STARTED.de.md), um alles Schritt für Schritt auf Ihrem eigenen Rechner zu installieren und auszuführen (kein Passwort nötig).

## Schnellstart (Windows PowerShell)

```powershell
cd NeuroCore
# Ein Befehl: Umgebung prüfen → .venv anlegen → passendes PyTorch für die GPU installieren → Tests → Demo
powershell -ExecutionPolicy Bypass -File .\deploy.ps1
```

Schritt für Schritt:

```powershell
.\scripts\check_env.ps1                 # nur prüfen, nichts installieren (PASS / WARN / FAIL + Lösungsbefehle)
.\scripts\check_env.ps1 -ReportPath env_report.json
.\scripts\setup_env.ps1                 # einrichten (GPU-Erkennung)
.\scripts\setup_env.ps1 -Cuda cpu       # CPU-Version erzwingen
.\scripts\setup_env.ps1 -Cuda cu130 -TorchVersion 2.14.0 -Recreate
.\scripts\run.ps1 -Task demo            # kleine Trainingsdemos der vier Netze
.\scripts\run.ps1 -Task demo -Model dit -Steps 300
.\scripts\run.ps1 -Task test            # Tests ausführen (pytest)
.\scripts\run.ps1 -Task list            # registrierte Modelle
.\scripts\run.ps1 -Task info            # PyTorch-/GPU-Status
```

### Was `check_env.ps1` prüft
64-Bit-System, PowerShell ≥ 5.1, Python 3.10–3.14 (bevorzugt den `py`-Launcher), pip, venv, Git, NVIDIA-GPU und die vom Treiber unterstützte CUDA-Version, Arbeitsspeicher, Festplatte, lange Windows-Pfade, Ausführungsrichtlinie und ob ein installiertes PyTorch die GPU nutzen kann. Zu jedem Punkt, der nicht PASS ist, gibt es einen Lösungsbefehl.

### Automatische Wahl der PyTorch-Quelle
`setup_env.ps1` liest die CUDA-Version des Treibers aus `nvidia-smi` und versucht der Reihe nach:

| Treiber-CUDA | Reihenfolge |
|---|---|
| ≥ 13.2 | cu132 → cu130 → cu126 → cpu |
| 13.0–13.1 | cu130 → cu126 → cpu |
| 12.6–12.9 | cu126 → cpu (PyTorch 2.14 ist die letzte Version mit CUDA 12.x) |
| Keine NVIDIA / älter | cpu |

**PyTorch schon installiert?** Das Skript findet die vorhandene Installation (z. B. global per `pip` installiertes 2.14.1) und legt `.venv` mit `--system-site-packages` an, um sie wiederzuverwenden – kein erneuter Download. Ist es eine CPU-Version und Sie haben eine NVIDIA-GPU, weist das Skript darauf hin; mit `-FreshTorch -Recreate` lässt sich eine CUDA-Version separat installieren.

Nach der Installation wird `torch.cuda.is_available()` tatsächlich aufgerufen; schlägt eine Quelle fehl, wird die nächste versucht. Alle Ausgaben landen in `deploy.log`. Unter macOS / Linux laufen dieselben Skripte mit PowerShell 7 (`pwsh`).

## Visuelle Trainingsoberfläche

```powershell
.\scripts\run.ps1 -Task ui        # öffnet http://127.0.0.1:8765 (fertig gebaute Oberfläche enthalten, kein Node.js nötig)
```

- **Netzgraph**: Die Standardaufgabe **MLP · Spiral-Klassifikation** zeichnet jedes Neuron und jedes Gewicht (orange = positiv, blau = negativ, Dicke = |w|); die Knotenfarbe ist die Aktivierung für das aktuelle Beispiel; türkise Impulse = Vorwärtsdurchlauf, magentafarbene Impulse = Rückwärtsdurchlauf (entlang der größten Gradienten). ViT / Transformer / U-Net / DiT zeigen einen **Schichtfluss**: eine Spalte pro Schicht in echter Ausführungsreihenfolge; Hover zeigt Form und Parameter, Klick springt zum Code.
- **Eine Trainingsiteration komplett**: ① Daten → ② Vorwärtsdurchlauf → ③ Verlust → ④ Backpropagation → ⑤ Gewichtsaktualisierung, jeweils mit Bedeutung, Formel (Kreuzentropie / MSE / Kettenregel / AdamW) und den zugehörigen Codezeilen.
- **Links**: Live-Kurven für Loss / Genauigkeit / Gradientennorm, Iterationen pro Sekunde und eine Lernvorschau (Klassifikationsergebnisse, Sequenzumkehr, Diffusion: Original → verrauscht → wiederhergestellt).
- **Rechts**: Mit **● Schritt erklären** zeichnet das Backend die nächste Iteration Zeile für Zeile auf und spielt sie in Zeitlupe ab: Das Code-Panel hebt die aktuelle Zeile samt Kommentar hervor, darunter erscheinen die von dieser Zeile erzeugten Tensoren (Form, Mittelwert ± Standardabweichung, Bereich, Verteilung) und der Aufrufstapel. Leertaste = Abspielen / Pause, ← → = Einzelschritt.
- **Warum nichts ruckelt**: Das Training läuft in einem Hintergrund-Thread; Metriken werden alle 100 ms gebündelt (≤ 10 Sendungen/s, automatisch ausgedünnt); Diagramme werden per Canvas gezeichnet; die Zeilenverfolgung ist nur für den erklärten Schritt aktiv.
- **Sprache und Layout**: Oben rechts mit „中 / EN“ zwischen Chinesisch und Englisch wechseln (Backend-Meldungen, Protokoll und Aufgabenbeschreibungen werden mitübersetzt; Code-Kommentare bleiben chinesisch). Die Trenner zwischen den beiden Spalten sowie zwischen Code und unterem Bereich lassen sich ziehen; jede Karte der linken Spalte lässt sich am unteren Rand in der Höhe ändern (Doppelklick setzt zurück). **Gedächtnis** und **Zeilenerklärung** können als Reiter, **nebeneinander**, als **schwebende**, verschieb- und skalierbare Fenster oder **abgelöst** in einem eigenen Browserfenster (z. B. auf einem zweiten Bildschirm) angezeigt werden. Das Layout wird gespeichert; ⟲ oben rechts stellt es zurück.
- **ONNX-Export**: Schaltfläche oben rechts, gespeichert unter `exports/<Aufgabe>.onnx` (benötigt `pip install -r requirements-export.txt`).

Frontend-Entwicklung (React + Vite, benötigt Node.js):

```powershell
.\scripts\run.ps1 -Task ui-dev     # Backend :8765 + Vite-Hot-Reload :5173
.\scripts\run.ps1 -Task build-ui   # web\dist neu bauen
```

## Eigene Daten importieren und analysieren

Oben auf **数据导入与分析 (Datenimport & Analyse)** umschalten. Lokalen Pfad eingeben, *Dateien hochladen* / *Ordner hochladen* klicken oder Dateien bzw. Ordner in das Feld ziehen:

| Daten | Formate | Automatische Verarbeitung |
|---|---|---|
| Tabellen | CSV / TSV / TXT / JSON / JSONL / Excel (.xlsx, ohne pandas) / .xls\* / Parquet\* | Spaltentypen werden erkannt: numerisch (fehlend → Mittelwert), kategorial (One-Hot), Text (gehashter Bag of Words), ID / konstant (verworfen); Zielspalte ist standardmäßig die letzte, wählbar |
| Bilder | Ein ganzer Ordner oder eine ZIP-Datei | Zentriert zugeschnitten und auf 16–64 px skaliert, Graustufen / Farbe erkannt → ViT |
| Audio | `.wav` (PCM / Float); `.flac .ogg .mp3` benötigen `pip install soundfile` | 56 akustische Merkmale → MLP; numerische Labels aus Metadaten ermöglichen Regression |
| Arrays | `.npz` / `.npy`: `X`+`y`, `x_train/y_train/x_test/y_test`, `X.npy`+`y.npy` in einem Ordner oder ein einzelnes 2-D-Array (letzte Spalte = Ziel) | 2-D → MLP; bildförmig → ViT |

\* .xls / Parquet benötigen `pip install -r requirements-data.txt`.

**Woher die Labels für Bilder / Audio kommen** (automatisch in dieser Reihenfolge erkannt und in der Oberfläche angezeigt):

1. **Metadatentabelle**: eine beliebige CSV / Excel / JSON im Ordner mit einer Dateinamen-Spalte (`img001.jpg`, `images/img001.jpg` oder ohne Endung) und Label-Spalten, z. B. `labels.csv`: `filename,breed,weight`. Die Label-Spalte lässt sich in der Oberfläche umschalten.
2. **Klassen-Unterordner**: `daten\katze\*.jpg`, `daten\hund\*.jpg`; auch `train\ val\ test\` mit Klassen darin.
3. **Dateinamen-Präfix**: `cat_001.jpg`, `dog.12.jpg` → `cat` / `dog`.

Nacheinander hochgeladene Ordner werden zum *aktuellen Datensatz* zusammengeführt: erst den cat-Ordner, dann den dog-Ordner hochladen = 2 Klassen. Uploads liegen unter `data/uploads/<Stapel>/`.

**Analyse**: Anzahl der Beispiele und Trainings-/Validierungsaufteilung, Zielverteilung, Warnung bei Klassenungleichgewicht, je Spalte Typ / fehlende Werte / Statistik / Verteilung, Zusammenhang zwischen Merkmalen und Ziel (Korrelationsverhältnis η² bei Klassifikation, |Pearson r| bei Regression), 2-D-PCA-Projektion, Beispielvorschau.

**Training**: *Mit diesen Daten trainieren* erzeugt automatisch eine *benutzerdefinierte* Aufgabe (Vektoren → MLP, Bilder → ViT; Kreuzentropie für Klassifikation, MSE für Regression). Netzgraph, Schritterklärung und die fünf Stufen funktionieren wie gewohnt.

**Modellbewertung**: Genauigkeit / R²·MAE·RMSE, Konfusionsmatrix, Precision / Recall / F1 je Klasse, Permutations-Merkmalswichtigkeit und die gröbsten Fehler.

> Meldet die Seite ein veraltetes Backend: das PowerShell-Fenster mit dem Backend schließen (oder Strg+C), `.\scripts\run.ps1 -Task ui` erneut starten und die Seite neu laden.

## Maschinelles Gedächtnis (Neural memory store)

Unten rechts auf der Trainingsseite: **🧠 机器记忆库 (Gedächtnis)**. Alle N Schritte (Standard: 1/40 des Laufs) wird der aktuelle Zustand des Netzes in eine lokale Datenbank geschrieben: `data/memory/neuro_memory.db` (SQLite, in Python enthalten). Gewichte liegen in `data/memory/ckpt/<Lauf>/latest.pt` und `best.pt`.

| Tabelle | Inhalt |
|---|---|
| `runs` | Eine Zeile pro Trainingslauf: Aufgabe, Datensatz, Hyperparameter, von welchem Snapshot fortgesetzt, bestes Validierungsergebnis |
| `snapshots` | Pro Snapshot: Loss, Trainings-/Validierungsgenauigkeit, Gradientennorm, Lernrate, gesamte Gewichtsänderung, Checkpoint |
| `layer_states` | Pro Schicht: Gewichtsnorm, Gradientennorm, Änderung seit dem vorigen Snapshot |
| `samples` | Metadaten der Beispiele: Quelldatei, Label, Training / Validierung |
| `embeddings` | Merkmalsvektor je Beispiel (vorletzte Schicht, float32) + Vorhersage + Konfidenz + richtig oder falsch |
| `sample_memory` | Ein über alle Läufe gesammeltes „Fehlerheft“: wie oft bewertet, wie oft falsch, geglätteter Verlust |

**Wie der nächste Lauf dieses Gedächtnis nutzt** (Optionen im Panel):

- **Start beim besten Gedächtnis / einem gewählten Snapshot**: lädt Gewichte + AdamW-Momente + Schrittzahl und lernt weiter, statt bei null zu beginnen;
- **Wiederholung schwerer Beispiele**: Trainingsbeispiele werden nach dem Fehlerheft gewichtet gezogen – frühere Fehler werden öfter geübt;
- **Bildaugmentierung**: zufälliges Spiegeln + Verschieben verringert Auswendiglernen bei kleinen Datensätzen (100 % Training bei niedriger Validierung ist genau dieses Problem);
- Der Snapshot mit der **höchsten Validierungsgenauigkeit** wird als *best* behalten, damit man nach Überanpassung zurückgehen kann.

Das Panel zeigt Wachstumskurven (Läufe im Vergleich + Herkunftskette des Gedächtnisses), den Merkmalsraum (PCA-Projektion je Snapshot + Trennungsmaß + Rohdatensätze), eine Heatmap der Änderungen je Schicht, das Fehlerheft, das Tabellenschema und eine schreibgeschützte SQL-Konsole.

## Online-Bereitstellung

Die Oberfläche läuft auf **Vercel**; das PyTorch-Trainings-Backend läuft auf **Ihrem eigenen Rechner** und wird über einen kostenlosen Cloudflare-Tunnel mit der Seite verbunden (`.\scripts\serve_public.ps1 -Publish`). Details in [DEPLOY.de.md](DEPLOY.de.md).

```powershell
.\scripts\publish_github.ps1 -User ihr-github-name     # pusht nach github.com/ihr-github-name/Visual-Network-Neuron
```

## Verwendung in Python

```python
import torch, neurocore as nc
from neurocore.diffusion import GaussianDiffusion

vit = nc.build_model("vit_small", img_size=224, num_classes=10)
logits = vit(torch.randn(2, 3, 224, 224))

seq2seq = nc.build_model("transformer", src_vocab_size=8000, tgt_vocab_size=8000)

diffusion = GaussianDiffusion(timesteps=1000, schedule="cosine")
dit = nc.build_model("dit_s_2", img_size=32, in_channels=4, num_classes=1000)
loss = diffusion.training_loss(dit, torch.randn(8, 4, 32, 32), torch.randint(0, 1000, (8,)))
samples = diffusion.sample(dit, (4, 4, 32, 32), y=torch.tensor([1, 2, 3, 4]), cfg_scale=4.0)

unet = nc.build_model("unet", in_channels=3, base_channels=64)      # Diffusions-Entrauscher
seg = nc.build_model("unet_seg", in_channels=3, num_classes=21)     # Segmentierung
```

Registrierte Modelle: `transformer`, `transformer_encoder`, `vit`, `vit_tiny/small/base`, `unet`, `unet_seg`, `dit`, `dit_s_2/b_2/xl_2`, `mlp`.

Attention ist zum Lernen explizit als Matrixrechnung ausgeschrieben; mit `use_sdpa=True` wird der fusionierte PyTorch-Kernel verwendet (die Tests prüfen, dass beide dasselbe Ergebnis liefern).

## Einen neuen Bereich hinzufügen

1. `neurocore/extensions/_template.py` kopieren, z. B. nach `gnn.py` (ohne Unterstrich).
2. Mit `@register_model("my_net", domain="graph")` registrieren.
3. Wird bei `import neurocore` automatisch geladen; Verwendung mit `nc.build_model("my_net")`. Eine fehlerhafte Erweiterung erzeugt nur eine Warnung und beeinträchtigt den Kern nicht.

Reservierte Bereichs-Tags: `nlp, vision, generative, graph, audio, multimodal, timeseries, recsys, security, rl, basic, other`.

## Verzeichnisstruktur

```
NeuroCore/
├─ deploy.ps1                  ein Befehl: prüfen → einrichten → Demo
├─ DEPLOY.md                   Online-Bereitstellung (Vercel + lokaler Backend-Tunnel)
├─ Dockerfile · vercel.json    Cloud-Konfiguration
├─ scripts/
│  ├─ common.ps1               gemeinsame Funktionen (Python finden, GPU lesen, Quelle wählen)
│  ├─ check_env.ps1            Umgebungsprüfung
│  ├─ setup_env.ps1            Einrichtung (.venv + PyTorch + Abhängigkeiten + Tests)
│  ├─ run.ps1                  demo / test / list / info / ui / build-ui
│  ├─ publish_github.ps1       mit einem Befehl auf GitHub pushen
│  └─ vercel_build.mjs         Vercel-Build-Schritt
├─ neurocore/
│  ├─ registry.py              Modell-Registry
│  ├─ layers/                  Attention, Embeddings
│  ├─ models/                  transformer / vit / unet / dit / mlp
│  ├─ diffusion/               DDPM
│  ├─ export.py                ONNX-Export
│  └─ extensions/              künftige Bereichsmodule (automatisch erkannt)
├─ server/                     Visualisierungs-Backend (Starlette + SSE)
│  ├─ tasks.py                 Trainingsaufgaben (zeilenweise kommentiert, in der Oberfläche verfolgt)
│  ├─ trainer.py               Trainings-Thread, Pause / Fortsetzen / Verfolgung / Gedächtnis
│  ├─ tracer.py                Zeilenverfolgung mit sys.settrace
│  ├─ graph.py · pipeline.py   Netzgraph und fünfstufiger Ablauf
│  ├─ datasets.py · evaluate.py Datenimport & Analyse, Modellbewertung
│  ├─ memory.py                maschinelles Gedächtnis (SQLite)
│  ├─ config.py                Cloud-Einstellungen (Passwort, CORS, Datenverzeichnis)
│  ├─ hub.py                   gedrosselte Übertragung (≤ 10/s)
│  └─ app.py                   HTTP-API + Oberfläche
├─ web/                        React-Oberfläche (web/dist ist das Build-Ergebnis)
├─ tests/                      Tests für Modelle / Backend / Daten / Gedächtnis
└─ demo.py
```
