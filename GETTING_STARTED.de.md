# Einsteiger-Anleitung: NeuroCore auf dem eigenen Rechner ausführen

[中文](GETTING_STARTED.md) | [English](GETTING_STARTED.en.md) | [한국어](GETTING_STARTED.ko.md) | **Deutsch**

Diese Anleitung ist für **alle, die dieses Repository zum ersten Mal besuchen**. Der Reihe nach durchgearbeitet dauert sie etwa 15–30 Minuten (vor allem der PyTorch-Download). Danach trainieren Sie neuronale Netze auf Ihrem eigenen Rechner und sehen im Browser zu, wie sie lernen.

> Alle Pfade verwenden `C:\NeuroCore` als Beispiel. Ein anderer Ort geht auch, aber **ein kurzer Pfad mit lateinischen Buchstaben und ohne Leerzeichen** macht am wenigsten Probleme.

---

## Zuerst: wie Webseite und Backend zusammenhängen

```
Webseite (Oberfläche)  https://visual-network-neuron.vercel.app   oder   http://127.0.0.1:8765
     │  zeigt nur an: Knöpfe, Kurven, Netzwerkgraph
     ▼
Backend (Rechnen)  python -m server auf Ihrem Rechner (127.0.0.1:8765)
        trainiert die Netze, analysiert Daten, speichert den „Maschinen-Gedächtnisspeicher"
```

- **Die Webseite ist nur die Oberfläche**; sie rechnet selbst nichts.
- **Das Backend erledigt die Arbeit.** Diese Anleitung installiert es auf **Ihrem eigenen Rechner**.
- `127.0.0.1` bedeutet „dieser Rechner selbst"; aus dem Internet ist er nicht erreichbar.

---

## Schritt 0: Voraussetzungen (einmalig)

| Benötigt | Prüfen | Falls nicht vorhanden |
|---|---|---|
| Windows 10 / 11 | — | macOS / Linux: siehe „Kein Windows?" am Ende |
| Python 3.10 – 3.14 | PowerShell öffnen, `python --version` eingeben | von <https://www.python.org/downloads/> installieren und **„Add python.exe to PATH" anhaken**; oder `winget install Python.Python.3.12` |
| Git (optional) | `git --version` | nicht nötig – Schritt 2 geht auch per ZIP |
| NVIDIA-Grafikkarte (optional) | — | ohne läuft das Training auf der CPU, nur langsamer |

**PowerShell öffnen:** `Win`-Taste → `PowerShell` tippen → Enter.

---

## Schritt 1: Ordner anlegen

In PowerShell (nach jeder Zeile Enter):

```powershell
mkdir C:\NeuroCore
cd C:\NeuroCore
```

Sie sind jetzt in `C:\NeuroCore`. Alle weiteren Befehle **in diesem Ordner** ausführen.

---

## Schritt 2: Code in den Ordner herunterladen

**Variante A: Git (empfohlen, später leicht zu aktualisieren)**

```powershell
cd C:\
git clone https://github.com/tianmingliu-pixel/Visual-Network-Neuron.git NeuroCore
cd C:\NeuroCore
```

> Haben Sie in Schritt 1 schon ein leeres `C:\NeuroCore` angelegt, verwendet `git clone` es trotzdem (es muss nur leer sein).

**Variante B: ZIP herunterladen**

1. <https://github.com/tianmingliu-pixel/Visual-Network-Neuron> öffnen
2. Grünen Knopf **Code** → **Download ZIP**
3. Entpacken ergibt einen Ordner `Visual-Network-Neuron-main`
4. **Alles darin** nach `C:\NeuroCore` kopieren

**Prüfen:** `dir` ausführen; Sie sollten `README.md`, `scripts`, `server`, `web` usw. sehen. Sehen Sie nur einen Ordner `Visual-Network-Neuron-main`, ist eine Ebene zu viel – dessen Inhalt eine Ebene nach oben verschieben.

---

## Schritt 3: Skripte erlauben (in jedem neuen PowerShell-Fenster)

Windows blockiert `.ps1`-Skripte standardmäßig. Dieser Befehl gilt **nur für das aktuelle Fenster** und ändert keine Systemeinstellungen:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Bei ZIP-Download zusätzlich einmal ausführen (entfernt die Markierung „aus dem Internet heruntergeladen"):

```powershell
Get-ChildItem C:\NeuroCore -Recurse | Unblock-File
```

---

## Schritt 4: Umgebung installieren (einmalig)

```powershell
cd C:\NeuroCore
.\scripts\check_env.ps1      # zuerst prüfen: jeder Punkt zeigt PASS / WARN / FAIL, bei FAIL steht die Lösung dabei
.\scripts\setup_env.ps1      # dann installieren: legt C:\NeuroCore\.venv an und installiert das passende PyTorch für Ihre GPU
```

- PyTorch wird heruntergeladen (einige hundert MB bis über 2 GB) – bitte Geduld.
- Alles landet in `C:\NeuroCore\.venv` und **beeinflusst keine andere Python-Installation**.
- Besteht der abschließende Selbsttest, ist alles fertig.

CPU-Version erzwingen (z. B. bei Treiberproblemen): `.\scripts\setup_env.ps1 -Cuda cpu`

---

## Schritt 5: Backend starten (bei jeder Nutzung)

```powershell
cd C:\NeuroCore
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run.ps1 -Task ui
```

- `NeuroCore UI -> http://127.0.0.1:8765` bedeutet: das Backend läuft.
- Der Browser öffnet `http://127.0.0.1:8765` automatisch.
- **Dieses PowerShell-Fenster offen lassen** – schließen = Backend stoppt. Zum Beenden `Strg + C` drücken oder das Fenster schließen.

---

## Schritt 6: Seite öffnen und prüfen, dass Ihr Rechner verwendet wird

Beide Adressen funktionieren **gleich**:

| Adresse | Hinweis |
|---|---|
| `http://127.0.0.1:8765` | von Ihrem eigenen Backend ausgeliefert; öffnet sich in Schritt 5 automatisch; **funktioniert auch offline** |
| `https://visual-network-neuron.vercel.app` | die Online-Seite; sie **findet** das Backend auf Ihrem Rechner **automatisch** |

Nach dem Laden **zuerst oben rechts schauen**:

| Anzeige oben rechts | Bedeutung | Zu tun |
|---|---|---|
| `● 后端 · 本页面` (this page) | Ihr eigener Rechner (geöffnet über 127.0.0.1:8765) | ✅ nichts |
| `● 后端 · 本机` (local) | Ihr eigener Rechner (geöffnet über die Online-Seite) | ✅ nichts |
| `● 后端 · 网站后端 · 只读观看` (site backend · view only) und „👀 观看模式" | verbunden mit **dem Rechner des Autors**, nur zuschauen | ❌ Ihr Backend läuft nicht – zurück zu Schritt 5 |
| `○ … · 未连接` (disconnected) | kein Backend erreichbar | ❌ zurück zu Schritt 5, prüfen, ob das Fenster noch offen ist |

> Die Oberflächensprache stellen Sie oben rechts mit **中 / EN** um (derzeit Chinesisch / Englisch). In Klammern steht jeweils die englische Anzeige.

### Was die Optionen im Backend-Feld bedeuten (normalerweise nichts ändern)

Ein Klick auf **后端 (Backend)** oben rechts öffnet „后端连接 (Backend connection)":

| Option | Bedeutung | Wann verwenden |
|---|---|---|
| 后端地址 (Backend URL) | wo die Seite das Backend sucht | normalerweise **leer lassen**, die Automatik entscheidet |
| 本页面同源 (Same as this page) | der Rechner, der diese Seite ausgeliefert hat | wenn über `127.0.0.1:8765` geöffnet |
| 本机 (local) `http://127.0.0.1:8765` | **Ihr eigener Rechner** | wenn die Online-Seite Ihr Backend nicht erkannt hat, manuell anklicken |
| 网站默认后端 (Site default backend) | der Rechner des Autors (über einen Tunnel) | nur zum **Zuschauen** beim Training des Autors |
| 口令 (Access token) | Passwort für das Backend des Autors | **beim eigenen Rechner leer lassen – kein Passwort nötig** |
| 自动选择 (Auto-select) | löscht manuelle Einstellungen, zurück zur Automatik | wenn etwas durcheinandergeraten ist |
| 保存并重新连接 (Save & reconnect) | speichert die Einstellungen und lädt neu | nach jeder Änderung |

**Reihenfolge der Automatik:** ① eine manuell gesetzte Adresse → ② die eigene Adresse der Seite → ③ `127.0.0.1:8765` auf Ihrem Rechner → ④ der Rechner des Autors (nur ansehen). Solange das Fenster aus Schritt 5 offen ist, verwendet die Seite also automatisch Ihren Rechner.

---

## Schritt 7: Das erste Training

1. Oben den Reiter **训练 (Training)** wählen.
2. Im Auswahlfeld eine Aufgabe wählen; für den Einstieg: **MLP · Spiral-Klassifikation (Neuronenebene)**.
3. Schritte, Lernrate und Batchgröße auf den Standardwerten lassen; **▶ 开始训练 (Start training)** klicken.
4. Sie sehen, wie sich jedes Neuron und Gewicht im Netzwerkgraph ändert, der Verlust sinkt, die Genauigkeit steigt, „逐行讲解 (Line-by-line)" die laufende Codezeile hervorhebt und der „机器记忆库 (Memory store)" unten rechts Schnappschüsse aufzeichnet.

**Eigene Daten:** Reiter **数据 (Data)** → Ordner hochladen (z. B. Bilder in den Unterordnern `cat\` und `dog\` oder eine Tabelle mit Label-Spalte) → 加载并分析 (Load & analyse) → trainieren. Die Daten bleiben in `C:\NeuroCore\data\`.

---

## Warum den eigenen Rechner als Backend verwenden?

| | Eigener Rechner | Rechner des Autors (zuschauen) |
|---|---|---|
| Passwort | **nicht nötig** | Passwort des Autors, sonst nur zuschauen |
| Trainieren / Hochladen / Exportieren | ✅ alles | ❌ nur zuschauen |
| Wohin Ihre Daten gehen | **nur auf Ihren Rechner** (`C:\NeuroCore\data`) | auf einen fremden Rechner |
| Geschwindigkeit | Ihre CPU / GPU, kein Netzwerk dazwischen | abhängig vom Rechner und Netz des Autors |
| Verfügbarkeit | jederzeit, auch offline | nur solange der Autor Rechner und Tunnel laufen lässt |
| Maschinen-Gedächtnisspeicher | Ihr eigener; wächst mit jedem Lauf; fortsetzbar | nicht sichtbar |

---

## Bei jeder weiteren Nutzung: nur drei Zeilen

```powershell
cd C:\NeuroCore
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run.ps1 -Task ui
```

**Aktualisieren** (Git): `cd C:\NeuroCore` → `git pull` → neu starten. Bei ZIP erneut herunterladen und überschreiben (der Ordner `data\` enthält Ihre Daten und den Gedächtnisspeicher – nicht löschen).

---

## Fehlerbehebung

| Symptom | Lösung |
|---|---|
| `Die Datei …ps1 kann nicht geladen werden, da die Ausführung von Skripts auf diesem System deaktiviert ist` | Schritt 3 ausführen |
| `python` wird nicht erkannt | Python mit angehaktem „Add to PATH" neu installieren, dann PowerShell **neu öffnen** |
| `setup_env.ps1` bemängelt die Python-Version | 3.10–3.14 nötig: `winget install Python.Python.3.12` |
| Port 8765 belegt | früheres `run.ps1`-Fenster schließen; oder `.\scripts\run.ps1 -Task ui -Port 8766` und im Backend-Feld `http://127.0.0.1:8766` eintragen |
| Online-Seite zeigt weiter „view only" oder „disconnected" | prüfen, ob das Fenster aus Schritt 5 offen ist; im Backend-Feld „本机 (local)" → „保存并重新连接"; hilft das nicht, `http://127.0.0.1:8765` direkt öffnen (manche Browser, z. B. Safari, verhindern, dass Online-Seiten den eigenen Rechner erreichen) |
| `UnicodeEncodeError: 'charmap' codec` | Problem einer alten Version – auf den neuesten Code aktualisieren |
| „开始训练" reagiert nicht | roten Text unten im „日志 (Log)" lesen; steht dort „Passwort erforderlich", sind Sie mit dem Rechner des Autors verbunden – zurück zu Schritt 6 |

---

## Kein Windows? (macOS / Linux)

Die Skripte sind PowerShell; unter macOS / Linux die gleichen Schritte von Hand:

```bash
mkdir -p ~/NeuroCore && cd ~/NeuroCore
git clone https://github.com/tianmingliu-pixel/Visual-Network-Neuron.git .
python3 -m venv .venv
source .venv/bin/activate
pip install torch            # mit NVIDIA-GPU den Befehl von https://pytorch.org verwenden
pip install -r requirements.txt
python -m server --open      # startet das Backend und öffnet http://127.0.0.1:8765
```

Jedes weitere Mal: `cd ~/NeuroCore && source .venv/bin/activate && python -m server --open`

---

Mehr: Projektüberblick in [README.de.md](README.de.md); wie der Repository-Eigentümer seinen Rechner für Besucher der Seite freigibt, steht in [DEPLOY.de.md](DEPLOY.de.md) (für normale Nutzer nicht nötig).
