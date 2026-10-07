# Online-Bereitstellung: GitHub → Vercel (Oberfläche) + Hugging Face Spaces (Trainings-Backend)

[中文](DEPLOY.md) | [English](DEPLOY.en.md) | [한국어](DEPLOY.ko.md) | **Deutsch**

```
Browser ──► https://visual-network-neuron.vercel.app           Oberfläche (Vercel, statische Dateien)
              │  /api/… (CORS + Zugangspasswort)
              ▼
        https://<hf-name>-visual-network-neuron.hf.space        PyTorch-Trainings-Backend (Docker, kostenlose CPU)
```

Vercel kann nur die Webseite ausliefern, aber kein PyTorch ausführen (zu groß, keine lang laufenden Prozesse, keine Festplatte). Das Trainings-Backend läuft deshalb in einem kostenlosen Docker-Space bei Hugging Face (2 CPU-Kerne, 16 GB RAM). Nach jedem Push auf GitHub werden **beide Seiten automatisch neu bereitgestellt**.

---

## ① Auf GitHub pushen (bei jeder Aktualisierung)

```powershell
cd D:\网络神经测试\核心\NeuroCore
.\scripts\publish_github.ps1 -User tianmingliu-pixel
```

- Pusht nach `https://github.com/tianmingliu-pixel/Visual-Network-Neuron`; beim ersten Mal öffnet sich ein GitHub-Anmeldefenster.
- Das Skript kopiert `deploy/deploy-backend-hf.yml` nach `.github/workflows/` (für die automatische Backend-Bereitstellung).
- Bereits vorhandene Inhalte im Repository (z. B. eine README) werden zusammengeführt. `data\` (Uploads, Gedächtnis), `exports\` und `.venv\` werden nie hochgeladen.
- Wenn Sie Frontend-Code (`web\src`) geändert haben, zuerst `.\scripts\run.ps1 -Task build-ui` ausführen (benötigt Node.js); Vercel liefert das eingecheckte `web\dist` aus.

## ② Backend: Hugging Face Space (einmalig einrichten)

1. Bei <https://huggingface.co> registrieren / anmelden → Avatar → **New Space**:
   Name `visual-network-neuron`, **SDK: Docker → Blank**, Hardware **CPU basic (kostenlos)**, öffentlich oder privat.
2. Space → **Settings → Variables and secrets**:
   - **New secret** `NEUROCORE_TOKEN` = ein selbst gewähltes Passwort (nötig zum Trainieren / Hochladen / Löschen des Gedächtnisses; Besucher ohne Passwort können nur zusehen)
   - (optional) **New variable** `NEUROCORE_ALLOWED_ORIGINS` = Ihre Vercel-Adresse aus Schritt ③, z. B. `https://visual-network-neuron.vercel.app`
3. Avatar → **Settings → Access Tokens → Create new token**, Typ **Write**, kopieren.
4. GitHub-Repository → **Settings → Secrets and variables → Actions**:
   - Reiter **Secrets**: New repository secret `HF_TOKEN` = das Token aus dem vorigen Schritt
   - Reiter **Variables**: New repository variable `HF_SPACE` = `ihr-hf-name/visual-network-neuron`
5. GitHub-Repository → **Actions → Deploy backend to Hugging Face Space → Run workflow**.
   Der Space wird gebaut (beim ersten Mal etwa 5–10 Minuten, PyTorch wird heruntergeladen). Danach
   `https://ihr-hf-name-visual-network-neuron.hf.space/api/version` öffnen; `{"version": 4, "cloud": true, …}` bedeutet: es funktioniert.

Ab jetzt wird das Backend bei jedem Push auf GitHub automatisch synchronisiert und neu gebaut.

## ③ Oberfläche: Vercel (einmalig einrichten)

1. Bei <https://vercel.com> mit dem GitHub-Konto anmelden → **Add New… → Project** → `Visual-Network-Neuron` wählen → **Import**.
2. Framework Preset: **Other**, alles andere unverändert lassen (`vercel.json` im Repository installiert nichts und veröffentlicht `web/dist`).
3. **Environment Variables** aufklappen und hinzufügen:
   `NEUROCORE_API_BASE` = `https://ihr-hf-name-visual-network-neuron.hf.space`
4. **Deploy**. Sie erhalten eine Adresse wie `https://visual-network-neuron.vercel.app`.
5. Öffnen → oben rechts **后端 (Backend)** → Passwort eingeben → speichern. Erscheinen „Cloud ☁“ und „Passwort korrekt“, können Sie trainieren und Daten hochladen.

Ab jetzt veröffentlicht Vercel die Seite bei jedem Push auf GitHub automatisch neu.

## Dieselbe Seite kann auch Ihren eigenen Rechner nutzen

Unter **后端 (Backend)** „本机 (lokal) http://127.0.0.1:8765“ wählen und speichern; die Seite nutzt dann das Backend auf Ihrem Rechner (vorher `.\scripts\run.ps1 -Task ui` starten): Training mit eigener CPU / GPU, die Daten verlassen den Rechner nicht. „Standard wiederherstellen“ wechselt zurück zur Cloud.

## Grenzen der kostenlosen Cloud

| Grenze | Erläuterung |
|---|---|
| Geschwindigkeit | Kostenlose CPU, keine GPU. Für die Demos und einige hundert kleine Bilder ausreichend; für große Datensätze das lokale Backend verwenden |
| Ruhezustand | Nach 48 Stunden ohne Besucher schläft der Space; der nächste Aufruf braucht etwa eine Minute zum Aufwachen |
| Keine Persistenz | Uploads und Gedächtnis werden beim Neustart des Space gelöscht. Zum Behalten in den Space-Einstellungen Persistent storage kaufen und die Variable `NEUROCORE_DATA_DIR=/data` setzen |
| Gemeinsame Sitzung | Alle Besucher sehen denselben Trainingslauf; das Passwort schützt schreibende Aktionen, ohne Passwort nur Zuschauen |
| Sicherheit | Im Cloud-Modus werden nur auf den Server hochgeladene Daten analysiert, keine beliebigen Serverpfade; Upload-Grenze 200 MB pro Datei (`NEUROCORE_MAX_UPLOAD_MB`) |

## Zugehörige Dateien

| Datei | Zweck |
|---|---|
| `scripts/publish_github.ps1` | Mit einem Befehl committen und auf GitHub pushen |
| `vercel.json`, `scripts/vercel_build.mjs` | Vercel veröffentlicht `web/dist` und schreibt die Backend-Adresse in `config.js` |
| `Dockerfile`, `.dockerignore` | Backend-Image (CPU-PyTorch, Port 7860) |
| `deploy/deploy-backend-hf.yml` | Vorlage für den GitHub-Actions-Workflow; wird vor dem Push nach `.github/workflows/` kopiert und synchronisiert danach bei jedem Push den Space |
| `server/config.py` | Cloud-Einstellungen: Passwort, CORS, Datenverzeichnis, Upload-Grenze |
