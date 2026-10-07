# Online-Bereitstellung: Vercel (Oberfläche) + eigener Rechner (Trainings-Backend über einen kostenlosen Cloudflare-Tunnel)

[中文](DEPLOY.md) | [English](DEPLOY.en.md) | [한국어](DEPLOY.ko.md) | **Deutsch**

```
Browser ──► https://visual-network-neuron.vercel.app        Oberfläche (Vercel, statische Dateien)
              │  /api/… (CORS + Zugangspasswort)
              ▼
        https://xxxx.trycloudflare.com                      kostenloser Cloudflare-Tunnel
              ▼
        python -m server auf Ihrem Rechner (127.0.0.1:8765)  PyTorch-Backend: Ihre CPU / GPU, die Daten bleiben bei Ihnen
```

> Dieses Dokument richtet sich an den **Repository-Eigentümer** (der seinen Rechner für Besucher der Seite freigibt). Wer NeuroCore nur auf dem eigenen Rechner nutzen will, liest [GETTING_STARTED.de.md](GETTING_STARTED.de.md). `C:\NeuroCore` unten durch den eigenen Projektordner ersetzen.

Vercel kann nur die Webseite ausliefern, kein PyTorch, und Docker-Spaces bei Hugging Face sind inzwischen kostenpflichtig. Das Trainings-Backend läuft deshalb auf Ihrem eigenen Rechner; ein kostenloser Cloudflare-Tunnel gibt ihm eine öffentliche https-Adresse – ohne Kosten, ohne Kreditkarte und mit Ihrer eigenen GPU.

---

## ① Auf GitHub pushen (bei jeder Aktualisierung)

```powershell
cd C:\NeuroCore
.\scripts\publish_github.ps1 -User tianmingliu-pixel
```

`data\` (Uploads, Gedächtnis, Passwort), `exports\` und `.venv\` werden nie hochgeladen. Wurde Frontend-Code (`web\src`) geändert, vorher `.\scripts\run.ps1 -Task build-ui` ausführen.

## ② Oberfläche: Vercel (einmalig)

1. Bei <https://vercel.com> mit GitHub anmelden → **Add New… → Project** → `Visual-Network-Neuron` → **Import**.
2. Framework Preset: **Other**, Rest unverändert (`vercel.json` installiert nichts und veröffentlicht `web/dist`) → **Deploy**.
3. Sie erhalten eine Adresse wie `https://visual-network-neuron.vercel.app`.

Jeder Push auf GitHub veröffentlicht die Seite neu.

## ③ Backend: vom eigenen Rechner freigeben (immer wenn Sie die Seite nutzen)

```powershell
.\scripts\serve_public.ps1 -Publish
```

Das Skript:

1. fragt beim ersten Start nach einem **Zugangspasswort** (gespeichert in `data\.public_token`, wird nie hochgeladen);
2. installiert Cloudflares `cloudflared` (winget);
3. startet das Backend im Hintergrund und öffnet einen Tunnel mit der Adresse `https://xxxx.trycloudflare.com` (in die Zwischenablage kopiert);
4. schreibt mit `-Publish` diese Adresse nach `deploy/backend_url.txt` und pusht sie – Vercel aktualisiert sich in etwa einer Minute, **andere Besucher verbinden sich automatisch**;
5. öffnet die Seite (feste Adresse `https://visual-network-neuron.vercel.app`). Auf Ihrem eigenen Rechner verbindet sich die Seite automatisch mit dem lokalen Backend; geben Sie das Passwort einmal unter **后端 (Backend)** oben rechts ein, um zu trainieren und hochzuladen (der Browser merkt es sich).

Schließen des PowerShell-Fensters (oder Strg+C) beendet Backend und Tunnel.

**Ändert sich die Adresse?** Die Seitenadresse `visual-network-neuron.vercel.app` ändert sich nie; die Tunneladresse `xxxx.trycloudflare.com` ist bei jedem Start zufällig, wird aber nur im Hintergrund verwendet – `-Publish` teilt der Seite die neue Adresse mit. Geben Sie anderen nur die Vercel-Adresse und öffnen Sie selbst ebenfalls diese (nicht die Tunneladresse: das Passwort wird pro Adresse gespeichert, bei einer neuen Adresse müssen Sie es erneut eingeben).

## Nach dem Öffnen der Seite: automatische Backend-Wahl

Die Seite ist nur die Oberfläche; Training und Datenanalyse laufen im Backend. Beim Laden sucht die Seite **automatisch** in dieser Reihenfolge:

| Situation | Verbindet mit | Anzeige oben rechts | Was geht |
|---|---|---|---|
| Besucher hat ein Backend auf dem eigenen Rechner laufen (`run.ps1 -Task ui`) | seinem Rechner `127.0.0.1:8765` | Backend · lokal | alles, mit eigener CPU/GPU; Daten bleiben auf seinem Rechner |
| Kein lokales Backend, Ihr Tunnel läuft | Ihrem Rechner (über den Tunnel) | Backend · Seiten-Backend · nur ansehen | nur zuschauen; Training / Upload nach Eingabe des Passworts |
| Keins von beiden | — | nicht verbunden | nur die Oberfläche, nichts läuft |

Manuell wählen: oben rechts **Backend** → Backend-Adresse eintragen → „Speichern & neu verbinden"; „Automatisch wählen" stellt die Automatik wieder her.

### Wo das Passwort eingegeben wird

1. Ganz oben rechts auf **Backend** klicken (rechts vom 中/EN-Umschalter) – das Feld „Backend-Verbindung" öffnet sich;
2. **Backend-Adresse** leer lassen (= automatisch / gleiche Adresse wie diese Seite);
3. ins Feld **Passwort** das Passwort eintragen, das Sie beim ersten Start von `serve_public.ps1` festgelegt haben (vergessen: `Get-Content data\.public_token`);
4. „Speichern & neu verbinden" klicken. Sobald „nur ansehen" verschwindet, können Sie trainieren.

> „Training starten" reagiert nicht und im Log steht „Passwort erforderlich" = das Passwort fehlt noch; es fehlen keine Dateien.

### Besucher, die auf dem eigenen Rechner trainieren wollen

Oben rechts erscheint „👀 Zuschauen · selbst trainieren?"; ein Klick zeigt die Schritte:

1. Code herunterladen: <https://github.com/tianmingliu-pixel/Visual-Network-Neuron>
2. Im Code-Ordner `.\scripts\setup_env.ps1` und dann `.\scripts\run.ps1 -Task ui` ausführen
3. Seite neu laden: sie wechselt automatisch zum Backend auf dem eigenen Rechner (kein Passwort nötig, die Daten gelangen nicht auf Ihren Rechner)

## Fehlerbehebung

| Symptom | Ursache / Lösung |
|---|---|
| `UnicodeEncodeError: 'charmap' codec can't encode` | Ältere Versionen schrieben chinesischen Text mit der Windows-Standardkodierung ins Log; behoben (Backend erzwingt UTF-8, Skripte setzen `PYTHONUTF8=1`). Neuesten Code holen |
| „Backend wurde nicht gestartet" | `data\backend.err.log` prüfen; ist Port 8765 belegt, zuerst andere `run.ps1 -Task ui`-Fenster schließen |
| Seite zeigt „nicht verbunden" | `serve_public.ps1` läuft nicht, oder Sie haben gerade mit `-Publish` gepusht und Vercel aktualisiert noch (ca. 1 Minute) |
| Trainingsknopf reagiert nicht | Passwort nicht eingegeben (siehe „Wo das Passwort eingegeben wird") |

## Sicherheit

| Wer | Darf |
|---|---|
| Sie (mit Passwort) | trainieren, Daten hochladen, Gedächtnis ansehen / löschen, lokale Pfade nutzen |
| Andere (ohne Passwort) | nur das Training ansehen (Kurven, Netzgraph, Schritterklärung); Ihre Daten, Uploads und das Gedächtnis bleiben verborgen |

Das Backend lauscht nur auf 127.0.0.1; von außen ist es nur über den Tunnel erreichbar, und ohne laufendes `serve_public.ps1` ist nichts freigegeben.

## Feste Adresse gewünscht?

- **Feste URL**: eigene Domain zu Cloudflare hinzufügen und einen Named Tunnel verwenden (`cloudflared tunnel create`).
- **Cloud-Server**: das `Dockerfile` im Repository läuft auf jedem Docker-Host (Hugging Face PRO, Google Cloud Run, kostenpflichtiger Render-Tarif …, mindestens 2 GB RAM); in Vercel `NEUROCORE_API_BASE` darauf setzen.

## Zugehörige Dateien

| Datei | Zweck |
|---|---|
| `scripts/publish_github.ps1` | Mit einem Befehl committen und auf GitHub pushen |
| `scripts/serve_public.ps1` | Backend + Cloudflare-Tunnel starten und der Seite die Adresse mitteilen |
| `vercel.json`, `scripts/vercel_build.mjs` | Vercel veröffentlicht `web/dist`; Backend-Adresse aus `NEUROCORE_API_BASE` oder `deploy/backend_url.txt` |
| `server/config.py` | Passwort, CORS, Lesesperre, Datenverzeichnis |
| `Dockerfile` | Für einen späteren Cloud-Server |
