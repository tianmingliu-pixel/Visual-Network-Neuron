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

Vercel kann nur die Webseite ausliefern, kein PyTorch, und Docker-Spaces bei Hugging Face sind inzwischen kostenpflichtig. Das Trainings-Backend läuft deshalb auf Ihrem eigenen Rechner; ein kostenloser Cloudflare-Tunnel gibt ihm eine öffentliche https-Adresse – ohne Kosten, ohne Kreditkarte und mit Ihrer eigenen GPU.

---

## ① Auf GitHub pushen (bei jeder Aktualisierung)

```powershell
cd D:\网络神经测试\核心\NeuroCore
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
5. öffnet `ihre-seite/?api=tunnel-adresse`; unter **后端 (Backend)** oben rechts das Passwort eingeben, um zu trainieren und hochzuladen.

Schließen des PowerShell-Fensters (oder Strg+C) beendet Backend und Tunnel. Die kostenlose Tunneladresse ändert sich bei jedem Start – einfach jedes Mal `-Publish` verwenden.

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
