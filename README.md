# Datenanalyse auf Basis von KI-Methoden – Wintersemester 2022/2023

## Projektübersicht
Dieses Repository enthält das finale Projekt für den Kurs "Datenanalyse auf Basis von KI-Methoden" im Wintersemester 2022/2023. Der Fokus liegt auf einem einfachen Chatbot zur Beantwortung von Kundenanfragen (Deutsch).

## Technologien
- Python, Flask
- TensorFlow (CPU), TFLearn, NLTK, NumPy

## Projektstruktur
- `chatbot/`: Anwendungscode, Modellartefakte, Templates und Static-Files
- `render.yaml`: Render-Deployment-Konfiguration
- `requirements.txt`: Python-Abhängigkeiten
- `runtime.txt`: Python-Version für Render

## Lokal ausführen
1. Abhängigkeiten installieren:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python -m nltk.downloader punkt stopwords -d chatbot/nltk_data
   ```
2. App starten:
   ```bash
   python chatbot/app.py
   ```
3. Browser öffnen: `http://localhost:5000`

## Modell neu trainieren (optional)
Wenn du `chat.json` änderst, kannst du das Modell neu trainieren:
```bash
./train.sh
```
Das erzeugt neue Dateien `chatbot/model.tflearn*` und `chatbot/trained_data`.
Das Training nutzt einen Validation-Split und Early-Stopping, damit es frueher stoppt, wenn keine Verbesserung mehr erreicht wird.

## Deployment auf Render
Empfohlen: Deployment über `render.yaml`.

1. Render Web Service erstellen und Repo verbinden.
2. Render erkennt `render.yaml` automatisch und nutzt:
   - Build Command: `pip install -r requirements.txt && python -m nltk.downloader punkt stopwords -d chatbot/nltk_data`
   - Start Command: `gunicorn chatbot.wsgi:app --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 120 --access-logfile - --error-logfile -`
3. Deploy starten.

Falls du manuell konfigurierst, verwende exakt die Build- und Start-Commands aus `render.yaml`.

## Hinweise
- NLTK-Daten werden beim Build nach `chatbot/nltk_data` geladen.
- Render nutzt Python 3.10.13 über `.python-version` bzw. `PYTHON_VERSION` in `render.yaml`.
- `tflearn` benötigt aktuell `pillow<10`, daher ist `pillow==9.5.0` gepinnt.
- Für stabile Inferenz läuft Gunicorn mit `--threads 1` und Modellvorhersage ist im Code serialisiert.
- Falls die Installation von TensorFlow/TFLearn fehlschlägt, prüfe die Python-Version und die Versions-Pins in `requirements.txt`.
- Bedienungshilfe im Chat: `/hilfe`, `/modus kurz`, `/modus normal`, `/modus ausfuehrlich`.

## Autor
AFL et al.
