# RAG Pipeline: Gesprächsnotiz

## In 30 Sekunden

„Ich habe eine lokal ausführbare RAG-Pipeline für Fragen zu
Unternehmensdokumenten aufgebaut. Sie extrahiert und indexiert Dokumente,
sucht passende Textstellen und erzeugt eine Antwort mit nachvollziehbaren
Quellenangaben. Mir war wichtig, nicht nur eine Demo zu bauen: Die Pipeline
prüft Modell- und Collection-Kompatibilität, erhält die Herkunft der Belege
und lässt sich mit reproduzierbaren Datensätzen und Offline-Tests bewerten.“

## So funktioniert sie

1. **Indexieren:** PDF, DOCX, Markdown, HTML oder Text werden zu LangChain-
   Dokumenten mit Quellenmetadaten; PDFs behalten die Seiteninformation.
   Standardmäßig entstehen überlappende, rekursiv gesplittete Chunks.
2. **Speichern:** Ein lokales Embedding-Modell wandelt Chunks in Vektoren um.
   Embedded Qdrant speichert Vektoren und Metadaten. Stabile Punkt-IDs und
   quellenbezogener Ersatz halten wiederholte Indexläufe nachvollziehbar.
3. **Suchen:** Die Frage wird mit demselben Embedding-Vertrag kodiert. Standard
   ist dichte Suche; optional kommen BM25-Hybridsuche mit Reciprocal Rank
   Fusion, Metadatenfilter und Cross-Encoder-Reranking hinzu.
4. **Antworten:** Nur ausgewählte Belege innerhalb des Token-Budgets gelangen
   in den versionierten Prompt. Ohne nutzbare Belege verweigert das System
   deterministisch die Antwort. Zitate entstehen aus den Belegmetadaten,
   nicht aus frei vom Sprachmodell erzeugten Quellenangaben.

## Entscheidungen, die ich erklären kann

- **LangChain** liefert Dokument-, Embedding- und Prompt-Schnittstellen. Die
  Orchestrierung, Validierung und Auswertung bleiben explizit im eigenen Code,
  damit Fehler und Trade-offs sichtbar sind.
- **Embedded Qdrant** ermöglicht einen reproduzierbaren lokalen Prototyp ohne
  Serverbetrieb. Für parallele Nutzer, Backups und Hochverfügbarkeit wäre ein
  separat betriebener Vektordienst nötig.
- **CLI und FastAPI** nutzen dieselben Anwendungsabläufe. Die API ist bewusst
  lesend; Dokumentimporte und Indexierung bleiben CLI-Operationen. FastAPI
  liefert validierte Pydantic-Schemata und OpenAPI, aber keine Benutzeroberfläche.
- **Kleine lokale Modelle** machen den vollständigen Ablauf auf CPU testbar.
  Sie sind ein funktionaler Ausgangspunkt, kein Nachweis guter Antwortqualität
  für echte Unternehmensdaten.

## Qualität und Sicherheitsgrenze

- Versionierte synthetische Datensätze prüfen Retrieval (u. a. Hit@k, Recall
  und negative Fragen) und Antworten (u. a. Textübereinstimmung,
  Abstention und Zitierverhalten). Isolierte Benchmarks erfassen Konfiguration,
  Provenienz und Laufzeiten. Diese Metriken beweisen keine faktische Treue.
- Die Tests laufen deterministisch und ohne Modell-Downloads oder API-Keys;
  CI prüft Python 3.11 bis 3.13, Typen, Stil, Tests, Coverage und Paket-Build.
- HTTP-Anfragen benötigen einen Bearer-Token. Optional begrenzt eine private
  Policy Quellen **vor** der Top-k-Auswahl; Policy v2 bindet Tokens an getrennte
  lokale Tenant-Stores. Das ist keine echte Nutzeridentität, kein Live-ACL-Sync
  und keine Betriebssystem- oder Cluster-Isolation.

## Was ich ausdrücklich nicht als fertig darstelle

Die SharePoint-, Confluence-, Google-Drive-, Slack- und Teams-Importe sind
manuelle, begrenzte Snapshots ohne synchronisierte Zugriffsrechte oder
Löschungen. Es gibt keine produktive Bereitstellung, kein SSO/Keycloak,
keine Rate Limits, keinen Audit-Trail, keine Hosted-LLM-Anbindung und keine UI.
Für ein Unternehmen würde ich zuerst Datenquellen, Berechtigungen und
Erfolgskriterien klären; danach Identitätsprüfung und ACL-Sync, getrennten
Speicher, asynchrone Indexjobs, Betriebssicherheit und Evaluation mit echten,
freigegebenen Fragen und Belegen ergänzen.

## Typische Rückfragen

**Warum RAG statt Fine-Tuning?** Änderbare Dokumente können neu indexiert und
als Belege gezeigt werden, ohne Modellgewichte für jede Richtlinie anzupassen.
RAG garantiert trotzdem keine korrekte Antwort.

**Wie gehst du mit Halluzinationen um?** Belege werden begrenzt in den Prompt
gepackt; bei fehlender Evidenz wird nicht generiert. Zitate zeigen die
verwendeten Chunks, beweisen aber keine Aussage auf Satzebene.

**Wie hast du Codex genutzt?** Als Entwicklungspartner für die Implementierung;
Architektur, Anforderungen, Tests und Designentscheidungen habe ich selbst
festgelegt und generierten Code geprüft und mit Tests validiert.

## Kurze Demo

Aus dem Repository-Verzeichnis, mit Python 3.11+ und `uv`:

```powershell
uv sync --locked
uv run rag-pipeline index evaluation/datasets/asteria-policies-v2/documents --collection-name asteria_demo
uv run rag-pipeline answer "When must expense claims be submitted?" --collection-name asteria_demo
```

Die fünf Dokumente sind fiktiv; der erste Lauf kann lokale Modelle herunterladen.
Für Details: [README](../README.md), [Architektur](../ARCHITECTURE.md),
[Technisches Handbuch](../TECHNICAL_MANUAL.md) und
[API-Vertrag](API_REFERENCE.md).
