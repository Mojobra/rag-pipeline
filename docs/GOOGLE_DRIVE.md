# Google Drive Folder Import

`index-google-drive` scans one Google Drive folder recursively and imports
supported files and native Google Docs into embedded Qdrant. It is a **manual,
read-only import** for a trusted local environment, not continuous sync or
permission-aware retrieval.

## Before You Start

- Enable the Google Drive API for the Google Cloud project supplying credentials.
- Configure [Application Default Credentials](https://docs.cloud.google.com/docs/authentication/application-default-credentials)
  with the `https://www.googleapis.com/auth/drive.readonly` scope. The identity
  must have access to the chosen folder and its files. A service account needs
  the folder shared with it; local user ADC for Drive needs an OAuth client and
  explicit Drive scope, not only the default Cloud Platform scope.
- Store credential files outside the repository. Long-lived service-account keys
  are a security risk; prefer managed or federated credentials where available.
- Install locked dependencies with `uv sync --locked`. First local model use may
  download model artifacts.
- Restrict the local collection to its intended readership. The default
  `google_drive_documents` collection shares the `.rag_data/qdrant` directory
  with other collections unless `--store-path` is changed.

Google references: [list and search files](https://developers.google.com/workspace/drive/api/reference/rest/v3/files/list),
[download or export content](https://developers.google.com/workspace/drive/api/guides/manage-downloads),
[Google Docs export formats](https://developers.google.com/workspace/drive/api/guides/ref-export-formats),
and [Drive scopes](https://developers.google.com/workspace/drive/api/guides/api-specific-auth).

## Configure And Run

With an ADC credential file stored outside the checkout, run in PowerShell:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = 'C:\private\drive-reader.json'
uv run rag-pipeline index-google-drive '<folder-id>' --acknowledge-no-acl
```

`GOOGLE_APPLICATION_CREDENTIALS` is optional when another ADC source is
configured. The CLI does not load `.env` automatically. The folder ID selects
the traversal root; the authenticated identity determines which items are
actually visible. Check `uv run rag-pipeline index-google-drive --help` for
indexing and limit options. Defaults permit 100 supported files, 20 MiB per
file, and 100 MiB of total downloaded content. Increasing these values costs
API calls, temporary disk, and RAM.

The importer accepts `.txt`, `.md`, `.markdown`, `.html`, `.htm`, `.pdf`, and
`.docx` blobs. Native Google Docs are exported as `text/plain`, losing rich
formatting. Sheets, Slides, shortcuts, and other file types are skipped. The
Google Docs export endpoint has a 10 MB provider limit; the command also
enforces its lower configured byte cap. Files whose `canDownload` capability
is false fail the import before indexing. Redirects are not followed.

Query with the same collection, embedding model/revision, search mode, and
store path used at import:

```powershell
uv run rag-pipeline retrieve 'What does the policy require?' `
  --collection-name google_drive_documents
```

## Data And Security Contract

- `gdrive://<file-id>` is the stable source key for chunks and citations;
  original name, root folder ID, and MIME type are metadata. Renaming or moving
  a file does not change its source key.
- Listing, bounded downloading, and extraction finish before model inference
  or Qdrant writes. Reimport replaces listed sources only; files missing from a
  later listing **remain indexed**. Drive pages are not a point-in-time snapshot
  and Qdrant updates are not atomic. Repeat after an interrupted write.
- HTTP calls target fixed Drive v3 URLs with a read-only bearer token. No
  redirect is followed or logged; errors omit response bodies and credentials.
  The token and downloaded plaintext still exist in process memory. Temporary
  extraction files are removed after each file.
- **Drive ACLs are not copied or enforced at retrieval time.** The required
  `--acknowledge-no-acl` flag acknowledges this gap, not fixes it. Do not expose
  the collection to users who should not see the imported content. The
  `drive.readonly` scope is broad and may require verification or a security
  assessment for a distributed app.
- There is no deletion reconciliation, incremental checkpoint, scheduled
  refresh, permission sync, retry/backoff, or live Google-account test in the
  default suite. Unsupported files and inaccessible folders are not indexed.

See [the architecture](../ARCHITECTURE.md) for component boundaries and the
[API reference](API_REFERENCE.md) for the separate local HTTP security model.
