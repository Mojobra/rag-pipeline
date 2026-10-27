# SharePoint Drive Import

`index-sharepoint` reads one Microsoft Graph document-library drive and indexes
supported files into embedded Qdrant. It is a **manual, read-only import**, not a
continuous sync or a permission-aware enterprise connector. The command is
intended for a trusted local evaluation environment.

## Before You Start

- Register an Entra application for **app-only** access and grant the smallest
  suitable Microsoft Graph application permission. Broad `Files.Read.All` can
  read more than this command needs. `Sites.Selected` requires an additional
  site-specific grant; merely adding it to the app registration is insufficient.
- Obtain the target document-library **drive ID** and create a client secret.
  Keep the secret out of Git, terminal history, screenshots, and logs.
- Install the locked dependencies with `uv sync --locked`. Initial local model
  use may download model artifacts.
- Restrict access to the imported collection. Its default name is
  `sharepoint_documents` in the **same** `.rag_data/qdrant` database; use
  `--store-path` if a separate local database is needed.

Microsoft's references: [app-only token acquisition](https://learn.microsoft.com/en-us/entra/msal/python/getting-started/acquiring-tokens),
[selected permissions](https://learn.microsoft.com/en-us/graph/permissions-selected-overview),
[listing children](https://learn.microsoft.com/en-us/graph/api/driveitem-list-children?view=graph-rest-1.0),
and [file content downloads](https://learn.microsoft.com/en-us/graph/api/driveitem-get-content?view=graph-rest-1.0).

## Configure And Run

Set these variables in the **same PowerShell process** that runs the command:

```powershell
$env:RAG_SHAREPOINT_TENANT_ID = '<tenant-uuid>'
$env:RAG_SHAREPOINT_CLIENT_ID = '<application-client-uuid>'
$env:RAG_SHAREPOINT_CLIENT_SECRET = Read-Host 'Graph client secret' -MaskInput
uv run rag-pipeline index-sharepoint '<drive-id>' --acknowledge-no-acl
```

The three variables have no defaults. The CLI does **not** read `.env` files
automatically and does not accept a client secret as a flag. The masked input
example requires PowerShell 7. To remove the secret from the current PowerShell
process afterwards:

```powershell
Remove-Item Env:RAG_SHAREPOINT_CLIENT_SECRET
```

Check `uv run rag-pipeline index-sharepoint --help` before tuning limits or
index settings. Defaults allow at most 100 supported files, 20 MiB per file,
and 100 MiB downloaded in total. `--max-files`, `--max-file-mib`, and
`--max-total-mib` are positive integer limits; raising them increases network
work, temporary disk, and memory. The command scans folders recursively and
accepts `.txt`, `.md`, `.markdown`, `.html`, `.htm`, `.pdf`, and `.docx`.
Unsupported file types are skipped.

The command reuses the local indexer's chunking, embedding, dense/hybrid mode,
and Qdrant options. Set `--collection-name` and `--store-path` explicitly when
isolating multiple drives. Query with the **same** embedding model, revision,
search mode, collection, and store path as the import. For example:

```powershell
uv run rag-pipeline retrieve 'What does the policy require?' `
  --collection-name sharepoint_documents
```

## Data And Security Contract

- The source key is `sharepoint://<drive-id>/<item-id>`. It stays stable across
  renames; the original name and Graph `webUrl`, when provided, are metadata.
  Citations use the stable source key, not a live permission-checked link.
- The importer finishes listing, downloading, and extraction before embedding
  or Qdrant writes. It checks declared and actual byte limits. It stages each
  file in a temporary local directory for existing extractors, then removes it.
- Reimport replaces chunks for files still listed. It **does not remove** items
  absent from a later Graph listing. A failed write can leave a partial update;
  rerun the same command after resolving the failure.
- Graph access uses an application token. The content redirect is fetched
  without forwarding that token and is restricted to HTTPS SharePoint hosts.
- **No SharePoint ACLs are copied or enforced.** The default shared-token API
  and local CLI/files can expose indexed content regardless of SharePoint
  permissions. The optional API source policy is manually maintained, not a
  live SharePoint permission check. The `--acknowledge-no-acl` flag acknowledges
  this risk; it does not mitigate it. Do not import restricted documents into
  a collection exposed to users who should not read them. CLI metadata filters
  are not authorization.
- There is no delta token, webhook, scheduled refresh, remote deletion, Graph
  throttling retry, audit trail, or live-tenant test in the default suite.
  Errors stop the import with sanitized messages; repeat it manually after
  correcting credentials, permissions, limits, or provider availability.

See [the architecture](../ARCHITECTURE.md) for package boundaries and the
[API reference](API_REFERENCE.md) for the separate local HTTP security model.
