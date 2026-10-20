# Confluence Cloud Space Import

`index-confluence` reads current pages from one Confluence Cloud space through
REST API v2 and indexes their visible text into embedded Qdrant. It is a
**manual, read-only import**, not a continuous sync or a permission-aware
connector. Run it only in a trusted local evaluation environment.

## Before You Start

- Obtain the numeric **space ID** and use an Atlassian account that can view
  the intended space and pages. The API only returns pages visible to that
  account; it does not transfer those visibility rules to Qdrant.
- Create an API token for that account. Atlassian documents email plus API
  token Basic Auth for manual scripts; a distributed cloud integration should
  use an app-based authentication design instead.
- Install dependencies with `uv sync --locked`. The first local embedding run
  may download model artifacts.
- Restrict access to the imported collection. Its default name is
  `confluence_documents` in the **same** `.rag_data/qdrant` database; use
  `--store-path` if a separate local database is needed.

See Atlassian's [pages-in-space API](https://developer.atlassian.com/cloud/confluence/rest/v2/api-group-page),
[cursor pagination](https://developer.atlassian.com/cloud/confluence/rest/v2/intro/),
and [API-token authentication guidance](https://developer.atlassian.com/cloud/confluence/basic-auth-for-rest-apis/).

## Configure And Run

Set these variables in the **same PowerShell 7 process** that runs the CLI:

```powershell
$env:RAG_CONFLUENCE_BASE_URL = 'https://your-site.atlassian.net'
$env:RAG_CONFLUENCE_EMAIL = 'reader@example.com'
$env:RAG_CONFLUENCE_API_TOKEN = Read-Host 'Confluence API token' -MaskInput
uv run rag-pipeline index-confluence '<numeric-space-id>' --acknowledge-no-acl
```

The three variables have no defaults. The site URL must be an HTTPS
`*.atlassian.net` root, not a `/wiki` path; custom-domain and Data Center
installations are not supported. The CLI does **not** load `.env` files
automatically or accept tokens as flags. Remove the token from this PowerShell
process afterwards:

```powershell
Remove-Item Env:RAG_CONFLUENCE_API_TOKEN
```

Run `uv run rag-pipeline index-confluence --help` for all settings. Defaults
allow at most 100 current pages, 2 MiB of UTF-8 Storage-HTML per page, and
20 MiB in total. `--max-pages`, `--max-page-mib`, and `--max-total-mib` are
positive integer limits; larger values increase API work and memory. Each
single API listing response also has a 64 MiB hard cap. This import reads page
bodies only: attachments, comments, blog posts, whiteboards, and macro-rendered
external content are not imported. Confluence Storage-HTML is normalized with
the same HTML parser used for local files; complex macros may not become
faithful visible text.

The command reuses local chunking, embedding, dense/hybrid mode, and Qdrant
options. Query with the **same** model, revision, search mode, collection, and
store path as the import. For example:

```powershell
uv run rag-pipeline retrieve 'When are expenses due?' `
  --collection-name confluence_documents
```

## Data And Security Contract

- The source key is `confluence://<site-host>/<page-id>`. It survives title
  changes and keeps citations stable. Title, space ID, page version, and the
  page URL when returned by the API are stored as metadata. Excerpt offsets
  refer to normalized text with the title prepended, not raw Storage-HTML.
- The importer fetches and validates the full bounded traversal before
  model inference or Qdrant writes. Reimport replaces chunks for pages still
  listed. It **does not delete** previously indexed pages absent from a later
  API response. Cursor requests are not a point-in-time snapshot, and Qdrant
  updates are not atomic; rerun after a write failure.
- Requests use an API token held in process memory, never in a URL. Redirects
  are not followed. Cursor links are accepted only for the same site and
  space-listing endpoint. Response bodies, tokens, and cursor links are not
  included in CLI errors.
- **No Confluence ACLs are copied or enforced at query time.** Anyone who can
  query this collection through the shared-token API or read its local Qdrant
  files may see its content, regardless of Confluence page restrictions. The
  `--acknowledge-no-acl` flag acknowledges this gap; it is not authorization.
  Retrieval metadata filters are not a substitute for permission checks.
- There is no scheduled refresh, webhook, deletion reconciliation, rate-limit
  retry, page-version checkpoint, audit log, OAuth app flow, or live-tenant
  test in the default suite. Correct credentials, permissions, and limits
  before manually rerunning a failed import.

See [the architecture](../ARCHITECTURE.md) for package boundaries and the
[API reference](API_REFERENCE.md) for the separate local HTTP security model.
