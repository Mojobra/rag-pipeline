# Slack And Teams Channel Imports

`index-slack` and `index-teams` manually read one channel, including threaded
replies, and index message text into local Qdrant. They are read-only provider
integrations for a trusted operator, **not** permission-aware search or a
continuous synchronization service. Messages can contain private or regulated
information; get organizational approval before importing them.

## Slack

Create an internal Slack app with a bot token and the relevant channel-history
scope (`channels:history` for public channels or `groups:history` for private
channels). Add the bot to the channel. The importer uses `auth.test` to obtain
the workspace ID, then `conversations.history` and `conversations.replies` to
read text. It does not read DMs, multi-party DMs, files, or message attachments.
Store the bot token outside Git and expose it to the process:

```powershell
$env:RAG_SLACK_BOT_TOKEN = '<bot-token>'
uv run rag-pipeline index-slack '<channel-id>' --acknowledge-no-acl
```

Slack [history](https://docs.slack.dev/reference/methods/conversations.history/)
and [replies](https://docs.slack.dev/reference/methods/conversations.replies/)
are cursor-paginated. This importer requests at most 15 messages per page to
stay within Slack's reduced page-size limit for some app distributions.
Rate limits still depend on the app: HTTP 429 aborts the import, with no retry
or checkpoint. For commercial non-Marketplace apps subject to one request per
minute, multi-page imports need a future paced/resumable workflow; this command
is intended for internal customer-built apps. See Slack's
[rate-limit policy](https://docs.slack.dev/changelog/2025/05/29/rate-limit-changes-for-non-marketplace-apps/).

## Teams

Register an Entra application in the tenant that owns the channel. Grant
Microsoft Graph application permission `ChannelMessage.Read.Group` with
resource-specific consent for the selected team, or the broader
`ChannelMessage.Read.All` only when organizational policy calls for it. The
importer uses client credentials and Graph v1.0 to list channel roots with
expanded replies, then follows any root or reply continuation URLs. It does
not read chats, meeting messages, files, or attachments.

Set these process environment variables to the Entra tenant UUID, client UUID,
and application secret (keep the secret out of the repository):

```powershell
$env:RAG_TEAMS_TENANT_ID = '<tenant-uuid>'
$env:RAG_TEAMS_CLIENT_ID = '<client-uuid>'
$env:RAG_TEAMS_CLIENT_SECRET = '<client-secret>'
uv run rag-pipeline index-teams '<team-uuid>' '<channel-id>' --acknowledge-no-acl
```

Microsoft documents the [channel message list](https://learn.microsoft.com/en-us/graph/api/channel-list-messages?view=graph-rest-1.0)
and [reply list](https://learn.microsoft.com/en-us/graph/api/chatmessage-list-replies?view=graph-rest-1.0)
APIs, including app permission requirements. The same MSAL/Graph client
credential mechanism used by SharePoint is reused, but `RAG_TEAMS_*` values
are independent. A 429 or other Graph failure aborts; no retry is implemented.

## Local Index Contract

- Defaults are `slack_messages` and `teams_messages` in the existing
  `.rag_data/qdrant` store. Use `--collection-name` and `--store-path` to
  isolate content further. Query with the same embedding model, revision,
  search mode, collection, and store path used at import.
- One import accepts at most 100 root/reply messages, 64 KiB of raw UTF-8 per
  message, and 2 MiB combined by default. `--max-messages`,
  `--max-message-kib`, and `--max-total-mib` can change these caps. The
  adapters also bound response bytes and HTTP request counts; oversized or
  malformed traversals abort before indexing.
- Stable `slack://<workspace>/<channel>/<timestamp>` and
  `teams://<team>/<channel>/<message-id>` root keys back chunks and citations;
  Teams reply keys include `/<parent-id>/replies/<message-id>` so IDs from
  different threads cannot collide. Teams HTML becomes visible text; Slack
  message markup remains as returned by the API. Empty-text messages can
  replace old chunks with no
  new chunks. Slack bot messages are included; non-message system events and
  deleted messages are skipped.
- A completed read is staged before model inference or Qdrant writes. A
  reimport replaces only messages returned in that run. Provider pages are
  **not** a point-in-time snapshot; messages deleted, made inaccessible, or
  absent from a later read **remain indexed**. Qdrant replacement is not
  atomic. There is no deletion reconciliation, scheduled refresh, backoff,
  incremental checkpoint, or live workspace/tenant test in the default suite.
- **Neither Slack nor Teams ACLs are copied or enforced on retrieval.** The
  required `--acknowledge-no-acl` flag is an operator warning, not an access
  control. The local HTTP API's shared bearer token is not a substitute for
  per-user message authorization. Restrict the collection, index files, model
  host, logs, and exports to the intended readership. Permissions-aware
  retrieval is a separate roadmap task.

Run `uv run rag-pipeline index-slack --help` or
`uv run rag-pipeline index-teams --help` for chunking, embedding, and storage
options. The CLI does not load `.env` automatically; [.env.example](../.env.example)
is a template, not a credential loader.
