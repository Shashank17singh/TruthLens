# Database schema

The target is PostgreSQL 18, which provides `uuidv7()`. Identifiers that appear in URLs use time-ordered UUIDs, so they cannot be guessed and they do not fragment the index. Internal rows use `bigint` identity keys.

The backend is the only database client, so ownership is enforced in queries and row-level security is not used.

```sql
create table users (
  id            uuid primary key default uuidv7(),
  email         text not null,
  password_hash text not null,
  display_name  text,
  role          text not null default 'user' check (role in ('user', 'admin')),
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);
create unique index users_email_lower_key on users (lower(email));

create table api_keys (
  id                  uuid primary key default uuidv7(),
  user_id             uuid not null references users (id) on delete cascade,
  name                text not null,
  key_prefix          text not null,
  key_hash            text not null unique,
  rate_limit_per_min  integer not null default 30 check (rate_limit_per_min > 0),
  last_used_at        timestamptz,
  expires_at          timestamptz,
  revoked_at          timestamptz,
  created_at          timestamptz not null default now()
);
create index api_keys_user_id_idx on api_keys (user_id);

create table analyses (
  id               uuid primary key default uuidv7(),
  user_id          uuid not null references users (id) on delete cascade,
  api_key_id       uuid references api_keys (id) on delete set null,
  modality         text check (modality in ('image', 'audio', 'text')),
  source_type      text not null check (source_type in ('upload', 'url')),
  source_url       text,
  status           text not null default 'queued'
                   check (status in ('queued', 'processing', 'completed', 'failed')),
  progress         smallint not null default 0 check (progress between 0 and 100),
  file_name        text,
  mime_type        text,
  size_bytes       bigint check (size_bytes >= 0),
  sha256           text check (sha256 ~ '^[0-9a-f]{64}$'),
  object_key       text,
  retain_media     boolean not null default false,
  media_expires_at timestamptz,
  media_deleted_at timestamptz,
  error_code       text,
  error_message    text,
  created_at       timestamptz not null default now(),
  started_at       timestamptz,
  completed_at     timestamptz,
  constraint analyses_url_has_source
    check (source_type <> 'url' or source_url is not null),
  constraint analyses_modality_known
    check (modality is not null or (source_type = 'url' and status <> 'completed')),
  constraint analyses_failure_has_code
    check (status <> 'failed' or error_code is not null)
);
create index analyses_user_history_idx on analyses (user_id, created_at desc);
create index analyses_api_key_id_idx on analyses (api_key_id);
create index analyses_media_purge_idx on analyses (media_expires_at)
  where media_deleted_at is null and retain_media = false;

create table analysis_results (
  id               bigint generated always as identity primary key,
  analysis_id      uuid not null references analyses (id) on delete cascade,
  model_name       text not null,
  model_version    text not null,
  probability      double precision not null check (probability between 0 and 1),
  decision_threshold double precision not null check (decision_threshold between 0 and 1),
  verdict          text not null check (verdict in ('likely_real', 'uncertain', 'likely_fake')),
  confidence_low   double precision not null check (confidence_low between 0 and 1),
  confidence_high  double precision not null check (confidence_high between 0 and 1),
  explanation      text not null,
  details          jsonb not null default '{}',
  artefacts        jsonb not null default '{}',
  latency_ms       integer not null check (latency_ms >= 0),
  created_at       timestamptz not null default now(),
  constraint analysis_results_confidence_order check (confidence_low <= confidence_high),
  constraint analysis_results_one_per_model unique (analysis_id, model_name, model_version)
);

create table reports (
  id           uuid primary key default uuidv7(),
  analysis_id  uuid not null unique references analyses (id) on delete cascade,
  status       text not null default 'queued'
               check (status in ('queued', 'rendering', 'ready', 'failed', 'expired')),
  object_key   text,
  sha256       text check (sha256 ~ '^[0-9a-f]{64}$'),
  hmac         text,
  hmac_key_id  text,
  public_token text not null unique,
  created_at   timestamptz not null default now(),
  ready_at     timestamptz,
  constraint reports_ready_is_signed
    check (status <> 'ready' or (object_key is not null and sha256 is not null
           and hmac is not null and hmac_key_id is not null))
);
create index reports_sha256_idx on reports (sha256);

create table feedback (
  id             bigint generated always as identity primary key,
  analysis_id    uuid not null references analyses (id) on delete cascade,
  user_id        uuid not null references users (id) on delete cascade,
  is_correct     boolean not null,
  reported_label text check (reported_label in ('real', 'fake', 'unsure')),
  comment        text,
  created_at     timestamptz not null default now(),
  constraint feedback_one_per_user unique (analysis_id, user_id)
);
create index feedback_user_id_idx on feedback (user_id);
create index feedback_flagged_idx on feedback (created_at desc) where is_correct = false;
```

## Column notes

- `analyses.modality` stays empty for a URL job until the worker has downloaded the media and identified its type.
- `analyses.sha256` is computed from the original upload, before EXIF is stripped, so the report matches the file the user holds.
- `analyses.media_expires_at` is 24 hours after creation. Saving an analysis sets `retain_media` to true and the purge job then skips it.
- `analysis_results` allows several rows per analysis, so a later re-run with a new model version does not overwrite the earlier result.
- `analysis_results.artefacts` holds MinIO keys for heatmaps, spectra and spectrogram saliency maps. `details` holds the modality-specific payload defined in [api.md](api.md#result-details-by-modality).
- `api_keys.key_hash` is a SHA-256 hash. The keys are long random strings, so a slow password hash is unnecessary, and the hash can be checked on every request.

## Purge job

The hourly purge selects rows where `media_expires_at < now()`, `retain_media` is false and `media_deleted_at` is null. For each row it:

1. Deletes the upload, the artefacts and any cached PDF from MinIO.
2. Removes sentence text from `analysis_results.details`.
3. Sets `object_key` to null on the analysis and its report, and sets `media_deleted_at`.
4. Marks a `ready` report as `expired`, keeping its `sha256` and `hmac`.

The verdict, probability, model version and file hash remain, so history still works after media is deleted, and `/verify` still recognises a PDF the user downloaded earlier.
