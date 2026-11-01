# API contract

The base path is `/api/v1`. All bodies are JSON unless a row says otherwise.

## Authentication

| Method | Used by | Header or transport |
|---|---|---|
| JWT access token, 15 minutes | Web app | `Authorization: Bearer <token>` |
| Refresh token, 7 days | Web app | `HttpOnly`, `Secure`, `SameSite=Strict` cookie, rotated on use |
| API key | Extension and public API | `X-API-Key: tl_<random>` |
| WebSocket ticket, 30 seconds, single use | Web app | `?ticket=` on the WebSocket URL |

Browsers cannot set headers on a WebSocket, and a JWT in a query string ends up in access logs. The short-lived ticket avoids both problems.

## Endpoints

### Accounts

| Method | Path | Auth | Request | Success | Errors |
|---|---|---|---|---|---|
| POST | `/auth/register` | None | `email`, `password`, `display_name?` | `201` user | `409`, `422` |
| POST | `/auth/login` | None | `email`, `password` | `200` access token, sets refresh cookie | `401`, `429` |
| POST | `/auth/refresh` | Cookie | None | `200` access token | `401` |
| POST | `/auth/logout` | Cookie | None | `204` | `401` |
| GET | `/auth/me` | JWT | None | `200` user | `401` |

### Analysis

| Method | Path | Auth | Request | Success | Errors |
|---|---|---|---|---|---|
| POST | `/analyze/image` | JWT or key | Multipart `file`: JPEG, PNG or WebP, up to 10 MB | `202` submission | `401`, `413`, `415`, `422`, `429` |
| POST | `/analyze/audio` | JWT or key | Multipart `file`: WAV, MP3, FLAC, OGG, M4A or MP4, up to 25 MB and 5 minutes | `202` submission | `401`, `413`, `415`, `422`, `429` |
| POST | `/analyze/text` | JWT or key | `text`, up to 20,000 characters | `202` submission | `401`, `413`, `422`, `429` |
| POST | `/analyze/url` | JWT or key | `url`, `modality_hint?` | `202` submission | `401`, `422`, `429` |
| GET | `/analyses/{id}` | Owner | None | `200` analysis | `401`, `404` |
| GET | `/analyses` | JWT or key | `page`, `page_size`, `modality?`, `verdict?`, `status?`, `q?`, `from?`, `to?` | `200` page | `401`, `422` |
| POST | `/analyses/{id}/save` | Owner | None | `200` analysis with `retain_media: true` | `401`, `404`, `410` |
| DELETE | `/analyses/{id}` | Owner | None | `204` | `401`, `404` |
| GET | `/analyses/{id}/artefacts/{name}` | Owner | None | `302` to a presigned URL valid for 5 minutes | `401`, `404`, `410` |
| POST | `/analyses/{id}/feedback` | Owner | `is_correct`, `reported_label?`, `comment?` | `201` feedback | `401`, `404`, `409`, `422` |
| POST | `/analyses/{id}/ws-ticket` | JWT | None | `201` ticket | `401`, `404` |
| WS | `/ws/analyses/{id}?ticket=` | Ticket | None | Event stream | Close `4401`, `4404` |

`/analyze/url` returns `202` before the download starts. A blocked or invalid URL found in the worker appears as a failed analysis with an `error.code`.

### Reports and verification

| Method | Path | Auth | Request | Success | Errors |
|---|---|---|---|---|---|
| POST | `/analyses/{id}/report` | Owner | None | `202` report, or `200` if already cached | `401`, `404`, `409`, `410` |
| GET | `/reports/{id}.pdf` | Owner | None | `200` PDF, or `202` with `Retry-After` while rendering | `401`, `404`, `410` |
| GET | `/public/reports/{token}` | None | None | `200` verdict summary with no media | `404`, `429` |
| POST | `/verify` | None | Multipart `file`: PDF, up to 10 MB | `200` with `genuine` and, if true, `report_id` and `issued_at` | `413`, `415`, `429` |

### API keys

| Method | Path | Auth | Request | Success | Errors |
|---|---|---|---|---|---|
| GET | `/api-keys` | JWT | None | `200` list showing prefixes only | `401` |
| POST | `/api-keys` | JWT | `name`, `expires_at?` | `201` with the full key, shown once | `401`, `422` |
| DELETE | `/api-keys/{id}` | JWT | None | `204` | `401`, `404` |

### Admin and operations

| Method | Path | Auth | Request | Success | Errors |
|---|---|---|---|---|---|
| GET | `/admin/stats` | Admin JWT | `from?`, `to?` | `200` usage, latency and feedback accuracy | `401`, `403` |
| GET | `/admin/feedback` | Admin JWT | `page`, `page_size`, `flagged?`, `modality?` | `200` page | `401`, `403` |
| GET | `/models` | None | None | `200` active model names and versions | None |
| GET | `/healthz` | None | None | `200` when the process is alive | None |
| GET | `/readyz` | None | None | `200` when Postgres, Redis and MinIO respond | `503` |

## Shapes

### Submission

```json
{
  "analysis_id": "01966f3e-7b1a-7c11-9d2a-3f2a1c9e4b10",
  "status": "queued",
  "status_url": "/api/v1/analyses/01966f3e-7b1a-7c11-9d2a-3f2a1c9e4b10"
}
```

### Analysis

```json
{
  "id": "01966f3e-7b1a-7c11-9d2a-3f2a1c9e4b10",
  "modality": "image",
  "status": "completed",
  "progress": 100,
  "created_at": "2026-11-01T10:30:00Z",
  "completed_at": "2026-11-01T10:30:03Z",
  "media_expires_at": "2026-11-02T10:30:00Z",
  "result": {
    "probability": 0.91,
    "confidence_band": { "low": 0.85, "high": 0.95 },
    "verdict": "likely_fake",
    "verdict_label": "Likely AI-generated",
    "explanation": "Unusual texture detected around the mouth and hairline.",
    "model_name": "image-convnext-tiny-freq",
    "model_version": "1.0.0",
    "warnings": [],
    "disclaimer": "Results are probabilistic and must not be used alone to make decisions about a person.",
    "details": {}
  },
  "error": null
}
```

`probability` is always the probability that the content is AI-generated or manipulated. `verdict` uses the same three values for every modality. `verdict_label` gives the wording for that modality, so text shows `Likely human`, `Mixed` or `Likely AI`.

### Result details by modality

```json
{
  "image": {
    "face_detected": true,
    "regions": [
      { "box": [84, 132, 140, 170], "score": 0.82, "part": "mouth" }
    ],
    "artefacts": {
      "heatmap": "heatmap.png",
      "spectrum": "spectrum.png"
    }
  },
  "audio": {
    "duration_seconds": 30.0,
    "aggregate": { "mean": 0.77, "max": 0.94 },
    "windows": [
      { "start": 0.0, "end": 4.0, "probability": 0.72 }
    ],
    "artefacts": { "saliency": "saliency.png" }
  },
  "text": {
    "word_count": 412,
    "sentences": [
      { "start": 0, "end": 88, "text": "The sentence text.", "probability": 0.64 }
    ]
  }
}
```

### WebSocket events

```json
{ "type": "progress", "analysis_id": "...", "stage": "inference", "progress": 60 }
{ "type": "completed", "analysis_id": "..." }
{ "type": "failed", "analysis_id": "...", "error": { "code": "corrupt_media", "message": "..." } }
```

Stages run in the order `queued`, `fetching`, `validating`, `inference`, `explaining`, `saving`. `fetching` occurs only for URL jobs. The client fetches `GET /analyses/{id}` after `completed`.

### Paging

```json
{ "items": [], "page": 1, "page_size": 20, "total": 0 }
```

`page_size` has a maximum of 100.

### Errors

```json
{
  "error": {
    "code": "file_too_large",
    "message": "Images must be 10 MB or smaller.",
    "details": { "max_bytes": 10485760 }
  }
}
```

| HTTP status | Meaning | Example codes |
|---|---|---|
| `401` | Credentials are missing or invalid | `invalid_token`, `invalid_api_key` |
| `403` | The role is not permitted | `admin_required` |
| `404` | The resource does not exist or is not yours | `not_found` |
| `409` | A conflicting state exists | `email_taken`, `feedback_exists`, `analysis_not_complete` |
| `410` | The media was deleted by the 24-hour policy | `media_expired` |
| `413` | The input is over a limit | `file_too_large`, `text_too_long`, `audio_too_long` |
| `415` | The type is not supported | `unsupported_type` |
| `422` | The request is invalid | `validation_error`, `invalid_url` |
| `429` | The rate limit was reached | `rate_limited` |

Another user's analysis returns `404`, not `403`, so a caller cannot learn which ids exist.

## Rate limits

Limits are kept in Redis and reported through `X-RateLimit-Limit`, `X-RateLimit-Remaining` and `Retry-After`.

| Caller | Limit |
|---|---|
| API key | `api_keys.rate_limit_per_min`, default 30 per minute |
| JWT user | 60 per minute |
| Unauthenticated, per IP | 10 per minute on `/auth/login`, `/verify` and `/public/reports/{token}` |

## URL fetching rules

`/analyze/url` downloads attacker-chosen addresses from inside the server network, so the worker applies these rules:

- Only `http` and `https` on ports 80 and 443 are allowed.
- Every resolved address must be public. Loopback, private, link-local and metadata addresses are refused.
- The connection goes to the checked address, so a DNS change between check and fetch cannot redirect it.
- Each redirect is checked the same way, with a maximum of 3 redirects.
- Downloads stop at the size limit for the detected type and after 20 seconds.
- `Content-Type` is treated as a hint. Magic-number checks decide the actual type.
- Only direct media files are supported. Streaming pages and `blob:` URLs return `unsupported_type`.

## Validation order

1. Authenticate and check the rate limit.
2. Check the declared size, then the actual bytes received.
3. Identify the type from magic numbers.
4. Decode the media fully. Files that do not decode return `corrupt_media`.
5. Compute the SHA-256 hash of the original.
6. Strip EXIF from images.
7. Run the virus-scan hook.
8. Store the file in MinIO.
