# Donated-Car Sale-Value Predictor — API Reference

This document describes the HTTP API exposed by the service, for teams integrating
against it. It covers:

- `POST /predict` — predict a donation sale value for one or more vehicles
- `GET /healthz` — service health check
- `POST /logs` — query historical application logs

It intentionally does not cover every route the service process exposes — only the
ones consumers are expected to call.

---

## Base URL

```
https://api.modulaire-app.com
```

---

## Authentication

Every endpoint below requires an API key passed as a request header:

```
x-api-key: <your key>
```

Each endpoint checks the key against a different configured value, so a key that
works for `/predict` will not work for `/logs`, and vice versa — use the key you
were issued for the specific endpoint you're calling.

An optional header identifies the caller for logging/auditing purposes:

```
x-user-id: <your identifier>
```

If omitted, requests are logged under `"anonymous"`.

**Auth error responses:**

| Status | Body | Meaning |
|---|---|---|
| `401` | `{"error": "Missing x-api-key header."}` | No `x-api-key` header was sent. |
| `403` | `{"error": "Invalid API key for this endpoint."}` | The key sent doesn't match what this endpoint expects. |

---

## Common conventions

- **Content type**: send `Content-Type: application/json`; all responses are JSON.
- **Request ID**: every response carries an `X-Request-ID` header (a UUID). Include
  it when reporting an issue — it's the join key for server-side logs.
- **Error shape**: outside of the specific validation format described under
  `/predict` below, errors are returned as:
  ```json
  { "error": "<human-readable message>" }
  ```

---

## `POST /predict`

Predicts a donation sale value for one vehicle, or a batch of vehicles in a single
call.

### Query parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `explain` | bool | `false` | If `true`, also generates a natural-language explanation of the prediction. Adds noticeable latency (see [Performance](#performance)). |
| `shap_quantile` | string | `p50` | Which prediction (`p5`, `p50`, or `p95`) the returned feature attributions explain. |
| `k_pos` | int | `5` | Max number of positive (value-increasing) feature attributions to return. `-1`–`20`, where `-1` returns all available. |
| `k_neg` | int | `5` | Max number of negative (value-decreasing) feature attributions to return. `-1`–`20`, where `-1` returns all available. |
| `explanation_units` | string | `both` | Units used inside the natural-language explanation text: `dollar`, `percentage`, or `both`. Only relevant when `explain=true`. |
| `prompt_version` | string | `v1` | Which prompt template generates the explanation — see the table below. Only relevant when `explain=true`. **Which version served a request is not recorded in logs.** |

#### `prompt_version` options

| Version | Features covered | Length | Per-feature context |
|---|---|---|---|
| `v1` (default) | top 5 | 40-60 words | no |
| `v2` | top 5 | no limit | yes |
| `v3` | top 3 | no limit | yes |
| `v4` | top 3 | ~50-70 words | yes |

"Per-feature context" adds a plain-English blurb about what each feature
represents — e.g. a market-trend feature reads as "current U.S. economic
conditions" — introduced in `v2` along with explicit null-value handling.

The feature count is a **ceiling, not a target**: `k_pos`/`k_neg` decide how many
attributions exist at all, so `v2` shows only 2 positive features if you sent
`k_pos=2`.

### Request body

A single vehicle object, **or a JSON array of vehicle objects** to score as a batch.
Send an array and you get an array back, in the same order; send a single object
and you get a single object back.

All fields are optional — omit anything you don't have, rather than sending an
empty string or a placeholder. **Unknown/unrecognized fields cause the whole
request to be rejected** (`422`), so don't send fields not listed below.

| Field | Type | Notes |
|---|---|---|
| `stock_id` | string | Your own identifier for this vehicle. Echoed back in the response and in logs — send it so you can correlate results. |
| `make` | string | |
| `model` | string | |
| `year` | integer | |
| `trim` | string | |
| `vehicle_type` | number | Picklist ID from your system (not free text). |
| `vehicle_category` | string | Free text (e.g. body style). |
| `body_subtype` | string | |
| `mileage` | number | |
| `vehicle_cond_picklist_id` | number | Picklist ID — overall vehicle condition. |
| `color` | number | Picklist ID. |
| `body_paint_cond_picklist_id` | number | Picklist ID — body/paint condition. |
| `engine_cond_picklist_id` | number | Picklist ID — engine condition. |
| `transmission_cond_picklist_id` | number | Picklist ID — transmission condition. |
| `tire_cond_picklist_id` | number | Picklist ID — tire condition. |
| `interior_cond_picklist_id` | number | Picklist ID — interior condition. |
| `other_damage_pklist_id` | number or array of numbers | Picklist ID(s) — damage type(s). Send a single ID, or a JSON array of IDs for multiple damage types. |
| `comment` | string | Free-text notes. |
| `zip` | number | 5-digit ZIP. Must be a real number or JSON `null` — do not send an empty string or `"N/A"` as a stand-in for "unknown". |
| `state_picklist_id` | number | Picklist ID — vehicle's state. |
| `state_title_picklist` | number | Picklist ID — state on the title. |
| `accessible_for_tow_truck` | string | Literally `"true"` or `"false"` (a plain JSON boolean is also accepted and converted). |
| `located_at_donation_c_a` | string | Literally `"true"` or `"false"` (a plain JSON boolean is also accepted and converted). |
| `creation_datetime` | string | ISO 8601 date or datetime (e.g. `2026-04-15`). Defaults to today if omitted. |

**Important — picklist ID fields must be numbers, not resolved text.** Fields
marked "Picklist ID" above (`vehicle_type`, `color`, every `*_picklist_id`
field, and `other_damage_pklist_id`) must be sent as the numeric ID from your
system's picklist tables — e.g. `22968`, not `"Runs & Drives"`. A numeric-looking
string like `"22968"` is accepted and converted automatically, but genuine text
is rejected with a `422` rather than silently producing a bad prediction.

#### Example request (single vehicle)

```json
{
  "stock_id": "2LSHB6741XG059386",
  "make": "Honda",
  "model": "Element",
  "year": 2006,
  "trim": null,
  "vehicle_type": 23119,
  "vehicle_category": null,
  "body_subtype": null,
  "mileage": 340000,
  "vehicle_cond_picklist_id": 22974,
  "color": 22946,
  "body_paint_cond_picklist_id": 23044,
  "engine_cond_picklist_id": 23057,
  "transmission_cond_picklist_id": 23060,
  "tire_cond_picklist_id": 23073,
  "interior_cond_picklist_id": 23050,
  "other_damage_pklist_id": null,
  "comment": "Tire Condition Note: ; Body and Paint Damage Note: ; Interior Damage Note: ; Other Damage Note: ; Additional Comments: Vehicle hasn't been started or driven for 2 years.",
  "zip": "50248",
  "state_picklist_id": 13329,
  "state_title_picklist": 13329,
  "accessible_for_tow_truck": "true",
  "located_at_donation_c_a": "true"
}
```

Note `zip` above is sent as the string `"50248"` — a numeric-looking string is
accepted and converted automatically, same as for the picklist ID fields.

Send an array of such objects to score a batch in one call.

### Response body

| Field | Type | Notes |
|---|---|---|
| `request_id` | string | Unique ID for this API call, identical to the `X-Request-ID` response header. On a batch request **every item carries the same value** — it identifies the HTTP request, not the row (pair it with `stock_id` to point at one row). Quote it when reporting a problem, or pass it to [`POST /logs`](#post-logs) to pull back every log line for this exact call. |
| `stock_id` | string or null | Echoes the `stock_id` you sent. |
| `is_cult` | bool or null | Whether this vehicle was flagged as a collectible/rare vehicle — this changes how its condition and value are weighed internally. |
| `predictions.low` | number | Low end of the 90% confidence range (5th percentile). |
| `predictions.predicted_price` | number | Point estimate (median / 50th percentile). Use this as "the" prediction. |
| `predictions.high` | number | High end of the 90% confidence range (95th percentile). |
| `feature_importances.top_positive` | array | Features that pushed the value **up**, largest first. Omitted/empty if `k_pos=0`. |
| `feature_importances.top_negative` | array | Features that pushed the value **down**, largest first. Omitted/empty if `k_neg=0`. |
| `explanation` | string or null | Natural-language summary of the prediction. Only present when `explain=true` was passed. |

Each entry in `top_positive` / `top_negative`:

| Field | Type | Notes |
|---|---|---|
| `feature_raw_key` | string | Internal key for the input this attribution is about (e.g. `make`, `mileage`). |
| `feature_label` | string | Human-readable label for the same (e.g. `"Make"`). |
| `value` | string or null | The value you submitted for this input, decoded to its display name where applicable (e.g. a picklist ID resolves to `"Runs & Drives"`). |
| `dollar_impact` | number | How many dollars this input added (positive) or subtracted (negative) from the baseline. |
| `pct_of_prediction` | number | Same impact, as a percentage of the final predicted price. |

#### Example response

```json
{
  "request_id": "21b7928e-7296-4a8a-9ac2-1a6b81502472",
  "stock_id": "STK-48213",
  "is_cult": false,
  "predictions": {
    "low": 412.50,
    "predicted_price": 750.20,
    "high": 1320.80
  },
  "feature_importances": {
    "top_positive": [
      {
        "feature_raw_key": "make",
        "feature_label": "Make",
        "value": "Toyota",
        "dollar_impact": 130.40,
        "pct_of_prediction": 17.4
      }
    ],
    "top_negative": [
      {
        "feature_raw_key": "vehicle_cond_picklist_id",
        "feature_label": "Vehicle condition",
        "value": "Runs & Drives",
        "dollar_impact": -210.80,
        "pct_of_prediction": -28.1
      }
    ]
  },
  "explanation": "This 2008 Toyota Camry has an estimated donation sale value of about $750..."
}
```

For a batch request, the response is a JSON array of objects in this same shape,
one per input vehicle, in the same order.

### Error responses

| Status | Body | Cause |
|---|---|---|
| `422` | `{"error": "Validation Failed", "diagnostics": "<details>"}` | Request body failed validation — missing-field and per-field messages are combined into `diagnostics`, e.g. a picklist field sent as text instead of a number. |
| `500` | `{"error": "Internal server error during prediction."}` | Unexpected failure while generating a prediction. |
| `401` / `403` | See [Authentication](#authentication). | Missing or wrong `x-api-key`. |

---

## `GET /healthz`

Lightweight liveness/readiness check — no auth, no body.

```json
{
  "status": "running",
  "model_loaded": true
}
```

If the prediction model failed to load at startup, this returns `503` instead:

```json
{ "error": "Service Unavailable: Models not loaded." }
```

Use this for load-balancer / container health checks, not for anything
prediction-related.

---

## `POST /logs`

Queries the service's own structured application logs (e.g. to look up what was
returned for a given `stock_id`, or to audit recent traffic). Requires the
`/logs`-specific API key.

### Request body

Choose **one** time-window mode — mixing them is rejected:

| Field | Type | Notes |
|---|---|---|
| `stock_id` | string or array of strings | Optional filter — only return log entries mentioning this stock ID. Pass an array to match entries mentioning **any** of several IDs. |
| `request_id` | string or array of strings | Optional filter — only return log entries for this exact API call. Use the `request_id` from a `/predict` response body (or the `X-Request-ID` header). Must be a UUID. Pass an array to match any of several. |
| `endpoint` | string or array of strings | Optional filter — restrict to one endpoint: `/predict`, `/healthz`, or `/logs`. Pass an array for several. Omit to get all three (the default). Any other value is rejected with a `422`. |
| `days_ago` | integer | Relative window: logs from the last N days. |
| `minutes_ago` | integer | Relative window: logs from the last N minutes. Combine with `days_ago` if useful. |
| `start_time` | datetime | Absolute window start. Must include a UTC/timezone offset (e.g. `2026-01-01T00:00:00Z`). Requires `end_time`. |
| `end_time` | datetime | Absolute window end. Same format rule. Requires `start_time`. |
| `limit` | integer | Max log entries to return. `1`–`200`, default `200`. |

If none of `days_ago`/`minutes_ago`/`start_time`/`end_time` are given, the query
defaults to the last 7 days.

Filters combine with AND: `{"endpoint": "/predict", "stock_id": "STK-48213"}`
returns only `/predict` entries that mention that stock ID.

#### Example request

```json
{
  "stock_id": "STK-48213",
  "days_ago": 1,
  "limit": 50
}
```

#### Example: trace one API call end to end

Take the `request_id` from any `/predict` response and ask for just that call's
log lines — typically the "requested" entry and its matching "completed" (or
error) entry:

```json
{
  "request_id": "21b7928e-7296-4a8a-9ac2-1a6b81502472",
  "days_ago": 1
}
```

### Response body

```json
{
  "request_id": "9c2f01ab-55de-4c77-b0a1-6d3e8f0c1122",
  "time_window": { "start": "2026-08-31T00:00:00+00:00", "end": "2026-09-01T00:00:00+00:00" },
  "log_count": 2,
  "logs": [ { "...": "one JSON object per matching log entry" } ]
}
```

The top-level `request_id` identifies **this `/logs` call itself** (same as its
`X-Request-ID` header) — not the entries being searched for. Each entry inside
`logs` carries its own `request_id`, which is what the `request_id` filter
matches on.

`logs` entries mirror the service's internal structured log fields and are not a
fixed, guaranteed schema — treat them as opaque records for display/debugging
rather than parsing specific fields out of them.

### Error responses

| Status | Body | Cause |
|---|---|---|
| `422` | Standard FastAPI validation error | e.g. both a relative and absolute time window given, or a naive (no-timezone) `start_time`/`end_time`. |
| `500` | `{"error": "Downstream service processing breakdown: <details>"}` | The underlying log query failed. |
| `401` / `403` | See [Authentication](#authentication). | Missing or wrong `x-api-key`. |

---

## Performance

- `/predict` without `explain`: typically 50–150 ms.
- `/predict` with `explain=true`: typically 2–8 seconds (dominated by the
  natural-language generation step) — don't set tight client-side timeouts on
  this path.
