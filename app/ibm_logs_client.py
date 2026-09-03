# ================
# File: app/ibm_logs_client.py
# ================
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

import httpx


# The endpoints this service logs against, and therefore the only values
# /logs can be filtered by. It lives here rather than in schemas.py because
# this module builds the DataPrime query AND has to keep working when run
# directly (`python app/ibm_logs_client.py` puts app/ on sys.path, not the
# repo root, so it could not import app.schemas). schemas.py imports this
# tuple for its allowlist validator instead of restating the values: an
# allowlist that drifted from this default would fail silently -- an
# accepted endpoint the default query never includes just returns zero
# rows, with no error anywhere.
#
# /models is deliberately absent: it does no logging at all, so filtering
# on it could only ever return an empty list.
LOG_QUERY_ENDPOINTS = ("/predict", "/healthz", "/logs")


async def fetch_and_format_logs(
    api_key: str,
    stock_id: Optional[Union[str, List[str]]] = None,
    endpoint: Optional[Union[str, List[str]]] = None,
    request_id: Optional[Union[str, List[str]]] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    days_ago: Optional[int] = 7,
    minutes_ago: Optional[int] = None,
    limit: int = None,
) -> Dict[str, Any]:
    """
    Core business logic to authenticate, query, and format logs from IBM Cloud.
    Requires IBM_LOGS_INSTANCE_ID and IBM_LOGS_REGION to be set in the environment.
    """
    instance_id = os.getenv("IBM_LOGS_INSTANCE_ID")
    region = os.getenv("IBM_LOGS_REGION", "us-south")

    if not instance_id:
        raise ValueError("IBM_LOGS_INSTANCE_ID environment variable is missing.")

    # Each filter below gets its OWN `| filter` pipeline stage. Stages are
    # ANDed implicitly, so the `||`s inside one stage can never bleed into
    # another -- no parenthesisation or operator-precedence juggling is
    # needed to combine "any of these endpoints" with "any of these stock
    # IDs" with "any of these request IDs".
    #
    # An unspecified endpoint falls back to every known endpoint, which
    # reproduces the previously hardcoded 3-way OR exactly. Dropping the
    # filter entirely would NOT be equivalent: the same subsystem also
    # emits startup and uvicorn lines that carry no $d.endpoint at all, and
    # those would suddenly start appearing in /logs output.
    #
    # The values are interpolated unescaped because they cannot be
    # caller-controlled: LogsQueryRequest validates them against
    # LOG_QUERY_ENDPOINTS, so every string reaching this f-string is one of
    # our own constants. That is a stronger guarantee than the character
    # denylist stock_id relies on below.
    endpoints = (
        [endpoint] if isinstance(endpoint, str) else list(endpoint or LOG_QUERY_ENDPOINTS)
    )
    unknown = [ep for ep in endpoints if ep not in LOG_QUERY_ENDPOINTS]
    if unknown:
        # Belt and braces: LogsQueryRequest is the gate for HTTP callers and
        # gives them a friendly 422, but this function is also called
        # directly (see __main__ below), and the interpolation above assumes
        # allowlisted values.
        raise ValueError(f"Unknown endpoint(s) for log query: {unknown}")

    endpoint_clauses = " || ".join(f"$d.endpoint == '{ep}'" for ep in endpoints)
    query = (
        "source logs | filter $l.subsystemname == 'car-resale-api'"
        f" | filter {endpoint_clauses}"
    )

    if stock_id:
        # input_stock_ids is logged as a JSON array (app/main.py always wraps
        # it in a list, even for single-item requests), so array membership
        # must be checked with arrayContains -- the string .contains() method
        # used here previously caused a type-mismatch 400 from the DataPrime
        # query engine whenever it matched a row.
        #
        # arrayContains takes one element, not a list, so multiple requested
        # IDs are ORed together as separate arrayContains(...) filters --
        # matches a log row containing ANY of the given stock IDs.
        stock_ids = [stock_id] if isinstance(stock_id, str) else list(stock_id)
        contains_clauses = " || ".join(
            f"$d.input_stock_ids.arrayContains('{sid}')" for sid in stock_ids
        )
        query += f" | filter {contains_clauses}"

    if request_id:
        # Scalar equality, NOT arrayContains: request_id is a plain string
        # on the log record (one id per HTTP request), unlike
        # input_stock_ids which is always a JSON array. Using arrayContains
        # here would be the same type-mismatch 400 that .contains() on
        # input_stock_ids caused.
        #
        # Interpolated unescaped because LogsQueryRequest constrains these
        # to UUID format -- hex digits and dashes only, nothing that can
        # break out of the surrounding string literal.
        request_ids = (
            [request_id] if isinstance(request_id, str) else list(request_id)
        )
        request_id_clauses = " || ".join(
            f"$d.request_id == '{rid}'" for rid in request_ids
        )
        query += f" | filter {request_id_clauses}"

    if limit:
        query += f" | limit {limit}"

    # Calculate the Time Window
    if start_time and end_time:
        start_utc = (
            start_time.astimezone(timezone.utc)
            if start_time.tzinfo
            else start_time.replace(tzinfo=timezone.utc)
        )
        end_utc = (
            end_time.astimezone(timezone.utc)
            if end_time.tzinfo
            else end_time.replace(tzinfo=timezone.utc)
        )
    else:
        now_utc = datetime.now(timezone.utc)
        end_utc = now_utc - timedelta(minutes=3)

        if days_ago is not None:
            start_utc = end_utc - timedelta(days=days_ago)
        else:
            start_utc = end_utc - timedelta(minutes=minutes_ago or 30)

    # Format explicitly to IBM's required ISO 8601 string format
    end_date_str = end_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    start_date_str = start_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    async with httpx.AsyncClient() as client:
        # Authenticate with IBM IAM
        try:
            auth_response = await client.post(
                "https://iam.cloud.ibm.com/identity/token",
                data={
                    "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                    "apikey": api_key,
                },
                headers={"Accept": "application/json"},
                timeout=10,
            )
            auth_response.raise_for_status()
            token = auth_response.json().get("access_token")
            if not token:
                raise ValueError("Token not found in response payload.")
        except Exception as e:
            raise ConnectionError(f"Failed to authenticate with IBM IAM: {str(e)}")

        # Fetch Logs from IBM Cloud
        url = f"https://{instance_id}.api.{region}.logs.cloud.ibm.com/v1/query"
        payload = {
            "query": query,
            "metadata": {
                "start_date": start_date_str,
                "end_date": end_date_str,
                "tier": "archive",
            },
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        try:
            response = await client.post(
                url, headers=headers, json=payload, timeout=300
            )
            response.raise_for_status()
        except Exception as e:
            raise ConnectionError(f"Failed to fetch logs from IBM API: {str(e)}")

        # Parse and Format the SSE Response
        formatted_logs: List[Dict[str, Any]] = []
        for line in response.text.splitlines():
            if line.startswith("data:") and not line.endswith("[DONE]"):
                raw_data = line[5:].strip()
                try:
                    parsed = json.loads(raw_data)
                    if "result" in parsed and "results" in parsed["result"]:
                        for entry in parsed["result"]["results"]:
                            timestamp = "UNKNOWN_TIME"
                            for meta in entry.get("metadata", []):
                                if meta.get("key") == "timestamp":
                                    timestamp = meta.get("value")
                                    break

                            user_data_str = entry.get("user_data", "{}")
                            try:
                                user_data = json.loads(user_data_str)
                                timestamp = user_data.get("timestamp", timestamp)
                                log_level = user_data.get("level", "INFO")
                                base_message = user_data.get(
                                    "message", user_data.get("log", "")
                                ).strip()

                                # entry_-prefixed, not endpoint/request_id:
                                # this function now takes filter parameters
                                # by both those names, which loop-locals of
                                # the same name would silently overwrite.
                                entry_endpoint = user_data.get("endpoint", "N/A")
                                client_ip = user_data.get("client_ip", "N/A")
                                user_id = user_data.get("user_id", "N/A")
                                entry_request_id = user_data.get("request_id", "N/A")

                                if (
                                    "Prediction" in base_message
                                    and "requested" in base_message
                                ):
                                    display_title = "INCOMING REQUEST: API Prediction"
                                elif (
                                    "Prediction" in base_message
                                    and "completed" in base_message
                                ):
                                    display_title = "SUCCESS: Prediction Completed"
                                elif (
                                    log_level in ["ERROR", "WARNING"]
                                    or "Exception" in base_message
                                ):
                                    display_title = f"ERROR: {base_message}"
                                else:
                                    display_title = f"INFO: {base_message}"

                                log_entry = {
                                    "timestamp": timestamp,
                                    "level": log_level,
                                    "title": display_title,
                                    "endpoint": entry_endpoint,
                                    "client_ip": client_ip,
                                    "user_id": user_id,
                                    # Same UUID the request's X-Request-ID
                                    # header and /predict response body
                                    # carry -- it is the join key between a
                                    # caller's bug report and these logs,
                                    # and what the request_id filter above
                                    # matches on.
                                    "request_id": entry_request_id,
                                }

                                # ==========================================
                                # Add Request Metadata
                                # ==========================================
                                if (
                                    "Prediction" in base_message
                                    and "requested" in base_message
                                ):
                                    request_meta = {}
                                    if "batch_size" in user_data:
                                        request_meta["batch_size"] = user_data.get(
                                            "batch_size"
                                        )
                                    if "input_stock_ids" in user_data:
                                        request_meta["stock_ids"] = user_data.get(
                                            "input_stock_ids"
                                        )

                                    # Support new project fields
                                    if "explain_requested" in user_data:
                                        request_meta["explain_requested"] = (
                                            user_data.get("explain_requested")
                                        )
                                    if "shap_quantile" in user_data:
                                        request_meta["shap_quantile"] = user_data.get(
                                            "shap_quantile"
                                        )

                                    if request_meta:
                                        log_entry["request_metadata"] = request_meta

                                # ==========================================
                                # Add Completion Metadata
                                # ==========================================
                                if (
                                    "Prediction" in base_message
                                    and "completed" in base_message
                                ):
                                    completion_meta = {}

                                    # Support new project fields
                                    if "total_ms" in user_data:
                                        completion_meta["duration_ms"] = user_data.get(
                                            "total_ms"
                                        )
                                    if "route" in user_data:
                                        completion_meta["route"] = user_data.get(
                                            "route"
                                        )
                                    if "is_cult" in user_data:
                                        completion_meta["is_cult"] = user_data.get(
                                            "is_cult"
                                        )

                                    if "inference_duration_seconds" in user_data:
                                        completion_meta["duration_seconds"] = (
                                            user_data.get("inference_duration_seconds")
                                        )

                                    if "output_data" in user_data:
                                        results_list = []
                                        for item in user_data["output_data"]:
                                            stock = item.get("stock_id", "N/A")
                                            val = item.get(
                                                "predicted_sale_value", "N/A"
                                            )
                                            rng = item.get("predicted_range", {})

                                            features = item.get(
                                                "feature_explanations", "N/A"
                                            )
                                            if isinstance(features, list):
                                                formatted_features = ", ".join(
                                                    [
                                                        f"{f.get('feature_name', 'Unknown')} ({f.get('contribution', '0%')})"
                                                        for f in features
                                                        if isinstance(f, dict)
                                                    ]
                                                )
                                            else:
                                                formatted_features = str(features)

                                            results_list.append(
                                                {
                                                    "stock_id": stock,
                                                    "predicted_sale_value": val,
                                                    "predicted_range": {
                                                        "low": rng.get("low", "N/A"),
                                                        "high": rng.get("high", "N/A"),
                                                    },
                                                    "features": formatted_features,
                                                }
                                            )
                                        completion_meta["results"] = results_list

                                    if completion_meta:
                                        log_entry["completion_metadata"] = (
                                            completion_meta
                                        )

                                formatted_logs.append(log_entry)

                            except json.JSONDecodeError:
                                formatted_logs.append(
                                    {
                                        "timestamp": timestamp,
                                        "level": "UNKNOWN",
                                        "title": "Raw parsing error",
                                    }
                                )
                except json.JSONDecodeError:
                    continue

        formatted_logs.sort(key=lambda x: x.get("timestamp", ""))

        return {
            "query_executed": query,
            "time_window": {"start": start_date_str, "end": end_date_str},
            "log_count": len(formatted_logs),
            "logs": formatted_logs,
        }


if __name__ == "__main__":
    import asyncio
    import sys

    # Attempt to load environment variables from a .env file if available
    try:
        from dotenv import load_dotenv, find_dotenv

        load_dotenv(find_dotenv())
    except ImportError:
        pass  # python-dotenv not installed, assuming env vars are set in the terminal

    async def main():
        # Fallback to WATSONX_API_KEY if IBM_CLOUD_API_KEY isn't set
        api_key = os.getenv("IBM_CLOUD_API_KEY") or os.getenv("WATSONX_API_KEY")

        if not api_key:
            print(
                json.dumps(
                    {
                        "error": "IBM_CLOUD_API_KEY or WATSONX_API_KEY is missing from the environment."
                    }
                )
            )
            sys.exit(1)

        # Optional: verify the instance ID is set so we don't fail deep inside the function
        if not os.getenv("IBM_LOGS_INSTANCE_ID"):
            print(
                json.dumps(
                    {"error": "IBM_LOGS_INSTANCE_ID is missing from the environment."}
                )
            )
            sys.exit(1)

        print("Fetching logs from IBM Cloud DataPrime...\n")
        try:
            # Test run: Fetch up to 20 logs from the last 14 days
            result = await fetch_and_format_logs(api_key=api_key, days_ago=30)
            print(json.dumps(result, indent=4))

        except Exception as e:
            print(json.dumps({"error": f"An error occurred: {str(e)}"}))

    # Run the async main function
    asyncio.run(main())
