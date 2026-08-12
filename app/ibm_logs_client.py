# ================
# File: app/ibm_logs_client.py
# ================
import os
import json
import httpx
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List


async def fetch_and_format_logs(
    api_key: str,
    stock_id: Optional[str] = None,
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

    # Updated query to match your current app's logger and endpoints
    query = "source logs | filter $l.subsystemname == 'car-resale-api' | filter $d.endpoint == '/predict'|| $d.endpoint == '/healthz' || $d.endpoint == '/logs'"

    if stock_id:
        query += f" | filter $d.input_stock_ids.contains('{stock_id}')"
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

                                endpoint = user_data.get("endpoint", "N/A")
                                client_ip = user_data.get("client_ip", "N/A")
                                user_id = user_data.get("user_id", "N/A")
                                model_version = user_data.get("model_version", "N/A")

                                if (
                                    "Prediction" in base_message
                                    and "requested" in base_message
                                ):
                                    display_title = f"INCOMING REQUEST: API Prediction ({model_version})"
                                elif (
                                    "Prediction" in base_message
                                    and "completed" in base_message
                                ):
                                    display_title = f"SUCCESS: Prediction Completed ({model_version})"
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
                                    "endpoint": endpoint,
                                    "client_ip": client_ip,
                                    "user_id": user_id,
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
