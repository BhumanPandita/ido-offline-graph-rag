#!/usr/bin/env python3
"""Minimal Azure OpenAI deployment diagnostic.

This intentionally uses the REST chat-completions route so the final URL and
Azure error body are visible when diagnosing a 404.
"""

from __future__ import annotations

import argparse
import os
import sys
from urllib.parse import quote

import requests
from dotenv import load_dotenv


def first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    return None


def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    for suffix in ("/openai/v1", "/openai"):
        if endpoint.lower().endswith(suffix):
            endpoint = endpoint[: -len(suffix)].rstrip("/")
    return endpoint


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default=first_env("AZURE_OPENAI_ENDPOINT", "ENDPOINT"),
        help="Azure resource endpoint, e.g. https://my-resource.openai.azure.com/",
    )
    parser.add_argument(
        "--api-key",
        default=first_env("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY", "API_KEY"),
    )
    parser.add_argument(
        "--deployment",
        default=first_env(
            "AZURE_OPENAI_DEPLOYMENT",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "AZURE_OPENAI_MODEL_NAME",
            "DEPLOYMENT_NAME",
        ),
        help="The custom Azure deployment name, not just the base model name",
    )
    parser.add_argument(
        "--api-version",
        default=first_env("AZURE_OPENAI_API_VERSION", "API_VERSION")
        or "2024-10-21",
    )
    args = parser.parse_args()

    missing = [
        name
        for name, value in (
            ("endpoint", args.endpoint),
            ("api-key", args.api_key),
            ("deployment", args.deployment),
            ("api-version", args.api_version),
        )
        if not value
    ]
    if missing:
        parser.error("Missing values: " + ", ".join(missing))

    endpoint = normalize_endpoint(args.endpoint)
    if "/deployments/" in endpoint.lower():
        parser.error(
            "Endpoint must be the Azure resource endpoint only; remove /openai/deployments/..."
        )

    url = (
        f"{endpoint}/openai/deployments/"
        f"{quote(args.deployment, safe='')}/chat/completions"
    )
    payload = {
        "messages": [
            {"role": "system", "content": "You are a connectivity test."},
            {"role": "user", "content": "Reply with exactly: Azure connection works."},
        ],
        "temperature": 0,
        "max_tokens": 20,
    }

    print(f"POST {url}?api-version={args.api_version}")
    print(f"Deployment: {args.deployment}")
    print(f"API key: {'*' * max(4, len(args.api_key) - 4)}{args.api_key[-4:]}")
    try:
        response = requests.post(
            url,
            params={"api-version": args.api_version},
            headers={"api-key": args.api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
    except requests.RequestException as exc:
        print(f"Network/TLS error: {exc}", file=sys.stderr)
        return 2

    print(f"HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError:
        print(response.text)
        return 1 if not response.ok else 0

    print(body)
    if response.ok:
        print("Azure connection works.")
        return 0

    error = body.get("error", {}) if isinstance(body, dict) else {}
    if response.status_code == 404:
        print(
            "404 checks: verify that the endpoint belongs to the resource containing "
            "this deployment, and that deployment is the exact custom deployment "
            "name shown in Azure, not merely 'gpt-4o'.",
            file=sys.stderr,
        )
    print(
        f"Azure error code={error.get('code', 'unknown')} "
        f"message={error.get('message', response.text)}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
