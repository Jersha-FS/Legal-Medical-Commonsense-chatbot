import requests
import streamlit as st


def generate_response(query: str) -> dict:
    """Call the separately hosted Flask/FastAPI backend."""
    backend_url = st.secrets.get("BACKEND_URL", "").strip().rstrip("/")
    if not backend_url:
        raise RuntimeError(
            "BACKEND_URL is missing. Add it in Streamlit Community Cloud → "
            "App settings → Secrets."
        )

    timeout = int(st.secrets.get("BACKEND_TIMEOUT", 120))
    response = requests.post(
        f"{backend_url}/generate",
        json={"query": query},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Expected the backend to return a JSON object.")
    return data
