
import streamlit as st
import requests
import os

st.set_page_config(
    page_title="AI Counselling Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ AI Counselling Assistant")
st.caption("Legal • Medical • Commonsense Knowledge Graph Retrieval")

st.info(
    "Enter your query below. The assistant will retrieve relevant "
    "knowledge and generate a response."
)

# The backend URL will be configured in deployment settings.
API_URL = os.getenv("BACKEND_URL", "").rstrip("/") + "/generate"

query = st.text_area(
    "Enter your query",
    placeholder="Example: Someone is threatening to share my private photos. What should I do?",
    height=120
)

if st.button("Get Answer", type="primary"):
    if not query.strip():
        st.warning("Please enter a query.")

    elif not os.getenv("BACKEND_URL"):
        st.warning(
            "The interface is ready, but the Qwen backend has not "
            "been configured yet."
        )

    else:
        try:
            with st.spinner("Retrieving evidence and generating your answer..."):
                result = requests.post(
                    API_URL,
                    json={"query": query.strip()},
                    timeout=300
                )

            if result.status_code != 200:
                st.error("The response could not be generated.")
            else:
                data = result.json()

                st.subheader("Counselling Response")
                st.write(data.get("response", "No response returned."))

                domains = data.get("selected_domains") or []
                if isinstance(domains, str):
                    domains = [domains]

                st.subheader("Selected Domains")
                if domains:
                    st.write("  ·  ".join(str(d).title() for d in domains))
                else:
                    st.caption("No domains returned.")

                scores = data.get("domain_scores")
                if isinstance(scores, dict) and scores:
                    with st.expander("View Domain Scores"):
                        for domain, score in scores.items():
                            st.write(str(domain).title())
                            try:
                                value = float(score)
                                progress = value / 100 if value > 1 else value
                                st.progress(max(0.0, min(1.0, progress)))
                                st.caption(f"Score: {value:g}")
                            except (TypeError, ValueError):
                                st.write(score)

                evidence = data.get("evidence") or []
                with st.expander(f"Retrieved Evidence ({len(evidence)})"):
                    if evidence:
                        for i, item in enumerate(evidence, start=1):
                            st.markdown(f"**Evidence {i}**")
                            st.caption(
                                f"{str(item.get('domain', 'Unknown')).title()} "
                                f"· Source: {item.get('source', 'Unknown')}"
                            )
                            st.write(item.get("text", "No content available."))
                            st.divider()
                    else:
                        st.write("No evidence was retrieved.")

                unsupported = data.get("unsupported_sections", [])
                if unsupported:
                    st.warning(
                        "Some legal section references were not found "
                        "in the retrieved evidence: "
                        + ", ".join(map(str, unsupported))
                    )

        except requests.RequestException:
            st.error(
                "Cannot connect to the backend. Please try again later."
            )
