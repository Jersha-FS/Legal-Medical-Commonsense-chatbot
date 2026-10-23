import streamlit as st
import requests
from client import generate_response
from router import route_query

st.set_page_config(page_title="AI Counselling Assistant", page_icon="⚖️", layout="wide")

st.title("⚖️ AI Counselling Assistant")
st.caption("Legal • Medical • Commonsense Knowledge Graph Retrieval")
st.warning(
    "This tool provides informational support only. It is not a substitute for a qualified "
    "lawyer, clinician, or emergency service. Avoid entering names, phone numbers, or other "
    "identifying details."
)

with st.sidebar:
    st.subheader("How it works")
    st.write("1. Classify the query domain")
    st.write("2. Retrieve knowledge-graph evidence in the backend")
    st.write("3. Generate a response with the configured language model")
    st.caption("The backend must be deployed separately and configured in Streamlit Secrets.")

with st.form("counselling_form"):
    query = st.text_area(
        "Describe your concern",
        placeholder="Example: Someone is threatening me online. What support is available?",
        height=140,
        max_chars=6000,
    )
    submitted = st.form_submit_button("Get guidance", type="primary")

if submitted:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        scores, domains = route_query(query)
        st.write("**Local domain routing:** " + ", ".join(domains))
        st.caption("Routing scores: " + " · ".join(f"{k.title()}: {v:.0%}" for k, v in scores.items()))
        try:
            with st.spinner("Contacting the counselling backend..."):
                result = generate_response(query.strip())
            answer = result.get("response") or result.get("answer") or result.get("output")
            if not answer:
                st.error("The backend responded, but no response text was found.")
                st.json(result)
            else:
                st.subheader("Counselling response")
                st.markdown(answer)

            selected = result.get("selected_domains")
            if selected:
                st.caption("Backend-selected domains: " + ", ".join(selected))

            evidence = result.get("evidence") or []
            with st.expander(f"Retrieved knowledge-graph evidence ({len(evidence)})"):
                if not evidence:
                    st.info("The backend returned no evidence.")
                for i, item in enumerate(evidence, 1):
                    st.markdown(f"**Evidence {i} — {str(item.get('domain', 'unknown')).title()}**")
                    st.caption(
                        f"Type: {item.get('type', 'retrieved')} · "
                        f"Source: {item.get('source', 'Unknown source')}"
                    )
                    st.write(item.get("text", ""))
                    if item.get("section_number"):
                        st.caption(f"Section: {item['section_number']}")
                    st.divider()

            unsupported = result.get("unsupported_sections") or []
            if unsupported:
                st.warning(
                    "Some section references were not matched to retrieved evidence: "
                    + ", ".join(map(str, unsupported))
                    + ". Verify with an authoritative legal source or qualified lawyer."
                )
        except requests.exceptions.Timeout:
            st.error("The backend timed out. Check that the backend is running and increase BACKEND_TIMEOUT if appropriate.")
        except requests.exceptions.RequestException as exc:
            st.error("Could not reach the backend. Check BACKEND_URL in Streamlit Community Cloud → Settings → Secrets.")
            st.caption(str(exc))
        except (ValueError, TypeError) as exc:
            st.error(f"The backend response could not be read: {exc}")
