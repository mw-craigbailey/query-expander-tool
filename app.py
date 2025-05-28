import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import time

st.set_page_config(page_title="Query Expander Export Tool", layout="wide")

# ---------- STYLES ----------
st.markdown("""
    <style>
    html, body {
        background-color: #f0f4ff;
    }
    h1 {
        color: #003399;
        text-align: center;
    }
    .stTextInput input, .stFileUploader input {
        background-color: #fff;
    }
    .dataframe-container {
        max-height: 400px;
        overflow-y: auto;
        margin-top: 20px;
        border: 1px solid #ddd;
        padding: 10px;
    }
    .custom-table table {
        width: 100%;
        table-layout: fixed;
    }
    .custom-table th, .custom-table td {
        word-wrap: break-word !important;
        white-space: normal !important;
        overflow-wrap: break-word;
        font-size: 0.95rem;
        padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<h1>📦 Query Expander Export Tool</h1>", unsafe_allow_html=True)

# ---------- API KEY ----------
api_key = st.text_input("Enter your Gemini API key", type="password")

if not api_key:
    st.info("Please enter your API key to begin.")
    st.stop()

genai.configure(api_key=api_key)

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload your CSV with a single column: 'seed_query'", type=["csv"])

# ---------- CONFIG ----------
num_expansions = st.slider("Number of expansions per seed query", min_value=1, max_value=25, value=5)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'seed_query' not in df.columns:
        st.error("Uploaded CSV must contain a column named 'seed_query'")
        st.stop()

    st.success(f"Loaded {len(df)} seed queries.")

    if st.button("🔁 Expand All Queries"):
        expanded_rows = []
        model = genai.GenerativeModel(model_name="models/gemini-2.5-pro-preview-03-25")

        with st.spinner("Running Gemini expansions..."):
            for index, row in df.iterrows():
                seed = row['seed_query']

                prompt = f"""
                The user provided a base query: "{seed}"

                Generate {num_expansions} short to mid-tail search queries that simulate diversified user intents (fanout) based on this input.
                Each query should be phrased like a real-world search input (max 12 words). Avoid full sentences or overly formal phrasing.

                For each query, output:
                - "query": [the generated search phrase]
                - "intent_type": [ambiguous, underspecified, exploratory, multi-faceted, comparative, task-oriented, or temporal]
                - "semantic_relationship": [describe how this relates to the original search]

                Respond only with valid raw JSON. Do not include markdown or backticks.
                """

                try:
                    response = model.generate_content(prompt)
                    raw = response.text.strip()

                    if raw.startswith("```json"):
                        raw = raw.replace("```json", "").strip().rstrip("```").strip()
                    elif raw.startswith("```"):
                        raw = raw.replace("```", "").strip()

                    parsed = json.loads(raw)

                    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                        for item in parsed:
                            expanded_rows.append({
                                "Original Seed Query": seed,
                                "Synth Query": item.get("query", "MISSING"),
                                "Intent Type": item.get("intent_type", "MISSING"),
                                "Semantic Relationship": item.get("semantic_relationship", "MISSING")
                            })
                    else:
                        raise ValueError("Gemini response was not a list of dictionaries")

                except Exception as e:
                    expanded_rows.append({
                        "Original Seed Query": seed,
                        "Synth Query": "ERROR",
                        "Intent Type": "ERROR",
                        "Semantic Relationship": f"{str(e)} — Raw: {raw[:150]}"
                    })

                time.sleep(0.2)  # throttle requests

        output_df = pd.DataFrame(expanded_rows)
        st.success("Expansion complete.")

        st.markdown("### Preview of expanded results")
        st.markdown("<div class='custom-table'>", unsafe_allow_html=True)
        st.write(output_df.sample(min(10, len(output_df))))
        st.markdown("</div>", unsafe_allow_html=True)

        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", data=csv, file_name="query_expansions.csv", mime="text/csv")

else:
    st.info("Upload a file to begin.")