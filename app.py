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

# ---------- EXPANSION STRATEGY ----------
expansion_strategy = st.radio("Expansion Strategy", ["User-defined", "Model-defined"], horizontal=True)

if expansion_strategy == "User-defined":
    num_expansions = st.slider("Number of expansions per seed query", min_value=1, max_value=25, value=5)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'seed_query' not in df.columns:
        st.error("Uploaded CSV must contain a column named 'seed_query'")
        st.stop()

    st.success(f"Loaded {len(df)} seed queries.")

    if st.button("Generate Synthetic Queries"):
        expanded_rows = []
        model = genai.GenerativeModel(model_name="models/gemini-2.5-pro-preview-03-25")

        with st.spinner("Running Gemini expansions..."):
            for index, row in df.iterrows():
                seed = row['seed_query']

                if expansion_strategy == "User-defined":
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
                else:
                    prompt = f"""
                    The user entered the query: "{seed}"

                    First, determine how many different search queries should be generated to explore this query from multiple dimensions. Think in terms of breadth, ambiguity, comparison, and user needs.

                    Then, generate the full list of that number of search queries using short to mid-tail phrasing. Each query should include:

                    - "query": [the generated query]
                    - "intent_type": [ambiguous, underspecified, exploratory, multi-faceted, comparative, task-oriented, or temporal]
                    - "semantic_relationship": [describe how this relates to the original query]

                    Finally, respond in JSON as:
                    {{
                      "reasoning": "...",
                      "target_number": 20,
                      "actual_queries": [ ... ]
                    }}

                    Do not include markdown or backticks. Do not return more than 100 total queries.
                    """

                try:
                    response = model.generate_content(prompt)
                    raw = response.text.strip()

                    if raw.startswith("```json"):
                        raw = raw.replace("```json", "").strip().rstrip("```").strip()
                    elif raw.startswith("```"):
                        raw = raw.replace("```", "").strip()

                    if expansion_strategy == "User-defined":
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
                    else:
                        structured = json.loads(raw)
                        queries = structured.get("actual_queries", [])
                        for item in queries:
                            expanded_rows.append({
                                "Original Seed Query": seed,
                                "Synth Query": item.get("query", "MISSING"),
                                "Intent Type": item.get("intent_type", "MISSING"),
                                "Semantic Relationship": item.get("semantic_relationship", "MISSING")
                            })

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

        st.markdown("### 📊 Intent Type Breakdown")
        count_df = output_df['Intent Type'].value_counts().rename_axis("Intent Type").reset_index(name="Count")
        st.bar_chart(count_df.set_index("Intent Type"))

        st.markdown("### 📋 Preview of Expanded Results")
        st.markdown("<div class='custom-table'>", unsafe_allow_html=True)
        st.dataframe(output_df.sample(min(10, len(output_df))), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", data=csv, file_name="query_expansions.csv", mime="text/csv")

else:
    st.info("Upload a file to begin.")
