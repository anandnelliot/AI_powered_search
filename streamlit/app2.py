import streamlit as st
import requests
import time
import pandas as pd
import os
from PIL import Image

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="🔍 Two-Step Product Search", layout="centered")

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'product_data' not in st.session_state:
    st.session_state.product_data = ""
if 'llm_output' not in st.session_state:
    st.session_state.llm_output = ""
if 'llm_done' not in st.session_state:
    st.session_state.llm_done = False

# -------------------------------------------------
# BACKEND URLS
# -------------------------------------------------
BASE_URL_TEXT    = "http://192.168.1.227:8008"
SEARCH_URL_TEXT  = f"{BASE_URL_TEXT}/product_search"
LLM_URL          = f"{BASE_URL_TEXT}/product_search_llm"

BASE_URL_IMAGE   = "http://192.168.1.227:8001"
SEARCH_URL_IMAGE = f"{BASE_URL_IMAGE}/search-image"

# -------------------------------------------------
# DATAFRAME FOR LOCAL IMAGE LOOKUP
# -------------------------------------------------
df = pd.read_csv(
    r"D:\Anand\Jsearch_ai\image search\data\fashion_with_image_paths.csv"
)

# -------------------------------------------------
# MAIN TITLE
# -------------------------------------------------
st.title("🔍 Two-Step Product Search with LLM Stream")

# -------------------------------------------------
# RADIO: CHOOSE SEARCH MODE
# -------------------------------------------------
search_mode = st.radio("Search mode:", ["Text", "Image"])

# -------------------------------------------------
# 1) TEXT MODE
# -------------------------------------------------
if search_mode == "Text":
    query   = st.text_input("Enter your text query",
                             placeholder="e.g. 'I need black sneakers'")
    k_value = st.number_input("Number of results (k)",
                              min_value=1, max_value=2000, value=5, step=1)

    # When user clicks "Search," call the text‐based endpoint
    if st.button("Search"):
        if not query.strip():
            st.error("Please enter a valid query.")
            st.stop()

        with st.spinner("Fetching product data..."):
            resp = requests.post(SEARCH_URL_TEXT,
                                 json={"query": query, "k": k_value},
                                 timeout=30)
            if resp.status_code != 200:
                st.error(f"Text Search Error {resp.status_code}: {resp.text}")
                st.stop()
            data = resp.json()

        # save results to session_state
        st.session_state.session_id    = data.get("session_id")
        st.session_state.product_data  = data.get("product_data", "")
        st.session_state.llm_output    = ""
        st.session_state.llm_done      = False

        st.success("Product data retrieved!")

    # once we have product data, show products + auto‐stream summary side by side
    if st.session_state.product_data:
        col_prod, col_llm = st.columns([2, 1])

        # ---- left: product list ----
        with col_prod:
            st.subheader("📦 Retrieved Products")
            st.text_area(
                "Product Results:",
                st.session_state.product_data,
                height=200
            )

        # ---- right: LLM summary ----
        with col_llm:
            st.subheader("📝 LLM Summary")
            placeholder = st.empty()

            # only call the LLM once
            if not st.session_state.llm_done:
                with st.spinner("Generating summary…"):
                    try:
                        r = requests.post(
                            LLM_URL,
                            json={"session_id": st.session_state.session_id},
                            stream=True,
                            timeout=120
                        )
                        r.raise_for_status()
                        for chunk in r.iter_content(chunk_size=128):
                            if chunk:
                                st.session_state.llm_output += chunk.decode("utf-8")
                                placeholder.markdown(st.session_state.llm_output + " ▌")
                                time.sleep(0.02)
                    except Exception as e:
                        placeholder.error(f"LLM Error: {e}")
                    finally:
                        st.session_state.llm_done = True

            # after streaming completes, show final text without cursor
            if st.session_state.llm_output:
                placeholder.markdown(st.session_state.llm_output)

# -------------------------------------------------
# 2) IMAGE MODE
# -------------------------------------------------
elif search_mode == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"]
    )
    top_k_img = st.slider("Number of results (top_k)", 1, 20, 5)

    if uploaded_file is not None and st.button("Search"):
        with st.spinner("Searching by image..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }
            params = {"top_k": top_k_img}
            resp = requests.post(
                SEARCH_URL_IMAGE, files=files, params=params, timeout=60
            )
            if resp.status_code != 200:
                st.error(f"Image Search Error {resp.status_code}: {resp.text}")
                st.stop()
            results = resp.json().get("results", [])

        st.write(f"Found {len(results)} image-based results.")

        # grid display, 3 images per row
        per_row = 3
        for i in range(0, len(results), per_row):
            row = results[i : i + per_row]
            cols = st.columns(len(row))
            for col, item in zip(cols, row):
                pid   = item["product_id"]
                score = item["similarity_score"]
                match = df.loc[df["ProductId"] == pid]
                if not match.empty:
                    path = match.iloc[0]["full_path"]
                    if os.path.isfile(path):
                        col.write(f"**PID:** {pid}, Score: {score:.4f}")
                        col.image(path, use_container_width=True)
                    else:
                        col.write(f"File not found: {path}")
                else:
                    col.write(f"No CSV row for PID={pid}")
