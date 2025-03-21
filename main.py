import asyncio
import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import altair as alt

# =====================
# 1. BASE CONFIGURATION
# =====================
# Using your provided Gemini API key:
GOOGLE_API_KEY = "AIzaSyCdoGJ77AtAzw9C7gf7mfk-cKDmUUgkf-4"
genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.0-flash-exp"

st.set_page_config(layout="centered")
st.title("4-Phase Car Parts Classification (No dash after colour, parentheses fix, consecutive duplicates)")

# Session state
if "log_text" not in st.session_state:
    st.session_state.log_text = ""
if "approved_categories" not in st.session_state:
    st.session_state.approved_categories = []
if "df_tokens" not in st.session_state:
    st.session_state.df_tokens = pd.DataFrame()
if "df_rewrites" not in st.session_state:
    st.session_state.df_rewrites = pd.DataFrame()
if "df_polished" not in st.session_state:
    st.session_state.df_polished = pd.DataFrame()

def log(message):
    st.session_state.log_text += message + "\n"

# ======================
# 2. GEMINI CALL UTILITY
# ======================
async def call_gemini(prompt, retries=10, max_tokens=2000):
    """
    Calls Gemini with a prompt and includes a retry mechanism for 429 or other errors.
    """
    model = genai.GenerativeModel(MODEL_NAME)
    backoff_seconds = 2
    for attempt in range(retries):
        try:
            log(f"[Gemini] Attempt {attempt+1}, max_tokens={max_tokens}, prompt len={len(prompt)}")
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config={"temperature": 0.5, "max_output_tokens": max_tokens}
            )
            if response and response.text:
                log("[Gemini] Success.")
                return response.text
            else:
                log("[Gemini] Returned no text.")
        except Exception as e:
            if "429" in str(e):
                log("[Gemini] 429 encountered. Backing off.")
            else:
                log(f"[Gemini] Exception: {e}")
        await asyncio.sleep(backoff_seconds)
        backoff_seconds *= 2
    raise RuntimeError("[Gemini] All attempts failed.")

# =============================================
# 3. PHASE 1: SUGGEST TAXONOMICAL CATEGORIES (OPTIONAL)
# =============================================
def suggest_categories(domain_context: str):
    """
    Prompt for taxonomical categories. Avoid mechanical function groups.
    """
    if not domain_context.strip():
        return []

    prompt = f"""
We are about to classify automotive e-commerce product titles for "{domain_context}".
We want categories that are typically found in such product titles, forming a taxonomical style:
- Brand (e.g. do88, GFB, Garrett)
- VehicleMake (e.g. Volvo, BMW, Audi)
- VehicleModel (e.g. 9-3 2.8T V6, 911 Turbo)
- YearRange (e.g. 2009–2014)
- CarPart (e.g. Intercooler, Radiator, Hose Kit, 'Turbo' if referencing forced induction)
- PartNumber (e.g. do88-kit73S)
- Colour (e.g. Black, Red, Blue)
- Unclassified (if truly unknown)

IMPORTANT: Avoid categories that group parts by mechanical system (e.g. Braking).
Return them as JSON with 'name' and 'definition'.
"""
    raw = asyncio.run(call_gemini(prompt, max_tokens=1000))
    start_index = raw.find("[")
    end_index = raw.rfind("]")
    if start_index == -1 or end_index == -1:
        log("[Categories] No JSON array found.")
        return []

    json_str = raw[start_index : end_index+1]
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
        else:
            log("[Categories] JSON was neither list nor dict.")
            return []
    except Exception as e:
        log(f"[Categories] JSON parse error: {e}")
        return []

# ====================================================
# 4. PHASE 2: CLASSIFY KEYWORDS & RANK MOST COMMON COMBOS
# ====================================================
async def classify_chunk(chunk_keywords):
    """
    Sends a batch to Gemini, requesting token/category assignment.
    Mentions that 'Turbo' might be part of VehicleModel or CarPart, etc.
    """
    if not chunk_keywords:
        return []

    # Build category instructions
    if st.session_state.approved_categories:
        cat_lines = []
        cat_list = []
        for c in st.session_state.approved_categories:
            nm = c.get("name","").strip()
            df = c.get("definition","").strip()
            cat_list.append(nm)
            cat_lines.append(f"{nm}: {df}")
        cats_str = "\n".join(cat_lines)
        cat_list_str = ", ".join(cat_list) + ", Unclassified"
        cat_instruction = f"""
We have these TAXONOMICAL categories:
{cats_str}

IMPORTANT:
- Keep multi-word model strings together if referencing the same trim (e.g. '911 Turbo').
- If 'Turbo' is forced induction hardware, CarPart; else part of VehicleModel.

Use only {cat_list_str}.
"""
    else:
        cat_instruction = """
Use these TAXONOMICAL categories:
- Brand
- CarPart
- VehicleMake
- VehicleModel
- YearRange
- PartNumber
- Colour
- Unclassified

'Turbo' might be VehicleModel if referencing e.g. 911 Turbo, or CarPart if hardware.
"""

    joined_kw = "\n".join(f"- {kw}" for kw in chunk_keywords)
    prompt = f"""
You have a list of automotive product titles. 
Split each into tokens, assign them to one category or Unclassified.

{cat_instruction}

Return strictly a JSON array:
[ 
  {{
    "keyword":"...",
    "tokens":[{{"text":"xx","category":"VehicleModel"}},...]
  }},
  ...
]

Titles:
{joined_kw}

No extra text.
"""
    raw_text = await call_gemini(prompt, max_tokens=2000)
    s_i = raw_text.find("[")
    e_i = raw_text.rfind("]")
    if s_i == -1 or e_i == -1:
        log("[Classifier] No JSON array found in chunk.")
        return []
    json_str = raw_text[s_i : e_i+1]
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
        else:
            log("[Classifier] JSON was neither list nor dict.")
            return []
    except Exception as e:
        log(f"[Classifier] JSON parse error: {e}")
        return []

async def classify_keywords_recursive(keywords, chunk_size=10, depth=0):
    """
    Classify the given list of keywords by chunking them into smaller subsets.
    If a chunk returns no data or fails, we split that chunk into halves
    and retry recursively.
    """
    if not keywords:
        return []

    if chunk_size < 1:
        chunk_size = 1

    results = []
    i = 0
    while i < len(keywords):
        chunk = keywords[i : i+chunk_size]
        log(f"[Recursive] depth={depth}, chunk {i} to {i+len(chunk)-1}, size={chunk_size}")
        chunk_res = await classify_chunk(chunk)
        if chunk_res:
            results.extend(chunk_res)
        else:
            log("[Recursive] No valid items in this chunk. Attempting to split further.")
            if chunk_size > 1:
                mid = len(chunk)//2
                first_half = chunk[:mid]
                second_half = chunk[mid:]
                r1 = await classify_keywords_recursive(first_half, max(1,chunk_size//2), depth+1)
                r2 = await classify_keywords_recursive(second_half, max(1,chunk_size//2), depth+1)
                results.extend(r1)
                results.extend(r2)
        i += chunk_size

    return results

def build_tokens_dataframe(items):
    rows = []
    for obj in items:
        kw = obj.get("keyword","")
        tokens = obj.get("tokens",[])
        for t in tokens:
            txt = t.get("text","")
            cat = t.get("category","")
            rows.append({"keyword": kw, "token": txt, "category": cat})
    return pd.DataFrame(rows, columns=["keyword","token","category"])

def display_classification_stats(df):
    st.write("### Token-Level Classification")
    st.dataframe(df)

    if df.empty:
        st.warning("No tokens found.")
        return

    cat_counts = df["category"].value_counts().reset_index()
    cat_counts.columns = ["Category","Count"]
    st.write("### Overall Category Frequency")
    st.dataframe(cat_counts)

    cat_chart = (
        alt.Chart(cat_counts)
        .mark_bar()
        .encode(
            x=alt.X("Category",sort=None),
            y="Count",
            tooltip=["Category","Count"]
        )
        .properties(width=600, height=300)
    )
    st.altair_chart(cat_chart)

    # Build sets of categories per phrase
    df_patterns = (
        df.groupby("keyword")["category"]
        .apply(lambda x: sorted(set(x)))
        .reset_index()
    )
    df_patterns.columns = ["keyword","CategorySet"]
    df_patterns["Pattern"] = df_patterns["CategorySet"].apply(lambda c: " + ".join(c) if c else "No Categories")
    pattern_counts = df_patterns["Pattern"].value_counts().reset_index()
    pattern_counts.columns = ["Pattern","Count"]

    st.write("### Most Common Category Combinations")
    st.dataframe(pattern_counts)

    pattern_chart = (
        alt.Chart(pattern_counts)
        .mark_bar()
        .encode(
            x=alt.X("Pattern",sort=None),
            y="Count",
            tooltip=["Pattern","Count"]
        )
        .properties(width=600, height=300)
    )
    st.altair_chart(pattern_chart)

# =======================================================
# GROUPING CONSECUTIVE VEHICLEMODEL TOKENS BEFORE REWRITE
# =======================================================
def group_consecutive_model_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges consecutive 'VehicleModel' tokens into one.
    E.g. '9-3' '2.8T' 'V6' => '9-3 2.8T V6'.
    """
    output_rows = []
    for kw, group_data in df.groupby("keyword"):
        group_data = group_data.sort_index()

        merged_tokens = []
        for _, row in group_data.iterrows():
            token = row["token"]
            cat = row["category"]
            if merged_tokens and cat == "VehicleModel" and merged_tokens[-1]["category"] == "VehicleModel":
                merged_tokens[-1]["token"] += " " + token
            else:
                merged_tokens.append({"token": token, "category": cat})

        for item in merged_tokens:
            output_rows.append({
                "keyword": kw,
                "token": item["token"],
                "category": item["category"]
            })

    return pd.DataFrame(output_rows, columns=["keyword","token","category"])


# ==========================================
# PHASE 3: REWRITE NEW TITLES
# ==========================================
def remove_consecutive_duplicates(tokens):
    cleaned = []
    for t in tokens:
        if not cleaned or cleaned[-1].lower() != t.lower():
            cleaned.append(t)
    return cleaned

def unify_parentheses(tokens):
    """
    If tokens appear like '(', 'stuff', ')' consecutively, unify them into '(stuff)'.
    Also remove trailing commas if a leftover token ends with ',' or '.,'
    """
    i = 0
    output = []
    while i < len(tokens):
        tk = tokens[i]
        if tk == "(" and i+2 < len(tokens) and tokens[i+2] == ")":
            new_token = "(" + tokens[i+1] + ")"
            output.append(new_token)
            i += 3
        else:
            if tk.endswith(","):
                tk = tk[:-1].strip()
            output.append(tk)
            i += 1
    return output

def restructure_title_from_tokens(token_cat_pairs):
    """
    Reorder tokens with minimal punctuation, merging leftover Colour/PartNumber with "| ".
    1) skip leftover dash tokens
    2) strip trailing dash if the leftover token is a Colour/PartNumber
    3) unify parentheses
    4) remove consecutive duplicates

    ADDITION: remove tokens if they are exactly 'for' (case-insensitive).
    """
    desired_order = ["Brand", "CarPart", "VehicleMake", "VehicleModel", "YearRange"]
    buckets = {o:[] for o in desired_order}
    leftover = []

    for token, cat in token_cat_pairs:
        # If the token is literally 'for' (case-insensitive), skip entirely
        if token.strip().lower() == "for":
            continue

        # If leftover token is literally '-'
        if token.strip() == "-":
            continue

        if cat in buckets:
            buckets[cat].append(token)
        else:
            leftover.append((token, cat))

    brand_str = " ".join(buckets["Brand"])
    part_str  = " ".join(buckets["CarPart"])
    make_str  = " ".join(buckets["VehicleMake"])
    model_str = " ".join(buckets["VehicleModel"])
    year_str  = " ".join(buckets["YearRange"])

    main_tokens = []
    if brand_str: main_tokens.append(brand_str)
    if part_str:  main_tokens.append(part_str)
    if make_str:  main_tokens.append(make_str)
    if model_str: main_tokens.append(model_str)
    if year_str:  main_tokens.append(year_str)

    base_string = " ".join(main_tokens).strip()

    final_segments = [base_string] if base_string else []

    leftover_fixed = []
    for (tok, cat) in leftover:
        # If cat is Colour or PartNumber, remove trailing dash if present
        if cat in ["Colour", "PartNumber"]:
            tok = tok.rstrip("-")
            leftover_fixed.append(f"| {tok}")
        else:
            tok = tok.rstrip("-")
            leftover_fixed.append(tok)

    leftover_str = " ".join(x for x in leftover_fixed if x.strip())

    raw_string = (base_string + " " + leftover_str).strip() if base_string else leftover_str

    splitted = raw_string.split()
    splitted = unify_parentheses(splitted)
    splitted = remove_consecutive_duplicates(splitted)

    return " ".join(splitted).strip()

async def classify_and_rewrite(titles_list):
    if not titles_list:
        return pd.DataFrame(columns=["original_title","restructured_title"])

    items = await classify_keywords_recursive(titles_list, chunk_size=10)
    df_tokens = build_tokens_dataframe(items)

    df_tokens = group_consecutive_model_tokens(df_tokens)

    restructured_rows = []
    for kw, group_data in df_tokens.groupby("keyword"):
        tlist = list(zip(group_data["token"], group_data["category"]))
        new_title = restructure_title_from_tokens(tlist)
        restructured_rows.append({
            "original_title": kw,
            "restructured_title": new_title
        })

    return pd.DataFrame(restructured_rows)

# ===========================================
# PHASE 4: FINAL PASS - PUNCTUATION POLISH
# ===========================================
async def punctuation_polish_chunk(chunk_titles):
    if not chunk_titles:
        return []

    bullet_list = "\n".join(f"- {t}" for t in chunk_titles)
    prompt = f"""
We have final product titles that may need minor punctuation or spacing fixes.
You must NOT add or remove any words, only fix punctuation, spacing, or capitalization.

Return strictly as a JSON array:
[
  {{
    "original": "...",
    "polished": "..."
  }},
  ...
]

Titles:
{bullet_list}

No extra text.
"""
    raw_text = await call_gemini(prompt, max_tokens=2000)
    s_i = raw_text.find("[")
    e_i = raw_text.rfind("]")
    if s_i == -1 or e_i == -1:
        log("[PunctuationPolish] No JSON array found.")
        return []

    js_str = raw_text[s_i:e_i+1]
    try:
        parsed = json.loads(js_str)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
        else:
            log("[PunctuationPolish] JSON was neither list nor dict.")
            return []
    except Exception as e:
        log(f"[PunctuationPolish] JSON parse error: {e}")
        return []

async def punctuation_polish_recursive(titles, chunk_size=10, depth=0):
    if not titles:
        return []

    if chunk_size<1:
        chunk_size=1

    results=[]
    i=0
    while i<len(titles):
        chunk=titles[i:i+chunk_size]
        log(f"[PolishRecursive] depth={depth}, chunk {i}-{i+len(chunk)-1}")
        chunk_res=await punctuation_polish_chunk(chunk)
        if chunk_res:
            validated=[]
            for obj in chunk_res:
                orig=obj.get("original","")
                pol=obj.get("polished","")
                # check tokens
                if sorted(orig.split())==sorted(pol.split()):
                    validated.append(obj)
                else:
                    log(f"[PolishValidation] mismatch for '{orig}'. Reverting.")
                    validated.append({"original":orig,"polished":orig})
            results.extend(validated)
        else:
            log("[PolishRecursive] chunk returned empty. Splitting if possible.")
            if chunk_size>1:
                mid=len(chunk)//2
                fh=chunk[:mid]
                sh=chunk[mid:]
                r1=await punctuation_polish_recursive(fh, max(1,chunk_size//2), depth+1)
                r2=await punctuation_polish_recursive(sh, max(1,chunk_size//2), depth+1)
                results.extend(r1)
                results.extend(r2)
        i+=chunk_size
    return results

# =====================
# STREAMLIT UI (MAIN)
# =====================
def main():
    st.header("Phase 1: Suggest Taxonomical Categories (Optional)")
    domain_box1, domain_box2 = st.columns([3,1])
    with domain_box1:
        domain_context=st.text_input("Domain Context (e.g. 'car parts')")
    with domain_box2:
        if st.button("Suggest Categories"):
            if domain_context.strip():
                cats=suggest_categories(domain_context)
                if cats:
                    st.session_state.approved_categories=cats
                    st.success("Categories suggested!")
                else:
                    st.warning("No suggestions returned. Check logs.")
            else:
                st.warning("Provide domain context first.")

    if st.session_state.approved_categories:
        st.write("**Approved Categories**")
        df_cats=pd.DataFrame(st.session_state.approved_categories)
        st.dataframe(df_cats)
        st.write("Edit JSON if needed:")
        def_json=json.dumps(st.session_state.approved_categories,indent=2)
        jtext=st.text_area("Approved Categories",def_json)
        if st.button("Update Approved Cats"):
            try:
                new_cats=json.loads(jtext)
                if isinstance(new_cats,list):
                    st.session_state.approved_categories=new_cats
                    st.success("Updated categories.")
                else:
                    st.warning("Must be JSON array.")
            except Exception as e:
                st.error(f"JSON parse error: {e}")
    else:
        st.info("No user-defined categories. We'll fallback to default set if needed.")

    st.header("Phase 2: Classify & Rank Most Common Combos")
    kw_input=st.text_area("Enter product titles to classify (one per line)")
    if st.button("Classify Keywords"):
        lines=[l.strip() for l in kw_input.split("\n") if l.strip()]
        if not lines:
            st.warning("Please provide lines first.")
        else:
            st.write("Classifying with Gemini...")
            items=asyncio.run(classify_keywords_recursive(lines, chunk_size=10))
            if items:
                df_tokens=build_tokens_dataframe(items)
                st.session_state.df_tokens=df_tokens
                display_classification_stats(df_tokens)
            else:
                st.warning("No results or parse error. Check logs.")

    st.subheader("Process Log (Classification)")
    st.text_area("", st.session_state.log_text, height=150, key="log_classify")

    st.header("Phase 3: Rewrite Titles (No dash after colour, parentheses fix, consecutive duplicates)")
    rewrite_input=st.text_area("Enter new page titles (one per line) to restructure")
    if st.button("Rewrite Titles"):
        lines2=[l.strip() for l in rewrite_input.split("\n") if l.strip()]
        if not lines2:
            st.warning("No lines to rewrite.")
        else:
            st.write("Classifying & rewriting in code...")
            df_rewrites=asyncio.run(classify_and_rewrite(lines2))
            st.session_state.df_rewrites=df_rewrites
            st.write("**Programmatically Restructured Titles**")
            st.dataframe(df_rewrites)
    else:
        st.info("Click 'Rewrite Titles' to see restructured output.")

    st.header("Phase 4: Final Punctuation Polish (Optional)")
    if st.session_state.df_rewrites.empty:
        st.info("Complete Phase 3 first.")
    else:
        st.write("Below are the titles to polish. They must keep the same tokens.")
        st.dataframe(st.session_state.df_rewrites)
        if st.button("Polish Punctuation"):
            lines_to_polish=st.session_state.df_rewrites["restructured_title"].tolist()
            polished_res=asyncio.run(punctuation_polish_recursive(lines_to_polish, chunk_size=10))
            pol_rows=[]
            for obj in polished_res:
                orig=obj.get("original","")
                pol=obj.get("polished","")
                pol_rows.append({"original":orig,"polished":pol})
            df_pol=pd.DataFrame(pol_rows)
            st.session_state.df_polished=df_pol
            st.write("**Polished Titles** (Gemini punctuation fix, with token checks)")
            st.dataframe(df_pol)
    st.subheader("Process Log (Overall)")
    st.text_area("", st.session_state.log_text, height=200, key="log_overall")


if __name__ == "__main__":
    main()
