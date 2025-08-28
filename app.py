# ----------------- IMPORTS (os must be imported before os.getenv) -----------------
import os
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ----------------- STREAMLIT SETUP (must be the first st.* call) -----------------
st.set_page_config(page_title="US University Recommender", page_icon="🎓", layout="wide")

# ----------------- ENV / SECRETS BOOTSTRAP -----------------
# Load .env for local runs
load_dotenv()

# On Streamlit Cloud, st.secrets exists; locally it may not.
# Try to read SCORECARD_API_KEY from st.secrets (harmless if unavailable).
try:
    key_from_secrets = st.secrets.get("SCORECARD_API_KEY", None)
except Exception:
    key_from_secrets = None

if key_from_secrets:
    os.environ["SCORECARD_API_KEY"] = key_from_secrets

# Fail fast with a clear message if the key is still missing.
if not os.getenv("SCORECARD_API_KEY"):
    st.error(
        "SCORECARD_API_KEY not found — add it in Streamlit Cloud **Settings → Secrets** "
        'as `SCORECARD_API_KEY="YOUR_REAL_KEY"` or create a local `.env` file with '
        "`SCORECARD_API_KEY=YOUR_REAL_KEY`."
    )
    st.stop()

# ----------------- PROJECT IMPORTS (after key is set) -----------------
from src.scorecard_api import ScorecardClient
# Try to import optional broadened search helpers; fall back safely if not present.
try:
    from src.cip_map import CIP_MAP, CIP_BUNDLES, TITLE_SYNONYMS
except Exception:
    from src.cip_map import CIP_MAP
    CIP_BUNDLES, TITLE_SYNONYMS = {}, {}

from src.lor_sop_nlp import score_document
from src.acceptance import acceptance_probability
from src.utils import USD, safe

# ----------------- BROADENED SEARCH HELPERS -----------------
def _dedup_results(rows):
    """Deduplicate results by (school.name, state)."""
    seen = set()
    out = []
    for r in rows:
        key = (r.get("school.name"), r.get("school.state"))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out

def broaden_search(client, *, state, city, zip_radius, course, first_results, cip4, max_pages=3):
    """
    If initial results are sparse, try:
      1) Related CIP bundle for the course (if defined)
      2) Title 'contains' synonyms (broadest)
    """
    if len(first_results) >= 20:
        return first_results

    all_rows = list(first_results)
    key = (course or "").strip().lower()

    # 1) CIP bundle candidates (skip the CIP we already used)
    cip_candidates = []
    if key in CIP_BUNDLES:
        cip_candidates = [c for c in CIP_BUNDLES[key] if c != cip4]

    for c in cip_candidates:
        for page in range(max_pages):
            data = client.search(state=state, city=city, zip_radius=zip_radius, cip4=c, page=page)
            rs = data.get("results", [])
            all_rows.extend(rs)
            if len(rs) < 100:
                break

    all_rows = _dedup_results(all_rows)
    if len(all_rows) >= 20:
        return all_rows

    # 2) Title synonyms (broadest)
    synonyms = TITLE_SYNONYMS.get(key, [course]) if course else []
    for kw in synonyms:
        for page in range(max_pages):
            data = client.search(state=state, city=city, zip_radius=zip_radius, title_contains=kw, page=page)
            rs = data.get("results", [])
            all_rows.extend(rs)
            if len(rs) < 100:
                break

    return _dedup_results(all_rows)

# ----------------- UI -----------------
st.title("🎓 US University Recommender – Real Data + NLP")

with st.sidebar:
    st.header("Your Profile")
    name = st.text_input("Name", placeholder="Your name")
    course = st.text_input("Preferred course / major", value="Data Science")

    loc_mode = st.radio("Location filter", ["State", "City", "ZIP radius", "Anywhere"], index=0)
    state = city = zipc = radius = None
    if loc_mode == "State":
        state = st.text_input("State (2-letter)", value="TX", max_chars=2)
    elif loc_mode == "City":
        city = st.text_input("City", value="Dallas")
        state = st.text_input("State (2-letter)", value="TX", max_chars=2)
    elif loc_mode == "ZIP radius":
        zipc = st.text_input("ZIP code", value="75201")
        radius = st.selectbox("Distance", ["10mi", "25mi", "50mi", "100mi"], index=1)

    st.markdown("---")
    st.subheader("Budget (2 years)")
    budget = st.number_input("Total budget for 2 years (USD)", min_value=10000, step=1000, value=80000)

    st.markdown("---")
    st.subheader("Academics")
    cgpa = st.number_input("CGPA (out of 4.0)", min_value=0.0, max_value=4.0, step=0.01, value=3.6)
    gre = st.number_input("GRE (optional)", min_value=0, max_value=340, step=1, value=320)
    ielts = st.number_input("IELTS (optional)", min_value=0.0, max_value=9.0, step=0.5, value=7.0)

    st.markdown("---")
    st.subheader("LOR / SOP Upload (PDF or .txt)")
    lor_sop_kind = st.radio("Document type", ["SOP", "LOR"], index=0)
    upload = st.file_uploader("Upload LOR/SOP (PDF or .txt)", type=["pdf", "txt"])

    st.markdown("---")
    st.caption("Costs use College Scorecard tuition + on-campus room/board + other on-campus expenses; books shown separately.")

if "go" not in st.session_state:
    st.session_state["go"] = False

colL, colR = st.columns([1, 2])
with colL:
    if st.button("🔎 Find Universities", use_container_width=True):
        st.session_state["go"] = True
with colR:
    st.info("Tip: you can export results at the bottom as CSV.")

# ----------------- SEARCH & RESULTS -----------------
if st.session_state.get("go"):
    client = ScorecardClient()

    # Initial fetch (exact CIP if known, else title search)
    key = (course or "").strip().lower()
    cip4 = CIP_MAP.get(key)
    title_contains = None if cip4 else course
    zip_radius = (zipc, radius) if (zipc and radius) else None

    results = []
    for page in range(0, 3):
        data = client.search(
            state=state,
            city=city,
            zip_radius=zip_radius,
            cip4=cip4,
            title_contains=title_contains,
            page=page,
        )
        rs = data.get("results", [])
        results.extend(rs)
        if len(rs) < 100:
            break

    # Broaden if too few (related CIPs + title synonyms)
    results = broaden_search(
        client,
        state=state,
        city=city,
        zip_radius=zip_radius,
        course=course,
        first_results=results,
        cip4=cip4,
        max_pages=3,
    )

    if not results:
        st.warning("No matching institutions found. Try broadening location or course.")
        st.stop()

    # LOR/SOP NLP score (optional)
    lor_score = 0.5
    lor_details = {}
    if upload is not None:
        bytes_data = upload.read()
        lor_score, lor_details = score_document(bytes_data, upload.name, kind_hint=lor_sop_kind.lower())

    # Build table
    rows = []
    for r in results:
        name_ = safe(r.get("school.name"), "")
        city_ = safe(r.get("school.city"), "")
        state_ = safe(r.get("school.state"), "")
        url_ = safe(r.get("school.school_url"), "")

        base_rate = r.get("latest.admissions.admission_rate.overall")
        in_tuition = r.get("latest.cost.tuition.in_state") or 0
        out_tuition = r.get("latest.cost.tuition.out_of_state") or 0
        room = r.get("latest.cost.roomboard.oncampus") or 0
        other = r.get("latest.cost.other_on_campus") or 0
        books = r.get("latest.cost.booksupply") or 0

        tuition = out_tuition if out_tuition else in_tuition
        living = (room or 0) + (other or 0)
        per_year = (tuition or 0) + living
        two_year = 2 * per_year
        meets_budget = two_year <= budget

        acc = acceptance_probability(
            base_rate, cgpa=cgpa, gre=(gre or None), ielts=(ielts or None), lor_sop=lor_score
        )

        rows.append(
            {
                "Institution": name_,
                "City": city_,
                "State": state_,
                "URL": url_,
                "Tuition (yr)": tuition,
                "Living (yr)": living,
                "Books (yr)": books,
                "Total (yr)": per_year,
                "Total (2y)": two_year,
                "Baseline admit": base_rate,
                "Your admit %": round(100 * acc, 1),
                "Within budget?": "✅" if meets_budget else "—",
            }
        )

    df = pd.DataFrame(rows)
    df.sort_values(["Within budget?", "Total (2y)", "Your admit %"], ascending=[False, True, False], inplace=True)

    st.subheader("Results")
    st.dataframe(
        df.assign(
            **{
                "Tuition (yr)": df["Tuition (yr)"].map(lambda x: USD.format(x) if x else "—"),
                "Living (yr)": df["Living (yr)"].map(lambda x: USD.format(x) if x else "—"),
                "Books (yr)": df["Books (yr)"].map(lambda x: USD.format(x) if x else "—"),
                "Total (yr)": df["Total (yr)"].map(lambda x: USD.format(x) if x else "—"),
                "Total (2y)": df["Total (2y)"].map(lambda x: USD.format(x) if x else "—"),
                "Baseline admit": df["Baseline admit"].map(lambda x: f"{x*100:.1f}%" if x else "—"),
            }
        ),
        use_container_width=True,
        height=520,
    )

    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="uni_recommendations.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("How the estimates are calculated"):
        st.markdown(
            """
            **Costs (per year)** = tuition (preferring out-of-state if available) + on-campus room & board + other on-campus expenses. 
            Books & supplies shown separately. **2-year total** simply multiplies by 2.

            **Acceptance probability** starts from the institution's latest undergraduate admission rate, then shifts using your profile (GPA, GRE, IELTS, and LOR/SOP NLP score). 
            This is a heuristic; program-level graduate admit rates are not published centrally, so treat the result as a relative indicator, not a guarantee.
            """
        )

    if lor_details:
        with st.expander("LOR/SOP NLP details"):
            st.json(lor_details)
