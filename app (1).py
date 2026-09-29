"""
app.py - Streamlit app for Mumbai House Price Prediction
Run locally:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

BUNDLE_PATH = "models/bundle.joblib"
FEATURES_CAT = ["locality", "region", "status", "furnishing"]

st.set_page_config(page_title="Mumbai House Price Predictor", page_icon="🏙️", layout="centered")


@st.cache_resource(show_spinner="Training models for the first time (about a minute)...")
def load_bundle():
    if not os.path.exists(BUNDLE_PATH):
        import train_model
        train_model.main()          # trains everything if models/ is missing (e.g. first run on Streamlit Cloud)
    return joblib.load(BUNDLE_PATH)


def fmt_price(lakh: float) -> str:
    """Show Rs lakh below 1 crore, Rs crore above."""
    return f"₹ {lakh / 100:.2f} Cr" if lakh >= 100 else f"₹ {lakh:.1f} Lakh"


bundle = load_bundle()
models, results = bundle["models"], bundle["results"]
loc_region, loc_psf = bundle["loc_region"], bundle["loc_psf"]

st.title("🏙️ Mumbai House Price Predictor")
st.caption("Predict the market price of a flat using Linear Regression, Gradient Boosting and a Deep Neural Network.")

# ---------------- Inputs ----------------
locality = st.selectbox("Locality", sorted(loc_region), index=sorted(loc_region).index("Andheri West"))
c1, c2 = st.columns(2)
bhk = c1.selectbox("BHK", [1, 2, 3, 4, 5], index=1)
area = c2.number_input("Area (sq ft)", 250, 5000, 750, step=25)
c3, c4 = st.columns(2)
total_floors = c3.number_input("Total floors in building", 1, 70, 14)
floor = c4.number_input("Floor number", 1, 70, 6)
c5, c6 = st.columns(2)
status = c5.selectbox("Status", ["Ready to Move", "Under Construction"])
furnishing = c6.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Furnished"])
c7, c8 = st.columns(2)
age = c7.slider("Age of building (years)", 0, 45, 5, disabled=(status == "Under Construction"))
parking = c8.selectbox("Parking spots", [0, 1, 2], index=1)
amenities = st.slider("Amenities score (0 = none, 10 = clubhouse, pool, gym, etc.)", 0, 10, 5)
dist = st.slider("Distance to nearest railway/metro station (km)", 0.1, 9.0, 1.2, step=0.1)
sea = st.checkbox("Sea facing")
model_name = st.radio("Model", list(models), index=1, horizontal=False)

if floor > total_floors:
    st.warning("Floor number cannot be more than total floors.")

if st.button("Predict price", type="primary", use_container_width=True, disabled=floor > total_floors):
    row = pd.DataFrame([{
        "locality": locality, "region": loc_region[locality], "status": status, "furnishing": furnishing,
        "bhk": bhk, "area_sqft": area, "floor": floor, "total_floors": total_floors,
        "age_years": 0 if status == "Under Construction" else age, "parking": parking,
        "amenities_score": amenities, "dist_station_km": dist, "sea_facing": int(sea)}])

    preds = {n: float(np.exp(m.predict(row)[0])) for n, m in models.items()}
    main_pred = preds[model_name]
    lo, hi = bundle["interval"]["low"], bundle["interval"]["high"]

    st.divider()
    st.subheader(f"{locality}  ·  {bhk} BHK  ·  {area} sq ft")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted price", fmt_price(main_pred))
    m2.metric("Price per sq ft", f"₹ {main_pred * 1e5 / area:,.0f}")
    m3.metric("Locality average", f"₹ {loc_psf[locality]:,.0f}/sq ft")
    gb = preds["Gradient Boosting"]
    st.info(f"Likely range (80% of test predictions fall inside): **{fmt_price(gb * lo)} – {fmt_price(gb * hi)}**")

    st.markdown("**All three models side by side**")
    st.dataframe(pd.DataFrame({"Model": list(preds), "Predicted price": [fmt_price(v) for v in preds.values()],
                               "Rs lakh": [round(v, 1) for v in preds.values()]}),
                 hide_index=True, use_container_width=True)
    st.bar_chart(pd.Series(preds, name="Rs lakh"))

# ---------------- Model info ----------------
st.divider()
st.subheader("Model performance (held-out test set)")
st.dataframe(results.round(3), use_container_width=True)
st.caption(f"Trained on {bundle['n_train']} listings, tested on {bundle['n_test']}. "
           "R² and error metrics are used because this is a regression problem (continuous target).")

with st.expander("What drives the price? (feature importance)"):
    imp = bundle["importance"].sort_values("importance", ascending=False).set_index("feature")
    st.bar_chart(imp["importance"])
    if os.path.exists("figures/fig7_actual_vs_predicted.png"):
        st.image("figures/fig7_actual_vs_predicted.png", caption="Actual vs predicted prices")

st.caption("⚠️ Educational project. Predictions come from a model trained on the dataset in /data and are estimates, "
           "not valuations or financial advice.")
