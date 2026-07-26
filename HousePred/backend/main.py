from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="House Price Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Load model + preprocessing artifacts once at startup
# ---------------------------------------------------------------------
try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    default_values = joblib.load("default_values.pkl")
    skewed_features = set(joblib.load("skewed_features.pkl"))
except FileNotFoundError as e:
    raise RuntimeError(
        "Model artifacts not found. Make sure best_model.pkl, scaler.pkl, "
        "feature_columns.pkl, default_values.pkl and skewed_features.pkl "
        "all exist in this folder."
    ) from e


# ---------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------
class HouseFeatures(BaseModel):
    OverallQual: int = Field(..., ge=1, le=10)
    GrLivArea: float = Field(..., gt=0)
    GarageCars: int = Field(..., ge=0, le=5)
    TotalBsmtSF: float = Field(..., ge=0)
    FullBath: int = Field(..., ge=0, le=5)
    TotRmsAbvGrd: int = Field(..., ge=1, le=20)
    YearBuilt: int = Field(..., ge=1870, le=2026)
    YearRemodAdd: int = Field(..., ge=1870, le=2026)


class FeatureImpact(BaseModel):
    feature: str
    label: str
    impact: (
        float  # dollar effect on price; positive = raised price, negative = lowered it
    )


class PredictionResponse(BaseModel):
    predicted_price: float
    baseline_price: float  # price of a "typical" house (all fields at median)
    explanation: list[FeatureImpact]
    summary: str  # short natural-language paragraph explaining the price


QUALITY_LABELS = {
    10: "Very Excellent",
    9: "Excellent",
    8: "Very Good",
    7: "Good",
    6: "Above Average",
    5: "Average",
    4: "Below Average",
    3: "Fair",
    2: "Poor",
    1: "Very Poor",
}

FIELD_DISPLAY_NAMES = {
    "OverallQual": "Overall Quality",
    "GrLivArea": "Living Area",
    "GarageCars": "Garage Capacity",
    "TotalBsmtSF": "Basement Area",
    "FullBath": "Full Bathrooms",
    "TotRmsAbvGrd": "Total Rooms",
    "YearBuilt": "Year Built",
    "YearRemodAdd": "Year Remodeled",
}


def build_row(values: dict) -> pd.DataFrame:
    """Take a dict of the 8 user-facing fields (raw scale), apply the same
    skew-transform used in training, merge onto the default template, and
    return a single-row dataframe in the exact column order the model expects."""
    transformed = {}
    for col, val in values.items():
        transformed[col] = np.log1p(val) if col in skewed_features else val

    row = default_values.copy()
    row.update(transformed)

    df = pd.DataFrame([row])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]


def predict_price(values: dict) -> float:
    row = build_row(values)
    scaled = scaler.transform(row)
    pred_log = model.predict(scaled)[0]
    return float(np.expm1(pred_log))


def format_value(field: str, value) -> str:
    if field == "OverallQual":
        return QUALITY_LABELS.get(int(value), str(value))
    if field in ("GrLivArea", "TotalBsmtSF"):
        return f"{int(value):,} sq ft"
    return str(value)


def format_dollars(value: float) -> str:
    return f"${abs(value):,.0f}"


def build_summary(
    full_price: float, baseline_price: float, explanation: list[FeatureImpact]
) -> str:
    """Turn the predicted price + sorted feature impacts into a short,
    human-readable paragraph explaining why the price came out the way it did."""
    diff = full_price - baseline_price
    direction = "above" if diff >= 0 else "below"

    positives = [e for e in explanation if e.impact > 0]
    negatives = [e for e in explanation if e.impact < 0]

    lead = (
        f"This house is predicted at {format_dollars(full_price)}, "
        f"{format_dollars(diff)} {direction} a typical home "
        f"({format_dollars(baseline_price)})."
    )

    parts = []
    if positives:
        top_pos = positives[:2]
        joined = " and ".join(
            f"{FIELD_DISPLAY_NAMES[p.feature]} (+{format_dollars(p.impact)})"
            for p in top_pos
        )
        parts.append(f"The biggest boosts come from {joined}.")

    if negatives:
        top_neg = negatives[:2]
        joined = " and ".join(
            f"{FIELD_DISPLAY_NAMES[n.feature]} (-{format_dollars(n.impact)})"
            for n in top_neg
        )
        parts.append(f"On the downside, {joined} pulled the price down.")

    if not positives and not negatives:
        parts.append(
            "Every feature is close to typical, so the price sits near the baseline."
        )

    return " ".join([lead] + parts)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "House Price Prediction API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: HouseFeatures):
    try:
        user_values = data.dict()

        # Full prediction with everything the user entered
        full_price = predict_price(user_values)

        # Baseline: a completely "typical" house (every field at its median)
        baseline_row = default_values.copy()
        baseline_df = pd.DataFrame([baseline_row])
        for col in feature_columns:
            if col not in baseline_df.columns:
                baseline_df[col] = 0
        baseline_df = baseline_df[feature_columns]
        baseline_price = float(
            np.expm1(model.predict(scaler.transform(baseline_df))[0])
        )

        # Leave-one-out: for each field, reset just that one to its default
        # (keeping the other 7 at the user's values) and see how much the
        # price drops. full_price - that_price = this field's contribution.
        explanation = []
        for field in user_values:
            reduced_values = user_values.copy()
            del reduced_values[field]  # this field falls back to the default template
            price_without = predict_price(reduced_values)
            impact = round(full_price - price_without, 2)

            explanation.append(
                FeatureImpact(
                    feature=field,
                    label=f"{FIELD_DISPLAY_NAMES[field]} ({format_value(field, user_values[field])})",
                    impact=impact,
                )
            )

        explanation.sort(key=lambda x: abs(x.impact), reverse=True)

        summary = build_summary(full_price, baseline_price, explanation)

        return PredictionResponse(
            predicted_price=round(full_price, 2),
            baseline_price=round(baseline_price, 2),
            explanation=explanation,
            summary=summary,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
