
class Constants:
    VETTING_SCHEMA = {
        "planet_candidate": bool,
        "confidence_score": float,
        "issues": list,
        "recommendations": list
    }
    R_EARTH_OVER_R_SUN = 0.0091577

    SystemMessage = f"You are an exoplanet vetting assistant. "
    "Given light-curve summaries, extracted features (TSFresh/XGBoost), and basic checks "
    "(odd-even depths, centroid offset, secondary eclipse, duration/period sanity), "
    "return ONLY a JSON object matching the provided schema. Be concise."