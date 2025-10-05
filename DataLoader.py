
import json, re, math
from typing import List, Dict, Any, Optional, Tuple
from Core.Constants import Constants
import numpy as np
import pandas as pd
import requests



class VettingDataLoader:


    def __init__(self):
        self.R_EARTH_OVER_R_SUN = Constants.R_EARTH_OVER_R_SUN
        pass

    @staticmethod
    def _to_num(v):
        if v is None:
            return np.nan
        try:
            if isinstance(v, str):
                v = re.sub("<.*?>", "", v)  # strip HTML tags
            return float(v)
        except Exception:
            return np.nan

    def depth_from_row(self, x: dict) -> float:
        kd = self._to_num(x.get("koi_depth"))
        if not math.isnan(kd) and kd > 0:
            return kd / 1e6  # ppm -> fraction
        pdp = self._to_num(x.get("pl_trandep"))
        if not math.isnan(pdp) and pdp > 0:
            return pdp/1e6 if pdp > 0.5 else pdp
        rp_re = self._to_num(x.get("pl_rade"))
        rs_rsun = self._to_num(x.get("st_rad"))
        if not math.isnan(rp_re) and not math.isnan(rs_rsun) and rs_rsun > 0:
            rprs = (rp_re * self.R_EARTH_OVER_R_SUN) / rs_rsun
            return (rprs ** 2)
        return np.nan

    def duration_from_row(self, x: dict) -> float:
        d = self._to_num(x.get("koi_duration"))
        if not math.isnan(d):
            return d
        d = self._to_num(x.get("pl_trandurh"))
        return d

    def epoch_from_row(self, x: dict):
        bk = self._to_num(x.get("koi_time0bk"))
        if not math.isnan(bk):
            return {"system":"BKJD", "value": float(bk), "bjd": float(bk) + 2454833.0}
        tm = self._to_num(x.get("pl_tranmid"))
        if not math.isnan(tm):
            return {"system":"BJD", "value": float(tm)}
        return None

    def flags_from_row(self, x: dict):
        tags = []
        mapping = [("koi_fpflag_nt","not-transit-like"),
                   ("koi_fpflag_ss","stellar-eclipse"),
                   ("koi_fpflag_co","centroid-offset"),
                   ("koi_fpflag_ec","ephemeris-match")]
        for k, tag in mapping:
            v = self._to_num(x.get(k))
            if v == 1:
                tags.append(tag)
        return tags

    def make_lc_summary_components(self, x: dict):
        name = x.get("pl_name") if pd.notna(x.get("pl_name")) else x.get("hostname")
        mission = x.get("disc_facility") if pd.notna(x.get("disc_facility")) else "Unknown facility"
        disp = x.get("disposition") if isinstance(x.get("disposition"), str) else None

        P = self._to_num(x.get("pl_orbper")) if pd.notna(x.get("pl_orbper")) else self._to_num(x.get("koi_period"))
        depth_frac = self.depth_from_row(x)
        dur_h = self.duration_from_row(x)
        snr = self._to_num(x.get("koi_model_snr"))
        impact = self._to_num(x.get("koi_impact"))
        epoch = self.epoch_from_row(x)
        flags = self.flags_from_row(x)

        parts = []
        parts.append(mission if mission else "Unknown facility")
        parts.append(str(name) if name else "Unknown target")
        parts.append(f"P={P:.6g} d" if not math.isnan(P) else "P=n/a")
        parts.append(f"dur≈{dur_h:.3g} h" if not math.isnan(dur_h) else "dur=n/a")
        if not math.isnan(depth_frac):
            parts.append(f"depth≈{depth_frac*100:.3g}% (~{depth_frac*1e6:.0f} ppm)")
        else:
            parts.append("depth=n/a")
        if epoch:
            if epoch["system"]=="BKJD":
                parts.append(f"t0≈{epoch['value']:.5f} BKJD (~{epoch['bjd']:.5f} BJD)")
            else:
                parts.append(f"t0≈{epoch['value']:.5f} BJD")
        if not math.isnan(snr):
            parts.append(f"SNR≈{snr:.2f}")
        if not math.isnan(impact):
            parts.append(f"b≈{impact:.2f}")

        s = " | ".join(parts)
        if disp:
            s += f" | disp={disp}"
        if flags:
            s += f" | flags=" + ",".join(flags)

        fields = {
            "period_days": None if math.isnan(P) else float(P),
            "duration_hours": None if math.isnan(dur_h) else float(dur_h),
            "depth_fraction": None if (depth_frac is None or math.isnan(depth_frac)) else float(depth_frac),
            "depth_ppm": None if (depth_frac is None or math.isnan(depth_frac)) else float(depth_frac*1e6),
            "snr": None if math.isnan(snr) else float(snr),
            "impact": None if math.isnan(impact) else float(impact),
            "epoch": epoch,
            "flags": flags,
            "disposition": disp,
            "mission": mission,
            "name": name,
            "host": x.get("hostname"),
        }
        return s, fields

    def add_lc_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        out = []
        for _, row in df.iterrows():
            s, fields = self.make_lc_summary_components(row.to_dict())
            out.append((s, fields))
        df = df.copy()
        df["lc_summary"] = [s for s, _ in out]
        df["lc_fields"] = [f for _, f in out]
        return df