import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fortitudo', 'tech'))
sys.path.insert(0, ROOT)
import re
import subprocess
from typing import List, Tuple, Dict
import pandas as pd

from fred_data import fetch_series_from_map, save_spreadsheet

PDF_PATH = "data/25-26-WGHIC-Approved-Treasury-Bonds-List.pdf"
TXT_PATH = "data/treasuries_identifiers.txt"
OUT_IDS_CSV = "data/treasuries_identifiers.csv"
OUT_MAP_CSV = "data/fred_mapping_treasuries.csv"
OUT_DATA_CSV = "data/fred_treasuries_timeseries.csv"
OUT_DATA_XLSX = "data/fred_treasuries_timeseries.xlsx"
START = "2016-01-01"
END = None


TENOR_MAP: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(1[\s-]*month|1[\s-]*mo|4[\s-]*week|4[\s-]*wk|1-?mo|4-?wk)\b", re.I), "DGS1MO"),
    (re.compile(r"\b(2[\s-]*month|2[\s-]*mo|8[\s-]*week|8[\s-]*wk|2-?mo|8-?wk)\b", re.I), "DGS2MO"),
    (re.compile(r"\b(3[\s-]*month|3[\s-]*mo|13[\s-]*week|13[\s-]*wk|3-?mo|13-?wk)\b", re.I), "DGS3MO"),
    (re.compile(r"\b(6[\s-]*month|6[\s-]*mo|26[\s-]*week|26[\s-]*wk|6-?mo|26-?wk)\b", re.I), "DGS6MO"),
    (re.compile(r"\b(1[\s-]*year|1[\s-]*yr|12[\s-]*month|1-?yr)\b", re.I), "DGS1"),
    (re.compile(r"\b(2[\s-]*year|2[\s-]*yr|2-?yr)\b", re.I), "DGS2"),
    (re.compile(r"\b(3[\s-]*year|3[\s-]*yr|3-?yr)\b", re.I), "DGS3"),
    (re.compile(r"\b(5[\s-]*year|5[\s-]*yr|5-?yr)\b", re.I), "DGS5"),
    (re.compile(r"\b(7[\s-]*year|7[\s-]*yr|7-?yr)\b", re.I), "DGS7"),
    (re.compile(r"\b(10[\s-]*year|10[\s-]*yr|10-?yr)\b", re.I), "DGS10"),
    (re.compile(r"\b(20[\s-]*year|20[\s-]*yr|20-?yr)\b", re.I), "DGS20"),
    (re.compile(r"\b(30[\s-]*year|30[\s-]*yr|30-?yr)\b", re.I), "DGS30"),
]


def ensure_pdftotext() -> None:
    try:
        subprocess.run(["pdftotext", "-v"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except FileNotFoundError:
        raise SystemExit("pdftotext not found. Please install 'poppler-utils' and retry.")


def pdf_to_text(pdf_path: str, txt_path: str) -> None:
    ensure_pdftotext()
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    subprocess.run(["pdftotext", "-layout", "-nopgbrk", pdf_path, txt_path], check=True)


def extract_identifiers_and_lines(text: str) -> Tuple[List[str], List[str], List[str]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # CUSIP: 9 alphanumeric
    cusips = sorted(set(re.findall(r"\b[0-9A-Z]{9}\b", text)))
    # ISIN: starts with 'US' then 10 alphanum
    isins = sorted(set(re.findall(r"\bUS[0-9A-Z]{10}\b", text)))
    return cusips, isins, lines


def infer_tenor_series(line: str) -> str:
    for pattern, series_id in TENOR_MAP:
        if pattern.search(line):
            return series_id
    return ""  # unknown; will be skipped from mapping


def build_mapping_from_lines(cusips: List[str], isins: List[str], lines: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    # For each line mentioning a tenor, create a mapping row and associate identifiers best-effort
    for ln in lines:
        fred_id = infer_tenor_series(ln)
        if not fred_id:
            continue
        # pick out identifiers present in this line to create symbols
        found_cusips = re.findall(r"\b[0-9A-Z]{9}\b", ln)
        found_isins = re.findall(r"\bUS[0-9A-Z]{10}\b", ln)
        syms = found_cusips + found_isins
        if not syms:
            # create a synthetic symbol from tenor text
            title = re.sub(r"\s+", "_", ln[:50])
            syms = [f"TENOR_{fred_id}_{title}"]
        for sym in syms:
            rows.append({
                "symbol": sym,
                "fred_series_id": fred_id,
                "frequency": "",
                "aggregation_method": "",
            })
    # Fallback: if no lines had tenor but we do have identifiers, map all identifiers to a generic set (10Y)
    if not rows and (cusips or isins):
        for sym in cusips + isins:
            rows.append({"symbol": sym, "fred_series_id": "DGS10", "frequency": "", "aggregation_method": ""})
    # Deduplicate
    map_df = pd.DataFrame(rows).drop_duplicates()
    return map_df


def build_country_mapping(lines: List[str], fred_series_id: str) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for ln in lines:
        found_cusips = re.findall(r"\b[0-9A-Z]{9}\b", ln)
        found_isins = re.findall(r"\b[A-Z]{2}[0-9A-Z]{10}\b", ln)
        syms = found_cusips + found_isins
        if not syms:
            title = re.sub(r"\s+", "_", ln[:60])
            syms = [f"{fred_series_id}_{title}"]
        for sym in syms:
            rows.append({
                "symbol": sym,
                "fred_series_id": fred_series_id,
                "frequency": "",
                "aggregation_method": "",
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Convert PDF to text
    pdf_to_text(PDF_PATH, TXT_PATH)
    with open(TXT_PATH, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    cusips, isins, lines = extract_identifiers_and_lines(text)

    # Save identifiers
    os.makedirs(os.path.dirname(OUT_IDS_CSV), exist_ok=True)
    ids_df = pd.DataFrame({
        "CUSIP": cusips,
    })
    if isins:
        # align lengths for CSV output convenience
        maxlen = max(len(cusips), len(isins))
        cusips_ext = cusips + [""] * (maxlen - len(cusips))
        isins_ext = isins + [""] * (maxlen - len(isins))
        ids_df = pd.DataFrame({"CUSIP": cusips_ext, "ISIN": isins_ext})
    ids_df.to_csv(OUT_IDS_CSV, index=False)

    # Filter by country blocks from the text
    lines_us = [ln for ln in lines if "United States" in ln or "United States Treasury" in ln]
    lines_uk = [ln for ln in lines if ln.startswith("United Kingdom") or "Gilt" in ln]
    lines_de = [ln for ln in lines if ln.startswith("Deutschland") or "BUND" in ln]
    lines_fr = [ln for ln in lines if ln.startswith("Obligation Assimilable du Trésor") or "OAT" in ln]
    lines_it = [ln for ln in lines if ln.startswith("Buoni del Tesoro") or "BTP" in ln]
    lines_nl = [ln for ln in lines if ln.startswith("Dutch Treasury") or "DSL" in ln]

    # US mapping by tenor (DGS*)
    map_us = build_mapping_from_lines(cusips, isins, lines_us)
    # Non-US: use long-term government bond yield series (monthly) available in FRED
    # UK, Germany, France, Italy, Netherlands
    map_uk = build_country_mapping(lines_uk, "IRLTLT01GBM156N")
    map_de = build_country_mapping(lines_de, "IRLTLT01DEM156N")
    map_fr = build_country_mapping(lines_fr, "IRLTLT01FRM156N")
    map_it = build_country_mapping(lines_it, "IRLTLT01ITM156N")
    map_nl = build_country_mapping(lines_nl, "IRLTLT01NLM156N")

    map_df = pd.concat([map_us, map_uk, map_de, map_fr, map_it, map_nl], axis=0, ignore_index=True)
    map_df = map_df.drop_duplicates()
    map_df.to_csv(OUT_MAP_CSV, index=False)

    # Fetch time series using FRED API (requires FRED_API_KEY env var)
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise SystemExit("Please set FRED_API_KEY environment variable before running.")
    data_df = fetch_series_from_map(map_df, api_key=api_key, start=START, end=END)
    save_spreadsheet(data_df, OUT_DATA_CSV, OUT_DATA_XLSX)
    print("Identifiers saved:", OUT_IDS_CSV)
    print("Mapping saved:", OUT_MAP_CSV)
    print("FRED data saved:", OUT_DATA_CSV)
    if os.path.exists(OUT_DATA_XLSX):
        print("FRED data saved:", OUT_DATA_XLSX)
