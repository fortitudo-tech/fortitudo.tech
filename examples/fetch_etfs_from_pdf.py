import os
import re
import subprocess
from typing import List, Set, Tuple

import numpy as np
import pandas as pd

# Reuse market data helpers without importing the full package (to avoid heavy deps)
import sys
MD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fortitudo', 'tech'))
sys.path.insert(0, MD_ROOT)
from market_data import fetch_prices, compute_returns, align_and_filter  # type: ignore


PDF_PATH = "data/25-26-WGHIC-Approved-ETF-List.pdf"
TXT_PATH = "data/etfs_identifiers.txt"
OUT_ETFS_CSV = "data/etfs_identifiers.csv"
OUT_PRICES_CSV = "data/etfs_prices.csv"
OUT_RETURNS_CSV = "data/etfs_returns.csv"
OUT_INVALID_CSV = "data/etfs_invalid.csv"
START = "2016-01-01"
END = None


def ensure_pdftotext() -> None:
	try:
		subprocess.run(["pdftotext", "-v"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
	except FileNotFoundError:
		raise SystemExit("pdftotext not found. Please install 'poppler-utils' and retry.")


def pdf_to_text(pdf_path: str, txt_path: str) -> None:
	ensure_pdftotext()
	os.makedirs(os.path.dirname(txt_path), exist_ok=True)
	subprocess.run(["pdftotext", "-layout", "-nopgbrk", pdf_path, txt_path], check=True)


STOPWORDS: Set[str] = {
	"ETF", "INDEX", "TRUST", "FUND", "INC", "PLC", "UCITS", "USD", "NAV", "ACC", "DIST",
	"CLASS", "SERIES", "SHARES", "LTD", "PLC", "PLC.", "ISHARES", "VANGUARD", "SPDR",
	"GLOBAL", "WORLD", "VALUE", "GROWTH", "INCOME", "DIVIDEND", "EQUITY", "BOND", "MARKET",
	"SECTOR", "CORE", "SELECT", "CAP", "MIDCAP", "SMALL", "LARGE", "ETF-OF-ETFS", "EMERGING",
}


def extract_tickers(text: str) -> List[str]:
	candidates: Set[str] = set()
	# 1) Strong patterns: parentheses and exchange-qualified
	for m in re.findall(r"\(([A-Z]{2,5})\)", text):
		candidates.add(m)
	for m in re.findall(r"\b(?:NYSEARCA|NASDAQ|NYSE|CBOE|AMEX)[:\s-]*([A-Z]{2,5})\b", text):
		candidates.add(m)

	# 2) Conservative tokens: 2-5 uppercase letters only, exclude common words/years
	tokens = re.findall(r"\b[A-Z]{2,5}\b", text)
	for t in tokens:
		if t in STOPWORDS:
			continue
		if re.fullmatch(r"\d{2,4}", t):
			continue
		candidates.add(t)

	# 3) Cleanup blacklist
	blacklist = {"CASH", "BOND", "FUNDS", "GRADE", "HEALTH", "ENERGY", "EUROPE"}
	tickers = sorted([c for c in candidates if c not in blacklist])
	return tickers


if __name__ == "__main__":
	# 1) Convert PDF to text
	pdf_to_text(PDF_PATH, TXT_PATH)
	with open(TXT_PATH, "r", encoding="utf-8", errors="ignore") as f:
		text = f.read()

	# 2) Extract tickers
	tickers = extract_tickers(text)
	os.makedirs(os.path.dirname(OUT_ETFS_CSV), exist_ok=True)
	pd.DataFrame({"ticker": tickers}).to_csv(OUT_ETFS_CSV, index=False)

	if not tickers:
		print("No ETF tickers found in the PDF text. Saved empty identifiers file.")
		raise SystemExit(0)

	# 3) Fetch adjusted prices via yfinance (using our helper)
	prices = fetch_prices(tickers, start=START, end=END)

	# Identify invalid tickers (no data or entirely NaN)
	invalid = [sym for sym in tickers if sym not in prices.columns]
	# Keep only valid columns going forward
	if invalid:
		valid_cols = [c for c in prices.columns if c in tickers]
	else:
		valid_cols = list(prices.columns)
	prices = prices[valid_cols]
	# Align and filter: drop columns with insufficient coverage and drop rows with any missing
	prices_aligned = align_and_filter(prices)

	# 4) Compute returns
	returns = compute_returns(prices_aligned, method='simple')

	# 5) Save outputs
	prices.to_csv(OUT_PRICES_CSV)
	returns.to_csv(OUT_RETURNS_CSV)
	if invalid:
		pd.DataFrame({"invalid_ticker": invalid}).to_csv(OUT_INVALID_CSV, index=False)

	print("ETF identifiers saved:", OUT_ETFS_CSV)
	print("ETF prices saved:", OUT_PRICES_CSV)
	print("ETF returns saved:", OUT_RETURNS_CSV)
	if invalid:
		print("Invalid/empty tickers saved:", OUT_INVALID_CSV)

