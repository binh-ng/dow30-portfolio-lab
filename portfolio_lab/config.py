"""Shared constants: the Dow 30 lineup used in the original project and a sector map."""

BENCHMARK = "DJIA"
TRADING_DAYS = 252

# Dow lineup used in the original analysis (roughly the Aug 2020 to Feb 2024 membership).
# AMZN, NVDA and SHW joined the index later and replaced WBA, INTC and DOW.
TICKERS = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT",
]

SECTORS = {
    "AAPL": "Technology", "AMGN": "Health Care", "AXP": "Financials",
    "BA": "Industrials", "CAT": "Industrials", "CRM": "Technology",
    "CSCO": "Technology", "CVX": "Energy", "DIS": "Communication",
    "DOW": "Materials", "GS": "Financials", "HD": "Consumer Discretionary",
    "HON": "Industrials", "IBM": "Technology", "INTC": "Technology",
    "JNJ": "Health Care", "JPM": "Financials", "KO": "Consumer Staples",
    "MCD": "Consumer Discretionary", "MMM": "Industrials", "MRK": "Health Care",
    "MSFT": "Technology", "NKE": "Consumer Discretionary", "PG": "Consumer Staples",
    "TRV": "Financials", "UNH": "Health Care", "V": "Financials",
    "VZ": "Communication", "WBA": "Consumer Staples", "WMT": "Consumer Staples",
}

# Original project settings, used for the exact replication check.
ORIGINAL_RISK_FREE = 0.0
ORIGINAL_MAX_WEIGHT = 0.25

NAMES = {
    "AAPL": "Apple", "AMGN": "Amgen", "AXP": "American Express", "BA": "Boeing", "CAT": "Caterpillar",
    "CRM": "Salesforce", "CSCO": "Cisco", "CVX": "Chevron", "DIS": "Disney", "DOW": "Dow Inc.",
    "GS": "Goldman Sachs", "HD": "Home Depot", "HON": "Honeywell", "IBM": "IBM", "INTC": "Intel",
    "JNJ": "Johnson & Johnson", "JPM": "JPMorgan Chase", "KO": "Coca-Cola", "MCD": "McDonald's",
    "MMM": "3M", "MRK": "Merck", "MSFT": "Microsoft", "NKE": "Nike", "PG": "Procter & Gamble",
    "TRV": "Travelers", "UNH": "UnitedHealth", "V": "Visa", "VZ": "Verizon",
    "WBA": "Walgreens Boots Alliance", "WMT": "Walmart",
}
