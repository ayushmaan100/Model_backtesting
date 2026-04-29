"""
Static sector mapping for the NSE 200 universe.

Used by analytics + dashboard for sector-exposure analysis. Mapping is based on
the sector grouping in nse200_tickers.py (which itself follows broad NSE/NIFTY
industry classification). This is a snapshot — sector reclassifications by NSE
or company restructurings are not tracked here. Documented limitation.

Schema: SECTOR_MAP[ticker] -> sector label (str).
Tickers absent from the map fall back to "Other".
"""

SECTOR_MAP: dict[str, str] = {}


def _add(sector: str, tickers: list[str]) -> None:
    for t in tickers:
        SECTOR_MAP[t if t.endswith(".NS") else f"{t}.NS"] = sector


_add("Banking", [
    "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
    "BANKBARODA", "PNB", "CANBK", "INDIANB", "UNIONBANK",
    "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "AUBANK",
])
_add("NBFC / Insurance", [
    "BAJFINANCE", "BAJAJFINSV", "SHRIRAMFIN", "CHOLAFIN", "MUTHOOTFIN",
    "PFC", "RECLTD", "IRFC", "LICHSGFIN", "SBILIFE",
    "HDFCLIFE", "ICICIGI", "LICI", "BSE", "CDSL", "CAMS",
    "SBICARD", "ABCAPITAL", "ANGELONE", "MFSL", "CANFINHOME", "PNBHOUSING",
])
_add("IT", [
    "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
    "LTIM", "MPHASIS", "COFORGE", "PERSISTENT", "OFSS",
])
_add("Energy", [
    "RELIANCE", "ONGC", "BPCL", "IOC", "COALINDIA",
    "NTPC", "POWERGRID", "TATAPOWER", "ADANIGREEN", "ADANIPORTS",
    "ADANIENT", "GAIL", "PETRONET", "OIL", "TORNTPOWER",
])
_add("FMCG", [
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
    "MARICO", "COLPAL", "GODREJCP", "EMAMILTD", "TATACONSUM",
    "VBL", "RADICO", "UBL",
])
_add("Auto", [
    "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "EICHERMOT",
    "TVSMOTOR", "HEROMOTOCO", "BOSCHLTD", "MOTHERSON", "BALKRISIND",
    "APOLLOTYRE", "EXIDEIND", "ASHOKLEY", "MINDINDS",
])
_add("Pharma & Healthcare", [
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN",
    "TORNTPHARM", "AUROPHARMA", "ALKEM", "ABBOTINDIA",
    "APOLLOHOSP", "MAXHEALTH", "FORTIS", "LALPATHLAB",
])
_add("Metals & Mining", [
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "SAIL",
    "NMDC", "NATIONALUM", "APLAPOLLO", "RATNAMANI",
])
_add("Cement", [
    "ULTRACEMCO", "GRASIM", "SHREECEM", "AMBUJACEM", "ACC",
    "DALMIACEM", "RAMCOCEM", "JKCEMENT",
])
_add("Capital Goods / Infra", [
    "LT", "SIEMENS", "ABB", "BHEL", "BEL", "HAL",
    "HAVELLS", "VOLTAS", "CUMMINSIND", "GMRAIRPORT", "NCC",
    "SCHAEFFLER", "TIMKEN", "ASTRAL",
])
_add("Chemicals & Paints", [
    "PIDILITIND", "DEEPAKNTR", "NAVINFLUOR", "AARTIIND",
    "GALAXYSURF", "FLUOROCHEM", "BERGEPAINT",
    "ASIANPAINT", "KANSAINER",
])
_add("Consumer Discretionary", [
    "DMART", "TRENT", "TITAN", "KALYANKJIL",
    "JUBLFOOD", "DIXON", "AMBER", "PVRINOX", "LODHA",
    "SUNTV", "ZEEL", "RAYMOND",
])
_add("Telecom", ["BHARTIARTL", "INDUSTOWER", "TATACOMM"])
_add("Logistics", ["INDIGO", "BLUEDART", "DELHIVERY", "CONCOR"])
_add("Real Estate", [
    "DLF", "GODREJPROP", "PRESTIGE", "BRIGADE", "OBEROIRLTY", "PHOENIXLTD",
])
_add("Agri & Fertilisers", ["UPL", "PIIND", "COROMANDEL", "CHAMBLFERT"])


def sector_of(ticker: str) -> str:
    """Return the sector label for a ticker, or 'Other' if unknown."""
    return SECTOR_MAP.get(ticker, "Other")
