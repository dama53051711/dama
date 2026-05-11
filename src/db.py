
import pandas as pd
import urllib.parse as up
from sqlalchemy import create_engine, text as sql_text
from os import path as osp

# ---------- DB ----------
def database_config(DB_USER: str, DB_HOST: str, DB_PASS: str, DB_NAME: str, port="13306"):
    url = (
        f"mysql+pymysql://{DB_USER}:{up.quote_plus(DB_PASS)}@{DB_HOST}:{port}/{DB_NAME}"
        "?charset=utf8mb4"
    )
    return create_engine(url, echo=False, pool_recycle=3600)


# ---------- QUERY BUILDER ----------
def query_builder(news_table: str, start_date: str, end_date: str) -> str:
    if news_table == "sa_market_news":
        return f"""
            SELECT id, publish_on, title, primary_tickers
            FROM sa_market_news
            WHERE publish_on >= '{start_date}' AND publish_on <= '{end_date}'
            ORDER BY publish_on ASC
        """
    elif news_table == "finspd":
        return f"""
            SELECT id, date, title, symbols
            FROM finspd
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date ASC
        """
    elif news_table == "eod_news":
        return f"""
            SELECT date, title, content, symbols
            FROM eod_news
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date ASC
        """
    else:
        # generic fallback
        return f"""
            SELECT *
            FROM {news_table}
            WHERE date >= '{start_date}' AND date <= '{end_date}'
        """


# ---------- NORMALIZERS ----------
def normalize_sa(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"publish_on": "date", "title": "text"})
    df = df[["date", "text"]]
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    return df


def normalize_finspd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"title": "text"})
    df = df[["date", "text"]]
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    return df


def normalize_eod(df: pd.DataFrame) -> pd.DataFrame:
    # build text from title + content if both exist
    if "title" in df.columns and "content" in df.columns:
        df["text"] = (df["title"].fillna("") + " " + df["content"].fillna("")).str.strip()
    elif "title" in df.columns:
        df["text"] = df["title"]
    df = df[["date", "text"]]
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    return df


NORMALIZE_FUNS = {
    "sa_market_news": normalize_sa,
    "finspd": normalize_finspd,
    "eod_news": normalize_eod,
}


# ---------- CORE READER FOR ONE TABLE ----------
def read_one_news_table(engine, table_name, start_date, end_date) -> pd.DataFrame:
    query = query_builder(table_name, start_date, end_date)
    with engine.connect() as conn:
        result = conn.execute(sql_text(query))
        cols = result.keys()
        df = pd.DataFrame(result.fetchall(), columns=cols)

    # normalize
    if table_name in NORMALIZE_FUNS:
        df = NORMALIZE_FUNS[table_name](df)
    return df


# ---------- PUBLIC FUNCTION ----------
def read_news_db(
    NEWS_ENGINE,
    NEWS_TABLE: str,
    TICKER=None,
    START_DATE="2010-01-01",
    END_DATE="2025-03-01",
):
    try:
        # case 1: ALL sources
        if NEWS_TABLE == "ALL":
            parts = []
            for src in ("sa_market_news", "finspd", "eod_news"):
                df_src = read_one_news_table(NEWS_ENGINE, src, START_DATE, END_DATE)
                parts.append(df_src)
            df = pd.concat(parts, ignore_index=True)
            return df

        # case 2: single source
        df = read_one_news_table(NEWS_ENGINE, NEWS_TABLE, START_DATE, END_DATE)

        # optional ticker filtering if the table actually has a ticker/symbols column
        if TICKER:
            # try to filter on common cols
            possible_cols = ["ticker", "symbol", "symbols", "primary_tickers"]
            col_to_use = None
            for c in possible_cols:
                if c in df.columns:
                    col_to_use = c
                    break

            if col_to_use:
                if isinstance(TICKER, (list, tuple, set)):
                    tickers = {str(t).upper() for t in TICKER}
                    df = df[df[col_to_use].astype(str).str.upper().isin(tickers)]
                else:
                    df = df[df[col_to_use].astype(str).str.upper() == str(TICKER).upper()]

        return df

    except Exception as e:
        print(f"Error while reading news data from db: {e}")
        return pd.DataFrame()


# ---------- tiny wrapper ----------
def read_news_data(start_date: str, end_date: str, tickers, news_source: str):
    NEWS_DATA = {
        "eod": "eod_news",
        "finspd": "finspd",
        "sa_news": "sa_market_news",
        "ALL": "ALL",
    }

    DB_USER = "deeptrade"
    DB_PASS = "Elqxmfpdlem12@"
    DB_NAME = "news"
    DB_HOST = '147.46.216.30'

    engine = database_config(DB_USER, DB_HOST, DB_PASS, DB_NAME)
    table = NEWS_DATA[news_source]

    return read_news_db(
        engine,
        table,
        TICKER=tickers,
        START_DATE=start_date,
        END_DATE=end_date,
    )

def news_multisource_chunks(table: str = "fin_news_multisource",
                                start_date: str = "2014-01-31", 
                                end_date:str = "2016-12-31",
                                columns = ("date", "text"),
                                chunksize: int = 200_000, 
                                ):
    """
    날짜 필터 없이 테이블 전체를 `chunksize` 단위로 스트리밍합니다.
    사용 예:
        for chunk in iter_news_multisource_chunks(engine, chunksize=200_000):
            # chunk는 DataFrame
            print(len(chunk))
    """
    DB_USER = "deeptrade"
    DB_PASS = "Elqxmfpdlem12@"
    DB_NAME = "news"
    DB_HOST = '147.46.216.30'

    engine = database_config(DB_USER, DB_HOST, DB_PASS, DB_NAME)
    cols_sql = ", ".join(f"`{c}`" for c in columns)
    q = f"SELECT {cols_sql} FROM `{table}` WHERE date >= '{start_date}' AND date <= '{end_date}' ORDER BY `id` ASC"

    with engine.connect() as conn:
        for chunk in pd.read_sql(sql_text(q), conn, chunksize=chunksize):
            yield chunk

def count_rows(table: str) -> int:
    DB_USER = "deeptrade"
    DB_PASS = "Elqxmfpdlem12@"
    DB_NAME = "news"
    DB_HOST = '147.46.216.30'

    engine = database_config(DB_USER, DB_HOST, DB_PASS, DB_NAME)
    with engine.connect() as conn:
        return conn.execute(sql_text(f"SELECT COUNT(*) FROM `{table}`")).scalar()
