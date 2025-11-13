import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_clean_data(path: str) -> pd.DataFrame:
    #load and clean stock price data
    df = pd.read_csv(path)

    #pick close col
    close_col = None
    for cand in ['Close/Last', 'Close', 'Adj Close']:
        if cand in df.columns:
            close_col = cand
            break
    if close_col is None:
        raise ValueError("No recognized closing price column found.")
    
    #clean $ and commas then float
    df[close_col] = df[close_col].astype(str).str.strip()
    df[close_col] = df[close_col].str.replace(r'[\$,]', '', regex=True).astype(float)

    #date parsing
    date_col = 'Date' if 'Date' in df.columns else None
    if date_col is None:
        raise ValueError("No recognized date column found.")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])

    #standardise column names
    df = df.rename(columns={close_col: 'Close', date_col: 'Date'})

    #sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def plot_closing_price(df: pd.DataFrame, outdir: Path, company_name: str = "Amazon") -> None:
    #plot closing price over time
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Close'], label=f'{company_name} Closing Price', color='blue')
    plt.title(f'{company_name} Closing Price Over 1 Year')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    outpath = outdir / "plots" / f"{company_name.lower()}_closing_price.png"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)    
    plt.close()

def calculate_daily_change(df: pd.DataFrame) -> pd.DataFrame:
    #calculate daily % change and add as new column
    df = df.copy()
    df['Daily % Change'] = df['Close'].pct_change() * 100
    df['Daily % Change'] = df['Daily % Change'].fillna(0)  #fill NaN for first row
    return df

def plot_daily_change(df: pd.DataFrame, outdir: Path, company_name: str = "Amazon") -> None:
    #plot daily % change
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Daily % Change'], label='Daily % Change', color='orange')
    plt.title(f'{company_name} Daily % Change')
    plt.xlabel('Date')
    plt.ylabel('Daily % Change (%)')
    plt.legend()
    plt.grid(True)

    outpath = outdir / "plots" / f"{company_name.lower()}_daily_change.png"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)    
    plt.close()

def calculate_std_dev(df: pd.DataFrame) -> float:
    #calculate standard deviation of daily % change
    return df['Daily % Change'].std()

