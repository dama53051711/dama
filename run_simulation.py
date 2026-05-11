import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def get_test_start_date(dataset_name):
    """
    Returns the start date of the test set for specific datasets.
    Adjust these dates if your test set range is different.
    """
    if dataset_name == 'acl18':
        return pd.Timestamp('2015-10-01')
    elif dataset_name == 'cikm18':
        return pd.Timestamp('2017-11-01')
    elif dataset_name == 'kdd17':
        return pd.Timestamp('2017-01-01') # Example
    else:
        # Default fallback
        return pd.Timestamp('2015-10-01')

def run_simulation(csv_path, dataset_name='acl18'):
    # 1. Read Data
    df = pd.read_csv(csv_path)
    
    # 2. Sort by Index to ensure correct chronological order per stock
    # (Assuming the CSV is ordered: Stock A Day 1, Stock A Day 2... OR Day 1 Stock A, Day 1 Stock B...)
    # Based on your snippet, it looks like it might be sorted by DayIdx within chunks.
    # To be safe, we sort by DayIdx.
    df = df.sort_values(by=['DayIdx', 'index'])

    # 3. Map DayIdx to Real Dates
    # We find the minimum DayIdx in the file and assume that corresponds to test_start_date
    min_day_idx = df['DayIdx'].min()
    start_date = get_test_start_date(dataset_name)
    
    # Create a mapping: DayIdx -> Date
    # We assume DayIdx increments represent Business Days (Monday-Friday)
    # or simple calendar days. Financial papers usually use Business Days.
    unique_day_idxs = sorted(df['DayIdx'].unique())
    date_map = {}
    current_date = start_date
    
    for d_idx in unique_day_idxs:
        date_map[d_idx] = current_date
        # Increment by 1 business day
        current_date = current_date + pd.tseries.offsets.BusinessDay(1)

    df['Date'] = df['DayIdx'].map(date_map)

    # 4. Calculate Returns
    # We need to calculate the % change of Price from Day T to Day T+1.
    # Since the CSV might contain multiple stocks mixed together, we strictly need to know
    # which row corresponds to the NEXT day for the SAME stock.
    #
    # LIMITATION: Your CSV lacks 'StockID'. 
    # ASSUMPTION: The data is grouped by Stock, then sorted by Day. 
    # OR: The standard test set order is preserved.
    #
    # To robustly calc returns without StockID, we look at the 'Price' column.
    # If Price[i+1] is very different from Price[i], it might be a different stock.
    # However, for simplicity, let's assume standard "Day T to Day T+1" logic per row group.
    
    # We will compute 'NextPrice' by shifting. 
    # Important: We must respect the boundaries (don't calc return between Stock A and Stock B).
    # Since we don't have Stock IDs, a safe heuristic is: 
    # If DayIdx[i+1] == DayIdx[i] + 1, it's likely the same stock sequence.
    
    df['NextPrice'] = df['Price'].shift(-1)
    df['NextDayIdx'] = df['DayIdx'].shift(-1)
    
    # Valid return calculation only if the next row is exactly the next day
    valid_transition = df['NextDayIdx'] == (df['DayIdx'] + 1)
    
    df.loc[valid_transition, 'Return'] = (df['NextPrice'] - df['Price']) / df['Price']
    df.loc[~valid_transition, 'Return'] = 0.0 # No action on transition boundaries

    # 5. Simulate Strategy
    # Group by Date (simulate trading all stocks available on that day)
    daily_groups = df.groupby('Date')
    
    dates = []
    portfolio_values = [1.0] # PV starts at 1.0
    market_values = [1.0]    # Market starts at 1.0
    
    print(f"Running simulation on {len(unique_day_idxs)} days...")

    for date, group in daily_groups:
        # Skip the very last date if we can't calculate a return for it (no T+1 data)
        if group['Return'].iloc[0] == 0.0 and date == list(daily_groups.groups.keys())[-1]:
            continue
            
        dates.append(date)
        
        # --- Market Index (Equal Weight of all stocks) ---
        market_return = group['Return'].mean()
        
        # --- Our Strategy (Long only if Prediction == 1) ---
        long_candidates = group[group['Prediction'] == 1]
        
        if len(long_candidates) > 0:
            strategy_return = long_candidates['Return'].mean()
        else:
            # If model predicts 0 for everything, we hold cash (0% return)
            # Or you could choose to Short. Here we assume Cash.
            strategy_return = 0.0
            
        # Update PV (Compound Return)
        # PV_t = PV_{t-1} * (1 + r_t)
        new_pv = portfolio_values[-1] * (1 + strategy_return)
        new_mkt = market_values[-1] * (1 + market_return)
        
        portfolio_values.append(new_pv)
        market_values.append(new_mkt)

    # 6. Plotting
    # Remove the initial '1.0' from plots if you want to align with dates (or repeat first date)
    # Usually we plot dates vs values. Since we have N dates of returns, we have N+1 values.
    # Let's align by plotting from index 1 to N.
    
    plot_dates = dates 
    plot_pv = portfolio_values[1:]
    plot_mkt = market_values[1:]
    
    plt.figure(figsize=(12, 6))
    plt.plot(plot_dates, plot_pv, label='Proposed (DTML)', color='#1f77b4', linewidth=2)
    plt.plot(plot_dates, plot_mkt, label='Market Index', color='orange', linestyle='--', linewidth=1.5)
    
    # Formatting
    plt.title(f'Investment Simulation: {dataset_name.upper()}', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value (Start=1.0)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Calculate Final Stats
    final_return = (portfolio_values[-1] - 1) * 100
    market_return = (market_values[-1] - 1) * 100
    
    # Add text annotation
    info_text = (f"Final Model PV: {portfolio_values[-1]:.4f} (+{final_return:.2f}%)\n"
                 f"Final Market PV: {market_values[-1]:.4f} (+{market_return:.2f}%)")
    
    plt.annotate(info_text, xy=(0.02, 0.85), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    # Save and Show
    save_path = csv_path.replace('.csv', '_plot.png')
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='src/out/acl18-0/pred_result.csv', help='Path to pred_result.csv')
    parser.add_argument('--data', type=str, default='acl18', help='Dataset name (acl18, cikm18)')
    args = parser.parse_args()
    
    run_simulation(args.path, args.data)

    # python simulation.py --path src/out/acl18-0/pred_result.csv --data acl18