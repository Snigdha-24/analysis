from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    if returns.empty or returns.isna().all():
        return 0.0
    excess_returns = returns - risk_free_rate/12  # Monthly risk-free rate
    if excess_returns.std() == 0:
        return 0.0
    return float(np.sqrt(12) * (excess_returns.mean() / excess_returns.std()))

def get_monthly_returns(ticker, start_date, end_date):
    try:
        print(f"\nFetching data for {ticker}...")
        stock = yf.Ticker(ticker)
        
        # Download daily data first to verify the ticker exists
        daily_data = stock.history(period='1d')
        if daily_data.empty:
            print(f"No data available for ticker {ticker}")
            return pd.Series(dtype='float64')
        
        # If we have valid daily data, proceed with monthly data
        df = stock.history(start=start_date, end=end_date, interval='1mo')
        if df.empty:
            print(f"No monthly data available for {ticker} between {start_date} and {end_date}")
            return pd.Series(dtype='float64')
        
        # Use Adj Close for returns calculation
        returns = df['Adj Close'].pct_change().dropna()
        print(f"Successfully fetched {len(returns)} monthly returns for {ticker}")
        return returns
    except Exception as e:
        print(f"Error fetching data for {ticker}:")
        print(traceback.format_exc())
        return pd.Series(dtype='float64')

@app.route('/api/stock-data', methods=['POST'])
def get_stock_analysis():
    try:
        data = request.json
        tickers = data['tickers']
        
        # Use more recent date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        print(f"Fetching data from {start_date} to {end_date}")
        
        # Try both S&P 500 and NASDAQ as benchmark
        benchmark_options = ['^IXIC', '^GSPC']  # Try NASDAQ first
        benchmark_returns = pd.Series(dtype='float64')
        benchmark_used = None
        
        for benchmark in benchmark_options:
            print(f"\nTrying benchmark {benchmark}...")
            benchmark_returns = get_monthly_returns(benchmark, start_date, end_date)
            if not benchmark_returns.empty:
                print(f"Successfully using {benchmark} as benchmark")
                benchmark_used = benchmark
                break
        
        if benchmark_returns.empty:
            error_msg = f"Unable to fetch benchmark data. Tried {benchmark_options}. Please try again later."
            print(error_msg)
            return jsonify({'error': error_msg}), 400
        
        results = []
        correlations = {}
        
        for ticker in tickers:
            stock_returns = get_monthly_returns(ticker, start_date, end_date)
            
            if not stock_returns.empty and not benchmark_returns.empty:
                # Align the returns series
                aligned_data = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
                if not aligned_data.empty:
                    correlation = float(aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1]))
                    print(f"Correlation calculated for {ticker}: {correlation}")
                else:
                    correlation = 0.0
                    print(f"No aligned data available for {ticker}")
            else:
                correlation = 0.0
                print(f"Missing returns data for {ticker}")
            
            correlations[ticker] = correlation
            
            # Calculate Sharpe ratio
            sharpe = calculate_sharpe_ratio(stock_returns)
            
            results.append({
                'ticker': ticker,
                'correlation': correlation if not np.isnan(correlation) else 0.0,
                'sharpe_ratio': sharpe if not np.isnan(sharpe) else 0.0,
                'returns': [float(x) if not np.isnan(x) else 0.0 for x in stock_returns.tolist()]
            })
        
        # Sort by correlation to find highest and lowest
        sorted_correlations = sorted(correlations.items(), key=lambda x: x[1])
        lowest_corr_ticker = sorted_correlations[0][0] if sorted_correlations else tickers[0]
        highest_corr_ticker = sorted_correlations[-1][0] if sorted_correlations else tickers[0]
        
        return jsonify({
            'stockData': results,
            'benchmarkReturns': [float(x) if not np.isnan(x) else 0.0 for x in benchmark_returns.tolist()],
            'benchmarkUsed': benchmark_used,
            'highestCorrTicker': highest_corr_ticker,
            'lowestCorrTicker': lowest_corr_ticker
        })

    except Exception as e:
        error_msg = f"Error in analysis: {str(e)}"
        print(f"Error details: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
