"""Feature configuration mapping raw data factors to logical categories."""

import pandas as pd

FEATURE_CONFIG = {
    # Value Metrics
    'value': {
        'enabled': True,
        'features': {
            'Book to Price': True,
            'Earnings to Price': True,
            'Operating Cash Flow to Price': True,
            'Free Cash Flow to Price': True,
            'Net Cash Flow to Price': True,
            'Net Cash Flow to Enterprise Value': True,
            'Operating Cash Flow to Enterprise Value': True,
            'Free Cash Flow to Enterprise Value': True,
            'EBITDA to Enterprise Value': True,
            'Sales to EV Ratio': True,
            'Assets to Price Ratio': True,
            'Operating Earnings to Price Ratio': True,
            'Pretax Income Per Share to Price': True,
            'Sales to Price Ratio': True,
            'Future Cashflow Yield': True,
            'Free Cash Flow to Common Equity': True,
            'Operating Cash Flow to Equity': True,
            'Operating Earnings to Assets Ratio': True,
            'Inverse P/E Ratio Adj for Growth and Yield': True,
            'Tobin\'s Q Ratio': True,
            'EBITDA to Price': True,
            'Cash to Price': True,
            'Dividends to Price Ratio': True,
            'Growth Flow to Price': True,
            'Net Current Assets to Price Ratio': True,
            'Discounted Future Cash Flows to Market Cap': True,
            'Proved Reserves to Market Cap': True,
            'Cash to EV Ratio': True,
            'Book Value to Enterprise Value': True,
            'Forward Earnings to Price': True,
            'Forward Free Cash Flow to Price Ratio': True,
            'Core EPS to Price (Fundamentals)': True,
            'Current Liability to Price': True,
            'Adjusted EBIT to Price': True,
            'Adjusted Forward Earnings to Price': True,
            'EBITDA to Assets': True,
            'Free Cash Flow to Sales': True,
            'Cash Flow to Total Asset': True,
            'Cash Flow to Equity': True,
            'Cash Flow to Invested Capital': True,
            'Cash to Sales': True,
            'Gross Profit to Assets': True
        }
    },
    
    # Quality Metrics
    'quality': {
        'enabled': True,
        'features': {
            'Return on Invested Capital': True,
            'Return on Equity': True,
            'Return on Assets': True,
            'Cash Flow Return on Invested Capital': True,
            'Operating Margin': True,
            'Net Profit Margin': True,
            'Gross Profit Margin': True,
            'EBIT Margin': True,
            'Balance Sheet Accruals': True,
            'Accrual Ratio - Balance Sheet': True,
            'Accrual Ratio - Cash Flows': True,
            'Working Capital Accruals': True,
            'Earnings Quality': True,
            'Piotroski F-Score': True,
            'Altman Z-Score': True,
            'Adjusted Interest Coverage Ratio': True,
            'Depreciation to Capex Ratio': True,
            'Return on Net Tangible Assets': True,
            'Return on Risk Weighted Assets (Fundamentals)': True,
            'EBIT to Assets': True,
            'EBIT to Assets v2': True,
            'EPS Stability': True,
            'Net Income Stability': True,
            'Sales to Invested Capital': True,
            'Dividends to Cash Flow': True,
            'Adjusted Accruals': True,
            'Asset Adjusted Capital Investments': True,
            'Pretax Profit Margin': True,
            'Reserve Exhaustion Rate': True,
            'Pretax Income to Net Operating Assets': True,
            'Return on Assets v2': True,
            'Inverse PEG': True
        }
    },
    
    # Growth Metrics
    'growth': {
        'enabled': True,
        'features': {
            '3Y Average Annual Sales Growth': True,
            '3Y Average Annual Earnings Growth': True,
            'Expected Earnings Growth: Fiscal Year 2/Fiscal Year 1': True,
            'Expected LTG': True,
            'Sustainable Growth Rate': True,
            'Production Growth': True,
            'Reserve Growth': True,
            'Dividend Growth': True,
            'Dividend per Share Growth': True,
            'Historical Growth': True,
            'Asset Growth - 1 Year': True,
            'Developed Reserves per Share Growth': True,
            '1Y Chg in Asset Adjusted EPS': True,
            '1Y Chg in Asset Adjusted Cash Flow': True,
            '1Y Chg in Asset Adjusted Free Cash Flow': True,
            '1Y Chg in Asset Adjusted Operating Cash Flow': True,
            '1Y Chg in Operating Margin': True,
            '1Y Chg in Cash Flow per Share': True,
            '1Y Chg in Free Cash Flow per Share': True,
            '1Y Chg in Operating Cash Flow per Share': True,
            '1Y Chg in Earnings per Share': True,
            '1Y Chg in Sales Turnover': True,
            '3Y Chg in Asset Adjusted EPS': True,
            '3Y Chg in Asset Adjusted Cash Flow': True,
            '3Y Chg in Asset Adjusted Free Cash Flow': True,
            '3Y Chg in Asset Adjusted Operating Cash Flow': True,
            '3Y Chg in Operating Margin': True,
            '3Y Chg in Cash Flow per Share': True,
            '3Y Chg in Free Cash Flow per Share': True,
            '3Y Chg in Operating Cash Flow per Share': True,
            '3Y Chg in Earnings per Share': True,
            '3Y Chg in Sales Turnover': True,
            '1Y Chg in Cash Flow to Price': True,
            '1Y Chg in Earnings to Price': True,
            '1Y Chg in Free Cash Flow to Price': True,
            '1Y Chg in Operating Cash Flow to Price': True,
            '1Y Chg in Sales to Price': True,
            '1Y Chg in Stock Buybacks': True,
            '1Y Chg in Discounted Future Cash Flows to Market Cap': True,
            '3Y Chg in Cash Flow to Price': True,
            '3Y Chg in Earnings to Price': True,
            '3Y Chg in Free Cash Flow to Price': True,
            '3Y Chg in Operating Cash Flow to Price': True,
            '3Y Chg in Sales to Price': True,
            'Sales Acceleration': True,
            'Same Restaurant Sales Growth': True,
            'Surgeries Growth': True
        }
    },
    
    # Balance Sheet Strength
    'balance_sheet': {
        'enabled': True,
        'features': {
            'Current Ratio': True,
            'Quick Ratio': True,
            'Cash Ratio': True,
            'Debt to Assets Ratio': True,
            'Long Term Debt to Assets Ratio': True,
            'Long Term Debt to Equity Ratio': True,
            'Interest Coverage Ratio': True,
            'Cash to Total Assets': True,
            'Working Capital to Assets Ratio': True,
            'Working Capital to Total Assets': True,
            'Retained Earnings to Total Assets': True,
            'Retained Earnings to Total Assets v2': True,
            'Capital Expenditure to Total Assets': True,
            'Financial Leverage (Equity Multiplier)': True,
            'Book Leverage': True,
            'Degree of Financial Leverage': True,
            'Solvency Ratio': True,
            'Total Market Value of Equity to Total Liabilities': True,
            'Problem Loans to Total Equity plus Loan Loss Reserves (Fundamentals)': True,
            'Savings & Money Market Deposit Accounts to Total Deposits (Fundamentals)': True,
            'Tier 1 Capital Ratio  (Fundamentals)': True,
            'Working Capital to Assets Ratio v2': True,
            'Inventory to Assets Ratio': True,
            '1Y Chg in Debt to Assets': True,
            'Payout Ratio': True,
            'Retention Ratio': True,
            'Year over Year Change of Total Debt': True,
            'Year over Year Change of Inventory to Assets': True
        }
    },
    
    # Efficiency/Activity
    'efficiency': {
        'enabled': True,
        'features': {
            'Asset Turnover': True,
            'Asset Turnover v2': True,
            'Inventory Turnover': True,
            'Receivables Turnover Ratio': True,
            'Cash Conversion Cycle': True,
            'Working Capital Turnover Ratio': True,
            'Equity Turnover': True,
            'Capital Expenditure to Sales': True,
            'SG&A to Sales': True,
            'Capital Acquisition Ratio': True,
            'Capital Efficiency': True,
            'Research & Development Intensity': True,
            'Total Revenue per Facility': True,
            'Operating Exp per Available Seat Miles': True,
            'Fuel Consumed per Available Seat Mile': True,
            'Average Age of Aircraft': True,
            'Cash Burn Rate': True,
            'Claims Ratio': True,
            'Combined Ratio': True,
            'Expense Ratio': True,
            'Change of Asset Turnover v2': True,
            'Change in TTM Depr. to Capex': True,
            'Change of Sales to Change of EPS': True,
            'Change EPS to Sales': True,
            'Sales to Gross Profit Margin Growth': True,
            'Working Capital to Sales': True,
            'Trailing 12-Month Overhead/Trailing 12-Month Sales': True,
            'Trailing 12-Month Receivables and Inventories/Trailing 12-Month Sales': True,
            'Year over Year Change in SGA to Sales': True,
            'Year over Year Change of EPS to Sales': True
        }
    },
    
    # Price Momentum
    'momentum': {
        'enabled': True,
        'features': {
            '12M - 1M Price Momentum': True,
            '9 Month Price Momentum': True,
            '6 Months Price Reversal': True,
            '1M Price Reversal': True,
            '5 Day Price Reversal': True,
            '14-Day Relative Strength Index': True,
            '10 Day MACD Trend': True,
            'Closing Price to 52 Week High': True,
            '4W to 52W Price Oscillator': True,
            '52W High Low': True,
            '130 Day Minimum Return': True,
            '180D Price TStat': True,
            '26W Price Stat': True,
            '26W Relative Price Strength': True,
            '39W Lagged Return': True,
            '50 Day to 200 Day Stock Price': True,
            'Slope of 52W Price Trend': True,
            'Risk Adjusted Relative Strength': True,
            '15/36 Week Price Ratio': True,
            '1M Price High - 1M Price Low': True,
            'Closing Price to 260 Day Low': True,
            'Percent Above 260-Day Low': True,
            'Price Momentum': True
        }
    },
    
    # Risk/Volatility
    'risk': {
        'enabled': True,
        'features': {
            '60M CAPM Beta': True,
            '60M CAPM Alpha': True,
            '12M Realized Price Volatility': True,
            '1M Realized Price Volatility': True,
            '24M Residual Return Variance': True,
            '90 Day Coefficient of Variation': True,
            'Sharpe Ratio': True,
            '20 Day Coefficient of Variation of Volume to Price': True,
            '30 Day Coefficient of Variation of Volume to Price': True,
            '60 Day Coefficient of Variation of Volume to Price': True,
            'Volatility': True,
            'Volatility adjusted 12M return': True,
            '6M Chg in 12M CAPM Alpha': True,
            '6M Chg in 18M CAPM Alpha': True,
            '6M Chg in 36M CAPM Alpha': True,
            'Max Daily Return in the Past 6 Months': True,
            'Amihud Illiquidity Measure': True
        }
    },
    
    # Market Microstructure
    'market': {
        'enabled': True,
        'features': {
            'Log Market Cap': True,
            'Share Turnover': True,
            '50 Day Volume Signal': True,
            'Adj 50 Day Volume Signal': True,
            '52-Week Volume Price Trend with 20-Day Lag': True,
            '5 Day Volume Signal': True,
            '5-Day Money Flow/Volume': True,
            '6 Month Average Share Turn Over': True,
            'Adjusted 6 Month Average Share Turn Over': True,
            '1Y Chg in Share Turnover': True,
            '1Y Chg in Shares Outstanding': True,
            'Change in Number of Buyers': True,
            'Change in Number of Holders': True,
            'Change of Ownership Level - All Managers': True,
            'Foreign Institution Ownership': True,
            'Institution Ownership Level': True,
            'Institution Ownership Level - Active Manager': True,
            'Institution Ownership Concentration': True,
            'Institution Ownership Turnover': True,
            'Ownership Breadth Stability of Institution Ownership by Buyers': True,
            'Ownership Breadth Stability of Institution Ownership by Holders': True,
            'Stability of Change in Number of Holders': True,
            'Log of Market Capitalization Cubed': True,
            'Size': True
        }
    },
    
    # Analyst/Market Sentiment
    'sentiment': {
        'enabled': True,
        'features': {
            'Earnings Surprise': True,
            'Analyst Dispersion for FY1 EPS': True,
            'Analyst Dispersion for FY2 EPS': True,
            'Analyst Earnings Estimate Diffusion': True,
            'Analyst Expectations': True,
            'EBIT Estimate Dispersion': True,
            'Revenue Estimate Dispersion': True,
            'Adjusted Number of EPS FY1 Revisions': True,
            'Adj Number of EPS FY2 Revisions': True,
            'Number of EPS FY1 Revisions': True,
            'Adjusted Revision Magnitude': True,
            'Street Revision Magnitude': True,
            '3M Revision in FY1 EPS Estimate': True,
            '3M Revision in FY2 EPS Estimate': True,
            '6M Chg in Target Price': True,
            '6M Chg in Target Price Gap': True,
            'Target Price Gap to 6M EMA': True,
            'Buy-to-Sell Recommendation Ratio Less 3M EMA': True,
            'Buy-to-Sell Recommendation Ratio Less 3M SMA': True,
            '6M Avg Chg 1M Recommendation': True,
            'Standardized Unexpected Earnings': True,
            '12 month Relative Price Strength(Ind Grp Rel VolAdjRtn12M)': True,
            '4-Week Change in 12-Month Forward Earnings Consensus Estimate/Price': True,
            '8-Week Change in 12-Month Forward Earnings Consensus Estimate/Price': True,
            '6M Momentum in Trailing 12M Sales': True,
            '3M Momentum in Trailing 12M Sales': True
        }
    },
    
    # Industry Relative
    'industry_relative': {
        'enabled': True,
        'features': {
            'Ind Grp Rel Return on Equity': True,
            'Ind Grp Rel Return on Assets': True,
            'Ind Grp Rel Return on Invested Capital': True,
            'Ind Grp Rel EBIT Margin': True,
            'Ind Grp Rel Operating Margin': True,
            'Ind Grp Rel Pretax Profit Margin': True,
            'Ind Grp Rel Gross Profit Margin': True,
            'Ind Grp Rel Operating Earnings to Assets Ratio': True,
            'Ind Grp Rel Operating Earnings to Price Ratio': True,
            'Ind Grp Rel Book to Price': True,
            'Ind Grp Rel Cash to Price': True,
            'Ind Grp Rel Free Cash Flow to Price': True,
            'Ind Grp Rel Operating Cash Flow to Price': True,
            'Ind Grp Rel Net Cash Flow to Price': True,
            'Ind Grp Rel Net Cash Flow to Enterprise Value': True,
            'Ind Grp Rel EBITDA to Price': True,
            'Ind Grp Rel EBITDA to Enterprise Value': True,
            'Ind Grp Rel Assets to Price Ratio': True,
            'Ind Grp Rel Sales to Price Ratio': True,
            'Ind Grp Rel Current Ratio': True,
            'Ind Grp Rel Quick Ratio': True,
            'Ind Grp Rel Cash to Total Assets': True,
            'Ind Grp Rel Working Capital to Assets Ratio': True,
            'Ind Grp Rel Long Term Debt to Assets Ratio': True,
            'Ind Grp Rel Inventory Turnover': True,
            'Ind Grp Rel Receivables Turnover Ratio': True,
            'Ind Grp Rel Working Capital Turnover Ratio': True,
            'Ind Grp Rel Sales to Gross Profit Margin Growth': True,
            'Ind Grp Rel 1Y Chg in Earnings per Share': True,
            'Ind Grp Rel 1Y Chg in Operating Margin': True,
            'Ind Grp Rel 1Y Chg in Cash Flow to Price': True,
            'Ind Grp Rel 1Y Chg in Earnings to Price': True,
            'Ind Grp Rel 1Y Chg in Free Cash Flow per Share': True,
            'Ind Grp Rel 1Y Chg in Free Cash Flow to Price': True,
            'Ind Grp Rel 1Y Chg in Operating Cash Flow per Share': True,
            'Ind Grp Rel 1Y Chg in Sales to Price': True,
            'Ind Grp Rel 1Y Chg in Shares Outstanding': True,
            'Ind Grp Rel 1Y Chg in Stock Buybacks': True,
            'Ind Grp Rel 3Y Chg in Sales to Price': True,
            'Ind Grp Rel 5 Day Price Momentum': True,
            'Ind Grp Rel 9M Price Momentum': True,
            'Ind Grp Rel 12M - 1M Price Momentum': True,
            'Ind Grp Rel 50 Day Volume Signal': True,
            'Ind Grp Rel Accrual Ratio - Cash Flows': True,
            'Ind Grp Rel Adjusted Accruals': True,
            'Ind Grp Rel Analyst Dispersion for FY1 EPS': True,
            'Ind Grp Rel Asset Adjusted Capital Investments': True,
            'Ind Grp Rel Balance Sheet Accruals': True,
            'Ind Grp Rel Inverse P/E Ratio Adj for Growth and Yield': True,
            'Ind Grp Rel Max Daily Return in the Past 6 Months': True,
            'Ind Grp Rel Standardized Unexpected Earnings': True,
            'Ind Grp Rel Sustainable Growth Rate': True,
            'Ind Grp Rel Tobin\'s Q Ratio': True,
            'Ind Grp Rel 1M Price Reversal': True,
            'Ind Grp Rel Earnings Surprise': True,
            'Ind Grp Rel Operating Cash Flow to Enterprise Value': True,
            'Ind Grp Rel Pretax Income Per Share to Price': True,
            'Ind Grp Rel Asset Turnover': True,
            'Ind Grp Rel Capital Acquisition Ratio': True,
            'Ind Grp Rel Capital Expenditure to Total Assets': True,
            'Ind Grp Rel Cash Conversion Cycle': True,
            'Ind Grp Rel Cash Flow Return on Invested Capital': True,
            'Ind Grp Rel Cash to EV Ratio': True,
            'Ind Grp Rel Change in TTM Depr. to Capex': True,
            'Ind Grp Rel Debt to Assets Ratio': True,
            'Ind Grp Rel Depreciation to Capex Ratio': True,
            'Ind Grp Rel Dividends to Price Ratio': True,
            'Ind Grp Rel Earnings to Price': True,
            'Ind Grp Rel Expected LTG': True,
            'Ind Grp Rel Forward Earnings to Price': True,
            'Ind Grp Rel Forward Free Cash Flow to Price Ratio': True,
            'Ind Grp Rel Free Cash Flow to Common Equity': True,
            'Ind Grp Rel Free Cash Flow to Enterprise Value': True,
            'Ind Grp Rel Growth Flow to Price': True,
            'Ind Grp Rel Interest Coverage Ratio': True,
            'Ind Grp Rel Inventory to Assets Ratio': True,
            'Ind Grp Rel Inverse PEG': True,
            'Ind Grp Rel Long Term Debt to Equity Ratio': True,
            'Ind Grp Rel Net Current Assets to Price Ratio': True,
            'Ind Grp Rel Net Profit Margin': True,
            'Ind Grp Rel Price Momentum - 6 Months': True,
            'Ind Grp Rel SG&A to Sales': True,
            'Ind Grp Rel Sales to EV Ratio': True,
            'Ind Grp Rel Working Capital Accruals': True,
            'Ind Grp Rel Working Capital to Sales': True,
            'Ind Grp Rel Year over Year Change of EPS to Sales': True,
            'Ind Grp Rel Year over Year Change of Total Debt': True,
            'Ind Grp Rel YoY Change in SGA to Sales': True
        }
    },

    # Historical Relative
    'historical_relative': {
        'enabled': True,
        'features': {
            '5 Yr Hist Rel 1Y Chg in Earnings per Share': True,
            '5 Yr Hist Rel Adjusted Accruals': True,
            '5 Yr Hist Rel Assets to Price Ratio': True,
            '5 Yr Hist Rel Book to Price': True,
            '5 Yr Hist Rel Cash to Total Assets': True,
            '5 Yr Hist Rel Current Ratio': True,
            '5 Yr Hist Rel EBIT Margin': True,
            '5 Yr Hist Rel EBITDA to Price': True,
            '5 Yr Hist Rel Free Cash Flow to Price': True,
            '5 Yr Hist Rel Gross Profit Margin': True,
            '5 Yr Hist Rel Inverse P/E Ratio Adj for Growth and Yield': True,
            '5 Yr Hist Rel Long Term Debt to Assets Ratio': True,
            '5 Yr Hist Rel Net Cash Flow to Enterprise Value': True,
            '5 Yr Hist Rel Net Cash Flow to Price': True,
            '5 Yr Hist Rel Operating Cash Flow to Price': True,
            '5 Yr Hist Rel Operating Earnings to Assets Ratio': True,
            '5 Yr Hist Rel Operating Margin': True,
            '5 Yr Hist Rel Pretax Profit Margin': True,
            '5 Yr Hist Rel Quick Ratio': True,
            '5 Yr Hist Rel Receivables Turnover Ratio': True,
            '5 Yr Hist Rel Return on Assets': True,
            '5 Yr Hist Rel Return on Equity': True,
            '5 Yr Hist Rel Return on Invested Capital': True,
            '5 Yr Hist Rel Sales to Gross Profit Margin Growth': True,
            '5 Yr Hist Rel Working Capital Turnover Ratio': True,
            '5 Yr Hist Rel Working Capital to Assets Ratio': True
        }
    },

    # Uncategorized Factors
    'uncategorized': {
        'enabled': True,
        'features': {
            '1 Yr Chg in Admissions': True,
            '1 Yr Chg in Number of Properties': True,
            '1 Yr Chg in Number of Restuarants': True,
            '1 Yr Chg in Number of Rooms': True,
            '1 Yr Chg in Revenue per Room': True,
            '1 Yr Grw Avg Length of Stay': True,
            '1 Yr Grw Broadband Subscribers': True,
            '1 Yr Grw Churn Rate': True,
            '1 Yr Grw House Sales Revenue': True,
            '1 Yr Grw Net New Orders': True,
            '1 Yr Grw Net Wireless Subs Additions': True,
            '1 Yr Grw Num of Patent Applications': True,
            '1 Yr Grw Num of Products in Phase 3': True,
            '1 Yr Grw Number of Patents': True,
            '1 Yr Grw Revenue Passenger Miles': True,
            '1 Yr Grw Revenue Passengers Carried': True,
            '1 Yr Grw Tot Home Rev to Const in Prog': True,
            '1 Yr Grw Tot Home Rev to Inventory': True,
            '1 Yr Grw Wireless Subscribers': True,
            '1 Yr Grw in Net Premiums Earned': True,
            '1 Yr Grw in Total Investments': True,
            '1 Yr Grw in Underwriting Profit': True,
            '1 Yr Same Rest Sales Acceleration': True,
            '1-Year Change in Current Ratio': True,
            '1-Year Change in Gross Profit Margin': True,
            '1-Year Change in Long Term Debt to Avg Total Assets': True,
            '1-Year Change in ROA': True,
            '1Y Change in Modified Texas Ratio (Fundamentals)': True,
            '1Y Change in Nonaccrual Loans to Total Assets (Fundamentals)': True,
            '1Y Chg in Amihud': True,
            '1Y Chg in EPS to Operating Cash Flow': True,
            '1Y Chg in Future Cash Flows per Share': True,
            '1Y Chg in Future Cash Flows to Total Assets': True,
            '1Y Chg in Future Cashflow Yield': True,
            '1Y Chg in Proved Reserves to Market Cap': True,
            '1Y Chg in Reserve Acquired to Total Costs': True,
            '1Y Chg in Reserve Exhaustion Rate': True,
            '1Y Chg in Sales to Earnings': True,
            '20-Day Lane\'s Stochastic Indicator': True,
            '3Y Change in Operating Income to Tier 1 Common Equity (Fundamentals)': True,
            '3Y Change in Total Pretax Expense to Average Assets (Fundamentals)': True,
            'Actual Production to Market Cap': True,
            'Backlog Homes Value Grw': True,
            'Churn Rate': True,
            'Deliv Homes Value Grw to Mkt Cap': True,
            'Gas Margin': True,
            'Home Gross Margin': True,
            'Hospitals and Facilities Growth': True,
            'Hotel & Casino Rev to Oper Exp': True,
            'Implied Cap Rate': True,
            'Insurance: Net Premium Written to Net Assets': True,
            'Investment Duration': True,
            'Licensed Beds Growth': True,
            'Log TTM Sales': True,
            'Log of Total Last Quarter Assets': True,
            'Log of Unadjusted Stock Price': True,
            'Mine Life - Gold Mining': True,
            'Net Prem Written to Stat Surplus': True,
            'Normalized EP': True,
            'Normalized ROA': True,
            'Normalized ROE': True,
            'Number of Patents to Mkt Cap': True,
            'Oil Margin': True,
            'Operating Cash Flow Ratio': True,
            'Operating Cash Flow to Asset v2': True,
            'Passenger Load Factor': True,
            'Patient Days Growth': True,
            'Pre-Provision Net Revenue / Price  (Fundamentals)': True,
            'Profit per Room': True,
            'Proved & Prob Reserves to Market Cap': True,
            'ROA 20 Qtr Standard Deviation': True,
            'ROA 60 Month Slope': True,
            'ROE 20 Qtr Standard Deviation': True,
            'Reserve Acquisition Cost': True,
            'Reserve Acquisition Yield': True,
            'Reserve Replacement Ratio': True,
            'Restaurant Operating Margin': True,
            'Restaurants Closing Momentum': True,
            'Restaurants Opening Momentum': True,
            'Revenue per Room': True,
            'Room Margin': True,
            'TTM Rest Closed & Sold to Total Rest': True,
            'TTM Rest Open & Acquire to Total Rest': True,
            'Total Profit per Available Seat Mile': True,
            'Underwriting Margin': True,
            'Unexpected Inventory Change': True,
            'Unexpected Receivables Change': True,
            'Valuation': True,
            'Wireless Penetration Rate': True
        }
    }
}

# Parameters for feature calculation
PARAMS = {
    'standardize': True,      # Whether to standardize features
    'winsorize': 0.01,       # Winsorization threshold
    'min_history': 252,      # Minimum history days required
    'industry_groups': True  # Whether to use industry grouping
}

def get_enabled_features():
    """Get list of enabled features."""
    enabled = []
    for category, config in FEATURE_CONFIG.items():
        if config['enabled']:
            for feature, is_enabled in config['features'].items():
                if is_enabled:
                    enabled.append(feature)  # Return the actual feature name, not category.feature
    return enabled

def get_feature_category(feature_name):
    """Get category for a given feature."""
    for category, config in FEATURE_CONFIG.items():
        if feature_name in config['features']:
            return category
    return None

def validate_feature_coverage(factor_list_path: str):
    """Validate feature config against actual factor list."""
    # Load factor list
    with open(factor_list_path, 'r') as f:
        raw_factors = {line.strip() for line in f.readlines()}
    
    # Get configured features and spot duplicates
    configured_features_list = list()
    for category in FEATURE_CONFIG.values():
        for feature in category['features'].keys():
            configured_features_list.append(feature)    
    configured_features = set(configured_features_list)

    duplicates = len(configured_features_list) - len(configured_features)
    if duplicates:
        print(f"{duplicates} duplicate features found in configured features")
    
    # Find discrepancies
    missing_in_config = raw_factors - configured_features
    missing_in_data = configured_features - raw_factors
    
    print("\n=== Validation Report ===")
    print(f"Total Factors in Data: {len(raw_factors)}")
    print(f"Total Configured Features: {len(configured_features)}")
    print(f"Missing in Config: {len(missing_in_config)}")
    print(f"Extra in Config: {len(missing_in_data)}")
    

def validate_company_names(data: pd.DataFrame):
    """Ensure COMPANYNAME exists and is non-null"""
    if 'COMPANYNAME' not in data.columns:
        raise KeyError("COMPANYNAME column missing from dataset")
    if data['COMPANYNAME'].isnull().any():
        raise ValueError("Null values found in COMPANYNAME column")

def validate_price_columns(data: pd.DataFrame):
    """Validate existence of price-related columns"""
    required = ['PRICECLOSE', 'COMPANYNAME', 'ASOFDATE']
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise KeyError(f"Missing price columns: {missing}")

if __name__ == "__main__":
    # --- Overall Statistics ---
    total_categories = len(FEATURE_CONFIG)
    total_features = sum(len(c['features']) for c in FEATURE_CONFIG.values())
    category_counts = {cat: len(config['features']) for cat, config in FEATURE_CONFIG.items()}
    
    # --- Enabled Statistics ---
    enabled_categories = 0
    total_enabled = 0
    enabled_counts = {}
    
    for cat, config in FEATURE_CONFIG.items():
        if config['enabled']:
            enabled_categories += 1
            enabled = sum(1 for f in config['features'].values() if f)
            enabled_counts[cat] = enabled
            total_enabled += enabled

    # --- Print Report ---
    print("=== Overall Statistics ===")
    print(f"Total Categories: {total_categories}")
    print(f"Total Features: {total_features}")
    print("\nFeatures per Category:")
    for cat, count in category_counts.items():
        print(f"  - {cat.title():<20} {count:>3} features")
    
    print("\n=== Enabled Statistics ===")
    print(f"Enabled Categories: {enabled_categories}/{total_categories}")
    print(f"Enabled Features: {total_enabled}/{total_features}")
    print("\nEnabled Features per Category:")
    for cat, count in enabled_counts.items():
        print(f"  - {cat.title():<20} {count:>3} features")
    
    validate_feature_coverage("data/raw/factor_list.txt")