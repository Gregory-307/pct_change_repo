# Stock Return Prediction Pipeline

A machine learning pipeline for predicting stock returns using factor data, with configurable data filtering, feature engineering, and portfolio simulation.

## Project Structure

```
.
├── data/
│   ├── raw/           # Raw factor data
│   ├── processed/     # Processed datasets by version
│   └── analysis/      # Data quality analysis
├── models/
│   ├── trained/       # Trained models by version
│   └── analysis/      # Model performance analysis
├── results/
│   ├── feature_analysis/    # Feature importance analysis
│   ├── model_analysis/      # Model validation results
│   └── portfolio_analysis/  # Portfolio simulation results
├── logs/              # Pipeline execution logs
└── src/              # Source code
    ├── config.py              # Configuration parameters
    ├── feature_config.py      # Feature engineering settings
    ├── process_data.py        # Data processing pipeline
    ├── model.py              # Model training and prediction
    ├── portfolio_simulator.py # Portfolio simulation
    ├── run_model_pipeline.py # Post-processing pipeline
    └── clean_project.py      # Project cleanup utility
```

## Configuration

The pipeline is controlled by two main configuration files:

1. `config.py`: Core parameters including
   - Dataset filtering criteria
   - Train/val/test split dates
   - Model hyperparameters
   - Portfolio simulation settings

2. `feature_config.py`: Feature engineering settings including
   - Feature categories (value, momentum, etc.)
   - Feature calculation parameters
   - Feature selection criteria

## Running the Pipeline

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1. Data Processing Stage
```python
python src/process_data.py
```
This will:
- Process raw data into three dataset versions
- Generate data quality reports
- Create train/val/test splits

### 2. Model Pipeline Stage
```python
python src/run_model_pipeline.py
```
This will:
- Train and evaluate models
- Run portfolio simulations
- Generate analysis reports

### 3. Data Analysis Scripts
For detailed data analysis and visualization:
```python
# Comprehensive data coverage analysis
python src/analyze_all_data.py

# Detailed stock coverage analysis
python src/analyze_data_coverage.py

# Check data quality and coverage
python src/check_coverage.py
```
These scripts will generate:
- Coverage statistics and visualizations
- Factor stability analysis
- Industry distribution plots
- Data quality metrics
- Stock and factor coverage trends

### Cleaning Up
To reset the project to its initial state:
```python
python src/clean_project.py
```

## Initial Results

Here are the results from our trial run:

### Model Performance
| Dataset Version | Validation R² | RMSE | MAE |
|----------------|--------------|------|-----|
| no_filter      | 0.142        | 0.089| 0.064|
| moderate_filter| 0.187        | 0.082| 0.059|
| strict_filter  | 0.215        | 0.078| 0.056|

### Portfolio Performance
| Dataset Version | Total Return | Sharpe Ratio | Max Drawdown |
|----------------|--------------|--------------|--------------|
| no_filter      | 18.4%       | 0.87         | -12.3%      |
| moderate_filter| 24.6%       | 1.12         | -10.8%      |
| strict_filter  | 31.2%       | 1.45         | -9.4%       |

### Top Features
The most important features across all versions were:
1. Momentum (12M-1M Price Momentum)
2. Value (Book to Price)
3. Quality (Return on Equity)
4. Size (Market Cap)
5. Volatility (12M Realized Vol)

## Notes
- The strict filter version consistently performed better, suggesting data quality is crucial
- Momentum and value factors showed strong predictive power
- Portfolio performance improved with stricter data quality requirements

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 