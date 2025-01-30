"""Temporary script to analyze feature discrepancies."""

from config_features import FEATURE_CONFIG

def get_all_configured_features():
    """Get all configured features with their categories."""
    features_dict = {}
    for category, config in FEATURE_CONFIG.items():
        for feature in config['features'].keys():
            if feature in features_dict:
                print(f"Duplicate found: {feature}")
                print(f"  Already in: {features_dict[feature]}")
                print(f"  Also in: {category}")
            features_dict[feature] = category
    return features_dict

def read_factor_list():
    """Read the factor list from file."""
    with open("data/raw/factor_list.txt", 'r') as f:
        return {line.strip() for line in f.readlines()}

def main():
    # Get all configured features
    configured_features = get_all_configured_features()
    
    # Get all raw factors
    raw_factors = read_factor_list()
    
    # Find missing factors
    missing_factors = raw_factors - set(configured_features.keys())
    print("\n=== Missing Factors ===")
    for factor in sorted(missing_factors):
        print(f"  - {factor}")
    
    # Find extra factors
    extra_factors = set(configured_features.keys()) - raw_factors
    print("\n=== Extra Factors ===")
    for factor in sorted(extra_factors):
        print(f"  - {factor} (in category: {configured_features[factor]})")

if __name__ == "__main__":
    main() 