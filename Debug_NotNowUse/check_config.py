import yaml

with open('pose_transfer/config/default.yaml') as f:
    config = yaml.safe_load(f)
    cross_filter = config.get('cross_filter', {})
    print(f"Cross-Filter enabled: {cross_filter.get('enabled', False)}")
    print(f"Body confidence threshold: {cross_filter.get('body_confidence_threshold', 0.3)}")
