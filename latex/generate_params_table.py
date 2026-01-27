import json
import math

# Load best params
with open('task_data/models_optuna_tscv/best_params.json', 'r') as f:
    data = json.load(f)
    params = data['best_params']

# Define ranges
ranges = {
    'eta': (0.01, 0.1, True), # (low, high, is_log)
    'max_depth': (2, 8, False),
    'min_child_weight': (1, 50, True),
    'subsample': (0.6, 1.0, False),
    'colsample_bytree': (0.6, 1.0, False),
    'lambda': (0.01, 10.0, True),
    'alpha': (0.0001, 1.0, True),
    'gamma': (0.0001, 5.0, True)
}

def get_position(val, low, high, is_log):
    if is_log:
        # Map to log space [log(low), log(high)]
        pos = (math.log(val) - math.log(low)) / (math.log(high) - math.log(low))
    else:
        pos = (val - low) / (high - low)
    
    if pos < 0.2: return "Lower bound"
    if pos < 0.4: return "Lower-mid"
    if pos < 0.6: return "Center"
    if pos < 0.8: return "Upper-mid"
    return "Upper bound"

# Header
print(r"\begin{table}[ht]")
print(r"\centering")
print(r"\caption{Main Model: Hyperparameter Position Analysis}")
print(r"\label{tab:hyperparameters}")
print(r"\begin{tabular}{lllll}")
print(r"\hline")
print(r"\textbf{Parameter} & \textbf{Status} & \textbf{Range} & \textbf{Final Value} & \textbf{Position} \\")
print(r"\hline")

for key, (low, high, is_log) in ranges.items():
    val = params.get(key)
    # Check for name mapping if necessary (lambda/alpha)
    if key == 'lambda': val = params.get('lambda', params.get('reg_lambda'))
    if key == 'alpha': val = params.get('alpha', params.get('reg_alpha'))
    
    pos_str = get_position(val, low, high, is_log)
    
    # Format for LaTeX
    range_str = f"[{low},\,{high}]" + (" (log)" if is_log else "")
    if key == 'max_depth': range_str = r"\{2,\dots,8}"
    
    label = key.replace('_', r'\_')
    if key == 'lambda': label = r"lambda (L2)"
    if key == 'alpha': label = r"alpha (L1)"
    
    print(f"{label} & tuned & ${range_str}$ & {val:.4f} & {pos_str} \\")

print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")
