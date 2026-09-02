import pandas as pd

def process(file_name):
    df = pd.read_csv(file_name, header=None, names=['x', 'y'])
    std_vals = [10, 20, 50, 100, 200, 500]
    df['n'] = df['x'].apply(lambda x: min(std_vals, key=lambda v: abs(v-x)))
    
    results = {}
    for n, group in df.groupby('n'):
        y_vals = group['y'].sort_values().values
        if len(y_vals) == 3:
            lower, center, upper = y_vals
            std = (upper - lower) / 2
            results[n] = {'mean': center, 'std': std}
    return results

ppo = process('runs/scaling_agents_seeded/data_original_plot_ppo.csv')
sac = process('runs/scaling_agents_seeded/data_original_plot_sac.csv')

ours_data = {
    10: {'mean': 21.966, 'std': 0.028},
    20: {'mean': 23.378, 'std': 0.102},
    50: {'mean': 26.485, 'std': 0.112},
    100: {'mean': 27.976, 'std': 0.121},
    200: {'mean': 29.630, 'std': 0.079},
    500: {'mean': 35.767, 'std': 0.113}
}

print(r"\begin{tabular}{r c c S[table-format=3.0] c S[table-format=3.0]}")
print(r"    \toprule")
print(r"    & {ours} & {PPO-CPU} & {speedup} & {SAC-CPU} & {speedup} \\")
print(r"    $n$ & {(s)} & {(h)} & {($\times$)} & {(h)} & {($\times$)} \\")
print(r"    \midrule")

for n in [10, 20, 50, 100, 200, 500]:
    o_mean, o_std = ours_data[n]['mean'], ours_data[n]['std']
    p_mean, p_std = ppo[n]['mean'], ppo[n]['std']
    s_mean, s_std = sac[n]['mean'], sac[n]['std']
    
    speedup_ppo = round((p_mean * 3600) / o_mean)
    speedup_sac = round((s_mean * 3600) / o_mean)
    
    # Format with 1 decimal for ours, 2 for reference
    ours_str = f"${o_mean:.1f} \\pm {o_std:.1f}$"
    ppo_str = f"${p_mean:.2f} \\pm {p_std:.2f}$"
    sac_str = f"${s_mean:.2f} \\pm {s_std:.2f}$"
    
    print(f"    {n:<3} & {ours_str} & {ppo_str} & {speedup_ppo:>3} & {sac_str} & {speedup_sac:>3} \\\\")

print(r"    \bottomrule")
print(r"\end{tabular}")
