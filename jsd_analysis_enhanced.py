"""
Enhanced Jensen-Shannon Divergence (JSD) Analysis for Real vs Synthetic BGP Data

This module compares probability distributions between real and synthetic datasets.

WHAT IS JSD?
============
Jensen-Shannon Divergence measures how different two probability distributions are:
- JSD = 0.0  → Identical distributions (perfect synthetic data)
- JSD = 0.5  → Moderately different distributions
- JSD = 1.0  → Completely different distributions (worst case)

WHY USE JSD?
============
- Unlike KL-divergence, JSD is symmetric and always finite
- Bounded [0,1] makes it easy to interpret
- Low JSD means a classifier cannot easily distinguish real from synthetic

COPY THE CODE BELOW INTO YOUR NOTEBOOK
======================================
"""

# ============================================================================
# CELL 1: PASTE THIS - Configuration and Helper Functions
# ============================================================================

CELL_1_CODE = '''
# ========================================
# JENSEN-SHANNON DIVERGENCE (JSD) ANALYSIS
# Enhanced Version with Clear Interpretations
# ========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CONFIGURATION - Adjust these thresholds
# ========================================
JSD_EXCELLENT_THRESHOLD = 0.1   # JSD < 0.1 = Excellent match
JSD_GOOD_THRESHOLD = 0.3        # JSD < 0.3 = Good match
N_BINS = 50                      # Number of histogram bins
LOW_PERCENTILE = 1               # Lower percentile for range (handles outliers)
HIGH_PERCENTILE = 99             # Upper percentile for range
RANDOM_STATE = 42

# ========================================
# HELPER FUNCTIONS
# ========================================

def jsd_quality_label(jsd_value):
    """
    Convert JSD value to human-readable quality label.

    Returns: (label, color) tuple
    """
    if np.isnan(jsd_value):
        return 'N/A', 'gray'
    if jsd_value < JSD_EXCELLENT_THRESHOLD:
        return 'Excellent', 'green'
    elif jsd_value < JSD_GOOD_THRESHOLD:
        return 'Good', 'orange'
    else:
        return 'Needs Work', 'red'


def create_histogram(data, n_bins=50, range_tuple=None, epsilon=1e-10):
    """
    Create normalized probability distribution from data.

    Returns uniform distribution for empty data (numerically stable).
    """
    data = data[np.isfinite(data)]

    # Return uniform distribution if no valid data
    if len(data) == 0:
        return np.full(n_bins, 1.0 / n_bins)

    hist, _ = np.histogram(data, bins=n_bins, range=range_tuple, density=False)
    hist = hist.astype(float)
    total = hist.sum()

    # Return uniform if all zeros
    if total == 0:
        return np.full(n_bins, 1.0 / n_bins)

    # Normalize to probability distribution
    hist /= total

    # Add epsilon for numerical stability (avoids log(0) in KL divergence)
    hist += epsilon
    hist /= hist.sum()  # Renormalize after adding epsilon

    return hist


def compute_jsd_divergence(p, q):
    """
    Compute Jensen-Shannon DIVERGENCE between two probability distributions.

    Note: scipy.jensenshannon returns the DISTANCE (sqrt of divergence).
    We square it to get the divergence, which is in [0, 1] with log base 2.

    Args:
        p: First probability distribution (must sum to 1)
        q: Second probability distribution (must sum to 1)

    Returns:
        JSD divergence value in [0, 1]
        - 0 = identical distributions
        - 1 = completely different distributions
    """
    return jensenshannon(p, q, base=2) ** 2


def compute_jsd_for_feature(real_data, synth_data, n_bins=50, low_q=1, high_q=99):
    """
    Compute JSD between real and synthetic data for a single feature.

    Uses percentile-based range to handle outliers robustly.

    Args:
        real_data: Array of real feature values
        synth_data: Array of synthetic feature values
        n_bins: Number of histogram bins
        low_q: Lower percentile for range (default 1)
        high_q: Upper percentile for range (default 99)

    Returns:
        Tuple of (jsd_value, real_histogram, synth_histogram, bin_range)
    """
    # Clean data once
    real_clean = real_data[np.isfinite(real_data)]
    synth_clean = synth_data[np.isfinite(synth_data)]
    all_data = np.concatenate([real_clean, synth_clean])

    # Handle empty data case
    if len(all_data) == 0:
        return np.nan, np.zeros(n_bins), np.zeros(n_bins), (0, 1)

    # Compute range using percentiles (robust to outliers)
    range_min, range_max = np.percentile(all_data, [low_q, high_q])

    # Handle constant data
    if range_min == range_max:
        range_max = range_min + 1

    bin_range = (range_min, range_max)

    # Create histograms with same bins for fair comparison
    real_hist = create_histogram(real_clean, n_bins, bin_range)
    synth_hist = create_histogram(synth_clean, n_bins, bin_range)

    # Compute JSD
    jsd = compute_jsd_divergence(real_hist, synth_hist)

    return jsd, real_hist, synth_hist, bin_range


print("JSD Analysis functions loaded successfully!")
print(f"  - Excellent threshold: JSD < {JSD_EXCELLENT_THRESHOLD}")
print(f"  - Good threshold: JSD < {JSD_GOOD_THRESHOLD}")
print(f"  - Histogram bins: {N_BINS}")
'''

# ============================================================================
# CELL 2: PASTE THIS - Data Verification
# ============================================================================

CELL_2_CODE = '''
# ========================================
# STEP 1: VERIFY YOUR DATA SOURCES
# ========================================
# This shows exactly what data will be compared

print("="*75)
print("STEP 1: DATA VERIFICATION")
print("="*75)

print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                     WHAT ARE WE COMPARING?                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   REAL DATA (from actual BGP traffic)                                   │
│   ├── Normal samples:  Discovered normal traffic (rrc05)                │
│   └── Anomaly samples: Known incidents with confirmed anomalies         │
│                                                                         │
│   SYNTHETIC DATA (generated by SMOTE)                                   │
│   ├── Normal samples:  SMOTE-generated normal patterns                  │
│   └── Anomaly samples: SMOTE-generated anomaly patterns                 │
│                                                                         │
│   GOAL: Check if synthetic distributions match real distributions       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
""")

print("\\n📁 REAL DATA DETAILS:")
print(f"   Shape: {X_real.shape[0]} samples × {X_real.shape[1]} features")
print(f"   Normal (y=0):  {sum(y_real==0):,} samples")
print(f"   Anomaly (y=1): {sum(y_real==1):,} samples")

print("\\n📁 SYNTHETIC DATA DETAILS:")
print(f"   Shape: {X_synthetic.shape[0]} samples × {X_synthetic.shape[1]} features")
print(f"   Normal (y=0):  {sum(y_synthetic==0):,} samples")
print(f"   Anomaly (y=1): {sum(y_synthetic==1):,} samples")

print("\\n📊 FEATURES TO ANALYZE:")
print(f"   Total: {len(common_features_all)} features")

# Verify data is valid
assert len(X_real) > 0, "ERROR: X_real is empty!"
assert len(X_synthetic) > 0, "ERROR: X_synthetic is empty!"
assert len(X_real.columns) == len(X_synthetic.columns), "ERROR: Feature mismatch!"

print("\\n✅ Data verification passed! Ready for JSD analysis.")
'''

# ============================================================================
# CELL 3: PASTE THIS - Compute JSD for All Features
# ============================================================================

CELL_3_CODE = '''
# ========================================
# STEP 2: COMPUTE JSD FOR ALL FEATURES
# ========================================

print("\\n" + "="*75)
print("STEP 2: COMPUTING JSD FOR ALL FEATURES")
print("="*75)

# Get top 10 features by importance (for focused analysis)
print("\\nFinding most important features using Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_real.fillna(X_real.median()), y_real)  # Use median imputation

feature_importance_df = pd.DataFrame({
    'feature': X_real.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

TOP_10_FEATURES = feature_importance_df.head(10)['feature'].tolist()

print("\\n🏆 TOP 10 MOST IMPORTANT FEATURES:")
print("   (These features are most useful for distinguishing Normal vs Anomaly)")
for i, row in feature_importance_df.head(10).iterrows():
    feat = row['feature']
    imp = row['importance']
    print(f"   {TOP_10_FEATURES.index(feat)+1:2d}. {feat:<35} (importance: {imp:.4f})")

# Compute JSD for all features
print("\\n\\nComputing JSD for each feature...")
print("   (This compares real vs synthetic distributions)")

jsd_results = []

for idx, feature in enumerate(common_features_all):
    # Get data for this feature
    real_data = X_real[feature].values
    synth_data = X_synthetic[feature].values

    # Overall JSD (all samples)
    jsd_overall, _, _, _ = compute_jsd_for_feature(
        real_data, synth_data, N_BINS, LOW_PERCENTILE, HIGH_PERCENTILE
    )

    # JSD for Normal class only
    jsd_normal, _, _, _ = compute_jsd_for_feature(
        X_real[y_real == 0][feature].values,
        X_synthetic[y_synthetic == 0][feature].values,
        N_BINS, LOW_PERCENTILE, HIGH_PERCENTILE
    )

    # JSD for Anomaly class only
    jsd_anomaly, _, _, _ = compute_jsd_for_feature(
        X_real[y_real == 1][feature].values,
        X_synthetic[y_synthetic == 1][feature].values,
        N_BINS, LOW_PERCENTILE, HIGH_PERCENTILE
    )

    # Get quality labels
    overall_quality, _ = jsd_quality_label(jsd_overall)
    normal_quality, _ = jsd_quality_label(jsd_normal)
    anomaly_quality, _ = jsd_quality_label(jsd_anomaly)

    jsd_results.append({
        'feature': feature,
        'jsd_overall': jsd_overall,
        'jsd_normal': jsd_normal,
        'jsd_anomaly': jsd_anomaly,
        'similarity_pct': (1 - jsd_overall) * 100 if not np.isnan(jsd_overall) else np.nan,
        'overall_quality': overall_quality,
        'normal_quality': normal_quality,
        'anomaly_quality': anomaly_quality,
        'is_top10': feature in TOP_10_FEATURES,
        'importance_rank': TOP_10_FEATURES.index(feature) + 1 if feature in TOP_10_FEATURES else None
    })

# Create results DataFrame
df_jsd = pd.DataFrame(jsd_results)
df_jsd = df_jsd.sort_values('jsd_overall')

print("\\n✅ JSD computation complete!")
'''

# ============================================================================
# CELL 4: PASTE THIS - Summary Statistics
# ============================================================================

CELL_4_CODE = '''
# ========================================
# STEP 3: SUMMARY STATISTICS
# ========================================

print("\\n" + "="*75)
print("STEP 3: JSD SUMMARY STATISTICS")
print("="*75)

print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                    HOW TO INTERPRET JSD VALUES                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   JSD = 0.00      Perfect match (distributions are identical)           │
│   JSD < 0.10      EXCELLENT - Very similar distributions                │
│   JSD < 0.30      GOOD - Reasonably similar distributions               │
│   JSD ≥ 0.30      NEEDS WORK - Distributions differ significantly       │
│   JSD = 1.00      Worst case (completely different distributions)       │
│                                                                         │
│   SIMILARITY % = (1 - JSD) × 100                                        │
│   Example: JSD = 0.15 → Similarity = 85%                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
""")

# Calculate statistics
mean_jsd = df_jsd['jsd_overall'].mean()
median_jsd = df_jsd['jsd_overall'].median()
min_jsd = df_jsd['jsd_overall'].min()
max_jsd = df_jsd['jsd_overall'].max()
min_feature = df_jsd.loc[df_jsd['jsd_overall'].idxmin(), 'feature']
max_feature = df_jsd.loc[df_jsd['jsd_overall'].idxmax(), 'feature']

n_excellent = (df_jsd['jsd_overall'] < JSD_EXCELLENT_THRESHOLD).sum()
n_good = ((df_jsd['jsd_overall'] >= JSD_EXCELLENT_THRESHOLD) &
          (df_jsd['jsd_overall'] < JSD_GOOD_THRESHOLD)).sum()
n_needs_work = (df_jsd['jsd_overall'] >= JSD_GOOD_THRESHOLD).sum()
total = len(df_jsd)

print("\\n📊 OVERALL STATISTICS:")
print(f"   ┌{'─'*50}┐")
print(f"   │ Mean JSD:        {mean_jsd:.4f} ({(1-mean_jsd)*100:.1f}% similar)         │")
print(f"   │ Median JSD:      {median_jsd:.4f} ({(1-median_jsd)*100:.1f}% similar)         │")
print(f"   │ Best (lowest):   {min_jsd:.4f} → {min_feature:<20} │")
print(f"   │ Worst (highest): {max_jsd:.4f} → {max_feature:<20} │")
print(f"   └{'─'*50}┘")

print("\\n📈 QUALITY BREAKDOWN:")
print(f"   ┌{'─'*55}┐")
print(f"   │ 🟢 EXCELLENT (JSD < {JSD_EXCELLENT_THRESHOLD}):  {n_excellent:2d} features ({n_excellent/total*100:5.1f}%)           │")
print(f"   │ 🟡 GOOD (JSD < {JSD_GOOD_THRESHOLD}):       {n_good:2d} features ({n_good/total*100:5.1f}%)           │")
print(f"   │ 🔴 NEEDS WORK (JSD ≥ {JSD_GOOD_THRESHOLD}): {n_needs_work:2d} features ({n_needs_work/total*100:5.1f}%)           │")
print(f"   └{'─'*55}┘")

# Class-specific analysis
mean_normal = df_jsd['jsd_normal'].mean()
mean_anomaly = df_jsd['jsd_anomaly'].mean()

print("\\n🔍 CLASS-SPECIFIC ANALYSIS:")
print(f"   ┌{'─'*60}┐")
print(f"   │ Normal Class:  Mean JSD = {mean_normal:.4f} ({(1-mean_normal)*100:.1f}% similar)        │")
print(f"   │ Anomaly Class: Mean JSD = {mean_anomaly:.4f} ({(1-mean_anomaly)*100:.1f}% similar)        │")
print(f"   └{'─'*60}┘")

if mean_normal > mean_anomaly + 0.1:
    print("\\n   ⚠️  WARNING: Synthetic NORMAL data matches poorly!")
    print("      The synthetic normal patterns differ significantly from real normal traffic.")
    print("      This may cause issues with TSTR (Train Synthetic, Test Real) performance.")
elif mean_anomaly > mean_normal + 0.1:
    print("\\n   ⚠️  WARNING: Synthetic ANOMALY data matches poorly!")
    print("      The synthetic anomaly patterns differ significantly from real anomalies.")
else:
    print("\\n   ✅ Both classes have similar matching quality.")
'''

# ============================================================================
# CELL 5: PASTE THIS - Top 10 Features Table
# ============================================================================

CELL_5_CODE = '''
# ========================================
# STEP 4: DETAILED RESULTS FOR TOP 10 FEATURES
# ========================================

print("\\n" + "="*75)
print("STEP 4: JSD RESULTS FOR TOP 10 MOST IMPORTANT FEATURES")
print("="*75)

print("""
These are the 10 features most important for anomaly detection.
We check how well synthetic data matches real data for each.
""")

# Filter to top 10 and sort by importance rank
df_top10 = df_jsd[df_jsd['is_top10']].copy()
df_top10 = df_top10.sort_values('importance_rank')

print(f"\\n{'Rank':<5} {'Feature':<30} {'Overall':<10} {'Normal':<10} {'Anomaly':<10} {'Quality'}")
print("─"*80)

for _, row in df_top10.iterrows():
    rank = int(row['importance_rank'])
    feature = row['feature'][:28]
    jsd_o = row['jsd_overall']
    jsd_n = row['jsd_normal']
    jsd_a = row['jsd_anomaly']
    quality = row['overall_quality']

    # Add emoji based on quality
    emoji = "🟢" if quality == "Excellent" else ("🟡" if quality == "Good" else "🔴")

    print(f"{rank:<5} {feature:<30} {jsd_o:.4f}     {jsd_n:.4f}     {jsd_a:.4f}     {emoji} {quality}")

print("─"*80)

# Interpretation
print("\\n📋 INTERPRETATION:")
print("   • Overall: JSD comparing ALL real vs ALL synthetic samples")
print("   • Normal:  JSD comparing only Normal class (real vs synthetic)")
print("   • Anomaly: JSD comparing only Anomaly class (real vs synthetic)")
print("\\n   Lower JSD = Better match = Synthetic data is more realistic")
'''

# ============================================================================
# CELL 6: PASTE THIS - Visualization 1 (Main Bar Chart)
# ============================================================================

CELL_6_CODE = '''
# ========================================
# STEP 5: VISUALIZATION - JSD Overview
# ========================================

print("\\n" + "="*75)
print("STEP 5: CREATING VISUALIZATIONS")
print("="*75)

# Prepare data for plotting
df_plot = df_top10.sort_values('jsd_overall')

fig, axes = plt.subplots(1, 2, figsize=(16, 9))

# ─────────────────────────────────────────────────────────────────
# LEFT PLOT: JSD Values (Lower is Better)
# ─────────────────────────────────────────────────────────────────
ax1 = axes[0]

# Color bars based on quality
colors = []
for jsd in df_plot['jsd_overall']:
    if jsd < JSD_EXCELLENT_THRESHOLD:
        colors.append('#2ecc71')  # Green
    elif jsd < JSD_GOOD_THRESHOLD:
        colors.append('#f39c12')  # Orange
    else:
        colors.append('#e74c3c')  # Red

bars1 = ax1.barh(range(len(df_plot)), df_plot['jsd_overall'], color=colors, edgecolor='black', linewidth=0.5)
ax1.set_yticks(range(len(df_plot)))
ax1.set_yticklabels(df_plot['feature'], fontsize=11)
ax1.invert_yaxis()

# Add threshold lines
ax1.axvline(x=JSD_EXCELLENT_THRESHOLD, color='green', linestyle='--', linewidth=2, label=f'Excellent (< {JSD_EXCELLENT_THRESHOLD})')
ax1.axvline(x=JSD_GOOD_THRESHOLD, color='orange', linestyle='--', linewidth=2, label=f'Good (< {JSD_GOOD_THRESHOLD})')

# Add value labels
for i, (bar, val) in enumerate(zip(bars1, df_plot['jsd_overall'])):
    ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

ax1.set_xlabel('Jensen-Shannon Divergence', fontsize=12, fontweight='bold')
ax1.set_title('JSD for Top 10 Features\\n(LOWER = More Similar = Better)',
              fontsize=14, fontweight='bold', color='darkblue')
ax1.legend(loc='lower right', fontsize=10)
ax1.set_xlim(0, max(0.4, df_plot['jsd_overall'].max() * 1.2))
ax1.grid(axis='x', alpha=0.3)

# ─────────────────────────────────────────────────────────────────
# RIGHT PLOT: Similarity Percentage (Higher is Better)
# ─────────────────────────────────────────────────────────────────
ax2 = axes[1]

similarity = df_plot['similarity_pct']
colors2 = []
for sim in similarity:
    if sim >= (1 - JSD_EXCELLENT_THRESHOLD) * 100:
        colors2.append('#2ecc71')  # Green
    elif sim >= (1 - JSD_GOOD_THRESHOLD) * 100:
        colors2.append('#f39c12')  # Orange
    else:
        colors2.append('#e74c3c')  # Red

bars2 = ax2.barh(range(len(df_plot)), similarity, color=colors2, edgecolor='black', linewidth=0.5)
ax2.set_yticks(range(len(df_plot)))
ax2.set_yticklabels(df_plot['feature'], fontsize=11)
ax2.invert_yaxis()

# Add threshold lines
ax2.axvline(x=90, color='green', linestyle='--', linewidth=2, label='Excellent (≥ 90%)')
ax2.axvline(x=70, color='orange', linestyle='--', linewidth=2, label='Good (≥ 70%)')

# Add value labels
for bar, val in zip(bars2, similarity):
    ax2.text(val + 1, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

ax2.set_xlabel('Similarity Score (%)', fontsize=12, fontweight='bold')
ax2.set_title('Distribution Similarity\\n(HIGHER = More Similar = Better)',
              fontsize=14, fontweight='bold', color='darkgreen')
ax2.legend(loc='lower right', fontsize=10)
ax2.set_xlim(0, 105)
ax2.grid(axis='x', alpha=0.3)

# Main title
fig.suptitle('How Well Does Synthetic Data Match Real Data?\\n(Top 10 Most Important Features)',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/jsd_overview.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\\n📊 WHAT THIS CHART SHOWS:")
print("   • Left:  JSD values - lower bars are better (more similar)")
print("   • Right: Similarity % - higher bars are better")
print("   • Green = Excellent match, Orange = Good match, Red = Needs improvement")
'''

# ============================================================================
# CELL 7: PASTE THIS - Visualization 2 (Class Comparison)
# ============================================================================

CELL_7_CODE = '''
# ========================================
# STEP 6: VISUALIZATION - Normal vs Anomaly Comparison
# ========================================

fig, ax = plt.subplots(figsize=(14, 10))

df_plot = df_top10.sort_values('jsd_overall')
x = np.arange(len(df_plot))
width = 0.35

# Create grouped bars
bars_normal = ax.barh(x - width/2, df_plot['jsd_normal'], width,
                       label='Normal Class', color='#3498db', edgecolor='black', linewidth=0.5)
bars_anomaly = ax.barh(x + width/2, df_plot['jsd_anomaly'], width,
                        label='Anomaly Class', color='#e74c3c', edgecolor='black', linewidth=0.5)

ax.set_yticks(x)
ax.set_yticklabels(df_plot['feature'], fontsize=11)
ax.invert_yaxis()

# Add threshold lines
ax.axvline(x=JSD_EXCELLENT_THRESHOLD, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(x=JSD_GOOD_THRESHOLD, color='orange', linestyle='--', linewidth=2, alpha=0.7)

# Add value labels
for bar in bars_normal:
    w = bar.get_width()
    if w > 0.01:
        ax.text(w + 0.01, bar.get_y() + bar.get_height()/2,
                f'{w:.2f}', va='center', fontsize=9, color='#2980b9', fontweight='bold')

for bar in bars_anomaly:
    w = bar.get_width()
    if w > 0.01:
        ax.text(w + 0.01, bar.get_y() + bar.get_height()/2,
                f'{w:.2f}', va='center', fontsize=9, color='#c0392b', fontweight='bold')

ax.set_xlabel('Jensen-Shannon Divergence (Lower = Better)', fontsize=12, fontweight='bold')
ax.set_title('JSD Comparison: Normal Class vs Anomaly Class\\n(How well does synthetic match real for each class?)',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=12)
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, max(0.8, max(df_plot['jsd_normal'].max(), df_plot['jsd_anomaly'].max()) * 1.1))

# Add interpretation box
textstr = """INTERPRETATION:
• Blue bars: How well synthetic Normal matches real Normal
• Red bars: How well synthetic Anomaly matches real Anomaly
• If blue >> red: Synthetic struggles with Normal patterns
• If red >> blue: Synthetic struggles with Anomaly patterns"""

props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/jsd_by_class.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

# Print analysis
mean_normal = df_plot['jsd_normal'].mean()
mean_anomaly = df_plot['jsd_anomaly'].mean()

print("\\n📊 CLASS-SPECIFIC ANALYSIS:")
print(f"   Average JSD for Normal class:  {mean_normal:.4f}")
print(f"   Average JSD for Anomaly class: {mean_anomaly:.4f}")

if mean_normal > mean_anomaly + 0.1:
    print("\\n   ⚠️  FINDING: Synthetic data matches ANOMALIES better than NORMAL traffic!")
    print("      → Your SMOTE-generated normal patterns don't capture real normal behavior well.")
    print("      → This may cause higher false positive rates in TSTR experiments.")
elif mean_anomaly > mean_normal + 0.1:
    print("\\n   ⚠️  FINDING: Synthetic data matches NORMAL better than ANOMALIES!")
    print("      → Your SMOTE-generated anomaly patterns don't capture real anomalies well.")
    print("      → This may cause missed detections in TSTR experiments.")
else:
    print("\\n   ✅ FINDING: Synthetic data matches both classes reasonably well.")
'''

# ============================================================================
# CELL 8: PASTE THIS - Visualization 3 (Distribution Histograms)
# ============================================================================

CELL_8_CODE = '''
# ========================================
# STEP 7: VISUALIZATION - Distribution Histograms
# ========================================

print("\\nCreating distribution comparison plots...")

n_features = len(TOP_10_FEATURES)
fig, axes = plt.subplots(5, 2, figsize=(16, 25))
axes = axes.flatten()

for i, feature in enumerate(TOP_10_FEATURES):
    ax = axes[i]

    # Get data
    real_data = X_real[feature].values
    synth_data = X_synthetic[feature].values

    # Compute JSD and get histogram info
    jsd_val, real_hist, synth_hist, bin_range = compute_jsd_for_feature(
        real_data, synth_data, N_BINS, LOW_PERCENTILE, HIGH_PERCENTILE
    )

    # Create bins
    bins = np.linspace(bin_range[0], bin_range[1], N_BINS + 1)

    # Filter data to range for cleaner visualization
    real_filtered = real_data[(real_data >= bin_range[0]) & (real_data <= bin_range[1])]
    synth_filtered = synth_data[(synth_data >= bin_range[0]) & (synth_data <= bin_range[1])]

    # Plot histograms
    ax.hist(real_filtered, bins=bins, alpha=0.6, label='Real Data',
            color='#3498db', density=True, edgecolor='black', linewidth=0.3)
    ax.hist(synth_filtered, bins=bins, alpha=0.6, label='Synthetic Data',
            color='#e74c3c', density=True, edgecolor='black', linewidth=0.3)

    # Get quality label and color
    quality, color = jsd_quality_label(jsd_val)

    # Title with JSD info
    ax.set_title(f'{feature}\\nJSD = {jsd_val:.4f} ({quality})',
                 fontsize=12, fontweight='bold', color=color)
    ax.set_xlabel('Value', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add similarity annotation
    similarity = (1 - jsd_val) * 100
    ax.annotate(f'Similarity: {similarity:.1f}%', xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=10, fontweight='bold', color=color,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('Distribution Comparison: Real vs Synthetic Data\\n(How well do the distributions overlap?)',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/jsd_distributions.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\\n📊 WHAT THESE HISTOGRAMS SHOW:")
print("   • Blue = Real data distribution")
print("   • Red = Synthetic data distribution")
print("   • More overlap = Better match = Lower JSD")
print("   • Green title = Excellent, Orange = Good, Red = Needs work")
'''

# ============================================================================
# CELL 9: PASTE THIS - Visualization 4 (Heatmap)
# ============================================================================

CELL_9_CODE = '''
# ========================================
# STEP 8: VISUALIZATION - JSD Heatmap
# ========================================

fig, ax = plt.subplots(figsize=(10, 12))

# Prepare heatmap data
df_heatmap = df_top10[['feature', 'jsd_normal', 'jsd_anomaly', 'jsd_overall']].copy()
df_heatmap = df_heatmap.set_index('feature')
df_heatmap.columns = ['Normal\\nClass', 'Anomaly\\nClass', 'Overall']
df_heatmap = df_heatmap.sort_values('Overall')

# Create heatmap
sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='RdYlGn_r',
            vmin=0, vmax=0.5, ax=ax,
            cbar_kws={'label': 'JSD (Lower = Better)', 'shrink': 0.8},
            linewidths=0.5, linecolor='white')

ax.set_title('JSD Heatmap: Real vs Synthetic by Feature and Class\\n(Green = Good Match, Red = Poor Match)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('\\nComparison Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')

# Rotate x labels for better readability
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/jsd_heatmap.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\\n📊 HOW TO READ THE HEATMAP:")
print("   • Each row is a feature")
print("   • Each column shows JSD for: Normal class, Anomaly class, Overall")
print("   • GREEN cells = Low JSD = Good match (synthetic ≈ real)")
print("   • RED cells = High JSD = Poor match (synthetic ≠ real)")
print("   • Look for patterns: Are certain features consistently red?")
'''

# ============================================================================
# CELL 10: PASTE THIS - Final Summary and Save Results
# ============================================================================

CELL_10_CODE = '''
# ========================================
# STEP 9: FINAL SUMMARY AND SAVE RESULTS
# ========================================

print("\\n" + "="*75)
print("STEP 9: FINAL SUMMARY")
print("="*75)

# Calculate key metrics
overall_similarity = (1 - df_jsd['jsd_overall'].mean()) * 100
normal_similarity = (1 - df_jsd['jsd_normal'].mean()) * 100
anomaly_similarity = (1 - df_jsd['jsd_anomaly'].mean()) * 100

n_excellent = (df_jsd['jsd_overall'] < JSD_EXCELLENT_THRESHOLD).sum()
n_good = ((df_jsd['jsd_overall'] >= JSD_EXCELLENT_THRESHOLD) &
          (df_jsd['jsd_overall'] < JSD_GOOD_THRESHOLD)).sum()
n_needs_work = (df_jsd['jsd_overall'] >= JSD_GOOD_THRESHOLD).sum()

print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                    JSD ANALYSIS FINAL REPORT                            │
├─────────────────────────────────────────────────────────────────────────┤""")

print(f"""│                                                                         │
│   OVERALL SIMILARITY: {overall_similarity:5.1f}%                                         │
│   ├── Normal Class:   {normal_similarity:5.1f}%                                         │
│   └── Anomaly Class:  {anomaly_similarity:5.1f}%                                         │
│                                                                         │
│   FEATURE QUALITY BREAKDOWN:                                            │
│   ├── 🟢 Excellent: {n_excellent:2d} features ({n_excellent/len(df_jsd)*100:5.1f}%)                             │
│   ├── 🟡 Good:      {n_good:2d} features ({n_good/len(df_jsd)*100:5.1f}%)                             │
│   └── 🔴 Needs Work: {n_needs_work:2d} features ({n_needs_work/len(df_jsd)*100:5.1f}%)                             │
│                                                                         │""")

# Determine overall verdict
if overall_similarity >= 90:
    verdict = "EXCELLENT - Synthetic data closely matches real data"
    verdict_emoji = "🎉"
elif overall_similarity >= 80:
    verdict = "GOOD - Synthetic data reasonably matches real data"
    verdict_emoji = "✅"
elif overall_similarity >= 70:
    verdict = "FAIR - Some features need improvement"
    verdict_emoji = "⚠️"
else:
    verdict = "NEEDS WORK - Significant distribution differences"
    verdict_emoji = "❌"

print(f"""│   VERDICT: {verdict_emoji} {verdict:<52} │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
""")

# Specific findings
print("\\n📋 KEY FINDINGS:")

if normal_similarity < anomaly_similarity - 10:
    print(f"   ⚠️  Synthetic NORMAL data is weaker ({normal_similarity:.1f}% vs {anomaly_similarity:.1f}% for anomaly)")
    print("      → Consider improving SMOTE parameters for normal class")
    print("      → This may affect TSTR performance (more false positives)")
elif anomaly_similarity < normal_similarity - 10:
    print(f"   ⚠️  Synthetic ANOMALY data is weaker ({anomaly_similarity:.1f}% vs {normal_similarity:.1f}% for normal)")
    print("      → Consider improving SMOTE parameters for anomaly class")
    print("      → This may affect TSTR performance (more missed detections)")
else:
    print("   ✅ Both classes have similar quality - balanced synthetic generation")

# Find problematic features
problematic = df_jsd[df_jsd['jsd_overall'] >= JSD_GOOD_THRESHOLD]['feature'].tolist()
if problematic:
    print(f"\\n   🔴 Features needing improvement: {', '.join(problematic)}")
else:
    print("\\n   ✅ No features with critical JSD values")

# Save results
df_jsd.to_csv(f"{OUTPUT_DIR}/jsd_results_all_features.csv", index=False)
df_top10.to_csv(f"{OUTPUT_DIR}/jsd_results_top10_features.csv", index=False)

print("\\n" + "─"*75)
print("📁 FILES SAVED:")
print(f"   • {OUTPUT_DIR}/jsd_results_all_features.csv")
print(f"   • {OUTPUT_DIR}/jsd_results_top10_features.csv")
print(f"   • {OUTPUT_DIR}/jsd_overview.png")
print(f"   • {OUTPUT_DIR}/jsd_by_class.png")
print(f"   • {OUTPUT_DIR}/jsd_distributions.png")
print(f"   • {OUTPUT_DIR}/jsd_heatmap.png")
print("─"*75)
'''

# Print instructions
if __name__ == "__main__":
    print("="*75)
    print("ENHANCED JSD ANALYSIS - COPY INSTRUCTIONS")
    print("="*75)
    print("""
    Copy each CELL_X_CODE into separate cells in your Jupyter notebook.

    Order:
    1. CELL_1_CODE - Configuration and helper functions
    2. CELL_2_CODE - Data verification
    3. CELL_3_CODE - Compute JSD for all features
    4. CELL_4_CODE - Summary statistics
    5. CELL_5_CODE - Top 10 features table
    6. CELL_6_CODE - Visualization: JSD Overview
    7. CELL_7_CODE - Visualization: Class comparison
    8. CELL_8_CODE - Visualization: Distribution histograms
    9. CELL_9_CODE - Visualization: Heatmap
    10. CELL_10_CODE - Final summary and save

    Run cells in order after loading your data!
    """)
