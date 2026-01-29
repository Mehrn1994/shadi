"""
Jensen-Shannon Divergence (JSD) Analysis for Real vs Synthetic BGP Data

This module provides functions to compute and visualize JSD between
real and synthetic datasets for the top features.

JSD is a symmetric measure of divergence between two probability distributions:
- JSD = 0: Identical distributions
- JSD = 1: Maximally different distributions (using log base 2)

Usage:
    # After loading X_real, X_synthetic, y_real, y_synthetic in your notebook:
    from jsd_distribution_analysis import run_jsd_analysis
    results = run_jsd_analysis(X_real, X_synthetic, y_real, y_synthetic, common_features_all)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon Divergence between two probability distributions.

    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)

    Uses log base 2 so JSD is in [0, 1].

    Args:
        p: First probability distribution (normalized)
        q: Second probability distribution (normalized)

    Returns:
        JSD value in [0, 1]
    """
    # scipy.spatial.distance.jensenshannon returns sqrt(JSD), so we square it
    # It uses log base e by default, but we want log base 2
    # Actually, jensenshannon with base=2 returns sqrt(JSD) in [0, 1]
    return jensenshannon(p, q, base=2) ** 2


def create_histogram(data: np.ndarray, n_bins: int = 50,
                     range_tuple: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Create normalized histogram (probability distribution) from data.

    Args:
        data: Input data array
        n_bins: Number of histogram bins
        range_tuple: Optional (min, max) range for bins

    Returns:
        Normalized histogram (sums to 1)
    """
    # Remove NaN and infinite values
    data = data[np.isfinite(data)]

    if len(data) == 0:
        return np.zeros(n_bins)

    # Compute histogram
    hist, _ = np.histogram(data, bins=n_bins, range=range_tuple, density=False)

    # Normalize to probability distribution
    hist = hist.astype(float)
    if hist.sum() > 0:
        hist = hist / hist.sum()

    # Add small epsilon to avoid zeros (for KL divergence stability)
    epsilon = 1e-10
    hist = hist + epsilon
    hist = hist / hist.sum()  # Renormalize

    return hist


def compute_jsd_for_feature(real_data: np.ndarray, synth_data: np.ndarray,
                            n_bins: int = 50) -> Tuple[float, np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    Compute JSD between real and synthetic data for a single feature.

    Args:
        real_data: Real data for the feature
        synth_data: Synthetic data for the feature
        n_bins: Number of histogram bins

    Returns:
        Tuple of (jsd_value, real_hist, synth_hist, bin_range)
    """
    # Combine to find common range
    all_data = np.concatenate([real_data[np.isfinite(real_data)],
                               synth_data[np.isfinite(synth_data)]])

    if len(all_data) == 0:
        return 1.0, np.zeros(n_bins), np.zeros(n_bins), (0, 1)

    # Use percentiles to handle outliers
    range_min = np.percentile(all_data, 1)
    range_max = np.percentile(all_data, 99)

    if range_min == range_max:
        range_max = range_min + 1

    bin_range = (range_min, range_max)

    # Create histograms with same bins
    real_hist = create_histogram(real_data, n_bins, bin_range)
    synth_hist = create_histogram(synth_data, n_bins, bin_range)

    # Compute JSD
    jsd = compute_jsd(real_hist, synth_hist)

    return jsd, real_hist, synth_hist, bin_range


def compute_jsd_all_features(X_real: pd.DataFrame, X_synth: pd.DataFrame,
                             features: List[str], n_bins: int = 50) -> pd.DataFrame:
    """
    Compute JSD for all features between real and synthetic data.

    Args:
        X_real: Real data DataFrame
        X_synth: Synthetic data DataFrame
        features: List of feature names
        n_bins: Number of histogram bins

    Returns:
        DataFrame with JSD results for each feature
    """
    results = []

    for feature in features:
        if feature not in X_real.columns or feature not in X_synth.columns:
            continue

        real_data = X_real[feature].values
        synth_data = X_synth[feature].values

        jsd, _, _, _ = compute_jsd_for_feature(real_data, synth_data, n_bins)

        # Compute additional statistics
        real_mean = np.nanmean(real_data)
        synth_mean = np.nanmean(synth_data)
        real_std = np.nanstd(real_data)
        synth_std = np.nanstd(synth_data)

        results.append({
            'feature': feature,
            'jsd': jsd,
            'jsd_similarity': 1 - jsd,  # Higher is better
            'real_mean': real_mean,
            'synth_mean': synth_mean,
            'mean_diff_pct': abs(real_mean - synth_mean) / (abs(real_mean) + 1e-10) * 100,
            'real_std': real_std,
            'synth_std': synth_std
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('jsd', ascending=True)

    return df_results


def compute_jsd_by_class(X_real: pd.DataFrame, X_synth: pd.DataFrame,
                         y_real: np.ndarray, y_synth: np.ndarray,
                         features: List[str], n_bins: int = 50) -> pd.DataFrame:
    """
    Compute JSD separately for Normal (class 0) and Anomaly (class 1) samples.

    Args:
        X_real: Real data DataFrame
        X_synth: Synthetic data DataFrame
        y_real: Real labels (0=Normal, 1=Anomaly)
        y_synth: Synthetic labels
        features: List of feature names
        n_bins: Number of histogram bins

    Returns:
        DataFrame with JSD results by class
    """
    results = []

    for feature in features:
        if feature not in X_real.columns or feature not in X_synth.columns:
            continue

        for class_val, class_name in [(0, 'Normal'), (1, 'Anomaly')]:
            real_mask = y_real == class_val
            synth_mask = y_synth == class_val

            real_data = X_real.loc[real_mask, feature].values
            synth_data = X_synth.loc[synth_mask, feature].values

            if len(real_data) == 0 or len(synth_data) == 0:
                continue

            jsd, _, _, _ = compute_jsd_for_feature(real_data, synth_data, n_bins)

            results.append({
                'feature': feature,
                'class': class_name,
                'jsd': jsd,
                'jsd_similarity': 1 - jsd,
                'n_real': len(real_data),
                'n_synth': len(synth_data)
            })

    df_results = pd.DataFrame(results)
    return df_results


def plot_jsd_bar_chart(df_jsd: pd.DataFrame, top_n: int = 10,
                       title: str = 'JSD Comparison: Real vs Synthetic',
                       figsize: Tuple[int, int] = (14, 8),
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Create bar chart showing JSD values for top features.

    Args:
        df_jsd: DataFrame with JSD results
        top_n: Number of top features to show
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Select top N features (sorted by JSD ascending = best first)
    df_plot = df_jsd.head(top_n).copy()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: JSD values (lower is better)
    ax1 = axes[0]
    colors = plt.cm.RdYlGn_r(df_plot['jsd'].values / df_plot['jsd'].max())
    bars1 = ax1.barh(df_plot['feature'], df_plot['jsd'], color=colors)
    ax1.set_xlabel('Jensen-Shannon Divergence', fontsize=12)
    ax1.set_ylabel('Feature', fontsize=12)
    ax1.set_title('JSD (Lower = More Similar)', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.axvline(x=0.1, color='green', linestyle='--', alpha=0.7, label='Excellent (< 0.1)')
    ax1.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='Good (< 0.3)')
    ax1.legend(loc='lower right')

    # Add value labels
    for bar, val in zip(bars1, df_plot['jsd']):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)

    # Plot 2: JSD Similarity (higher is better)
    ax2 = axes[1]
    colors = plt.cm.RdYlGn(df_plot['jsd_similarity'].values)
    bars2 = ax2.barh(df_plot['feature'], df_plot['jsd_similarity'] * 100, color=colors)
    ax2.set_xlabel('JSD Similarity Score (%)', fontsize=12)
    ax2.set_ylabel('Feature', fontsize=12)
    ax2.set_title('JSD Similarity (Higher = More Similar)', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    ax2.axvline(x=90, color='green', linestyle='--', alpha=0.7, label='Excellent (> 90%)')
    ax2.axvline(x=70, color='orange', linestyle='--', alpha=0.7, label='Good (> 70%)')
    ax2.legend(loc='lower right')

    # Add value labels
    for bar, val in zip(bars2, df_plot['jsd_similarity']):
        ax2.text(val * 100 + 1, bar.get_y() + bar.get_height()/2,
                f'{val*100:.1f}%', va='center', fontsize=10)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_jsd_by_class(df_jsd_by_class: pd.DataFrame, top_n: int = 10,
                      figsize: Tuple[int, int] = (14, 10),
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Create grouped bar chart showing JSD by class (Normal vs Anomaly).

    Args:
        df_jsd_by_class: DataFrame with JSD results by class
        top_n: Number of features to show
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Pivot for easier plotting
    df_pivot = df_jsd_by_class.pivot(index='feature', columns='class', values='jsd')

    # Select top features based on average JSD
    df_pivot['avg_jsd'] = df_pivot.mean(axis=1)
    df_pivot = df_pivot.sort_values('avg_jsd').head(top_n)
    df_pivot = df_pivot.drop('avg_jsd', axis=1)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(df_pivot))
    width = 0.35

    bars1 = ax.barh(x - width/2, df_pivot['Normal'], width, label='Normal', color='#3498db', alpha=0.8)
    bars2 = ax.barh(x + width/2, df_pivot['Anomaly'], width, label='Anomaly', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Jensen-Shannon Divergence', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('JSD by Class: Real vs Synthetic\n(Lower = More Similar)',
                 fontsize=14, fontweight='bold')
    ax.set_yticks(x)
    ax.set_yticklabels(df_pivot.index)
    ax.legend(loc='lower right')
    ax.invert_yaxis()

    # Add reference lines
    ax.axvline(x=0.1, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5, linewidth=1)

    # Add value labels
    for bar in bars1:
        width_val = bar.get_width()
        ax.text(width_val + 0.01, bar.get_y() + bar.get_height()/2,
               f'{width_val:.3f}', va='center', fontsize=9, color='#2980b9')
    for bar in bars2:
        width_val = bar.get_width()
        ax.text(width_val + 0.01, bar.get_y() + bar.get_height()/2,
               f'{width_val:.3f}', va='center', fontsize=9, color='#c0392b')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_distribution_comparison(X_real: pd.DataFrame, X_synth: pd.DataFrame,
                                  features: List[str], n_bins: int = 50,
                                  figsize: Tuple[int, int] = (16, 20),
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Create histogram overlays comparing real vs synthetic distributions.

    Args:
        X_real: Real data DataFrame
        X_synth: Synthetic data DataFrame
        features: List of features to plot
        n_bins: Number of histogram bins
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]

        real_data = X_real[feature].values
        synth_data = X_synth[feature].values

        # Compute JSD
        jsd, real_hist, synth_hist, bin_range = compute_jsd_for_feature(
            real_data, synth_data, n_bins
        )

        # Create bins
        bins = np.linspace(bin_range[0], bin_range[1], n_bins + 1)

        # Plot histograms
        ax.hist(real_data[(real_data >= bin_range[0]) & (real_data <= bin_range[1])],
                bins=bins, alpha=0.5, label='Real', color='#3498db', density=True)
        ax.hist(synth_data[(synth_data >= bin_range[0]) & (synth_data <= bin_range[1])],
                bins=bins, alpha=0.5, label='Synthetic', color='#e74c3c', density=True)

        # Color code title based on JSD quality
        if jsd < 0.1:
            title_color = 'green'
            quality = 'Excellent'
        elif jsd < 0.3:
            title_color = 'orange'
            quality = 'Good'
        else:
            title_color = 'red'
            quality = 'Needs Improvement'

        ax.set_title(f'{feature}\nJSD = {jsd:.4f} ({quality})',
                    fontsize=11, fontweight='bold', color=title_color)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Distribution Comparison: Real vs Synthetic\n(Histogram Overlay with JSD)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_jsd_heatmap_by_class(df_jsd_by_class: pd.DataFrame,
                               figsize: Tuple[int, int] = (12, 10),
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create heatmap showing JSD values by feature and class.

    Args:
        df_jsd_by_class: DataFrame with JSD results by class
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Pivot for heatmap
    df_pivot = df_jsd_by_class.pivot(index='feature', columns='class', values='jsd')

    # Sort by average JSD
    df_pivot['avg'] = df_pivot.mean(axis=1)
    df_pivot = df_pivot.sort_values('avg')
    df_pivot = df_pivot.drop('avg', axis=1)

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(df_pivot, annot=True, fmt='.3f', cmap='RdYlGn_r',
                vmin=0, vmax=0.5, ax=ax,
                cbar_kws={'label': 'JSD (Lower = Better)'})

    ax.set_title('JSD Heatmap: Real vs Synthetic by Class\n(Green = Similar, Red = Different)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def get_top_features_by_importance(X_real: pd.DataFrame, y_real: np.ndarray,
                                   n_top: int = 10) -> List[str]:
    """
    Get top N features by Random Forest feature importance.

    Args:
        X_real: Feature DataFrame
        y_real: Labels
        n_top: Number of top features

    Returns:
        List of top feature names
    """
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_real.fillna(0), y_real)

    importances = pd.DataFrame({
        'feature': X_real.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    return importances.head(n_top)['feature'].tolist()


def create_jsd_summary_table(df_jsd: pd.DataFrame, df_jsd_by_class: pd.DataFrame,
                             save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create a comprehensive summary table with all JSD metrics.

    Args:
        df_jsd: Overall JSD results
        df_jsd_by_class: JSD results by class
        save_path: Optional path to save CSV

    Returns:
        Summary DataFrame
    """
    # Pivot class data
    df_class_pivot = df_jsd_by_class.pivot(index='feature', columns='class', values='jsd')
    df_class_pivot.columns = ['jsd_normal', 'jsd_anomaly']

    # Merge with overall results
    df_summary = df_jsd.merge(df_class_pivot, on='feature', how='left')

    # Add quality ratings
    def rate_jsd(jsd):
        if jsd < 0.05:
            return 'Excellent'
        elif jsd < 0.1:
            return 'Very Good'
        elif jsd < 0.2:
            return 'Good'
        elif jsd < 0.3:
            return 'Fair'
        else:
            return 'Needs Improvement'

    df_summary['quality_rating'] = df_summary['jsd'].apply(rate_jsd)
    df_summary['similarity_score'] = (df_summary['jsd_similarity'] * 100).round(1)

    # Reorder columns
    columns = ['feature', 'jsd', 'jsd_similarity', 'similarity_score', 'quality_rating',
               'jsd_normal', 'jsd_anomaly', 'real_mean', 'synth_mean',
               'mean_diff_pct', 'real_std', 'synth_std']
    df_summary = df_summary[[c for c in columns if c in df_summary.columns]]

    if save_path:
        df_summary.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")

    return df_summary


def print_jsd_summary(df_jsd: pd.DataFrame):
    """Print formatted JSD summary statistics."""
    print("\n" + "="*80)
    print("JENSEN-SHANNON DIVERGENCE SUMMARY")
    print("="*80)

    print(f"\nOverall Statistics:")
    print(f"  Mean JSD:     {df_jsd['jsd'].mean():.4f}")
    print(f"  Median JSD:   {df_jsd['jsd'].median():.4f}")
    print(f"  Min JSD:      {df_jsd['jsd'].min():.4f} ({df_jsd.loc[df_jsd['jsd'].idxmin(), 'feature']})")
    print(f"  Max JSD:      {df_jsd['jsd'].max():.4f} ({df_jsd.loc[df_jsd['jsd'].idxmax(), 'feature']})")

    print(f"\nQuality Breakdown:")
    excellent = (df_jsd['jsd'] < 0.1).sum()
    good = ((df_jsd['jsd'] >= 0.1) & (df_jsd['jsd'] < 0.3)).sum()
    needs_work = (df_jsd['jsd'] >= 0.3).sum()

    print(f"  Excellent (JSD < 0.1):  {excellent} features ({excellent/len(df_jsd)*100:.1f}%)")
    print(f"  Good (0.1 <= JSD < 0.3): {good} features ({good/len(df_jsd)*100:.1f}%)")
    print(f"  Needs Work (JSD >= 0.3): {needs_work} features ({needs_work/len(df_jsd)*100:.1f}%)")

    print(f"\nAverage Similarity Score: {df_jsd['jsd_similarity'].mean()*100:.1f}%")

    print("\n" + "-"*80)
    print("TOP 10 BEST MATCHING FEATURES (Lowest JSD):")
    print("-"*80)
    for idx, row in df_jsd.head(10).iterrows():
        quality = 'Excellent' if row['jsd'] < 0.1 else ('Good' if row['jsd'] < 0.3 else 'Fair')
        print(f"  {row['feature']:30s} JSD={row['jsd']:.4f}  Similarity={row['jsd_similarity']*100:.1f}%  [{quality}]")

    if len(df_jsd) > 10:
        print("\n" + "-"*80)
        print("FEATURES NEEDING IMPROVEMENT (Highest JSD):")
        print("-"*80)
        worst = df_jsd.tail(5).iloc[::-1]
        for idx, row in worst.iterrows():
            print(f"  {row['feature']:30s} JSD={row['jsd']:.4f}  Similarity={row['jsd_similarity']*100:.1f}%")


def run_jsd_analysis(X_real: pd.DataFrame, X_synth: pd.DataFrame,
                     y_real: np.ndarray, y_synth: np.ndarray,
                     features: Optional[List[str]] = None,
                     top_n_features: int = 10,
                     n_bins: int = 50,
                     output_dir: Optional[str] = None,
                     use_feature_importance: bool = True) -> Dict:
    """
    Run complete JSD analysis between real and synthetic datasets.

    This is the main entry point for the analysis.

    Args:
        X_real: Real data DataFrame
        X_synth: Synthetic data DataFrame
        y_real: Real labels (0=Normal, 1=Anomaly)
        y_synth: Synthetic labels
        features: Optional list of features (uses all common if None)
        top_n_features: Number of top features for detailed analysis
        n_bins: Number of histogram bins for JSD computation
        output_dir: Optional directory to save outputs
        use_feature_importance: If True, rank features by RF importance

    Returns:
        Dictionary with all results and figures
    """
    print("\n" + "="*80)
    print("JENSEN-SHANNON DIVERGENCE ANALYSIS")
    print("Real vs Synthetic Dataset Distribution Comparison")
    print("="*80)

    # Use provided features or find common
    if features is None:
        features = [c for c in X_real.columns if c in X_synth.columns]

    print(f"\nAnalyzing {len(features)} features...")
    print(f"Real samples: {len(X_real)} | Synthetic samples: {len(X_synth)}")

    # Get top features by importance if requested
    if use_feature_importance and len(features) > top_n_features:
        print("\nComputing feature importance to select top features...")
        top_features = get_top_features_by_importance(X_real[features], y_real, top_n_features)
        print(f"Top {top_n_features} features by importance: {top_features}")
    else:
        top_features = features[:top_n_features]

    # Compute JSD for all features
    print("\nComputing JSD for all features...")
    df_jsd_all = compute_jsd_all_features(X_real, X_synth, features, n_bins)

    # Compute JSD by class
    print("Computing JSD by class (Normal vs Anomaly)...")
    df_jsd_by_class = compute_jsd_by_class(X_real, X_synth, y_real, y_synth, features, n_bins)

    # Compute JSD for top features
    df_jsd_top = compute_jsd_all_features(X_real, X_synth, top_features, n_bins)
    df_jsd_top_by_class = compute_jsd_by_class(X_real, X_synth, y_real, y_synth, top_features, n_bins)

    # Print summary
    print_jsd_summary(df_jsd_all)

    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    results = {
        'df_jsd_all': df_jsd_all,
        'df_jsd_by_class': df_jsd_by_class,
        'df_jsd_top': df_jsd_top,
        'df_jsd_top_by_class': df_jsd_top_by_class,
        'top_features': top_features,
        'figures': {}
    }

    # Save paths
    save_jsd_bar = f"{output_dir}/jsd_bar_chart.png" if output_dir else None
    save_jsd_class = f"{output_dir}/jsd_by_class.png" if output_dir else None
    save_distributions = f"{output_dir}/jsd_distributions.png" if output_dir else None
    save_heatmap = f"{output_dir}/jsd_heatmap.png" if output_dir else None
    save_csv = f"{output_dir}/jsd_summary.csv" if output_dir else None

    # Plot 1: JSD Bar Chart
    print("\n1. Creating JSD bar chart...")
    fig1 = plot_jsd_bar_chart(df_jsd_top, top_n=top_n_features,
                              title=f'JSD Analysis: Top {top_n_features} Features',
                              save_path=save_jsd_bar)
    results['figures']['jsd_bar_chart'] = fig1
    plt.show()

    # Plot 2: JSD by Class
    print("\n2. Creating JSD by class comparison...")
    fig2 = plot_jsd_by_class(df_jsd_top_by_class, top_n=top_n_features,
                             save_path=save_jsd_class)
    results['figures']['jsd_by_class'] = fig2
    plt.show()

    # Plot 3: Distribution Comparison
    print("\n3. Creating distribution comparison histograms...")
    fig3 = plot_distribution_comparison(X_real, X_synth, top_features, n_bins,
                                        save_path=save_distributions)
    results['figures']['distributions'] = fig3
    plt.show()

    # Plot 4: Heatmap
    print("\n4. Creating JSD heatmap...")
    fig4 = plot_jsd_heatmap_by_class(df_jsd_top_by_class, save_path=save_heatmap)
    results['figures']['heatmap'] = fig4
    plt.show()

    # Create summary table
    print("\n5. Creating summary table...")
    df_summary = create_jsd_summary_table(df_jsd_all, df_jsd_by_class, save_csv)
    results['df_summary'] = df_summary

    print("\n" + "="*80)
    print("JSD ANALYSIS COMPLETE")
    print("="*80)

    return results


# ============================================================
# NOTEBOOK CELL CODE - Copy this section into a new cell
# ============================================================

NOTEBOOK_CELL_CODE = '''
# ========================================
# JENSEN-SHANNON DIVERGENCE (JSD) ANALYSIS
# Comparing Real vs Synthetic Distributions
# ========================================

from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

def compute_jsd(p, q):
    """Compute JSD between two distributions. Returns value in [0, 1] using log base 2."""
    return jensenshannon(p, q, base=2) ** 2

def create_histogram(data, n_bins=50, range_tuple=None):
    """Create normalized histogram from data."""
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return np.zeros(n_bins)
    hist, _ = np.histogram(data, bins=n_bins, range=range_tuple, density=False)
    hist = hist.astype(float)
    if hist.sum() > 0:
        hist = hist / hist.sum()
    epsilon = 1e-10
    hist = hist + epsilon
    hist = hist / hist.sum()
    return hist

def compute_jsd_for_feature(real_data, synth_data, n_bins=50):
    """Compute JSD between real and synthetic for one feature."""
    all_data = np.concatenate([real_data[np.isfinite(real_data)],
                               synth_data[np.isfinite(synth_data)]])
    if len(all_data) == 0:
        return 1.0, np.zeros(n_bins), np.zeros(n_bins), (0, 1)
    range_min, range_max = np.percentile(all_data, 1), np.percentile(all_data, 99)
    if range_min == range_max:
        range_max = range_min + 1
    bin_range = (range_min, range_max)
    real_hist = create_histogram(real_data, n_bins, bin_range)
    synth_hist = create_histogram(synth_data, n_bins, bin_range)
    jsd = compute_jsd(real_hist, synth_hist)
    return jsd, real_hist, synth_hist, bin_range

# ========================================
# Get Top 10 Features by Feature Importance
# ========================================
print("Computing feature importance to select top 10 features...")
from sklearn.ensemble import RandomForestClassifier

rf_importance = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_importance.fit(X_real.fillna(0), y_real)

feature_importance = pd.DataFrame({
    'feature': X_real.columns,
    'importance': rf_importance.feature_importances_
}).sort_values('importance', ascending=False)

TOP_10_FEATURES = feature_importance.head(10)['feature'].tolist()
print(f"\\nTop 10 Features by Importance:")
for i, feat in enumerate(TOP_10_FEATURES, 1):
    imp = feature_importance[feature_importance['feature'] == feat]['importance'].values[0]
    print(f"  {i}. {feat} (importance: {imp:.4f})")

# ========================================
# Compute JSD for All Features
# ========================================
print("\\n" + "="*70)
print("Computing JSD for all features...")
print("="*70)

n_bins = 50
jsd_results = []

for feature in common_features_all:
    real_data = X_real[feature].values
    synth_data = X_synthetic[feature].values

    jsd_val, _, _, _ = compute_jsd_for_feature(real_data, synth_data, n_bins)

    # Compute JSD by class
    jsd_normal, _, _, _ = compute_jsd_for_feature(
        X_real[y_real == 0][feature].values,
        X_synthetic[y_synthetic == 0][feature].values, n_bins
    )
    jsd_anomaly, _, _, _ = compute_jsd_for_feature(
        X_real[y_real == 1][feature].values,
        X_synthetic[y_synthetic == 1][feature].values, n_bins
    )

    jsd_results.append({
        'feature': feature,
        'jsd': jsd_val,
        'jsd_similarity': 1 - jsd_val,
        'jsd_normal': jsd_normal,
        'jsd_anomaly': jsd_anomaly,
        'real_mean': np.nanmean(real_data),
        'synth_mean': np.nanmean(synth_data),
        'real_std': np.nanstd(real_data),
        'synth_std': np.nanstd(synth_data)
    })

df_jsd = pd.DataFrame(jsd_results).sort_values('jsd')

# Print summary
print(f"\\nJSD Summary Statistics:")
print(f"  Mean JSD:     {df_jsd['jsd'].mean():.4f}")
print(f"  Median JSD:   {df_jsd['jsd'].median():.4f}")
print(f"  Min JSD:      {df_jsd['jsd'].min():.4f} ({df_jsd.iloc[0]['feature']})")
print(f"  Max JSD:      {df_jsd['jsd'].max():.4f} ({df_jsd.iloc[-1]['feature']})")

excellent = (df_jsd['jsd'] < 0.1).sum()
good = ((df_jsd['jsd'] >= 0.1) & (df_jsd['jsd'] < 0.3)).sum()
needs_work = (df_jsd['jsd'] >= 0.3).sum()

print(f"\\nQuality Breakdown:")
print(f"  Excellent (JSD < 0.1):   {excellent} features ({excellent/len(df_jsd)*100:.1f}%)")
print(f"  Good (0.1 <= JSD < 0.3): {good} features ({good/len(df_jsd)*100:.1f}%)")
print(f"  Needs Work (JSD >= 0.3): {needs_work} features ({needs_work/len(df_jsd)*100:.1f}%)")
print(f"\\nAverage Similarity Score: {df_jsd['jsd_similarity'].mean()*100:.1f}%")

# ========================================
# JSD Results Table for Top 10 Features
# ========================================
print("\\n" + "="*70)
print("JSD RESULTS FOR TOP 10 FEATURES")
print("="*70)

df_jsd_top10 = df_jsd[df_jsd['feature'].isin(TOP_10_FEATURES)].copy()
df_jsd_top10['rank'] = df_jsd_top10['feature'].apply(lambda x: TOP_10_FEATURES.index(x) + 1)
df_jsd_top10 = df_jsd_top10.sort_values('rank')

print(f"\\n{'Rank':<6}{'Feature':<30}{'JSD':<10}{'Similarity':<12}{'JSD Normal':<12}{'JSD Anomaly':<12}{'Quality'}")
print("-"*94)
for _, row in df_jsd_top10.iterrows():
    quality = 'Excellent' if row['jsd'] < 0.1 else ('Good' if row['jsd'] < 0.3 else 'Fair')
    print(f"{int(row['rank']):<6}{row['feature']:<30}{row['jsd']:.4f}    {row['jsd_similarity']*100:.1f}%       "
          f"{row['jsd_normal']:.4f}      {row['jsd_anomaly']:.4f}      {quality}")

# ========================================
# Visualization 1: JSD Bar Chart
# ========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot JSD values
ax1 = axes[0]
df_plot = df_jsd_top10.sort_values('jsd')
colors = plt.cm.RdYlGn_r(df_plot['jsd'].values / max(df_plot['jsd'].max(), 0.5))
bars1 = ax1.barh(df_plot['feature'], df_plot['jsd'], color=colors)
ax1.set_xlabel('Jensen-Shannon Divergence', fontsize=12)
ax1.set_ylabel('Feature', fontsize=12)
ax1.set_title('JSD for Top 10 Features\\n(Lower = More Similar)', fontsize=14, fontweight='bold')
ax1.invert_yaxis()
ax1.axvline(x=0.1, color='green', linestyle='--', alpha=0.7, label='Excellent (< 0.1)')
ax1.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='Good (< 0.3)')
ax1.legend(loc='lower right')
for bar, val in zip(bars1, df_plot['jsd']):
    ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=10)

# Plot Similarity scores
ax2 = axes[1]
colors = plt.cm.RdYlGn(df_plot['jsd_similarity'].values)
bars2 = ax2.barh(df_plot['feature'], df_plot['jsd_similarity'] * 100, color=colors)
ax2.set_xlabel('JSD Similarity Score (%)', fontsize=12)
ax2.set_title('JSD Similarity for Top 10 Features\\n(Higher = More Similar)', fontsize=14, fontweight='bold')
ax2.invert_yaxis()
ax2.set_xlim(0, 100)
ax2.axvline(x=90, color='green', linestyle='--', alpha=0.7, label='Excellent (> 90%)')
ax2.axvline(x=70, color='orange', linestyle='--', alpha=0.7, label='Good (> 70%)')
ax2.legend(loc='lower right')
for bar, val in zip(bars2, df_plot['jsd_similarity']):
    ax2.text(val * 100 + 1, bar.get_y() + bar.get_height()/2, f'{val*100:.1f}%', va='center', fontsize=10)

plt.suptitle('Jensen-Shannon Divergence Analysis: Real vs Synthetic', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/jsd_bar_chart.png", dpi=150, bbox_inches='tight')
plt.show()

# ========================================
# Visualization 2: JSD by Class (Normal vs Anomaly)
# ========================================
fig, ax = plt.subplots(figsize=(14, 10))

df_plot = df_jsd_top10.sort_values('jsd')
x = np.arange(len(df_plot))
width = 0.35

bars1 = ax.barh(x - width/2, df_plot['jsd_normal'], width, label='Normal', color='#3498db', alpha=0.8)
bars2 = ax.barh(x + width/2, df_plot['jsd_anomaly'], width, label='Anomaly', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Jensen-Shannon Divergence', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title('JSD by Class: Real vs Synthetic (Top 10 Features)\\n(Lower = More Similar)', fontsize=14, fontweight='bold')
ax.set_yticks(x)
ax.set_yticklabels(df_plot['feature'])
ax.legend(loc='lower right')
ax.invert_yaxis()
ax.axvline(x=0.1, color='green', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5, linewidth=1)

for bar in bars1:
    w = bar.get_width()
    ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f'{w:.3f}', va='center', fontsize=9, color='#2980b9')
for bar in bars2:
    w = bar.get_width()
    ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f'{w:.3f}', va='center', fontsize=9, color='#c0392b')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/jsd_by_class.png", dpi=150, bbox_inches='tight')
plt.show()

# ========================================
# Visualization 3: Distribution Histograms
# ========================================
n_features = len(TOP_10_FEATURES)
n_cols = 2
n_rows = (n_features + 1) // 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
axes = axes.flatten()

for i, feature in enumerate(TOP_10_FEATURES):
    ax = axes[i]

    real_data = X_real[feature].values
    synth_data = X_synthetic[feature].values

    jsd_val, real_hist, synth_hist, bin_range = compute_jsd_for_feature(real_data, synth_data, n_bins)
    bins = np.linspace(bin_range[0], bin_range[1], n_bins + 1)

    ax.hist(real_data[(real_data >= bin_range[0]) & (real_data <= bin_range[1])],
            bins=bins, alpha=0.5, label='Real', color='#3498db', density=True)
    ax.hist(synth_data[(synth_data >= bin_range[0]) & (synth_data <= bin_range[1])],
            bins=bins, alpha=0.5, label='Synthetic', color='#e74c3c', density=True)

    quality = 'Excellent' if jsd_val < 0.1 else ('Good' if jsd_val < 0.3 else 'Needs Improvement')
    title_color = 'green' if jsd_val < 0.1 else ('orange' if jsd_val < 0.3 else 'red')

    ax.set_title(f'{feature}\\nJSD = {jsd_val:.4f} ({quality})', fontsize=11, fontweight='bold', color=title_color)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution Comparison: Real vs Synthetic (Top 10 Features)', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/jsd_distributions.png", dpi=150, bbox_inches='tight')
plt.show()

# ========================================
# Visualization 4: JSD Heatmap
# ========================================
fig, ax = plt.subplots(figsize=(10, 10))

df_heatmap = df_jsd_top10[['feature', 'jsd_normal', 'jsd_anomaly']].copy()
df_heatmap = df_heatmap.set_index('feature')
df_heatmap.columns = ['Normal', 'Anomaly']
df_heatmap = df_heatmap.sort_values(by=['Normal', 'Anomaly'])

sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='RdYlGn_r',
            vmin=0, vmax=0.5, ax=ax, cbar_kws={'label': 'JSD (Lower = Better)'})

ax.set_title('JSD Heatmap by Class: Real vs Synthetic (Top 10 Features)\\n(Green = Similar, Red = Different)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/jsd_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()

# ========================================
# Save JSD Results to CSV
# ========================================
df_jsd.to_csv(f"{OUTPUT_DIR}/jsd_results_all_features.csv", index=False)
df_jsd_top10.to_csv(f"{OUTPUT_DIR}/jsd_results_top10_features.csv", index=False)
print(f"\\nSaved JSD results to:")
print(f"  {OUTPUT_DIR}/jsd_results_all_features.csv")
print(f"  {OUTPUT_DIR}/jsd_results_top10_features.csv")
print(f"  {OUTPUT_DIR}/jsd_bar_chart.png")
print(f"  {OUTPUT_DIR}/jsd_by_class.png")
print(f"  {OUTPUT_DIR}/jsd_distributions.png")
print(f"  {OUTPUT_DIR}/jsd_heatmap.png")
'''

if __name__ == "__main__":
    print("JSD Distribution Analysis Module")
    print("="*50)
    print("\nTo use in your notebook, either:")
    print("1. Import and call run_jsd_analysis()")
    print("2. Copy the NOTEBOOK_CELL_CODE into a new cell")
    print("\nExample:")
    print("  from jsd_distribution_analysis import run_jsd_analysis")
    print("  results = run_jsd_analysis(X_real, X_synthetic, y_real, y_synthetic, common_features_all)")
