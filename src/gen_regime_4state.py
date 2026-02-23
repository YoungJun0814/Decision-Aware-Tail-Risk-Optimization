"""
4-State Hierarchical Regime Generator
======================================
Level 1: Risk-On / Risk-Off (2-state HMM)
Level 2a: Bull / Sideways (Risk-On 세분화)
Level 2b: Correction / Crisis (Risk-Off 세분화)

→ 최종 출력: [Prob_Bull, Prob_Sideways, Prob_Correction, Prob_Crisis]

Usage:
    python -m src.gen_regime_4state
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("[WARNING] hmmlearn not installed. Run: pip install hmmlearn")


# =============================================================================
# Data Collection
# =============================================================================

def get_regime_features(start_date='2007-01-01', end_date='2025-01-01'):
    """
    Regime 분류용 feature 수집.
    
    Features:
        - SPY monthly returns
        - Rolling 3M volatility (연율화)
        - VIX monthly level
    """
    print("[INFO] Downloading SPY and VIX data...")
    
    # SPY
    spy = yf.download('SPY', start=start_date, end=end_date,
                       progress=False, auto_adjust=True)['Close']
    if isinstance(spy, pd.DataFrame):
        spy = spy.squeeze()
    if isinstance(spy.index, pd.MultiIndex):
        spy.index = spy.index.get_level_values(0)
    
    spy_monthly = spy.resample('ME').last()
    spy_ret = spy_monthly.pct_change().dropna()
    spy_ret.name = 'Returns'
    
    # Rolling volatility (3M, annualized)
    spy_vol = spy_ret.rolling(3).std() * np.sqrt(12)
    spy_vol.name = 'RollingVol'
    
    # VIX
    vix = yf.download('^VIX', start=start_date, end=end_date,
                       progress=False, auto_adjust=True)['Close']
    if isinstance(vix, pd.DataFrame):
        vix = vix.squeeze()
    if isinstance(vix.index, pd.MultiIndex):
        vix.index = vix.index.get_level_values(0)
    vix_monthly = vix.resample('ME').last()
    vix_monthly.name = 'VIX'
    
    # Merge
    features = pd.concat([spy_ret, spy_vol, vix_monthly], axis=1).dropna()
    print(f"[INFO] Features shape: {features.shape}")
    print(f"[INFO] Period: {features.index[0].date()} ~ {features.index[-1].date()}")
    
    return features


# =============================================================================
# Hierarchical HMM
# =============================================================================

def fit_level1_hmm(features, n_iter=200, random_state=42):
    """
    Level 1: Risk-On / Risk-Off (2-state).
    Features: Returns + RollingVol + VIX
    """
    X = features[['Returns', 'RollingVol', 'VIX']].values
    
    model = GaussianHMM(
        n_components=2,
        covariance_type='full',
        n_iter=n_iter,
        random_state=random_state,
        init_params='stmc',
    )
    model.fit(X)
    
    # State 0 vs 1 식별: 평균 return이 높은 state를 Risk-On으로
    state_means = model.means_[:, 0]  # Returns 컬럼의 평균
    risk_on_state = np.argmax(state_means)
    risk_off_state = 1 - risk_on_state
    
    # 확률 추출
    posteriors = model.predict_proba(X)
    prob_risk_on = posteriors[:, risk_on_state]
    prob_risk_off = posteriors[:, risk_off_state]
    
    print(f"\n[Level 1] Risk-On/Risk-Off HMM")
    print(f"  Risk-On  mean return: {state_means[risk_on_state]:.4f}")
    print(f"  Risk-Off mean return: {state_means[risk_off_state]:.4f}")
    print(f"  Risk-On  months: {(prob_risk_on > 0.5).sum()}")
    print(f"  Risk-Off months: {(prob_risk_off > 0.5).sum()}")
    
    return prob_risk_on, prob_risk_off, model


def fit_level2_hmm(features, parent_probs, level_name, n_iter=200, random_state=42):
    """
    Level 2: 부모 state 내 세분화 (2-state).
    가중 HMM: parent_probs로 가중치 적용.
    
    Returns:
        sub_prob_high: 세부 state 중 'high return' 확률
        sub_prob_low: 세부 state 중 'low return' 확률
    """
    X = features[['Returns', 'RollingVol']].values
    
    # Parent probability가 0.3 이상인 구간만 사용 (노이즈 제거)
    mask = parent_probs > 0.3
    if mask.sum() < 20:
        # 충분한 데이터가 없으면 균등 분배
        print(f"  [WARNING] {level_name}: only {mask.sum()} months > 0.3, using uniform split")
        return np.full(len(features), 0.5), np.full(len(features), 0.5)
    
    X_filtered = X[mask]
    
    model = GaussianHMM(
        n_components=2,
        covariance_type='diag',
        n_iter=n_iter,
        random_state=random_state,
        init_params='stmc',
    )
    model.fit(X_filtered)
    
    # 전체 데이터에 대해 확률 추출
    posteriors = model.predict_proba(X)
    
    # State 식별: higher return mean = "good" state
    state_means = model.means_[:, 0]
    good_state = np.argmax(state_means)
    bad_state = 1 - good_state
    
    sub_prob_good = posteriors[:, good_state]
    sub_prob_bad = posteriors[:, bad_state]
    
    print(f"\n[Level 2: {level_name}]")
    print(f"  Good state mean: {state_means[good_state]:.4f}")
    print(f"  Bad  state mean: {state_means[bad_state]:.4f}")
    
    return sub_prob_good, sub_prob_bad


def generate_4state_probs(features):
    """
    Hierarchical HMM으로 4-state 확률 생성.
    
    Level 1: Risk-On / Risk-Off
    Level 2a (Risk-On → Bull / Sideways): Risk-On 구간 세분화
    Level 2b (Risk-Off → Correction / Crisis): Risk-Off 구간 세분화
    
    최종: P(Bull), P(Sideways), P(Correction), P(Crisis)
         = P(Risk-On)·P(Bull|Risk-On), P(Risk-On)·P(Sideways|Risk-On),
           P(Risk-Off)·P(Corr|Risk-Off), P(Risk-Off)·P(Crisis|Risk-Off)
    """
    if not HMM_AVAILABLE:
        raise ImportError("hmmlearn required. Install: pip install hmmlearn")
    
    # Level 1
    prob_on, prob_off, _ = fit_level1_hmm(features)
    
    # Level 2a: Risk-On 세분화 (Bull vs Sideways)
    sub_bull, sub_side = fit_level2_hmm(
        features, prob_on, "Risk-On → Bull/Sideways", random_state=42)
    
    # Level 2b: Risk-Off 세분화 (Correction vs Crisis)
    sub_corr, sub_crisis = fit_level2_hmm(
        features, prob_off, "Risk-Off → Correction/Crisis", random_state=43)
    
    # 결합: P(leaf) = P(parent) · P(child|parent)
    prob_bull   = prob_on  * sub_bull
    prob_side   = prob_on  * sub_side
    prob_corr   = prob_off * sub_corr
    prob_crisis = prob_off * sub_crisis
    
    # 정규화 (합=1 보장)
    total = prob_bull + prob_side + prob_corr + prob_crisis
    prob_bull   /= total
    prob_side   /= total
    prob_corr   /= total
    prob_crisis /= total
    
    result = pd.DataFrame({
        'Prob_Bull': prob_bull,
        'Prob_Sideways': prob_side,
        'Prob_Correction': prob_corr,
        'Prob_Crisis': prob_crisis,
    }, index=features.index)
    
    return result


# =============================================================================
# Smoothing
# =============================================================================

def smooth_probs(probs_df, alpha=0.3):
    """
    Exponential smoothing으로 급격한 regime 전환 완화.
    α 작을수록 더 부드러움.
    """
    smoothed = probs_df.copy()
    for col in probs_df.columns:
        smoothed[col] = probs_df[col].ewm(alpha=alpha, adjust=False).mean()
    
    # 정규화
    row_sums = smoothed.sum(axis=1)
    smoothed = smoothed.div(row_sums, axis=0)
    
    return smoothed


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("  4-State Hierarchical Regime Generator")
    print("=" * 60)
    
    features = get_regime_features('2007-01-01', '2025-01-01')
    
    probs = generate_4state_probs(features)
    probs = smooth_probs(probs, alpha=0.3)
    
    # 통계 요약
    print("\n" + "=" * 60)
    print("  Regime Distribution Summary")
    print("=" * 60)
    
    for col in probs.columns:
        dominant = (probs[col] > 0.4).sum()
        print(f"  {col:<20s}: mean={probs[col].mean():.3f}, "
              f"dominant months(>0.4)={dominant}")
    
    # 검증: 위기 구간이 실제 위기와 매칭되는지
    print("\n--- Crisis Detection Validation ---")
    crisis_months = probs[probs['Prob_Crisis'] > 0.3].index
    if len(crisis_months) > 0:
        print(f"  High crisis months ({len(crisis_months)}):")
        for date in crisis_months[:10]:
            print(f"    {date.strftime('%Y-%m')}: "
                  f"Bull={probs.loc[date, 'Prob_Bull']:.2f}, "
                  f"Side={probs.loc[date, 'Prob_Sideways']:.2f}, "
                  f"Corr={probs.loc[date, 'Prob_Correction']:.2f}, "
                  f"Crisis={probs.loc[date, 'Prob_Crisis']:.2f}")
    
    # 저장
    output_path = Path(__file__).parent.parent / 'data' / 'processed' / 'regime_4state.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    probs.index.name = 'Date'
    probs.to_csv(output_path)
    print(f"\n[SAVED] {output_path}")
    print(f"  Shape: {probs.shape}")
    
    return probs


if __name__ == "__main__":
    main()
