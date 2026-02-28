"""
run_compare.py — 통합 비교 파이프라인 (Phase 7 STEP 0)
=======================================================
실행 한 번으로:
  1. AI 모델 (v4 / Phase7) Walk-forward 결과 로드
  2. 전통 모델 6종 백테스트:
       ① 1/N Equal Weight   ② 60/40 (SPY+TLT)
       ③ Risk Parity         ④ Min Variance
       ⑤ HRP                 ⑥ Momentum (Top-5)
  3. (선택) DCC-GARCH, FF Prior 벤치마크
  4. 4-Panel 비교 그림 자동 저장
  5. 성과 테이블 CSV 저장

Usage:
    python run_compare.py                    # v4 기준
    python run_compare.py --label phase7_p3  # 실험 후 비교
    python run_compare.py --no-rerun         # AI 재학습 없이 기존 결과만 시각화
    python run_compare.py --no-dcc --no-ff   # DCC/FF 벤치마크 스킵
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 서버 환경(RunPod) 호환
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─── Constants ───────────────────────────────────────────────────────────────

CRISIS_BANDS = [
    ('2008-09', '2009-03', '2008 Financial Crisis'),
    ('2020-02', '2020-04', 'COVID-19 Crash'),
    ('2022-01', '2022-10', 'Rate Hike'),
]

# AI 모델 선은 빨간 계열, 전통 모델은 회색 계열로 구분
COLOR_AI = '#e63946'
COLOR_TRAD = '#adb5bd'


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(rets: pd.Series) -> dict:
    """월간 수익률 Series → 연환산 성과 지표."""
    if len(rets) == 0:
        return dict(Return=0, Vol=0, Sharpe=0, MDD=0, Calmar=0)
    
    ann_ret = rets.mean() * 12
    ann_vol = rets.std() * np.sqrt(12)
    sharpe = ann_ret / (ann_vol + 1e-9)
    
    cum = (1 + rets).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    mdd = drawdown.min()
    
    calmar = ann_ret / (abs(mdd) + 1e-9)
    
    return dict(
        Return=round(ann_ret, 6),
        Vol=round(ann_vol, 6),
        Sharpe=round(sharpe, 4),
        MDD=round(mdd, 6),
        Calmar=round(calmar, 4),
    )


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_4panel(all_rets: dict, save_path: str) -> dict:
    """
    4-Panel 비교 그림 생성:
      [0,:] 누적수익률 (전체 가로)
      [1,0] 낙폭
      [1,1] Sharpe 바 차트
    """
    n_strategies = len(all_rets)
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_strategies, 10)))
    
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    fig.suptitle('AI Portfolio vs Traditional Strategies',
                 fontsize=18, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30,
                           height_ratios=[1.2, 1])
    
    ax_cum = fig.add_subplot(gs[0, :])   # 누적수익률 (전체 가로)
    ax_dd = fig.add_subplot(gs[1, 0])    # 낙폭
    ax_bar = fig.add_subplot(gs[1, 1])   # Sharpe 바 차트
    
    # ── Panel 1: Cumulative Return ──
    for i, (name, rets) in enumerate(all_rets.items()):
        cum = (1 + rets).cumprod()
        is_ai = 'AI' in name
        lw = 2.8 if is_ai else 1.0
        ls = '-' if is_ai else '--'
        alpha = 1.0 if is_ai else 0.7
        color = COLOR_AI if is_ai else colors[i % len(colors)]
        zorder = 10 if is_ai else 1
        ax_cum.plot(cum.index, cum.values, label=name, color=color,
                    lw=lw, ls=ls, alpha=alpha, zorder=zorder)
    
    # 위기 구간 음영
    for start, end, label in CRISIS_BANDS:
        try:
            ax_cum.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                          alpha=0.08, color='red', zorder=0)
        except Exception:
            pass
    
    ax_cum.set_title('Cumulative Return ($1 invested)', fontweight='bold', fontsize=13)
    ax_cum.legend(fontsize=8, ncol=3, loc='upper left', framealpha=0.9)
    ax_cum.grid(alpha=0.3, linestyle=':')
    ax_cum.set_ylabel('Growth of $1')
    
    # ── Panel 2: Drawdown ──
    for i, (name, rets) in enumerate(all_rets.items()):
        cum = (1 + rets).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax() * 100
        is_ai = 'AI' in name
        color = COLOR_AI if is_ai else colors[i % len(colors)]
        lw = 2.5 if is_ai else 0.8
        ax_dd.plot(dd.index, dd.values, color=color, lw=lw, label=name,
                   alpha=1.0 if is_ai else 0.6)
        if is_ai:
            ax_dd.fill_between(dd.index, dd.values, 0, alpha=0.15, color=color)
    
    ax_dd.set_title('Drawdown (%)', fontweight='bold', fontsize=13)
    ax_dd.grid(alpha=0.3, linestyle=':')
    ax_dd.set_ylabel('Drawdown %')
    
    # ── Panel 3: Sharpe Bar Chart ──
    metrics = {name: compute_metrics(rets) for name, rets in all_rets.items()}
    names = list(metrics.keys())
    sharpes = [metrics[n]['Sharpe'] for n in names]
    bar_colors = [COLOR_AI if 'AI' in n else COLOR_TRAD for n in names]
    
    bars = ax_bar.bar(range(len(names)), sharpes, color=bar_colors,
                      edgecolor='white', linewidth=0.5, width=0.7)
    ax_bar.bar_label(bars, fmt='%.2f', padding=3, fontsize=9, fontweight='bold')
    ax_bar.set_title('Sharpe Ratio Comparison', fontweight='bold', fontsize=13)
    ax_bar.set_xticks(range(len(names)))
    ax_bar.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax_bar.grid(axis='y', alpha=0.3, linestyle=':')
    ax_bar.set_ylabel('Sharpe Ratio')
    
    # 0 기준선
    ax_bar.axhline(y=0, color='black', linewidth=0.5)
    
    # 저장
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"[SAVED] {save_path}")
    return metrics


def save_metrics_table(metrics: dict, out_dir: str, label: str):
    """성과 비교 테이블을 CSV와 텍스트로 저장."""
    df = pd.DataFrame(metrics).T
    df = df[['Sharpe', 'Return', 'Vol', 'MDD', 'Calmar']]
    df = df.sort_values('Sharpe', ascending=False)
    
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f'comparison_{label}.csv')
    df.to_csv(csv_path)
    
    print(f"\n{'='*70}")
    print(f"  Performance Comparison Table ({label})")
    print(f"{'='*70}")
    print(df.to_string(float_format='{:.4f}'.format))
    print(f"\n[SAVED] {csv_path}")
    
    return df


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='AI vs Traditional Strategies Comparison Dashboard')
    parser.add_argument('--label', default='v4',
                        help='실험 라벨 (결과 파일명에 반영)')
    parser.add_argument('--no-rerun', action='store_true',
                        help='AI 모델 재학습 없이 기존 결과만 시각화')
    parser.add_argument('--no-dcc', action='store_true',
                        help='DCC-GARCH 벤치마크 스킵')
    parser.add_argument('--no-ff', action='store_true',
                        help='FF Factor Prior 벤치마크 스킵')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  run_compare.py — AI vs Traditional Strategy Dashboard")
    print(f"  Label: {args.label}")
    print("=" * 70)
    
    # ── 1. AI 모델 결과 로드 (또는 재실행) ──
    ai_csv = f'results/walkforward/{args.label}_port_returns.csv'
    
    if not args.no_rerun:
        print("\n[1/4] Running AI Walk-Forward...")
        try:
            from run_walkforward import main as run_walkforward
            run_walkforward()
        except Exception as e:
            print(f"  [ERROR] Walk-forward 실행 실패: {e}")
            print(f"  기존 결과 파일을 사용합니다.")
    
    if not os.path.exists(ai_csv):
        print(f"\n[ERROR] AI 모델 수익률 파일이 없습니다: {ai_csv}")
        print(f"  먼저 python run_walkforward.py를 실행하세요.")
        sys.exit(1)
    
    ai_rets = pd.read_csv(ai_csv, index_col=0, parse_dates=True).squeeze()
    if isinstance(ai_rets, pd.DataFrame):
        ai_rets = ai_rets.iloc[:, 0]
    print(f"  AI 모델: {len(ai_rets)} months loaded from {ai_csv}")
    
    # ── 2. 전통 모델 6종 백테스트 ──
    print("\n[2/4] Running Traditional Baselines...")
    try:
        from run_baselines import run_baselines
        from run_walkforward import CONFIG
        base_rets = run_baselines(CONFIG)
        print(f"  전통 전략 {len(base_rets)}종 완료")
    except Exception as e:
        print(f"  [ERROR] 전통 모델 실행 실패: {e}")
        base_rets = {}
    
    # 모든 수익률 합치기 (AI가 맨 위에 오도록)
    all_rets = {f'AI ({args.label})': ai_rets}
    all_rets.update(base_rets)
    
    # ── 3. 확장 벤치마크 (선택) ──
    print("\n[3/4] Extended Benchmarks...")
    
    if not args.no_dcc:
        try:
            from src.benchmark_dcc import run_dcc_benchmark
            from run_walkforward import CONFIG
            dcc_rets = run_dcc_benchmark(CONFIG)
            all_rets['DCC-GARCH MVO'] = dcc_rets
            print("  ✅ DCC-GARCH MVO 추가됨")
        except ImportError:
            print("  ⏭️  DCC-GARCH 스킵 (arch 패키지 없음. pip install arch)")
        except Exception as e:
            print(f"  ⚠️  DCC-GARCH 실패: {e}")
    else:
        print("  ⏭️  DCC-GARCH 스킵 (--no-dcc)")
    
    if not args.no_ff:
        try:
            from src.benchmark_ff import run_ff_prior_benchmark
            from run_walkforward import CONFIG
            ff_rets = run_ff_prior_benchmark(CONFIG)
            all_rets['FF-5Factor Prior'] = ff_rets
            print("  ✅ FF 5-Factor Prior 추가됨")
        except ImportError:
            print("  ⏭️  FF Prior 스킵 (benchmark_ff.py 미구현)")
        except Exception as e:
            print(f"  ⚠️  FF Prior 실패: {e}")
    else:
        print("  ⏭️  FF Prior 스킵 (--no-ff)")
    
    # ── 4. 시각화 + 테이블 저장 ──
    print(f"\n[4/4] Generating 4-Panel Comparison (strategies={len(all_rets)})...")
    
    plot_path = f'results/plots/comparison_{args.label}.png'
    metrics = plot_4panel(all_rets, plot_path)
    
    table_df = save_metrics_table(metrics, 'results/plots', args.label)
    
    # ── 요약 ──
    ai_sharpe = metrics.get(f'AI ({args.label})', {}).get('Sharpe', 0)
    best_trad = max(
        (v['Sharpe'] for k, v in metrics.items() if 'AI' not in k),
        default=0
    )
    
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  AI Sharpe:               {ai_sharpe:.4f}")
    print(f"  Best Traditional Sharpe: {best_trad:.4f}")
    print(f"  AI Advantage:            {ai_sharpe - best_trad:+.4f}")
    print(f"  Plot saved:              {plot_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
