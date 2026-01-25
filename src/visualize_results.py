
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_performance():
    # 1. 데이터 로드 (시계열 수익률)
    try:
        df_returns = pd.read_csv("benchmark_returns.csv", index_col=0, parse_dates=True)
        print("benchmark_returns.csv 로드 완료")
    except FileNotFoundError:
        print("[Error] benchmark_returns.csv 파일을 찾을 수 없습니다.")
        print("python -m src.benchmark 명령어를 먼저 실행하여 벤치마크 결과를 생성해주세요.")
        return

    # 2. 누적 수익률 계산 (Cumulative Returns)
    # 1달러 투자 기준
    cumulative_returns = (1 + df_returns).cumprod()
    
    # 3. 낙폭 계산 (Drawdown)
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max

    # 4. 시각화 (2개 서브플롯: 누적 수익률, Drawdown)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # (1) Cumulative Return Plot
    for col in cumulative_returns.columns:
        # 마지막 수익률 순서로 범례 정렬을 위해
        final_ret = cumulative_returns[col].iloc[-1]
        label = f"{col.upper()} ({final_ret:.2f})"
        axes[0].plot(cumulative_returns.index, cumulative_returns[col], label=label, linewidth=2)
        
    axes[0].set_title("Cumulative Returns (Wealth of $1)", fontsize=14, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=10)
    axes[0].set_ylabel("Wealth Index")

    # (2) Drawdown Plot
    for col in drawdown.columns:
        # Min MDD 순서로 범례 정렬? (여기선 그냥 일관성 유지)
        min_dd = drawdown[col].min()
        label = f"{col.upper()} (MDD: {min_dd:.2%})"
        axes[1].plot(drawdown.index, drawdown[col], label=label, linewidth=1.5)
        # 영역 칠하기 (선택 사항)
        axes[1].fill_between(drawdown.index, drawdown[col], 0, alpha=0.1)

    axes[1].set_title("Drawdown (Risk Analysis)", fontsize=14, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=10)
    axes[1].set_ylabel("Drawdown %")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    plt.savefig("benchmark_comparison.png", dpi=300)
    print("그래프가 benchmark_comparison.png 파일로 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    visualize_performance()
