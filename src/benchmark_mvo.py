"""
Benchmark MVO Module
====================
벤치마크 구현: Mean-Variance Optimization (MVO)

- PyPortfolioOpt 라이브러리를 사용한 MVO 구현
- 샤프지수 최대화 포트폴리오 생성
- 성과 지표 출력 (수익률, MDD, Sharpe Ratio)

사용법:
    python -m src.benchmark_mvo
"""

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns


def calculate_mdd(returns: pd.Series) -> float:
    """
    Maximum Drawdown (MDD) 계산

    Args:
        returns: 수익률 시리즈

    Returns:
        MDD 값 (음수, 예: -0.25 = -25%)
    """
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    연간화된 샤프 비율 계산

    Args:
        returns: 월간 수익률 시리즈
        risk_free_rate: 연간 무위험 이자율 (기본값 2%)

    Returns:
        연간화된 샤프 비율
    """
    excess_returns = returns - (risk_free_rate / 12)  # 월간 무위험 이자율
    annualized_return = excess_returns.mean() * 12
    annualized_std = returns.std() * np.sqrt(12)

    if annualized_std == 0:
        return 0.0

    return annualized_return / annualized_std


def run_mvo_benchmark(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    rebalance_frequency: int = 12,
    risk_free_rate: float = 0.02
) -> dict:
    """
    MVO 벤치마크 실행

    PyPortfolioOpt을 사용해 샤프비율 최대화 포트폴리오를 만들고 백테스트합니다.

    Args:
        prices: 월간 가격 DataFrame
        returns: 월간 수익률 DataFrame
        rebalance_frequency: 리밸런싱 주기 (월 단위, 기본값 12 = 연간)
        risk_free_rate: 무위험 이자율

    Returns:
        결과 딕셔너리 (weights, portfolio_returns, metrics)
    """
    print("\n" + "=" * 60)
    print("Running MVO Benchmark (Max Sharpe)")
    print("=" * 60)

    # 결과 저장용
    portfolio_returns = []
    weight_history = []
    dates = returns.index.tolist()

    # 최소 학습 기간 (예: 24개월)
    min_history = 24

    for i in range(min_history, len(returns)):
        # 리밸런싱 시점인지 확인
        if (i - min_history) % rebalance_frequency == 0:
            # 과거 데이터로 MVO 최적화
            historical_prices = prices.iloc[:i]

            try:
                # 기대 수익률 계산
                mu = expected_returns.mean_historical_return(historical_prices)

                # 공분산 행렬 계산
                S = risk_models.sample_cov(historical_prices)

                # Efficient Frontier 최적화
                ef = EfficientFrontier(mu, S)
                ef.max_sharpe(risk_free_rate=risk_free_rate)
                weights = ef.clean_weights()

            except Exception as e:
                # 최적화 실패 시 동일 비중
                print(f"[WARNING] Optimization failed at {dates[i]}: {e}")
                weights = {ticker: 1.0 / len(returns.columns) for ticker in returns.columns}

        # 현재 비중으로 포트폴리오 수익률 계산
        current_returns = returns.iloc[i]
        port_return = sum(weights[ticker] * current_returns[ticker] for ticker in returns.columns)

        portfolio_returns.append(port_return)
        weight_history.append(weights.copy())

    # 결과 정리
    portfolio_returns = pd.Series(portfolio_returns, index=dates[min_history:])

    # 성과 지표 계산
    total_return = (1 + portfolio_returns).prod() - 1
    annualized_return = (1 + total_return) ** (12 / len(portfolio_returns)) - 1
    annualized_std = portfolio_returns.std() * np.sqrt(12)
    sharpe = calculate_sharpe_ratio(portfolio_returns, risk_free_rate)
    mdd = calculate_mdd(portfolio_returns)

    # 결과 출력
    print(f"\n--- MVO Benchmark Results ---")
    print(f"Period: {portfolio_returns.index[0]} ~ {portfolio_returns.index[-1]}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Annualized Volatility: {annualized_std:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {mdd:.2%}")

    print(f"\n--- Final Weights ---")
    for ticker, weight in weight_history[-1].items():
        print(f"  {ticker}: {weight:.2%}")

    return {
        'weights_history': weight_history,
        'portfolio_returns': portfolio_returns,
        'metrics': {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_std': annualized_std,
            'sharpe_ratio': sharpe,
            'max_drawdown': mdd
        }
    }


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    from src.data_loader import get_monthly_asset_data, ASSET_TICKERS

    print("=" * 60)
    print("MVO Benchmark Test")
    print("=" * 60)

    # 데이터 로드
    prices, returns = get_monthly_asset_data(
        ASSET_TICKERS,
        start_date='2007-07-01',
        end_date='2024-01-01'
    )

    # MVO 벤치마크 실행
    results = run_mvo_benchmark(prices, returns, rebalance_frequency=12)

    print("\n[Success] MVO Benchmark completed!")
