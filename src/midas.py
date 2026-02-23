"""
MIDAS Feature Engineering Module
=================================
Mixed Data Sampling (MIDAS) — Almon Polynomial 기반 혼합주파수 Feature 추출

핵심 아이디어:
  기존 파이프라인은 일간 데이터를 월말 스냅샷으로 리샘플링하여 월 중 변동성 역학 정보를 손실합니다.
  MIDAS는 Almon Polynomial 가중 함수를 사용하여 일간 관측치들에 최적 가중치를 부여하고,
  이를 단일 월간 Feature로 집약하여 훨씬 풍부한 정보를 보존합니다.

주요 구성:
  - Almon Polynomial 가중 함수: w(k; θ₁, θ₂) = exp(θ₁k + θ₂k²) / Σexp(θ₁j + θ₂j²)
  - Rolling Window OLS: Look-ahead Bias 방지를 위해 각 월 t에서 t-1까지만 사용
  - fit/transform 패턴: scikit-learn 스타일 인터페이스

참고 문헌:
  Ghysels, E., Santa-Clara, P., & Valkanov, R. (2004).
  "The MIDAS Touch: Mixed Data Sampling Regression Models"
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Tuple, List
import warnings


class MIDASFeatureExtractor:
    """
    MIDAS (Mixed Data Sampling) Feature Extractor

    일간 데이터를 Almon Polynomial 가중치로 집약하여 월간 Feature를 생성합니다.

    Parameters
    ----------
    K : int, default=22
        일간 래그 수 (한 달의 거래일 수). 각 월에 대해 K개의 일간 관측치를 사용합니다.
    poly_degree : int, default=2
        Almon 다항식 차수. 2차(quadratic)가 일반적으로 충분한 유연성을 제공합니다.
        - 1차: 단순 지수 감쇠 (monotonic decay)
        - 2차: hump-shaped 패턴 포착 가능 (최근도 먼 과거도 낮고 중간이 높은 가중치)

    Examples
    --------
    >>> extractor = MIDASFeatureExtractor(K=22, poly_degree=2)
    >>> midas_features = extractor.fit_transform(daily_vix, monthly_spy_returns, window=60)
    """

    def __init__(self, K: int = 22, poly_degree: int = 2):
        self.K = K
        self.poly_degree = poly_degree

        # 적합 결과 저장
        self.theta_history_: Optional[List[np.ndarray]] = None  # 각 월의 추정된 θ
        self.fitted_: bool = False
        self.theta_latest_: Optional[np.ndarray] = None  # 가장 최근 추정된 θ

    def _almon_weights(self, theta: np.ndarray, K: int) -> np.ndarray:
        """
        Almon Polynomial 가중치를 계산합니다.

        가중 함수:
            w(k; θ) = exp(Σ_{p=1}^{P} θ_p * k^p) / Σ_{j=1}^{K} exp(Σ_{p=1}^{P} θ_p * j^p)

        여기서 k = 1, 2, ..., K (k=1이 가장 최근, k=K가 가장 먼 과거)

        Parameters
        ----------
        theta : np.ndarray, shape (poly_degree,)
            Almon 다항식 파라미터
        K : int
            래그 수

        Returns
        -------
        weights : np.ndarray, shape (K,)
            정규화된 가중치 (합 = 1). 인덱스 0이 가장 최근 관측치.
        """
        k = np.arange(1, K + 1, dtype=np.float64)

        # 다항식 값 계산: Σ_{p=1}^{P} θ_p * k^p
        poly_val = np.zeros(K, dtype=np.float64)
        for p in range(len(theta)):
            poly_val += theta[p] * (k ** (p + 1))

        # 수치 안정성: overflow 방지를 위해 최대값을 빼줌
        poly_val -= np.max(poly_val)

        # softmax 스타일 정규화
        exp_val = np.exp(poly_val)
        weights = exp_val / np.sum(exp_val)

        return weights

    def _compute_midas_weighted_value(
        self, daily_window: np.ndarray, theta: np.ndarray
    ) -> float:
        """
        일간 윈도우에 MIDAS 가중치를 적용하여 단일 집약값을 계산합니다.

        Parameters
        ----------
        daily_window : np.ndarray, shape (K,)
            K개의 일간 관측치 (시간순 정렬, 인덱스 0이 가장 오래된 값)
        theta : np.ndarray
            Almon 파라미터

        Returns
        -------
        float
            가중 집약값
        """
        weights = self._almon_weights(theta, len(daily_window))
        # weights[0]이 k=1(가장 최근)에 대응하므로, daily_window를 뒤집어서 곱함
        # daily_window[-1]이 가장 최근 관측치
        return np.dot(weights, daily_window[::-1])

    def _objective(
        self,
        theta: np.ndarray,
        daily_windows: List[np.ndarray],
        monthly_targets: np.ndarray,
    ) -> float:
        """
        OLS 목적함수: Σ(y_t - β₀ - β₁ * MIDAS_x_t)²

        MIDAS 회귀에서:
        - y_t = 월간 타겟 (예: SPY 수익률)
        - MIDAS_x_t = Almon 가중치로 집약된 일간 Feature (예: VIX)
        - θ는 Almon 가중 함수의 파라미터
        - β₀, β₁은 Concentrated Out (θ 최적화 시 OLS로 내부 계산)

        Parameters
        ----------
        theta : np.ndarray
            Almon 파라미터
        daily_windows : List[np.ndarray]
            각 월의 K개 일간 관측치 리스트
        monthly_targets : np.ndarray
            월간 타겟 값

        Returns
        -------
        float
            잔차 제곱합 (RSS)
        """
        n = len(monthly_targets)

        # 1. MIDAS 가중 Feature 계산
        midas_x = np.array([
            self._compute_midas_weighted_value(dw, theta)
            for dw in daily_windows
        ])

        # 2. OLS로 β₀, β₁ 추정 (Concentrated Out)
        X_matrix = np.column_stack([np.ones(n), midas_x])
        try:
            beta, residuals, _, _ = np.linalg.lstsq(X_matrix, monthly_targets, rcond=None)
            # 잔차 제곱합 계산
            y_pred = X_matrix @ beta
            rss = np.sum((monthly_targets - y_pred) ** 2)
        except np.linalg.LinAlgError:
            rss = 1e10  # 수치 오류 시 큰 값 반환

        return rss

    def _extract_daily_windows(
        self,
        daily_data: pd.Series,
        month_end_dates: pd.DatetimeIndex,
        K: int,
    ) -> Tuple[List[np.ndarray], List[bool]]:
        """
        각 월말 날짜에 대해 직전 K개 거래일의 일간 데이터를 추출합니다.

        Parameters
        ----------
        daily_data : pd.Series
            일간 데이터 (DatetimeIndex)
        month_end_dates : pd.DatetimeIndex
            월말 날짜 인덱스
        K : int
            추출할 일간 관측치 수

        Returns
        -------
        windows : List[np.ndarray]
            각 월의 일간 윈도우 리스트
        valid_mask : List[bool]
            유효한 윈도우 여부
        """
        windows = []
        valid_mask = []

        for date in month_end_dates:
            # 해당 월말 이전(포함)의 일간 데이터 추출
            mask = daily_data.index <= date
            available = daily_data[mask]

            if len(available) >= K:
                window = available.iloc[-K:].values.astype(np.float64)
                windows.append(window)
                valid_mask.append(True)
            else:
                # 충분한 데이터가 없는 경우
                windows.append(np.zeros(K, dtype=np.float64))
                valid_mask.append(False)

        return windows, valid_mask

    def fit(
        self,
        daily_data: pd.Series,
        monthly_target: pd.Series,
        window: int = 60,
    ) -> 'MIDASFeatureExtractor':
        """
        Rolling Window OLS로 Almon 파라미터를 추정합니다.

        각 월 t에서, 과거 window개월의 데이터만 사용하여 θ를 추정합니다.
        이를 통해 Look-ahead Bias를 엄격히 방지합니다.

        Parameters
        ----------
        daily_data : pd.Series
            일간 Feature 데이터 (예: VIX 레벨). DatetimeIndex 필요.
        monthly_target : pd.Series
            월간 타겟 데이터 (예: SPY 수익률). DatetimeIndex 필요.
        window : int, default=60
            롤링 윈도우 크기 (월 단위). 60 = 5년.

        Returns
        -------
        self
            적합된 MIDASFeatureExtractor
        """
        print(f"[MIDAS] Fitting with K={self.K}, poly_degree={self.poly_degree}, window={window}...")

        # 공통 인덱스 찾기
        common_dates = monthly_target.index
        print(f"[MIDAS] 타겟 데이터 기간: {common_dates[0].date()} ~ {common_dates[-1].date()}")
        print(f"[MIDAS] 일간 데이터 기간: {daily_data.index[0].date()} ~ {daily_data.index[-1].date()}")

        # 각 월에 대한 일간 윈도우 추출
        all_windows, valid_mask = self._extract_daily_windows(
            daily_data, common_dates, self.K
        )

        # Rolling Window OLS
        theta_history = []
        n_months = len(common_dates)

        # 초기 θ (약간의 감쇠를 가정)
        theta_init = np.zeros(self.poly_degree)
        theta_init[0] = -0.01  # 약한 감쇠

        n_fitted = 0
        n_skipped = 0

        for t in range(window, n_months):
            # 윈도우 내 유효한 데이터만 사용
            start_idx = t - window
            window_valid = [
                valid_mask[i] for i in range(start_idx, t)
            ]
            window_windows = [
                all_windows[i] for i in range(start_idx, t) if valid_mask[i]
            ]
            window_targets_list = [
                monthly_target.iloc[i] for i in range(start_idx, t) if valid_mask[i]
            ]

            if len(window_windows) < 20:  # 최소 20개월 필요
                theta_history.append(theta_init.copy())
                n_skipped += 1
                continue

            window_targets = np.array(window_targets_list)

            # θ 최적화 (Nelder-Mead — gradient-free, 비선형 최적화에 적합)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = minimize(
                        self._objective,
                        theta_init,
                        args=(window_windows, window_targets),
                        method='Nelder-Mead',
                        options={
                            'maxiter': 500,
                            'xatol': 1e-6,
                            'fatol': 1e-6,
                            'adaptive': True,
                        }
                    )
                theta_est = result.x
                theta_init = theta_est.copy()  # warm start
                n_fitted += 1
            except Exception as e:
                theta_est = theta_init.copy()
                n_skipped += 1

            theta_history.append(theta_est.copy())

        self.theta_history_ = theta_history
        self.theta_latest_ = theta_history[-1] if theta_history else theta_init
        self.fitted_ = True
        self._fit_daily_data = daily_data
        self._fit_month_dates = common_dates
        self._fit_window = window
        self._all_windows = all_windows
        self._valid_mask = valid_mask

        print(f"[MIDAS] Fitting 완료: {n_fitted}개월 추정, {n_skipped}개월 스킵")
        print(f"[MIDAS] 최종 θ = {self.theta_latest_}")

        return self

    def transform(
        self,
        daily_data: Optional[pd.Series] = None,
        month_end_dates: Optional[pd.DatetimeIndex] = None,
    ) -> pd.Series:
        """
        적합된 Almon 가중치를 사용하여 일간 데이터를 월간 MIDAS Feature로 변환합니다.

        Parameters
        ----------
        daily_data : pd.Series, optional
            일간 Feature 데이터. None이면 fit 시 사용한 데이터를 재사용.
        month_end_dates : pd.DatetimeIndex, optional
            변환할 월말 날짜. None이면 fit 시 사용한 날짜 인덱스 사용.

        Returns
        -------
        pd.Series
            월간 MIDAS Feature 값
        """
        if not self.fitted_:
            raise RuntimeError("fit()을 먼저 호출해야 합니다.")

        # 기본값: fit 시 사용한 데이터
        if daily_data is None:
            daily_data = self._fit_daily_data
        if month_end_dates is None:
            month_end_dates = self._fit_month_dates

        # 일간 윈도우 추출
        all_windows, valid_mask = self._extract_daily_windows(
            daily_data, month_end_dates, self.K
        )

        window = self._fit_window
        n_months = len(month_end_dates)

        # MIDAS Feature 생성
        midas_values = []
        midas_dates = []

        for t in range(window, n_months):
            if not valid_mask[t]:
                continue

            # 해당 시점의 θ 사용
            theta_idx = t - window
            if theta_idx < len(self.theta_history_):
                theta = self.theta_history_[theta_idx]
            else:
                theta = self.theta_latest_

            val = self._compute_midas_weighted_value(all_windows[t], theta)
            midas_values.append(val)
            midas_dates.append(month_end_dates[t])

        result = pd.Series(
            midas_values,
            index=pd.DatetimeIndex(midas_dates),
            name='MIDAS_VIX',
            dtype=np.float64,
        )

        print(f"[MIDAS] Transform 완료: {len(result)}개월 Feature 생성")
        return result

    def fit_transform(
        self,
        daily_data: pd.Series,
        monthly_target: pd.Series,
        window: int = 60,
    ) -> pd.Series:
        """
        fit() + transform() 편의 메서드.

        Parameters
        ----------
        daily_data : pd.Series
            일간 Feature 데이터
        monthly_target : pd.Series
            월간 타겟 데이터
        window : int
            롤링 윈도우 크기

        Returns
        -------
        pd.Series
            월간 MIDAS Feature 값
        """
        self.fit(daily_data, monthly_target, window)
        return self.transform(daily_data)

    def get_latest_weights(self) -> np.ndarray:
        """
        가장 최근에 추정된 Almon 가중치를 반환합니다.

        Returns
        -------
        np.ndarray, shape (K,)
            정규화된 가중치. 인덱스 0이 가장 최근 관측치.
        """
        if not self.fitted_:
            raise RuntimeError("fit()을 먼저 호출해야 합니다.")
        return self._almon_weights(self.theta_latest_, self.K)

    def get_weights_at(self, month_idx: int) -> np.ndarray:
        """
        특정 월의 Almon 가중치를 반환합니다.

        Parameters
        ----------
        month_idx : int
            theta_history_ 내 인덱스

        Returns
        -------
        np.ndarray, shape (K,)
            정규화된 가중치
        """
        if not self.fitted_ or self.theta_history_ is None:
            raise RuntimeError("fit()을 먼저 호출해야 합니다.")
        if month_idx >= len(self.theta_history_):
            raise IndexError(f"인덱스 {month_idx}는 범위 밖입니다 (총 {len(self.theta_history_)}개)")
        return self._almon_weights(self.theta_history_[month_idx], self.K)


# =============================================================================
# 유틸리티 함수
# =============================================================================

def download_daily_vix(start_date: str, end_date: str) -> pd.Series:
    """
    일간 VIX 데이터를 다운로드합니다.

    Parameters
    ----------
    start_date : str
        시작일 (YYYY-MM-DD)
    end_date : str
        종료일 (YYYY-MM-DD)

    Returns
    -------
    pd.Series
        일간 VIX 종가 데이터
    """
    import yfinance as yf

    print(f"[MIDAS] 일간 VIX 데이터 다운로드 중 ({start_date} ~ {end_date})...")
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)['Close']
    vix_data = vix_data.ffill().bfill()

    # MultiIndex 처리 (yfinance 버전에 따라)
    if isinstance(vix_data, pd.DataFrame):
        vix_data = vix_data.squeeze()
    if isinstance(vix_data.index, pd.MultiIndex):
        vix_data.index = vix_data.index.get_level_values(0)

    # 이름 설정
    vix_data.name = 'VIX'

    print(f"[MIDAS] VIX 데이터: {len(vix_data)}일, "
          f"{vix_data.index[0].date()} ~ {vix_data.index[-1].date()}")

    return vix_data


def download_daily_spy_returns(start_date: str, end_date: str) -> pd.Series:
    """
    일간 SPY 수익률 데이터를 다운로드하고 월간으로 집계합니다.
    MIDAS 모델의 타겟으로 사용됩니다.

    Parameters
    ----------
    start_date : str
        시작일 (YYYY-MM-DD)
    end_date : str
        종료일 (YYYY-MM-DD)

    Returns
    -------
    pd.Series
        월간 SPY 수익률
    """
    import yfinance as yf

    print(f"[MIDAS] SPY 데이터 다운로드 중 ({start_date} ~ {end_date})...")
    spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']

    # MultiIndex 처리
    if isinstance(spy_data, pd.DataFrame):
        spy_data = spy_data.squeeze()
    if isinstance(spy_data.index, pd.MultiIndex):
        spy_data.index = spy_data.index.get_level_values(0)

    # 월간 수익률 계산
    monthly_prices = spy_data.resample('ME').last()
    monthly_returns = monthly_prices.pct_change().dropna()
    monthly_returns.name = 'SPY_Return'

    print(f"[MIDAS] SPY 월간 수익률: {len(monthly_returns)}개월, "
          f"{monthly_returns.index[0].date()} ~ {monthly_returns.index[-1].date()}")

    return monthly_returns


def get_monthly_vix_snapshot(start_date: str, end_date: str) -> pd.Series:
    """
    기존 방식의 월말 VIX 스냅샷을 생성합니다 (비교 기준용).

    Parameters
    ----------
    start_date : str
        시작일 (YYYY-MM-DD)
    end_date : str
        종료일 (YYYY-MM-DD)

    Returns
    -------
    pd.Series
        월말 VIX 값
    """
    import yfinance as yf

    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)['Close']

    # MultiIndex 처리
    if isinstance(vix_data, pd.DataFrame):
        vix_data = vix_data.squeeze()
    if isinstance(vix_data.index, pd.MultiIndex):
        vix_data.index = vix_data.index.get_level_values(0)

    monthly_vix = vix_data.resample('ME').last()
    monthly_vix = monthly_vix.ffill().bfill()
    monthly_vix.name = 'Monthly_VIX'

    return monthly_vix


# =============================================================================
# 자체 테스트 (Self-Test)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MIDAS Feature Engineering — Self Test")
    print("=" * 70)

    # 설정
    START_DATE = '2007-01-01'
    END_DATE = '2024-01-01'
    K = 22            # 1거래월
    WINDOW = 60       # 5년 롤링 윈도우
    POLY_DEGREE = 2   # 2차 Almon

    # 1. 데이터 다운로드
    print("\n[Step 1] 데이터 다운로드")
    daily_vix = download_daily_vix(START_DATE, END_DATE)
    monthly_spy_returns = download_daily_spy_returns(START_DATE, END_DATE)
    monthly_vix_snapshot = get_monthly_vix_snapshot(START_DATE, END_DATE)

    # 2. MIDAS 모델 적합 + 변환
    print("\n[Step 2] MIDAS 모델 적합 + 변환")
    extractor = MIDASFeatureExtractor(K=K, poly_degree=POLY_DEGREE)
    midas_vix = extractor.fit_transform(daily_vix, monthly_spy_returns, window=WINDOW)

    # 3. 진단 정보 출력
    print("\n" + "=" * 70)
    print("[진단] MIDAS Feature 요약")
    print("=" * 70)
    print(f"  생성된 Feature 수: {len(midas_vix)}")
    print(f"  기간: {midas_vix.index[0].date()} ~ {midas_vix.index[-1].date()}")
    print(f"  NaN 수: {midas_vix.isna().sum()}")
    print(f"  기본 통계:")
    print(f"    Mean:   {midas_vix.mean():.4f}")
    print(f"    Std:    {midas_vix.std():.4f}")
    print(f"    Min:    {midas_vix.min():.4f}")
    print(f"    Max:    {midas_vix.max():.4f}")
    print(f"    Median: {midas_vix.median():.4f}")

    # 4. 검증
    print("\n[검증]")
    assert len(midas_vix) > 0, "MIDAS Feature가 생성되지 않았습니다!"
    assert midas_vix.isna().sum() == 0, "NaN 값이 존재합니다!"
    print("  ✅ Feature 생성 확인")
    print("  ✅ NaN 없음 확인")

    # 5. Almon 가중치 확인
    latest_weights = extractor.get_latest_weights()
    print(f"\n[Almon 가중치] 최근 추정값 (K={K}):")
    print(f"  최근 관측치 가중치 (w[0]): {latest_weights[0]:.6f}")
    print(f"  중간 관측치 가중치 (w[{K//2}]): {latest_weights[K//2]:.6f}")
    print(f"  먼 과거 가중치 (w[{K-1}]): {latest_weights[K-1]:.6f}")
    print(f"  가중치 합: {latest_weights.sum():.6f}")

    # 6. 기존 월말 VIX와 상관관계
    common_idx = midas_vix.index.intersection(monthly_vix_snapshot.index)
    if len(common_idx) > 10:
        correlation = midas_vix.loc[common_idx].corr(monthly_vix_snapshot.loc[common_idx])
        print(f"\n[상관분석] MIDAS VIX vs 월말 VIX 상관계수: {correlation:.4f}")
        if correlation < 0.9:
            print(f"  ✅ 성공 기준 충족 (< 0.9): MIDAS가 추가 정보를 포착")
        else:
            print(f"  ⚠️ 상관계수가 높음 (>= 0.9): MIDAS 가중치 조정 필요")

    # 7. 샘플 출력
    print(f"\n[샘플 데이터] 처음 5개:")
    print(midas_vix.head())

    print("\n" + "=" * 70)
    print("MIDAS Self-Test 완료! ✅")
    print("=" * 70)
