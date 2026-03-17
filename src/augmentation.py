"""
Augmentation Module (v5 신규)
==============================
위기 에피소드 데이터 증강 (Crisis-Augmented Training)

문제: ~200개월 학습 데이터 중 위기 에피소드는 2-3개뿐.
해결: 위기 구간을 변형하여 다양한 위기 시나리오를 합성하고,
      Regime-Conditional Bootstrap으로 위기 구간의 학습 기회를 증가.

핵심 기법:
1. RegimeConditionalBootstrap: Crisis 블록에 높은 샘플링 확률 부여
   → 위기 학습 기회를 5-10배 증가, 다양한 위기 강도 자동 조합
2. CrisisAugmentor: 실제 위기 에피소드를 시간 스트레칭/강도 변형하여
   합성 위기 시나리오 생성 (COVID, Inflation 등 다양한 유형)

주의: 모든 증강은 look-ahead bias가 없음.
      - 블록 부트스트랩은 시간 내부 순서 보존
      - 합성 시나리오는 미래 정보를 사용하지 않음
"""

import numpy as np
import torch
from typing import Optional, List, Tuple


# =============================================================================
# 1. Regime-Conditional Block Bootstrap
# =============================================================================

def regime_conditional_bootstrap(
    X: np.ndarray,
    y: np.ndarray,
    regime_probs: np.ndarray,
    vix: Optional[np.ndarray] = None,
    macro: Optional[np.ndarray] = None,
    block_size: int = 6,
    n_augment: int = 3,
    crisis_weight_mult: float = 3.0,
    noise_std: float = 0.001,
    seed: Optional[int] = None,
) -> Tuple:
    """
    Regime-Conditional Block Bootstrap 데이터 증강.

    위기 확률이 높은 블록에 더 높은 샘플링 가중치를 부여하여,
    위기 구간의 학습 기회를 최대 crisis_weight_mult배 증가.

    Args:
        X: (T, seq_len, n_features) 입력 시퀀스
        y: (T, n_assets) 미래 수익률
        regime_probs: (T, n_regimes) Regime 확률 (마지막 열 = Crisis)
        vix: (T,) VIX 값 (optional)
        macro: (T, macro_dim) 매크로 피처 (optional)
        block_size: 블록 크기 (월 수). 기본 6개월.
        n_augment: 증강 횟수. 전체 데이터를 n_augment번 추가 샘플링.
        crisis_weight_mult: Crisis 블록의 샘플링 가중치 배수 (기본 3배).
        noise_std: 증강 데이터에 추가할 Gaussian 노이즈의 표준편차.
        seed: 재현성을 위한 랜덤 시드.

    Returns:
        튜플: (X_aug, y_aug, regime_aug, [vix_aug], [macro_aug])
        - 원본 + n_augment개의 증강 데이터가 결합됨
        - 증강 데이터 크기: T × (n_augment + 1)
    """
    rng = np.random.default_rng(seed)
    T = len(X)
    n_blocks = max(1, T // block_size)

    # 블록별 Crisis 확률 계산
    crisis_col = regime_probs[:, -1]  # 마지막 열 = Crisis
    block_crisis_scores = []
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, T)
        score = float(crisis_col[start:end].mean())
        block_crisis_scores.append(score)

    # 샘플링 가중치: 기본 1.0 + Crisis 비중에 따른 추가 가중치
    sampling_weights = np.array([
        1.0 + (crisis_weight_mult - 1.0) * score
        for score in block_crisis_scores
    ], dtype=np.float64)
    sampling_probs = sampling_weights / sampling_weights.sum()

    # 결과 수집
    X_list = [X]
    y_list = [y]
    regime_list = [regime_probs]
    vix_list = [vix] if vix is not None else None
    macro_list = [macro] if macro is not None else None

    for _ in range(n_augment):
        # 블록 인덱스 샘플링 (가중치 반영)
        sampled_block_indices = rng.choice(
            n_blocks, size=n_blocks, replace=True, p=sampling_probs
        )

        # 선택된 블록을 이어 붙여 증강 시퀀스 구성
        aug_X_chunks, aug_y_chunks, aug_regime_chunks = [], [], []
        aug_vix_chunks = [] if vix is not None else None
        aug_macro_chunks = [] if macro is not None else None

        for bi in sampled_block_indices:
            start = bi * block_size
            end = min(start + block_size, T)
            aug_X_chunks.append(X[start:end])
            aug_y_chunks.append(y[start:end])
            aug_regime_chunks.append(regime_probs[start:end])
            if vix is not None:
                aug_vix_chunks.append(vix[start:end])
            if macro is not None:
                aug_macro_chunks.append(macro[start:end])

        aug_X = np.concatenate(aug_X_chunks, axis=0)
        aug_y = np.concatenate(aug_y_chunks, axis=0)
        aug_regime = np.concatenate(aug_regime_chunks, axis=0)

        # 노이즈 추가 (중복 방지, 모델 강건성 향상)
        if noise_std > 0:
            aug_X = aug_X + rng.normal(0, noise_std, aug_X.shape).astype(aug_X.dtype)
            aug_y = aug_y + rng.normal(0, noise_std * 0.1, aug_y.shape).astype(aug_y.dtype)

        # 길이 맞추기 (원본 T와 동일하게 자르거나 패딩)
        aug_X = aug_X[:T]
        aug_y = aug_y[:T]
        aug_regime = aug_regime[:T]

        X_list.append(aug_X)
        y_list.append(aug_y)
        regime_list.append(aug_regime)

        if vix is not None:
            aug_vix = np.concatenate(aug_vix_chunks, axis=0)[:T]
            vix_list.append(aug_vix)
        if macro is not None:
            aug_macro = np.concatenate(aug_macro_chunks, axis=0)[:T]
            macro_list.append(aug_macro)

    # 최종 결합
    X_out = np.concatenate(X_list, axis=0)
    y_out = np.concatenate(y_list, axis=0)
    regime_out = np.concatenate(regime_list, axis=0)

    result = (X_out, y_out, regime_out)
    if vix is not None:
        result = result + (np.concatenate(vix_list, axis=0),)
    if macro is not None:
        result = result + (np.concatenate(macro_list, axis=0),)

    return result


# =============================================================================
# 2. Crisis Augmentor (합성 위기 시나리오)
# =============================================================================

class CrisisAugmentor:
    """
    실제 위기 에피소드를 변형하여 합성 위기 시나리오를 생성합니다.

    지원하는 변형 유형:
    - 시간 스트레칭: 2개월 크래시를 1.5~4개월로 변형
    - 강도 조절: 동일 패턴의 손실률을 0.5~2배로 변형
    - 회복 패턴: V자(빠른 회복), U자(느린 회복), L자(회복 없음)
    - 위기 유형 혼합: COVID형(급락)과 Inflation형(점진적 하락) 패턴 교차

    Args:
        crisis_severity_range: (min, max) 위기 강도 배율 범위.
        time_stretch_range: (min, max) 시간 스트레칭 배율 범위.
        recovery_patterns: 허용할 회복 패턴 목록.
        seed: 재현성을 위한 랜덤 시드.
    """

    RECOVERY_PATTERNS = ['V', 'U', 'L']

    def __init__(
        self,
        crisis_severity_range: Tuple[float, float] = (0.5, 1.8),
        time_stretch_range: Tuple[float, float] = (0.6, 2.0),
        recovery_patterns: List[str] = None,
        seed: Optional[int] = None,
    ):
        self.severity_range = crisis_severity_range
        self.time_range = time_stretch_range
        self.recovery_patterns = recovery_patterns or self.RECOVERY_PATTERNS
        self.rng = np.random.default_rng(seed)

    def generate_synthetic_episodes(
        self,
        asset_returns: np.ndarray,
        crisis_start_indices: List[int],
        crisis_durations: List[int],
        n_synthetic: int = 3,
    ) -> np.ndarray:
        """
        위기 에피소드를 변형하여 합성 수익률 매트릭스를 생성합니다.

        Args:
            asset_returns: (T, N) 전체 자산 수익률 행렬.
            crisis_start_indices: 각 위기 에피소드의 시작 인덱스 목록.
            crisis_durations: 각 위기 에피소드의 지속기간(월) 목록.
            n_synthetic: 에피소드당 생성할 합성 시나리오 수.

        Returns:
            synthetic_returns: (n_episodes * n_synthetic * T, N) 합성 수익률.
                각 합성 시나리오는 전체 T 시점의 수익률을 포함.
        """
        T, N = asset_returns.shape
        synthetic_list = []

        for ep_idx, (start, duration) in enumerate(
                zip(crisis_start_indices, crisis_durations)):
            end = min(start + duration, T)
            episode_rets = asset_returns[start:end]  # (duration, N)

            for _ in range(n_synthetic):
                # 1. 전체 복사본 생성
                synth = asset_returns.copy()

                # 2. 위기 구간 변형
                modified_episode = self._transform_episode(episode_rets)

                # 3. 변형된 에피소드를 무작위 시점에 삽입
                insert_start = self.rng.integers(
                    low=12,  # 최소 12개월 이후 삽입 (학습 데이터 충분성)
                    high=max(13, T - len(modified_episode) - 3)
                )
                insert_end = min(insert_start + len(modified_episode), T)
                actual_len = insert_end - insert_start
                synth[insert_start:insert_end] = modified_episode[:actual_len]

                synthetic_list.append(synth)

        if not synthetic_list:
            return np.empty((0, N), dtype=asset_returns.dtype)

        return np.stack(synthetic_list, axis=0)  # (n_total_synthetic, T, N)

    def _transform_episode(self, episode_rets: np.ndarray) -> np.ndarray:
        """
        단일 위기 에피소드를 변형합니다.

        Args:
            episode_rets: (duration, N) 위기 에피소드 수익률.

        Returns:
            transformed: 변형된 수익률 배열 (duration 달라질 수 있음).
        """
        # 1. 강도 조절
        severity = self.rng.uniform(*self.severity_range)
        scaled = episode_rets * severity

        # 2. 시간 스트레칭 (선형 보간)
        stretch = self.rng.uniform(*self.time_range)
        new_duration = max(2, int(len(scaled) * stretch))
        stretched = self._time_stretch(scaled, new_duration)

        # 3. 회복 패턴 적용
        pattern = self.rng.choice(self.recovery_patterns)
        with_recovery = self._add_recovery(stretched, pattern)

        return with_recovery

    def _time_stretch(self, rets: np.ndarray, new_len: int) -> np.ndarray:
        """선형 보간으로 수익률 시퀀스를 시간 스트레칭."""
        T, N = rets.shape
        if T == new_len:
            return rets.copy()
        # 각 자산별로 보간
        result = np.zeros((new_len, N), dtype=rets.dtype)
        old_t = np.linspace(0, T - 1, T)
        new_t = np.linspace(0, T - 1, new_len)
        for n in range(N):
            result[:, n] = np.interp(new_t, old_t, rets[:, n])
        return result

    def _add_recovery(self, rets: np.ndarray, pattern: str) -> np.ndarray:
        """
        위기 후 회복 패턴을 추가합니다.

        - 'V': 빠른 회복 (2-3개월, COVID 이후와 유사)
        - 'U': 느린 회복 (6-12개월, 금융위기 이후와 유사)
        - 'L': 회복 없음 또는 매우 느린 회복 (경기침체 장기화)
        """
        T, N = rets.shape
        # 위기 구간의 누적 손실 추정
        cumulative_loss = np.cumprod(1 + rets, axis=0)[-1] - 1  # (N,)

        if pattern == 'V':
            # 2-3개월에 걸쳐 손실의 60-80% 회복
            recovery_months = self.rng.integers(2, 5)
            recovery_strength = self.rng.uniform(0.6, 0.8)
        elif pattern == 'U':
            # 6-12개월에 걸쳐 손실의 40-60% 회복
            recovery_months = self.rng.integers(6, 13)
            recovery_strength = self.rng.uniform(0.4, 0.6)
        else:  # 'L'
            # 최소 회복 (0-20%)
            recovery_months = self.rng.integers(3, 7)
            recovery_strength = self.rng.uniform(0.0, 0.2)

        # 회복 수익률: 손실 × 회복률을 recovery_months에 균등 분배
        monthly_recovery = (-cumulative_loss * recovery_strength) / recovery_months
        recovery_rets = np.tile(monthly_recovery, (recovery_months, 1))

        # 소음 추가 (단순 선형 회복이 아닌 자연스러운 패턴)
        noise = self.rng.normal(0, 0.002, recovery_rets.shape)
        recovery_rets = recovery_rets + noise

        return np.concatenate([rets, recovery_rets], axis=0)


# =============================================================================
# 3. 기존 블록 부트스트랩 (이전 버전 호환)
# =============================================================================

def block_bootstrap_augment(
    X: torch.Tensor,
    y: torch.Tensor,
    vix: Optional[torch.Tensor] = None,
    regime: Optional[torch.Tensor] = None,
    macro: Optional[torch.Tensor] = None,
    block_size: int = 6,
    n_augment: int = 3,
    noise_std: float = 0.001,
    seed: Optional[int] = None,
) -> tuple:
    """
    기본 Block Bootstrap 증강 (기존 방식, 하위 호환).

    시간적 구조를 보존하면서 학습 데이터를 확장합니다.
    모든 텐서는 동일한 블록 인덱스를 사용하여 일관성을 보장합니다.

    Args:
        X: (T, seq_len, n_features) 입력 텐서
        y: (T, n_assets) 타겟 텐서
        vix: (T,) VIX 텐서 (optional)
        regime: (T, n_regimes) Regime 확률 텐서 (optional)
        macro: (T, macro_dim) 매크로 피처 텐서 (optional)
        block_size: 블록 크기 (월 수). 기본 6개월.
        n_augment: 증강 횟수. 기본 3회 (→ 4배 데이터).
        noise_std: 노이즈 표준편차. 기본 0.001.
        seed: 랜덤 시드.

    Returns:
        증강된 텐서 튜플. 제공된 텐서와 동일한 순서.
    """
    rng = np.random.default_rng(seed)
    T = len(X)
    n_blocks = max(1, T // block_size)

    X_list = [X]
    y_list = [y]
    vix_list = [vix] if vix is not None else None
    regime_list = [regime] if regime is not None else None
    macro_list = [macro] if macro is not None else None

    for _ in range(n_augment):
        block_indices = rng.choice(n_blocks, size=n_blocks, replace=True)

        aug_chunks = {k: [] for k in ['X', 'y', 'vix', 'regime', 'macro']}
        for bi in block_indices:
            start = bi * block_size
            end = min(start + block_size, T)
            aug_chunks['X'].append(X[start:end])
            aug_chunks['y'].append(y[start:end])
            if vix is not None:
                aug_chunks['vix'].append(vix[start:end])
            if regime is not None:
                aug_chunks['regime'].append(regime[start:end])
            if macro is not None:
                aug_chunks['macro'].append(macro[start:end])

        aug_X = torch.cat(aug_chunks['X'], dim=0)[:T]
        aug_y = torch.cat(aug_chunks['y'], dim=0)[:T]

        # 노이즈 추가
        if noise_std > 0:
            aug_X = aug_X + torch.randn_like(aug_X) * noise_std
            aug_y = aug_y + torch.randn_like(aug_y) * (noise_std * 0.1)

        X_list.append(aug_X)
        y_list.append(aug_y)

        if vix is not None:
            vix_list.append(torch.cat(aug_chunks['vix'], dim=0)[:T])
        if regime is not None:
            regime_list.append(torch.cat(aug_chunks['regime'], dim=0)[:T])
        if macro is not None:
            macro_list.append(torch.cat(aug_chunks['macro'], dim=0)[:T])

    result = (torch.cat(X_list, dim=0), torch.cat(y_list, dim=0))
    if vix is not None:
        result += (torch.cat(vix_list, dim=0),)
    if regime is not None:
        result += (torch.cat(regime_list, dim=0),)
    if macro is not None:
        result += (torch.cat(macro_list, dim=0),)

    return result
