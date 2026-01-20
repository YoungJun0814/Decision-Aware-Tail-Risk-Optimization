"""
Explainability Module (XAI)
===========================
[Optional] 모델 해석 가능성 분석 도구

역할: AI가 "왜" 그런 포트폴리오 비중을 결정했는지 설명합니다.

주요 기능:
1. GradientSaliency: 입력 변수별 기여도 분석 (Gradient-based)
2. AttentionAnalyzer: 시점별 중요도 분석 (Temporal Attention)
3. CounterfactualAnalyzer: "만약 VIX가 더 높았다면?" 가상 시나리오 분석
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Dict, List, Union


# =============================================================================
# 1. Gradient-based Saliency Map
# =============================================================================

class GradientSaliency:
    """
    Gradient-based Saliency Map (입력 민감도 분석)
    
    모델의 출력(포트폴리오 비중)에 대한 입력 변수의 기여도를 계산합니다.
    "어떤 자산의 과거 데이터가 현재 의사결정에 가장 큰 영향을 주었는가?"
    
    Math:
        Saliency = |∂output / ∂input|
    
    Example:
        >>> saliency = GradientSaliency(model)
        >>> importance = saliency.compute(input_tensor, target_asset_idx=4)  # SH
        >>> saliency.plot_heatmap(importance, feature_names)
    """
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: 학습된 DecisionAwareNet 또는 TransformerNet
        """
        self.model = model
        self.model.eval()
    
    def compute(
        self, 
        x: torch.Tensor, 
        target_asset_idx: int = 0,
        absolute: bool = True
    ) -> torch.Tensor:
        """
        입력에 대한 Saliency Map 계산
        
        Args:
            x: (Batch, Seq, Features) 형태의 입력 텐서
            target_asset_idx: 분석할 타겟 자산 인덱스 (0-based)
                - 0: SPY, 1: XLV, 2: TLT, 3: GLD, 4: SH
            absolute: True면 절대값 반환 (기본값)
        
        Returns:
            saliency: (Batch, Seq, Features) 형태의 중요도 맵
        """
        # 입력에 대한 그래디언트 계산 활성화
        x_input = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = self.model(x_input)  # (Batch, Num_assets)
        
        # 타겟 자산의 비중에 대한 그래디언트
        target_output = output[:, target_asset_idx].sum()
        target_output.backward()
        
        # Saliency = 입력에 대한 그래디언트
        saliency = x_input.grad
        
        if absolute:
            saliency = saliency.abs()
        
        return saliency.detach()
    
    def compute_all_assets(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        모든 자산에 대한 Saliency Map 계산
        
        Args:
            x: (Batch, Seq, Features) 형태의 입력 텐서
        
        Returns:
            각 자산 인덱스를 키로 하는 Saliency 딕셔너리
        """
        num_assets = self.model.num_assets
        saliency_maps = {}
        
        for asset_idx in range(num_assets):
            saliency_maps[asset_idx] = self.compute(x, target_asset_idx=asset_idx)
        
        return saliency_maps
    
    def aggregate(
        self, 
        saliency: torch.Tensor, 
        mode: str = 'mean'
    ) -> np.ndarray:
        """
        Saliency를 시퀀스 차원에서 집계
        
        Args:
            saliency: (Batch, Seq, Features) 형태
            mode: 집계 방식 ('mean', 'max', 'sum')
        
        Returns:
            aggregated: (Features,) 형태의 중요도 배열
        """
        if mode == 'mean':
            agg = saliency.mean(dim=(0, 1))
        elif mode == 'max':
            agg = saliency.max(dim=1)[0].mean(dim=0)
        else:  # sum
            agg = saliency.sum(dim=(0, 1))
        
        return agg.cpu().numpy()
    
    def plot_heatmap(
        self,
        saliency: torch.Tensor,
        feature_names: List[str],
        sample_idx: int = 0,
        title: str = "Input Saliency Map",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Saliency Map 시각화 (Heatmap)
        
        Args:
            saliency: (Batch, Seq, Features) 형태의 Saliency 텐서
            feature_names: 특성 이름 리스트 (예: ['SPY', 'XLV', 'TLT', 'GLD', 'SH'])
            sample_idx: 시각화할 샘플 인덱스
            title: 그래프 제목
            figsize: 그래프 크기
            save_path: 저장 경로 (None이면 저장 안 함)
        
        Returns:
            matplotlib Figure 객체
        """
        # 단일 샘플 추출
        sal_np = saliency[sample_idx].cpu().numpy()  # (Seq, Features)
        
        # 시퀀스 인덱스 생성 (t-12, t-11, ..., t-1)
        seq_len = sal_np.shape[0]
        time_labels = [f"t-{seq_len - i}" for i in range(seq_len)]
        
        # 히트맵 생성
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            sal_np.T,  # (Features, Seq) 형태로 전치
            xticklabels=time_labels,
            yticklabels=feature_names,
            cmap='YlOrRd',
            annot=False,
            ax=ax,
            cbar_kws={'label': 'Importance'}
        )
        
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Input Feature")
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Saved saliency heatmap to: {save_path}")
        
        return fig
    
    def plot_feature_importance(
        self,
        saliency: torch.Tensor,
        feature_names: List[str],
        title: str = "Feature Importance (Aggregated Saliency)",
        figsize: Tuple[int, int] = (10, 5),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        집계된 Feature Importance 바 차트
        
        Args:
            saliency: (Batch, Seq, Features) 형태
            feature_names: 특성 이름 리스트
            title: 그래프 제목
            figsize: 그래프 크기
            save_path: 저장 경로
        
        Returns:
            matplotlib Figure 객체
        """
        # 집계
        importance = self.aggregate(saliency, mode='mean')
        
        # 정렬
        sorted_idx = np.argsort(importance)[::-1]
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_importance = importance[sorted_idx]
        
        # 바 차트
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(sorted_names)))[::-1]
        bars = ax.barh(sorted_names, sorted_importance, color=colors)
        
        ax.set_xlabel("Average Saliency (Importance)")
        ax.set_title(title)
        ax.invert_yaxis()  # 상위 항목이 위에 오도록
        
        # 값 라벨 추가
        for bar, val in zip(bars, sorted_importance):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Saved feature importance to: {save_path}")
        
        return fig


# =============================================================================
# 2. Attention Weights Analysis
# =============================================================================

class AttentionAnalyzer:
    """
    Temporal Attention Analysis (시간 가중치 분석)
    
    "모델이 과거 12개월 중 어느 시점을 가장 중요하게 참고했는가?"
    
    LSTM의 경우 Attention이 없으므로, 대리 지표로 Gradient-based attention을 사용합니다.
    Transformer의 경우 실제 Attention weights를 추출합니다.
    
    Example:
        >>> analyzer = AttentionAnalyzer(model)
        >>> temporal_weights = analyzer.compute_temporal_importance(input_tensor)
        >>> analyzer.plot_temporal_attention(temporal_weights)
    """
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: 학습된 모델 (DecisionAwareNet 또는 TransformerNet)
        """
        self.model = model
        self.model.eval()
        self._is_transformer = hasattr(model, 'transformer')
    
    def compute_temporal_importance(
        self, 
        x: torch.Tensor,
        target_asset_idx: int = 0
    ) -> np.ndarray:
        """
        시점별 중요도 계산 (Gradient-based Temporal Attention)
        
        LSTM 모델에서 각 시점(t-12, t-11, ..., t-1)이 
        최종 의사결정에 얼마나 기여했는지 계산합니다.
        
        Args:
            x: (Batch, Seq, Features) 형태의 입력
            target_asset_idx: 분석할 자산 인덱스
        
        Returns:
            temporal_importance: (Seq,) 형태의 시점별 중요도
        """
        x_input = x.clone().detach().requires_grad_(True)
        
        # Forward
        output = self.model(x_input)
        target = output[:, target_asset_idx].sum()
        target.backward()
        
        # 각 시점의 그래디언트 크기 계산 (L2 norm across features)
        grad = x_input.grad  # (Batch, Seq, Features)
        temporal_importance = grad.norm(dim=2).mean(dim=0)  # (Seq,)
        
        # 정규화 (합 = 1)
        temporal_importance = temporal_importance / temporal_importance.sum()
        
        return temporal_importance.cpu().numpy()
    
    def compute_multi_asset_temporal(
        self, 
        x: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        모든 자산에 대한 시점별 중요도 계산
        
        Returns:
            딕셔너리 {asset_name: temporal_importance}
        """
        asset_names = ['SPY', 'XLV', 'TLT', 'GLD', 'SH']
        num_assets = min(len(asset_names), self.model.num_assets)
        
        result = {}
        for idx in range(num_assets):
            result[asset_names[idx]] = self.compute_temporal_importance(x, idx)
        
        return result
    
    def plot_temporal_attention(
        self,
        temporal_importance: np.ndarray,
        title: str = "Temporal Attention (Which past months matter most?)",
        figsize: Tuple[int, int] = (10, 5),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        시점별 중요도 시각화
        
        Args:
            temporal_importance: (Seq,) 형태의 중요도 배열
            title: 그래프 제목
            figsize: 그래프 크기
            save_path: 저장 경로
        
        Returns:
            matplotlib Figure 객체
        """
        seq_len = len(temporal_importance)
        time_labels = [f"t-{seq_len - i}" for i in range(seq_len)]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 바 차트
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, seq_len))
        bars = ax.bar(time_labels, temporal_importance, color=colors)
        
        # 가장 중요한 시점 강조
        max_idx = np.argmax(temporal_importance)
        bars[max_idx].set_color('#e74c3c')  # 빨간색
        
        ax.set_xlabel("Time Step (Past Months)")
        ax.set_ylabel("Importance Weight")
        ax.set_title(title)
        
        # 최대값 표시
        ax.annotate(
            f'Most Important\n({time_labels[max_idx]})',
            xy=(max_idx, temporal_importance[max_idx]),
            xytext=(max_idx, temporal_importance[max_idx] + 0.02),
            ha='center',
            fontsize=9,
            color='#e74c3c'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Saved temporal attention to: {save_path}")
        
        return fig
    
    def plot_multi_asset_attention(
        self,
        temporal_dict: Dict[str, np.ndarray],
        title: str = "Temporal Attention by Asset",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        자산별 시점 중요도 비교 (Line Chart)
        
        Args:
            temporal_dict: {asset_name: temporal_importance} 딕셔너리
            title: 그래프 제목
            figsize: 그래프 크기
            save_path: 저장 경로
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        seq_len = len(list(temporal_dict.values())[0])
        time_labels = [f"t-{seq_len - i}" for i in range(seq_len)]
        x_pos = range(seq_len)
        
        colors = {'SPY': '#3498db', 'XLV': '#2ecc71', 'TLT': '#9b59b6', 
                  'GLD': '#f1c40f', 'SH': '#e74c3c'}
        
        for asset_name, importance in temporal_dict.items():
            color = colors.get(asset_name, '#95a5a6')
            ax.plot(x_pos, importance, marker='o', label=asset_name, 
                    color=color, linewidth=2, markersize=5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(time_labels)
        ax.set_xlabel("Time Step (Past Months)")
        ax.set_ylabel("Importance Weight")
        ax.set_title(title)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Saved multi-asset attention to: {save_path}")
        
        return fig


# =============================================================================
# 3. Counterfactual Analysis
# =============================================================================

class CounterfactualAnalyzer:
    """
    Counterfactual Analysis (가상 시나리오 분석)
    
    "만약 VIX가 지금보다 2배 높았다면 모델은 어떻게 결정했을까?"
    
    입력 변수를 인위적으로 조작(Perturbation)해서 모델 반응을 분석합니다.
    민감도 곡선을 통해 모델의 행동 패턴을 이해할 수 있습니다.
    
    Example:
        >>> cf = CounterfactualAnalyzer(model)
        >>> results = cf.perturb_feature(input_tensor, feature_idx=0, scales=[0.5, 1.0, 2.0])
        >>> cf.plot_sensitivity_curve(results)
    """
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: 학습된 모델
        """
        self.model = model
        self.model.eval()
    
    def perturb_feature(
        self,
        x: torch.Tensor,
        feature_idx: int,
        scales: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
        mode: str = 'multiply'
    ) -> Dict[str, np.ndarray]:
        """
        특정 Feature를 조작하고 모델 반응 관찰
        
        Args:
            x: (Batch, Seq, Features) 형태의 입력
            feature_idx: 조작할 특성 인덱스 (예: 0=SPY returns)
            scales: 스케일 리스트 (1.0 = 원본)
            mode: 조작 방식
                - 'multiply': 원본 값에 scale 곱함
                - 'add': 원본 값에 (scale * std) 더함
        
        Returns:
            {
                'scales': 스케일 리스트,
                'weights': 각 스케일에서의 포트폴리오 비중 (scales, batch, assets)
            }
        """
        results = []
        
        with torch.no_grad():
            for scale in scales:
                x_perturbed = x.clone()
                
                if mode == 'multiply':
                    x_perturbed[:, :, feature_idx] *= scale
                else:  # add
                    std = x[:, :, feature_idx].std()
                    x_perturbed[:, :, feature_idx] += scale * std
                
                weights = self.model(x_perturbed)
                results.append(weights.cpu().numpy())
        
        return {
            'scales': np.array(scales),
            'weights': np.stack(results, axis=0)  # (num_scales, batch, assets)
        }
    
    def analyze_vix_sensitivity(
        self,
        x: torch.Tensor,
        vix_values: List[float] = [10, 15, 20, 25, 30, 40, 50, 60, 80],
        vix_feature_idx: int = 0  # VIX가 features의 몇 번째인지 (없으면 -1)
    ) -> Dict[str, np.ndarray]:
        """
        VIX 레벨에 따른 포트폴리오 반응 분석
        
        실제 VIX 값을 주입하여 모델이 위기 상황에서 
        어떤 자산 배분을 하는지 분석합니다.
        
        Args:
            x: 입력 텐서
            vix_values: 테스트할 VIX 레벨 리스트
            vix_feature_idx: VIX 특성의 인덱스 (현재 모델에서 VIX는 별도 입력이므로 -1)
        
        Returns:
            {'vix_levels': [...], 'weights': (num_levels, assets)}
        """
        # 주의: 현재 모델은 VIX를 입력 Features로 직접 사용하지 않고
        # Loss 함수에서만 사용합니다. 따라서 이 함수는 시연용입니다.
        # 실제로는 VIX를 Features에 추가하거나, 별도 분석 방법이 필요합니다.
        
        results = []
        
        with torch.no_grad():
            for vix in vix_values:
                x_perturbed = x.clone()
                
                # VIX가 별도 feature라면 해당 인덱스에 값 주입
                # (현재는 모든 feature를 일정 비율로 스케일링하는 대안 사용)
                # 높은 VIX = 높은 변동성 = 일반적으로 자산 수익률 하락
                volatility_factor = 1.0 - (vix - 20) / 100  # 가상 변동성 효과
                x_perturbed = x_perturbed * volatility_factor
                
                weights = self.model(x_perturbed)
                # 배치 평균
                mean_weights = weights.mean(dim=0)
                results.append(mean_weights.cpu().numpy())
        
        return {
            'vix_levels': np.array(vix_values),
            'weights': np.stack(results, axis=0)  # (num_levels, assets)
        }
    
    def compare_scenarios(
        self,
        x_base: torch.Tensor,
        scenarios: Dict[str, torch.Tensor]
    ) -> pd.DataFrame:
        """
        여러 시나리오의 포트폴리오 비중 비교
        
        Args:
            x_base: 기준 입력 텐서
            scenarios: {시나리오명: 입력 텐서} 딕셔너리
        
        Returns:
            비교 테이블 (DataFrame)
        """
        asset_names = ['SPY', 'XLV', 'TLT', 'GLD', 'SH']
        
        results = {}
        
        with torch.no_grad():
            # 기준 시나리오
            base_weights = self.model(x_base).mean(dim=0).cpu().numpy()
            results['Baseline'] = base_weights
            
            # 각 시나리오
            for name, x_scenario in scenarios.items():
                weights = self.model(x_scenario).mean(dim=0).cpu().numpy()
                results[name] = weights
        
        # DataFrame 생성
        df = pd.DataFrame(results, index=asset_names[:len(base_weights)])
        df = df.T  # 행: 시나리오, 열: 자산
        
        return df
    
    def plot_sensitivity_curve(
        self,
        perturb_results: Dict[str, np.ndarray],
        asset_names: List[str] = ['SPY', 'XLV', 'TLT', 'GLD', 'SH'],
        feature_name: str = "Feature",
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Perturbation에 따른 자산 비중 변화 곡선
        
        Args:
            perturb_results: perturb_feature() 반환값
            asset_names: 자산 이름 리스트
            feature_name: 조작한 특성 이름
            title: 그래프 제목
            figsize: 그래프 크기
            save_path: 저장 경로
        
        Returns:
            matplotlib Figure 객체
        """
        scales = perturb_results['scales']
        weights = perturb_results['weights']  # (num_scales, batch, assets)
        
        # 배치 평균
        mean_weights = weights.mean(axis=1)  # (num_scales, assets)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = {'SPY': '#3498db', 'XLV': '#2ecc71', 'TLT': '#9b59b6', 
                  'GLD': '#f1c40f', 'SH': '#e74c3c'}
        
        num_assets = mean_weights.shape[1]
        for i in range(num_assets):
            name = asset_names[i] if i < len(asset_names) else f"Asset {i}"
            color = colors.get(name, '#95a5a6')
            ax.plot(scales, mean_weights[:, i], marker='o', label=name,
                    color=color, linewidth=2, markersize=6)
        
        # 기준선 (scale=1.0)
        if 1.0 in scales:
            ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='Baseline')
        
        ax.set_xlabel(f"{feature_name} Scale (1.0 = Original)")
        ax.set_ylabel("Portfolio Weight")
        ax.set_title(title or f"Sensitivity to {feature_name} Perturbation")
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(0.6, mean_weights.max() * 1.1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Saved sensitivity curve to: {save_path}")
        
        return fig
    
    def plot_vix_sensitivity(
        self,
        vix_results: Dict[str, np.ndarray],
        asset_names: List[str] = ['SPY', 'XLV', 'TLT', 'GLD', 'SH'],
        title: str = "Portfolio Response to VIX Levels",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        VIX 레벨에 따른 자산 배분 변화 시각화
        
        Args:
            vix_results: analyze_vix_sensitivity() 반환값
            asset_names: 자산 이름
            title: 그래프 제목
            figsize: 그래프 크기
            save_path: 저장 경로
        """
        vix_levels = vix_results['vix_levels']
        weights = vix_results['weights']  # (num_levels, assets)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        colors = {'SPY': '#3498db', 'XLV': '#2ecc71', 'TLT': '#9b59b6', 
                  'GLD': '#f1c40f', 'SH': '#e74c3c'}
        
        # 1. Line Chart
        ax1 = axes[0]
        num_assets = weights.shape[1]
        for i in range(num_assets):
            name = asset_names[i] if i < len(asset_names) else f"Asset {i}"
            color = colors.get(name, '#95a5a6')
            ax1.plot(vix_levels, weights[:, i], marker='o', label=name,
                    color=color, linewidth=2, markersize=5)
        
        # 위기 구간 표시
        ax1.axvspan(30, vix_levels.max(), color='red', alpha=0.1, label='Crisis Zone (VIX>30)')
        ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.5)  # SH Cap
        ax1.annotate('SH Cap (30%)', xy=(vix_levels.max(), 0.31), 
                    fontsize=8, color='red', ha='right')
        
        ax1.set_xlabel("VIX Level")
        ax1.set_ylabel("Portfolio Weight")
        ax1.set_title("Weight Changes by VIX")
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # 2. Stacked Area Chart
        ax2 = axes[1]
        ax2.stackplot(
            vix_levels, 
            [weights[:, i] for i in range(num_assets)],
            labels=[asset_names[i] if i < len(asset_names) else f"Asset {i}" 
                    for i in range(num_assets)],
            colors=[colors.get(asset_names[i] if i < len(asset_names) else '', '#95a5a6')
                    for i in range(num_assets)],
            alpha=0.8
        )
        ax2.set_xlabel("VIX Level")
        ax2.set_ylabel("Cumulative Weight")
        ax2.set_title("Portfolio Composition by VIX")
        ax2.legend(loc='upper left', framealpha=0.9)
        ax2.set_ylim(0, 1.05)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Saved VIX sensitivity to: {save_path}")
        
        return fig


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Explainability Module Test")
    print("=" * 60)
    
    # 모델 임포트 (모듈 실행 시 src. 접두사 필요)
    try:
        from src.models import DecisionAwareNet
    except ModuleNotFoundError:
        from models import DecisionAwareNet
    
    # 테스트 설정
    batch_size = 10
    seq_length = 12
    input_dim = 5  # 5 assets
    num_assets = 5
    
    # 모델 생성
    model = DecisionAwareNet(input_dim=input_dim, num_assets=num_assets)
    model.eval()
    
    # 더미 입력
    dummy_input = torch.randn(batch_size, seq_length, input_dim)
    asset_names = ['SPY', 'XLV', 'TLT', 'GLD', 'SH']
    
    # ==========================================================================
    # 1. Gradient Saliency Test
    # ==========================================================================
    print("\n--- 1. Gradient Saliency Test ---")
    
    saliency = GradientSaliency(model)
    
    # SH (인버스) 비중에 대한 민감도 분석
    sal_map = saliency.compute(dummy_input, target_asset_idx=4)  # SH
    print(f"Saliency Map Shape: {sal_map.shape}")  # (batch, seq, features)
    
    # 집계
    importance = saliency.aggregate(sal_map, mode='mean')
    print(f"Feature Importance: {dict(zip(asset_names, importance.round(4)))}")
    
    # 시각화 (저장 없이 생성만)
    fig1 = saliency.plot_heatmap(sal_map, asset_names, sample_idx=0, 
                                  title="Saliency for SH Weight Decision")
    print("[OK] Saliency heatmap created")
    
    fig2 = saliency.plot_feature_importance(sal_map, asset_names)
    print("[OK] Feature importance chart created")
    
    # ==========================================================================
    # 2. Attention Analysis Test
    # ==========================================================================
    print("\n--- 2. Attention Analysis Test ---")
    
    attention = AttentionAnalyzer(model)
    
    # 시점별 중요도
    temporal_imp = attention.compute_temporal_importance(dummy_input, target_asset_idx=4)
    print(f"Temporal Importance Shape: {temporal_imp.shape}")  # (seq,)
    print(f"Most Important Time Step: t-{seq_length - np.argmax(temporal_imp)}")
    
    # 멀티 자산 분석
    multi_temporal = attention.compute_multi_asset_temporal(dummy_input)
    print(f"Multi-asset temporal analysis done for: {list(multi_temporal.keys())}")
    
    # 시각화
    fig3 = attention.plot_temporal_attention(temporal_imp, title="Temporal Attention for SH")
    print("[OK] Temporal attention chart created")
    
    fig4 = attention.plot_multi_asset_attention(multi_temporal)
    print("[OK] Multi-asset attention chart created")
    
    # ==========================================================================
    # 3. Counterfactual Analysis Test
    # ==========================================================================
    print("\n--- 3. Counterfactual Analysis Test ---")
    
    cf = CounterfactualAnalyzer(model)
    
    # SPY 수익률 변화에 대한 반응
    perturb_results = cf.perturb_feature(
        dummy_input, 
        feature_idx=0,  # SPY
        scales=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    )
    print(f"Perturbation Scales: {perturb_results['scales']}")
    print(f"Weights Shape: {perturb_results['weights'].shape}")
    
    # VIX 민감도
    vix_results = cf.analyze_vix_sensitivity(dummy_input)
    print(f"VIX Levels Analyzed: {vix_results['vix_levels']}")
    
    # 시나리오 비교
    scenarios = {
        'High Volatility': dummy_input * 1.5,
        'Low Volatility': dummy_input * 0.5,
        'Crash Scenario': dummy_input * -0.5
    }
    comparison = cf.compare_scenarios(dummy_input, scenarios)
    print("\nScenario Comparison:")
    print(comparison.round(4))
    
    # 시각화
    fig5 = cf.plot_sensitivity_curve(perturb_results, asset_names, feature_name="SPY Returns")
    print("[OK] Sensitivity curve created")
    
    fig6 = cf.plot_vix_sensitivity(vix_results, asset_names)
    print("[OK] VIX sensitivity chart created")
    
    # ==========================================================================
    # 완료
    # ==========================================================================
    print("\n" + "=" * 60)
    print("[SUCCESS] All XAI Module Tests Passed!")
    print("=" * 60)
    
    # 플롯 표시 (인터랙티브 환경에서)
    # plt.show()
