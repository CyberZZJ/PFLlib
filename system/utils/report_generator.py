#!/usr/bin/env python
import os
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


class ReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def generate_full_report(self, results: List[Dict], dataset_name: str, 
                             algorithm: str, config: Dict) -> str:
        report_path = os.path.join(self.output_dir, "tuning_report.md")
        
        best_result = max(results, key=lambda x: x['best_accuracy'])
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 自适应差分隐私调参报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## 1. 实验配置\n\n")
            f.write(f"| 配置项 | 值 |\n")
            f.write(f"|--------|----|\n")
            f.write(f"| 数据集 | {dataset_name} |\n")
            f.write(f"| 算法 | {algorithm} |\n")
            f.write(f"| 测试参数组合数 | {len(results)} |\n")
            f.write(f"| 隐私预算范围 ε | {config.get('epsilon_range', 'N/A')} |\n")
            f.write(f"| 裁剪范数范围 C | {config.get('clip_norm_range', 'N/A')} |\n")
            f.write(f"| 噪声乘数范围 σ | {config.get('noise_multiplier_range', 'N/A')} |\n")
            f.write(f"| 近端项系数范围 μ | {config.get('mu_range', 'N/A')} |\n\n")
            
            f.write(f"## 2. 最佳结果\n\n")
            f.write(f"| 指标 | 值 |\n")
            f.write(f"|------|----|\n")
            f.write(f"| **最佳准确率** | **{best_result['best_accuracy']:.4f}** |\n")
            f.write(f"| 最佳轮次 | {best_result['best_round']} |\n")
            f.write(f"| 隐私预算 ε | {best_result['privacy_budget']:.4f} |\n")
            f.write(f"| 近端项系数 μ | {best_result['params']['mu']:.3f} |\n")
            f.write(f"| 裁剪范数 C | {best_result['params']['clip_norm']:.2f} |\n")
            f.write(f"| 噪声乘数 σ | {best_result['params']['noise_multiplier']:.2f} |\n\n")
            
            f.write(f"## 3. 所有参数组合结果\n\n")
            f.write(f"| 排名 | 准确率 | 轮次 | ε | μ | C | σ | 早停 |\n")
            f.write(f"|------|--------|------|---|---|---|---|------|\n")
            
            sorted_results = sorted(results, key=lambda x: x['best_accuracy'], reverse=True)
            for i, r in enumerate(sorted_results, 1):
                early_stop = "是" if r.get('terminated_early', False) else "否"
                f.write(f"| {i} | {r['best_accuracy']:.4f} | {r['best_round']} | "
                       f"{r['privacy_budget']:.2f} | {r['params']['mu']:.3f} | "
                       f"{r['params']['clip_norm']:.2f} | {r['params']['noise_multiplier']:.2f} | {early_stop} |\n")
            
            f.write(f"\n## 4. 隐私预算审计\n\n")
            f.write(f"### 4.1 隐私预算分布\n\n")
            
            epsilons = [r['privacy_budget'] for r in results]
            f.write(f"- 最小 ε: {min(epsilons):.4f}\n")
            f.write(f"- 最大 ε: {max(epsilons):.4f}\n")
            f.write(f"- 平均 ε: {np.mean(epsilons):.4f}\n")
            f.write(f"- 最佳结果 ε: {best_result['privacy_budget']:.4f}\n\n")
            
            f.write(f"### 4.2 隐私约束验证\n\n")
            epsilon_max = config.get('epsilon_range', (0, 10))[1]
            valid_count = sum(1 for e in epsilons if e <= epsilon_max)
            f.write(f"- 隐私预算上限: {epsilon_max}\n")
            f.write(f"- 满足约束的组合数: {valid_count}/{len(results)}\n")
            f.write(f"- 最佳结果是否满足约束: {'是' if best_result['privacy_budget'] <= epsilon_max else '否'}\n\n")
            
            f.write(f"## 5. 训练效率分析\n\n")
            total_rounds = sum(r['total_rounds'] for r in results)
            early_stop_count = sum(1 for r in results if r.get('terminated_early', False))
            f.write(f"- 总训练轮数: {total_rounds}\n")
            f.write(f"- 平均每组合轮数: {np.mean([r['total_rounds'] for r in results]):.1f}\n")
            f.write(f"- 早停组合数: {early_stop_count}/{len(results)}\n\n")
            
            f.write(f"## 6. 结论与建议\n\n")
            f.write(f"### 6.1 最优参数推荐\n\n")
            f.write(f"```python\n")
            f.write(f"recommended_params = {{\n")
            f.write(f"    'mu': {best_result['params']['mu']},\n")
            f.write(f"    'clip_norm': {best_result['params']['clip_norm']},\n")
            f.write(f"    'noise_multiplier': {best_result['params']['noise_multiplier']},\n")
            f.write(f"    'epsilon': {best_result['params']['epsilon']},\n")
            f.write(f"    'delta': {best_result['params']['delta']}\n")
            f.write(f"}}\n")
            f.write(f"```\n\n")
            
            f.write(f"### 6.2 性能总结\n\n")
            f.write(f"- 在{dataset_name}数据集上，使用{algorithm}算法\n")
            f.write(f"- 最佳准确率达到 **{best_result['best_accuracy']*100:.2f}%**\n")
            f.write(f"- 隐私预算消耗为 ε = {best_result['privacy_budget']:.4f}\n")
            f.write(f"- 训练在第{best_result['best_round']}轮达到最佳性能\n\n")
        
        print(f"报告已生成: {report_path}")
        return report_path
    
    def generate_accuracy_curve(self, results: List[Dict], 
                                title: str = "准确率收敛曲线") -> str:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        sorted_results = sorted(results, key=lambda x: x['best_accuracy'], reverse=True)
        
        colors = plt.cm.viridis(np.linspace(0, 1, min(5, len(sorted_results))))
        for i, r in enumerate(sorted_results[:5]):
            if 'accuracy_history' in r:
                ax1.plot(r['accuracy_history'], 
                        label=f"组合{r['param_idx']+1} (acc={r['best_accuracy']:.3f})",
                        color=colors[i])
        
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('准确率')
        ax1.set_title(f'{title} - Top 5参数组合')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        accuracies = [r['best_accuracy'] for r in results]
        param_indices = [r['param_idx'] + 1 for r in results]
        
        bars = ax2.bar(param_indices, accuracies, color='steelblue', alpha=0.7)
        
        best_idx = np.argmax(accuracies)
        bars[best_idx].set_color('red')
        
        ax2.set_xlabel('参数组合编号')
        ax2.set_ylabel('最佳准确率')
        ax2.set_title('各参数组合最佳准确率对比')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for i, (idx, acc) in enumerate(zip(param_indices, accuracies)):
            if i == best_idx:
                ax2.annotate(f'{acc:.4f}', xy=(idx, acc), 
                           xytext=(0, 5), textcoords='offset points',
                           ha='center', fontweight='bold', color='red')
        
        plt.tight_layout()
        
        curve_path = os.path.join(self.output_dir, "accuracy_curve.png")
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"准确率曲线已生成: {curve_path}")
        return curve_path
    
    def generate_privacy_analysis(self, results: List[Dict]) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1 = axes[0]
        epsilons = [r['privacy_budget'] for r in results]
        accuracies = [r['best_accuracy'] for r in results]
        
        scatter = ax1.scatter(epsilons, accuracies, c=accuracies, 
                             cmap='RdYlGn', s=100, alpha=0.7)
        plt.colorbar(scatter, ax=ax1, label='准确率')
        
        best_idx = np.argmax(accuracies)
        ax1.scatter([epsilons[best_idx]], [accuracies[best_idx]], 
                   c='red', s=200, marker='*', label='最佳结果')
        
        ax1.set_xlabel('隐私预算 ε')
        ax1.set_ylabel('最佳准确率')
        ax1.set_title('隐私-效用权衡')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        param_names = ['μ', 'C', 'σ', 'ε']
        
        correlations = []
        for param in ['mu', 'clip_norm', 'noise_multiplier', 'epsilon']:
            values = [r['params'][param] for r in results]
            corr = np.corrcoef(values, accuracies)[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        
        colors = ['green' if c > 0 else 'red' for c in correlations]
        bars = ax2.bar(param_names, correlations, color=colors, alpha=0.7)
        
        ax2.set_xlabel('参数')
        ax2.set_ylabel('与准确率的相关系数')
        ax2.set_title('参数-准确率相关性分析')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax2.annotate(f'{corr:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        
        analysis_path = os.path.join(self.output_dir, "privacy_analysis.png")
        plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"隐私分析图已生成: {analysis_path}")
        return analysis_path
    
    def generate_json_report(self, results: List[Dict], dataset_name: str,
                            algorithm: str, config: Dict) -> str:
        best_result = max(results, key=lambda x: x['best_accuracy'])
        
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "dataset": dataset_name,
                "algorithm": algorithm,
                "total_combinations": len(results)
            },
            "best_result": {
                "accuracy": best_result['best_accuracy'],
                "round": best_result['best_round'],
                "privacy_budget": best_result['privacy_budget'],
                "params": best_result['params']
            },
            "all_results": [
                {
                    "param_idx": r['param_idx'],
                    "accuracy": r['best_accuracy'],
                    "round": r['best_round'],
                    "privacy_budget": r['privacy_budget'],
                    "params": r['params'],
                    "terminated_early": r.get('terminated_early', False)
                }
                for r in sorted(results, key=lambda x: x['best_accuracy'], reverse=True)
            ],
            "statistics": {
                "accuracy": {
                    "max": max(r['best_accuracy'] for r in results),
                    "min": min(r['best_accuracy'] for r in results),
                    "mean": np.mean([r['best_accuracy'] for r in results]),
                    "std": np.std([r['best_accuracy'] for r in results])
                },
                "privacy_budget": {
                    "max": max(r['privacy_budget'] for r in results),
                    "min": min(r['privacy_budget'] for r in results),
                    "mean": np.mean([r['privacy_budget'] for r in results])
                }
            },
            "config": config
        }
        
        json_path = os.path.join(self.output_dir, "report.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"JSON报告已生成: {json_path}")
        return json_path


if __name__ == "__main__":
    test_results = [
        {
            "param_idx": 0,
            "best_accuracy": 0.85,
            "best_round": 15,
            "total_rounds": 20,
            "privacy_budget": 4.5,
            "params": {"mu": 0.1, "clip_norm": 0.5, "noise_multiplier": 1.0, "epsilon": 5.0, "delta": 1e-5},
            "terminated_early": True,
            "accuracy_history": [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.84, 0.85, 0.85, 0.85]
        },
        {
            "param_idx": 1,
            "best_accuracy": 0.82,
            "best_round": 18,
            "total_rounds": 25,
            "privacy_budget": 5.2,
            "params": {"mu": 0.15, "clip_norm": 0.6, "noise_multiplier": 0.8, "epsilon": 4.5, "delta": 1e-5},
            "terminated_early": False,
            "accuracy_history": [0.5, 0.55, 0.62, 0.68, 0.72, 0.75, 0.78, 0.80, 0.81, 0.82]
        }
    ]
    
    generator = ReportGenerator("./test_output")
    generator.generate_full_report(test_results, "MNIST", "DPProx", 
                                   {"epsilon_range": (4.0, 5.0)})
    generator.generate_accuracy_curve(test_results)
    generator.generate_privacy_analysis(test_results)
