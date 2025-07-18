"""
Performance Monitor for AI Dispatch Engine
Tracks learning progress and optimization metrics over time
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Any
import logging

class PerformanceMonitor:
    def __init__(self):
        """Initialize performance monitoring"""
        self.logger = logging.getLogger(__name__)
        
        # Performance history
        self.run_history = []
        self.learning_metrics = {
            'total_runs': 0,
            'best_distance': float('inf'),
            'best_balance': 0.0,
            'average_improvement': 0.0,
            'convergence_trend': []
        }
        
        # Set style for plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def track_run(self, result: Dict[str, Any]):
        """Track a single optimization run"""
        run_data = {
            'run_id': result['run_id'],
            'timestamp': datetime.now().isoformat(),
            'total_distance': result['total_distance'],
            'balance_score': result['balance_score'],
            'cluster_score': result['cluster_score'],
            'generations': result['generations'],
            'improvement_%': result['improvement_%'],
            'processing_time': result['processing_time']
        }
        
        self.run_history.append(run_data)
        self.learning_metrics['total_runs'] += 1
        
        # Update best records
        if result['total_distance'] < self.learning_metrics['best_distance']:
            self.learning_metrics['best_distance'] = result['total_distance']
        
        if result['balance_score'] > self.learning_metrics['best_balance']:
            self.learning_metrics['best_balance'] = result['balance_score']
        
        # Update convergence trend
        self.learning_metrics['convergence_trend'].append(result['generations'])
        
        # Calculate average improvement
        improvements = [r['improvement_%'] for r in self.run_history]
        self.learning_metrics['average_improvement'] = np.mean(improvements)
        
        self.logger.info(f"Tracked run {result['run_id']}: Distance={result['total_distance']:.2f}, "
                        f"Balance={result['balance_score']:.3f}, Improvement={result['improvement_%']:.1f}%")
    
    def show_learning_trends(self):
        """Display comprehensive learning trends"""
        if len(self.run_history) < 2:
            self.logger.warning("Need at least 2 runs to show trends")
            return None
        
        df = pd.DataFrame(self.run_history)
        
        # Create comprehensive dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('AI Dispatch Engine - Learning Progress', fontsize=16, fontweight='bold')
        
        # 1. Distance Optimization Over Time
        axes[0, 0].plot(df['run_id'], df['total_distance'], 'o-', linewidth=2, markersize=6)
        axes[0, 0].set_title('Total Distance Optimization', fontweight='bold')
        axes[0, 0].set_xlabel('Run Number')
        axes[0, 0].set_ylabel('Total Distance (km)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['run_id'], df['total_distance'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(df['run_id'], p(df['run_id']), "--", alpha=0.8, color='red')
        
        # 2. AI Learning Progress (Improvement over Random)
        axes[0, 1].plot(df['run_id'], df['improvement_%'], 'o-', linewidth=2, markersize=6, color='green')
        axes[0, 1].set_title('AI Learning Progress', fontweight='bold')
        axes[0, 1].set_xlabel('Run Number')
        axes[0, 1].set_ylabel('Improvement Over Random (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add target line
        axes[0, 1].axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Target (20%)')
        axes[0, 1].legend()
        
        # 3. Workload Balance Improvement
        axes[0, 2].plot(df['run_id'], df['balance_score'], 'o-', linewidth=2, markersize=6, color='orange')
        axes[0, 2].set_title('Workload Balance Improvement', fontweight='bold')
        axes[0, 2].set_xlabel('Run Number')
        axes[0, 2].set_ylabel('Balance Score')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 1)
        
        # 4. Convergence Speed Analysis
        axes[1, 0].plot(df['run_id'], df['generations'], 'o-', linewidth=2, markersize=6, color='purple')
        axes[1, 0].set_title('AI Convergence Speed', fontweight='bold')
        axes[1, 0].set_xlabel('Run Number')
        axes[1, 0].set_ylabel('Generations to Converge')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Processing Time Efficiency
        axes[1, 1].plot(df['run_id'], df['processing_time'], 'o-', linewidth=2, markersize=6, color='brown')
        axes[1, 1].set_title('Processing Time Efficiency', fontweight='bold')
        axes[1, 1].set_xlabel('Run Number')
        axes[1, 1].set_ylabel('Processing Time (seconds)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Performance Distribution
        metrics = ['total_distance', 'balance_score', 'improvement_%']
        normalized_data = []
        
        for metric in metrics:
            # Normalize to 0-1 scale for comparison
            values = df[metric].values
            if metric == 'total_distance':
                # Lower is better for distance
                normalized = 1 - (values - values.min()) / (values.max() - values.min() + 1e-6)
            else:
                # Higher is better for balance and improvement
                normalized = (values - values.min()) / (values.max() - values.min() + 1e-6)
            normalized_data.append(normalized)
        
        box_data = [normalized_data[0], normalized_data[1], normalized_data[2]]
        axes[1, 2].boxplot(box_data, labels=['Distance\n(Optimized)', 'Balance\nScore', 'Improvement\n(%)'])
        axes[1, 2].set_title('Performance Distribution', fontweight='bold')
        axes[1, 2].set_ylabel('Normalized Performance')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        self._print_learning_summary(df)
        
        return fig
    
    def _print_learning_summary(self, df: pd.DataFrame):
        """Print detailed learning summary"""
        print("\n" + "="*60)
        print("AI DISPATCH ENGINE - LEARNING SUMMARY")
        print("="*60)
        
        print(f"📊 Total Runs: {len(df)}")
        print(f"🎯 Best Distance: {df['total_distance'].min():.2f} km")
        print(f"⚖️  Best Balance: {df['balance_score'].max():.3f}")
        print(f"📈 Average Improvement: {df['improvement_%'].mean():.1f}%")
        
        # Learning trends
        if len(df) >= 3:
            # Distance trend
            recent_distance = df['total_distance'].tail(3).mean()
            early_distance = df['total_distance'].head(3).mean()
            distance_improvement = ((early_distance - recent_distance) / early_distance) * 100
            
            # Balance trend
            recent_balance = df['balance_score'].tail(3).mean()
            early_balance = df['balance_score'].head(3).mean()
            balance_improvement = ((recent_balance - early_balance) / early_balance) * 100
            
            print(f"📉 Distance Learning: {distance_improvement:.1f}% improvement")
            print(f"📊 Balance Learning: {balance_improvement:.1f}% improvement")
        
        # Convergence analysis
        avg_convergence = df['generations'].mean()
        print(f"🔄 Average Convergence: {avg_convergence:.1f} generations")
        
        # Efficiency metrics
        avg_time = df['processing_time'].mean()
        print(f"⏱️  Average Processing: {avg_time:.1f} seconds")
        
        # AI Intelligence indicators
        improvement_trend = np.polyfit(df['run_id'], df['improvement_%'], 1)[0]
        convergence_trend = np.polyfit(df['run_id'], df['generations'], 1)[0]
        
        print(f"\n🧠 AI INTELLIGENCE INDICATORS:")
        print(f"   Learning Rate: {improvement_trend:.2f}% per run")
        print(f"   Convergence Trend: {convergence_trend:.1f} gen/run")
        
        if improvement_trend > 0.1:
            print("   Status: ✅ AI is actively learning and improving")
        elif improvement_trend > -0.1:
            print("   Status: ⚖️  AI performance is stable")
        else:
            print("   Status: ⚠️  AI may need parameter tuning")
        
        print("="*60)
    
    def create_performance_comparison(self, baseline_results: Dict = None):
        """Compare AI performance against baselines"""
        if not self.run_history:
            self.logger.warning("No run history available for comparison")
            return None
        
        df = pd.DataFrame(self.run_history)
        latest_run = df.iloc[-1]
        
        # Default baseline if not provided
        if baseline_results is None:
            baseline_results = {
                'Random Assignment': {
                    'total_distance': latest_run['total_distance'] * 1.3,
                    'balance_score': 0.4,
                    'processing_time': 0.1
                },
                'Greedy Assignment': {
                    'total_distance': latest_run['total_distance'] * 1.15,
                    'balance_score': 0.6,
                    'processing_time': 2.0
                }
            }
        
        # Add AI result
        comparison_data = baseline_results.copy()
        comparison_data['AI Genetic Algorithm'] = {
            'total_distance': latest_run['total_distance'],
            'balance_score': latest_run['balance_score'],
            'processing_time': latest_run['processing_time']
        }
        
        # Create comparison chart
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        methods = list(comparison_data.keys())
        distances = [comparison_data[m]['total_distance'] for m in methods]
        balances = [comparison_data[m]['balance_score'] for m in methods]
        times = [comparison_data[m]['processing_time'] for m in methods]
        
        # Distance comparison
        bars1 = axes[0].bar(methods, distances, color=['red', 'orange', 'green'])
        axes[0].set_title('Total Distance Comparison', fontweight='bold')
        axes[0].set_ylabel('Distance (km)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Balance comparison
        bars2 = axes[1].bar(methods, balances, color=['red', 'orange', 'green'])
        axes[1].set_title('Workload Balance Comparison', fontweight='bold')
        axes[1].set_ylabel('Balance Score')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_ylim(0, 1)
        
        # Time comparison
        bars3 = axes[2].bar(methods, times, color=['red', 'orange', 'green'])
        axes[2].set_title('Processing Time Comparison', fontweight='bold')
        axes[2].set_ylabel('Time (seconds)')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def export_performance_data(self, filename: str = None):
        """Export performance data to CSV"""
        if not self.run_history:
            self.logger.warning("No performance data to export")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_dispatch_performance_{timestamp}.csv"
        
        df = pd.DataFrame(self.run_history)
        df.to_csv(filename, index=False)
        
        self.logger.info(f"Performance data exported to {filename}")
        return filename
    
    def save_learning_state(self, filename: str = None):
        """Save learning state for future analysis"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_learning_state_{timestamp}.json"
        
        state = {
            'run_history': self.run_history,
            'learning_metrics': self.learning_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Learning state saved to {filename}")
        return filename
    
    def load_learning_state(self, filename: str):
        """Load previous learning state"""
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            
            self.run_history = state['run_history']
            self.learning_metrics = state['learning_metrics']
            
            self.logger.info(f"Learning state loaded from {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load learning state: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Test performance monitor
    monitor = PerformanceMonitor()
    
    # Simulate some runs with improving performance
    for i in range(10):
        # Simulate improving AI performance
        base_distance = 1000
        improvement_factor = 1 - (i * 0.05)  # 5% improvement per run
        
        mock_result = {
            'run_id': i,
            'total_distance': base_distance * improvement_factor + np.random.normal(0, 10),
            'balance_score': 0.5 + (i * 0.04) + np.random.normal(0, 0.05),
            'cluster_score': 0.7 + np.random.normal(0, 0.1),
            'generations': 300 - (i * 10) + np.random.randint(-20, 20),
            'improvement_%': 10 + (i * 2) + np.random.normal(0, 2),
            'processing_time': 25 + np.random.normal(0, 3)
        }
        
        monitor.track_run(mock_result)
    
    # Show learning trends
    monitor.show_learning_trends()
    
    # Create performance comparison
    monitor.create_performance_comparison()
    
    # Export data
    monitor.export_performance_data("test_performance.csv")
    monitor.save_learning_state("test_learning_state.json")
    
    print("Performance monitor test completed!")