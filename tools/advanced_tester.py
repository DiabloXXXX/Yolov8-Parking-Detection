"""
Enhanced testing framework with statistical analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from datetime import datetime
import json

import sys
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

class AdvancedTester:
    """Advanced testing with statistical analysis"""
    
    def __init__(self, detector, config):
        self.detector = detector
        self.config = config
        self.detailed_results = []
    
    def run_statistical_test(self, video_path: str, iterations: int = 10):
        """Run multiple iterations for statistical analysis"""
        print(f"📊 Statistical Testing: {video_path} ({iterations} iterations)")
        
        results = []
        for i in range(iterations):
            print(f"   Run {i+1}/{iterations}...")
            result = self._single_test_run(video_path)
            if result:
                results.append(result)
        
        if results:
            stats_result = self._calculate_statistics(results, video_path)
            self.detailed_results.append(stats_result)
            self._print_statistics(stats_result)
            return stats_result
        
        return None
    
    def _single_test_run(self, video_path: str):
        """Single test run implementation"""
        # Implementation similar to existing test but return detailed metrics
        pass
    
    def _calculate_statistics(self, results: list, video_path: str):
        """Calculate statistical metrics"""
        detections = [r['detection_count'] for r in results]
        inference_times = [r['inference_time'] for r in results]
        
        stats_result = {
            'video_path': video_path,
            'iterations': len(results),
            'detection_stats': {
                'mean': np.mean(detections),
                'std': np.std(detections),
                'min': np.min(detections),
                'max': np.max(detections),
                'median': np.median(detections),
                'confidence_interval': stats.t.interval(0.95, len(detections)-1, 
                                                      loc=np.mean(detections), 
                                                      scale=stats.sem(detections))
            },
            'performance_stats': {
                'mean_inference': np.mean(inference_times),
                'std_inference': np.std(inference_times)
            },
            'raw_data': results,
            'timestamp': datetime.now().isoformat()
        }
        
        return stats_result
    
    def generate_statistical_report(self):
        """Generate comprehensive statistical report"""
        if not self.detailed_results:
            print("❌ No statistical data available")
            return
        
        print("\n📈 STATISTICAL ANALYSIS REPORT")
        print("=" * 70)
        
        # Create comparison table
        df_data = []
        for result in self.detailed_results:
            video_name = result['video_path'].split('/')[-1]
            stats = result['detection_stats']
            df_data.append({
                'Video': video_name,
                'Mean': f"{stats['mean']:.2f}",
                'Std Dev': f"{stats['std']:.2f}",
                'Min': stats['min'],
                'Max': stats['max'],
                'Median': f"{stats['median']:.2f}",
                'CI Lower': f"{stats['confidence_interval'][0]:.2f}",
                'CI Upper': f"{stats['confidence_interval'][1]:.2f}"
            })
        
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        
        # Save detailed results
        output_json = PROJECT_ROOT / 'output' / 'statistical_analysis.json'
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(self.detailed_results, f, indent=2)
        print(f"\n💾 Detailed results saved to: {output_json}")
    
    def plot_statistical_analysis(self):
        """Create statistical visualization plots"""
        if not self.detailed_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Box plots for detection counts
        video_names = []
        all_detections = []
        
        for result in self.detailed_results:
            video_name = result['video_path'].split('/')[-1]
            detections = [r['detection_count'] for r in result['raw_data']]
            video_names.append(video_name)
            all_detections.append(detections)
        
        axes[0,0].boxplot(all_detections, labels=video_names)
        axes[0,0].set_title('Detection Count Distribution')
        axes[0,0].set_ylabel('Vehicle Count')
        
        # Confidence intervals
        means = [r['detection_stats']['mean'] for r in self.detailed_results]
        cis_lower = [r['detection_stats']['confidence_interval'][0] for r in self.detailed_results]
        cis_upper = [r['detection_stats']['confidence_interval'][1] for r in self.detailed_results]
        
        x_pos = range(len(video_names))
        axes[0,1].errorbar(x_pos, means, 
                          yerr=[np.array(means) - np.array(cis_lower),
                                np.array(cis_upper) - np.array(means)],
                          fmt='o', capsize=5)
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels(video_names, rotation=45)
        axes[0,1].set_title('95% Confidence Intervals')
        axes[0,1].set_ylabel('Vehicle Count')
        
        # Standard deviation comparison
        std_devs = [r['detection_stats']['std'] for r in self.detailed_results]
        axes[1,0].bar(video_names, std_devs)
        axes[1,0].set_title('Detection Variability (Std Dev)')
        axes[1,0].set_ylabel('Standard Deviation')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Performance comparison
        perf_means = [r['performance_stats']['mean_inference'] for r in self.detailed_results]
        axes[1,1].bar(video_names, perf_means)
        axes[1,1].set_title('Average Inference Time')
        axes[1,1].set_ylabel('Time (ms)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_png = PROJECT_ROOT / 'output' / 'statistical_analysis.png'
        output_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Statistical plots saved to: {output_png}")

print("✅ Advanced testing framework ready!")
