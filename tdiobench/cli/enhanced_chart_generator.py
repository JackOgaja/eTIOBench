#!/usr/bin/env python3
"""
Enhanced chart generation module for eTIOBench reports.

This module integrates the advanced chart generators with the existing report command
to provide comprehensive visualizations directly through the CLI.

Author: Jack Ogaja
Date: 2025-07-18
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configure styling
plt.style.use('default')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class EnhancedChartGenerator:
    """Enhanced chart generator for eTIOBench reports."""
    
    def __init__(self, output_dir: Path, chart_width: int = 12, chart_height: int = 8):
        self.output_dir = Path(output_dir)
        self.charts_dir = self.output_dir / "charts"
        self.charts_dir.mkdir(exist_ok=True)
        self.generated_charts = []
        self.chart_width = chart_width / 100  # Convert pixels to inches (approx)
        self.chart_height = chart_height / 100
    
    def generate_all_charts(self, benchmark_data: Dict[str, Any], benchmark_id: str, 
                          chart_types: List[str] = None) -> List[str]:
        """Generate all requested chart types for a benchmark."""
        
        chart_types = chart_types or ["comprehensive", "heatmap", "comparison"]
        
        print(f"📊 Generating enhanced charts for benchmark {benchmark_id[:12]}...")
        
        # Extract performance data
        performance_data = self._extract_performance_data(benchmark_data)
        
        if not performance_data['patterns']:
            print("⚠️  No performance data found for chart generation")
            return []
        
        print(f"✅ Found {performance_data['summary']['total_tests']} performance patterns")
        
        for chart_type in chart_types:
            try:
                if chart_type in ["comprehensive", "dashboard", "enhanced_dashboard", "all"]:
                    self._create_comprehensive_dashboard(performance_data, benchmark_id)
                
                if chart_type in ["heatmap", "heat", "performance_heatmap", "all"]:
                    self._create_performance_heatmap(performance_data, benchmark_id)
                
                if chart_type in ["comparison", "compare", "operation_comparison", "all"]:
                    self._create_operation_comparison(performance_data, benchmark_id)
                
                if chart_type in ["trends", "trend", "block_size_trends", "all"]:
                    self._create_block_size_trends(performance_data, benchmark_id)
                    
            except Exception as e:
                logger.error(f"Error generating {chart_type} chart: {str(e)}")
                continue
        
        print(f"📈 Generated {len(self.generated_charts)} charts in {self.charts_dir}")
        return self.generated_charts
    
    def _extract_performance_data(self, benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured performance data from benchmark results."""
        
        performance_data = {
            'patterns': {},
            'summary': {
                'max_throughput': 0,
                'max_iops': 0,
                'min_latency': float('inf'),
                'total_tests': 0
            }
        }
        
        metrics = benchmark_data.get('metrics', {})
        
        # Process each tier's metrics
        for tier_name, tier_metrics in metrics.items():
            if not isinstance(tier_metrics, dict):
                continue
                
            # Process each test pattern
            for pattern_name, pattern_data in tier_metrics.items():
                if not isinstance(pattern_data, dict):
                    continue
                
                # Extract metrics
                throughput = pattern_data.get('throughput_MBps', 0)
                iops = pattern_data.get('iops', 0)
                latency = pattern_data.get('latency_ms', 0)
                
                if throughput > 0 or iops > 0:
                    # Parse pattern name
                    operation, block_size = self._parse_pattern_name(pattern_name)
                    
                    performance_data['patterns'][pattern_name] = {
                        'operation': operation,
                        'block_size': block_size,
                        'throughput_MBps': throughput,
                        'iops': iops,
                        'latency_ms': latency,
                        'tier': tier_name
                    }
                    
                    # Update summary
                    if throughput > performance_data['summary']['max_throughput']:
                        performance_data['summary']['max_throughput'] = throughput
                    if iops > performance_data['summary']['max_iops']:
                        performance_data['summary']['max_iops'] = iops
                    if latency > 0 and latency < performance_data['summary']['min_latency']:
                        performance_data['summary']['min_latency'] = latency
                    
                    performance_data['summary']['total_tests'] += 1
        
        return performance_data
    
    def _parse_pattern_name(self, pattern_name: str) -> tuple:
        """Parse pattern name to extract operation and block size."""
        parts = pattern_name.split('_')
        if len(parts) >= 2:
            operation = parts[0]
            block_size = '_'.join(parts[1:])
            return operation, block_size
        return pattern_name, 'unknown'
    
    def _create_comprehensive_dashboard(self, performance_data: Dict[str, Any], benchmark_id: str):
        """Create comprehensive performance dashboard."""
        
        patterns = performance_data['patterns']
        if not patterns:
            return
        
        # Organize data
        operations = {}
        block_sizes = set()
        
        for pattern_name, data in patterns.items():
            operation = data['operation']
            block_size = data['block_size']
            
            if operation not in operations:
                operations[operation] = {}
            operations[operation][block_size] = data
            block_sizes.add(block_size)
        
        # Sort block sizes
        block_size_order = {'4k': 1, '64k': 2, '1m': 3}
        block_sizes = sorted(list(block_sizes), key=lambda x: block_size_order.get(x, 999))
        
        # Create dashboard
        fig = plt.figure(figsize=(self.chart_width * 1.6, self.chart_height * 2))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # Colors for operations
        colors = {
            'read': '#2E86AB',
            'write': '#A23B72',
            'randrw': '#F18F01',
            'randread': '#C73E1D',
            'randwrite': '#8E44AD'
        }
        
        # 1. Throughput comparison
        ax1 = fig.add_subplot(gs[0, :2])
        self._create_metric_comparison(operations, block_sizes, ax1, 'throughput_MBps',
                                     'Throughput Performance by Operation and Block Size',
                                     'Throughput (MB/s)', colors)
        
        # 2. IOPS comparison
        ax2 = fig.add_subplot(gs[1, :2])
        self._create_metric_comparison(operations, block_sizes, ax2, 'iops',
                                     'IOPS Performance by Operation and Block Size',
                                     'IOPS', colors)
        
        # 3. Latency comparison
        ax3 = fig.add_subplot(gs[2, :2])
        self._create_metric_comparison(operations, block_sizes, ax3, 'latency_ms',
                                     'Latency by Operation and Block Size',
                                     'Latency (ms)', colors)
        
        # 4. Summary stats
        ax4 = fig.add_subplot(gs[0, 2])
        self._create_summary_panel(performance_data, ax4, benchmark_id)
        
        # 5. Operation totals
        ax5 = fig.add_subplot(gs[1, 2])
        self._create_operation_totals(operations, ax5, colors)
        
        # 6. Block size analysis
        ax6 = fig.add_subplot(gs[2, 2])
        self._create_block_size_averages(operations, block_sizes, ax6)
        
        # Title
        fig.suptitle(f'📊 Enhanced Performance Dashboard - {benchmark_id[:8]}',
                    fontsize=24, fontweight='bold', y=0.96)
        
        # Save
        chart_path = self.charts_dir / f"{benchmark_id}_enhanced_dashboard.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.generated_charts.append(chart_path.name)
        print(f"  ✅ Enhanced dashboard: {chart_path.name}")
    
    def _create_performance_heatmap(self, performance_data: Dict[str, Any], benchmark_id: str):
        """Create performance heatmap."""
        
        patterns = performance_data['patterns']
        if not patterns:
            return
        
        pattern_names = list(patterns.keys())
        metrics = ['throughput_MBps', 'iops', 'latency_ms']
        
        # Prepare data matrix
        data_matrix = []
        for metric in metrics:
            row = []
            for pattern_name in pattern_names:
                value = patterns[pattern_name].get(metric, 0)
                row.append(value)
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # Normalize for visualization
        normalized_matrix = np.zeros_like(data_matrix)
        for i, row in enumerate(data_matrix):
            if row.max() > 0:
                normalized_matrix[i] = row / row.max()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(self.chart_width, len(pattern_names) * 1.2), self.chart_height))
        
        im = ax.imshow(normalized_matrix, cmap='YlOrRd', aspect='auto')
        
        # Set labels
        ax.set_xticks(np.arange(len(pattern_names)))
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_xticklabels([name.replace('_', '\n') for name in pattern_names], 
                          rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(['Throughput (MB/s)', 'IOPS', 'Latency (ms)'], fontsize=12)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Normalized Performance (0=min, 1=max)', 
                          rotation=-90, va="bottom", fontsize=12)
        
        # Add value annotations
        for i in range(len(metrics)):
            for j in range(len(pattern_names)):
                value = data_matrix[i, j]
                if value > 0:
                    text_color = "white" if normalized_matrix[i, j] > 0.5 else "black"
                    if i == 0:  # Throughput
                        text = f'{value:.0f}'
                    elif i == 1:  # IOPS
                        text = f'{value:,.0f}'
                    else:  # Latency
                        text = f'{value:.2f}'
                    
                    ax.text(j, i, text, ha="center", va="center",
                           color=text_color, fontsize=9, fontweight='bold')
        
        ax.set_title(f'Performance Heatmap - {benchmark_id[:8]}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        chart_path = self.charts_dir / f"{benchmark_id}_enhanced_heatmap.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.generated_charts.append(chart_path.name)
        print(f"  ✅ Enhanced heatmap: {chart_path.name}")
    
    def _create_operation_comparison(self, performance_data: Dict[str, Any], benchmark_id: str):
        """Create operation comparison chart."""
        
        patterns = performance_data['patterns']
        if not patterns:
            return
        
        # Group by operation
        operations = {}
        for pattern_name, data in patterns.items():
            operation = data['operation']
            if operation not in operations:
                operations[operation] = []
            operations[operation].append(data)
        
        fig, axes = plt.subplots(1, 3, figsize=(self.chart_width * 1.5, self.chart_height * 0.75))
        fig.suptitle(f'Operation Performance Comparison - {benchmark_id[:8]}',
                    fontsize=16, fontweight='bold')
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E44AD']
        
        # Throughput comparison
        ax1 = axes[0]
        operation_names = list(operations.keys())
        throughputs = [max(data['throughput_MBps'] for data in operations[op]) 
                      for op in operation_names]
        
        bars = ax1.bar(operation_names, throughputs, color=colors[:len(operation_names)], alpha=0.8)
        ax1.set_title('Peak Throughput by Operation', fontweight='bold')
        ax1.set_ylabel('Throughput (MB/s)')
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, throughputs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # IOPS comparison
        ax2 = axes[1]
        iops_values = [max(data['iops'] for data in operations[op]) 
                      for op in operation_names]
        
        bars = ax2.bar(operation_names, iops_values, color=colors[:len(operation_names)], alpha=0.8)
        ax2.set_title('Peak IOPS by Operation', fontweight='bold')
        ax2.set_ylabel('IOPS')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, iops_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f'{value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Latency comparison
        ax3 = axes[2]
        latencies = [min(data['latency_ms'] for data in operations[op] if data['latency_ms'] > 0) 
                    for op in operation_names]
        
        bars = ax3.bar(operation_names, latencies, color=colors[:len(operation_names)], alpha=0.8)
        ax3.set_title('Minimum Latency by Operation', fontweight='bold')
        ax3.set_ylabel('Latency (ms)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, latencies):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        chart_path = self.charts_dir / f"{benchmark_id}_operation_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.generated_charts.append(chart_path.name)
        print(f"  ✅ Operation comparison: {chart_path.name}")
    
    def _create_block_size_trends(self, performance_data: Dict[str, Any], benchmark_id: str):
        """Create block size performance trends."""
        
        patterns = performance_data['patterns']
        if not patterns:
            return
        
        # Group by block size
        block_sizes_data = {}
        for pattern_name, data in patterns.items():
            block_size = data['block_size']
            if block_size not in block_sizes_data:
                block_sizes_data[block_size] = []
            block_sizes_data[block_size].append(data)
        
        # Sort block sizes
        block_size_order = {'4k': 1, '64k': 2, '1m': 3}
        sorted_sizes = sorted(block_sizes_data.keys(), key=lambda x: block_size_order.get(x, 999))
        
        fig, ax = plt.subplots(figsize=(self.chart_width, self.chart_height))
        
        # Calculate averages for each block size
        avg_throughputs = []
        avg_iops_list = []
        avg_latencies = []
        
        for size in sorted_sizes:
            data_list = block_sizes_data[size]
            avg_throughput = sum(d['throughput_MBps'] for d in data_list) / len(data_list)
            avg_iops_val = sum(d['iops'] for d in data_list) / len(data_list)
            valid_latencies = [d['latency_ms'] for d in data_list if d['latency_ms'] > 0]
            avg_latency = sum(valid_latencies) / len(valid_latencies) if valid_latencies else 0
            
            avg_throughputs.append(avg_throughput)
            avg_iops_list.append(avg_iops_val)
            avg_latencies.append(avg_latency)
        
        # Create dual-axis plot
        x = range(len(sorted_sizes))
        
        ax1 = ax
        color1 = '#2E86AB'
        ax1.set_xlabel('Block Size', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Throughput (MB/s)', color=color1, fontsize=12, fontweight='bold')
        line1 = ax1.plot(x, avg_throughputs, marker='o', color=color1, linewidth=3, markersize=8, label='Throughput')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(sorted_sizes)
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        color2 = '#A23B72'
        ax2.set_ylabel('IOPS', color=color2, fontsize=12, fontweight='bold')
        line2 = ax2.plot(x, avg_iops_list, marker='s', color=color2, linewidth=3, markersize=8, label='IOPS')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Add value annotations
        for i, (tp, iops) in enumerate(zip(avg_throughputs, avg_iops_list)):
            ax1.annotate(f'{tp:.0f}', (i, tp), textcoords="offset points",
                        xytext=(0,10), ha='center', fontweight='bold', color=color1)
            ax2.annotate(f'{iops:,.0f}', (i, iops), textcoords="offset points",
                        xytext=(0,-15), ha='center', fontweight='bold', color=color2)
        
        ax1.set_title(f'Performance by Block Size - {benchmark_id[:8]}',
                     fontsize=16, fontweight='bold')
        
        # Legend - combine lines properly
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        
        # Save
        chart_path = self.charts_dir / f"{benchmark_id}_block_size_trends.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.generated_charts.append(chart_path.name)
        print(f"  ✅ Block size trends: {chart_path.name}")
    
    def _create_metric_comparison(self, operations, block_sizes, ax, metric_key, title, ylabel, colors):
        """Create metric comparison chart."""
        
        if not operations or not block_sizes:
            ax.text(0.5, 0.5, f'No data available for {metric_key}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontweight='bold')
            return
        
        x = np.arange(len(block_sizes))
        width = 0.25
        operation_names = list(operations.keys())
        
        for i, operation in enumerate(operation_names):
            values = []
            for block_size in block_sizes:
                value = operations[operation].get(block_size, {}).get(metric_key, 0)
                values.append(value)
            
            color = colors.get(operation, f'C{i}')
            bars = ax.bar(x + i * width, values, width,
                         label=operation.capitalize(), color=color, alpha=0.8)
            
            # Add value labels
            for bar, value in zip(bars, values):
                if value > 0:
                    if metric_key == 'throughput_MBps':
                        label = f'{value:.0f}'
                    elif metric_key == 'iops':
                        label = f'{value:,.0f}' if value >= 1000 else f'{value:.0f}'
                    elif metric_key == 'latency_ms':
                        label = f'{value:.2f}' if value < 10 else f'{value:.1f}'
                    else:
                        label = f'{value:.1f}'
                    
                    ax.text(bar.get_x() + bar.get_width()/2,
                           bar.get_height() * 1.02,
                           label, ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Block Size', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(operation_names) - 1) / 2)
        ax.set_xticklabels(block_sizes, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _create_summary_panel(self, performance_data, ax, benchmark_id):
        """Create summary statistics panel."""
        
        summary = performance_data['summary']
        
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Performance Summary', fontsize=16, fontweight='bold',
               ha='center', transform=ax.transAxes)
        
        # Benchmark ID
        ax.text(0.05, 0.85, f'ID: {benchmark_id[:12]}...', fontsize=11,
               transform=ax.transAxes, family='monospace')
        
        # Test count
        ax.text(0.05, 0.78, f'Total Tests: {summary["total_tests"]}', fontsize=11,
               transform=ax.transAxes)
        
        # Peak metrics
        ax.text(0.05, 0.65, 'Peak Performance:', fontsize=13, fontweight='bold',
               color='darkblue', transform=ax.transAxes)
        
        ax.text(0.05, 0.55, '• Max Throughput:', fontsize=11, transform=ax.transAxes)
        ax.text(0.1, 0.48, f'{summary["max_throughput"]:.1f} MB/s', fontsize=12,
               color='steelblue', fontweight='bold', transform=ax.transAxes)
        
        ax.text(0.05, 0.38, '• Max IOPS:', fontsize=11, transform=ax.transAxes)
        ax.text(0.1, 0.31, f'{summary["max_iops"]:,.0f}', fontsize=12,
               color='orange', fontweight='bold', transform=ax.transAxes)
        
        if summary["min_latency"] < float('inf'):
            ax.text(0.05, 0.21, '• Min Latency:', fontsize=11, transform=ax.transAxes)
            ax.text(0.1, 0.14, f'{summary["min_latency"]:.2f} ms', fontsize=12,
                   color='red', fontweight='bold', transform=ax.transAxes)
        
        # Add border
        rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, linewidth=2,
                            edgecolor='lightgray', facecolor='none', transform=ax.transAxes)
        ax.add_patch(rect)
    
    def _create_operation_totals(self, operations, ax, colors):
        """Create operation performance totals."""
        
        operation_names = list(operations.keys())
        total_throughputs = []
        
        for operation in operation_names:
            total = sum(data.get('throughput_MBps', 0)
                       for data in operations[operation].values())
            total_throughputs.append(total)
        
        if not total_throughputs or all(t == 0 for t in total_throughputs):
            ax.text(0.5, 0.5, 'No throughput data\navailable', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Total Throughput by Operation', fontweight='bold')
            return
        
        bars = ax.bar(operation_names, total_throughputs,
                     color=[colors.get(op, 'gray') for op in operation_names], alpha=0.8)
        
        for bar, value in zip(bars, total_throughputs):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                       f'{value:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title('Total Throughput by Operation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Throughput (MB/s)', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        if len(operation_names) > 3:
            ax.tick_params(axis='x', rotation=45)
    
    def _create_block_size_averages(self, operations, block_sizes, ax):
        """Create block size average performance analysis."""
        
        avg_throughputs = []
        
        for block_size in block_sizes:
            total_throughput = 0
            count = 0
            
            for operation, data in operations.items():
                if block_size in data:
                    total_throughput += data[block_size].get('throughput_MBps', 0)
                    count += 1
            
            avg_throughput = total_throughput / count if count > 0 else 0
            avg_throughputs.append(avg_throughput)
        
        if not avg_throughputs or all(t == 0 for t in avg_throughputs):
            ax.text(0.5, 0.5, 'No block size data\navailable', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Avg Performance by Block Size', fontweight='bold')
            return
        
        bars = ax.bar(block_sizes, avg_throughputs, color='purple', alpha=0.7)
        
        for bar, value in zip(bars, avg_throughputs):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                       f'{value:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title('Avg Performance by Block Size', fontsize=12, fontweight='bold')
        ax.set_xlabel('Block Size', fontsize=10)
        ax.set_ylabel('Avg Throughput (MB/s)', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    def create_enhanced_html_report(self, benchmark_data: Dict[str, Any], 
                                  benchmark_id: str, config: Dict[str, Any]) -> str:
        """Create enhanced HTML report with embedded charts."""
        
        run_id = benchmark_data.get('run_id', benchmark_id)
        start_time = benchmark_data.get('start_time', 'Unknown')
        end_time = benchmark_data.get('end_time', 'Unknown')
        duration = benchmark_data.get('duration_seconds', 0)
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{config.get('title', 'Enhanced Benchmark Report')}</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f7;
                    color: #1d1d1f;
                }}
                
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    background: white;
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }}
                
                .header h1 {{
                    font-size: 2.5em;
                    font-weight: 700;
                    margin: 0;
                    color: #1d1d1f;
                }}
                
                .benchmark-info {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                
                .info-card {{
                    background: #f0f0f2;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                }}
                
                .info-card h3 {{
                    margin: 0 0 10px 0;
                    color: #6e6e73;
                    font-size: 0.9em;
                }}
                
                .info-card .value {{
                    font-size: 1.2em;
                    font-weight: 600;
                    color: #1d1d1f;
                    font-family: monospace;
                }}
                
                .charts-section {{
                    background: white;
                    border-radius: 12px;
                    padding: 30px;
                    margin: 20px 0;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }}
                
                .charts-section h2 {{
                    margin: 0 0 30px 0;
                    font-size: 1.8em;
                    font-weight: 600;
                    color: #1d1d1f;
                    text-align: center;
                }}
                
                .chart-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                    gap: 30px;
                }}
                
                .chart-item {{
                    text-align: center;
                }}
                
                .chart-item h3 {{
                    margin: 0 0 15px 0;
                    font-size: 1.3em;
                    font-weight: 600;
                    color: #1d1d1f;
                }}
                
                .chart-item img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                }}
                
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding: 20px;
                    color: #6e6e73;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 {config.get('title', 'Enhanced Benchmark Report')}</h1>
                
                <div class="benchmark-info">
                    <div class="info-card">
                        <h3>Benchmark ID</h3>
                        <div class="value">{run_id[:12]}...</div>
                    </div>
                    <div class="info-card">
                        <h3>Start Time</h3>
                        <div class="value">{start_time}</div>
                    </div>
                    <div class="info-card">
                        <h3>Duration</h3>
                        <div class="value">{duration:.1f}s</div>
                    </div>
                    <div class="info-card">
                        <h3>Generated</h3>
                        <div class="value">{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
                    </div>
                </div>
            </div>
            
            <div class="charts-section">
                <h2>📈 Performance Visualizations</h2>
                
                <div class="chart-grid">
        """
        
        # Add chart images if they exist
        for chart_name in self.generated_charts:
            chart_title = chart_name.replace(f"{benchmark_id}_", "").replace(".png", "").replace("_", " ").title()
            html += f"""
                    <div class="chart-item">
                        <h3>{chart_title}</h3>
                        <img src="charts/{chart_name}" alt="{chart_title}" loading="lazy">
                    </div>
            """
        
        html += """
                </div>
            </div>
            
            <div class="footer">
                <p>Generated by eTIOBench Enhanced Report Generator</p>
                <p>📁 Chart files available in the charts/ directory</p>
            </div>
        </body>
        </html>
        """
        
        return html
