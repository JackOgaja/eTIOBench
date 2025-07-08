#!/usr/bin/env python3
"""
Result Aggregator (Tiered Storage I/O Benchmark)

This module provides simpple functionality for consolidating benchmark results mainly for CLI,

Author: Jack Ogaja
Date: 2025-06-29
"""

import json
import statistics
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

class ResultsAggregator:
    """Consolidates and formats benchmark results"""
    
    def __init__(self):
        """Initialize ResultsAggregator"""
        self.formatters = {
            'cli': self._format_cli,
            'json': self._format_json,
            'markdown': self._format_markdown,
            'html': self._format_html,
            'csv': self._format_csv
        }
    
    def aggregate_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate metrics from benchmark results
        
        Args:
            results: Raw benchmark results
            
        Returns:
            Aggregated metrics with summary statistics
        """
        aggregated = {
            'seq_read_mb': 0,
            'seq_write_mb': 0,
            'rand_read_iops': 0,
            'rand_write_iops': 0,
            'avg_read_latency_us': 0,
            'avg_write_latency_us': 0,
            'tiers': {}
        }
        
        tier_results = results.get('results', {})
        
        for tier, tests in tier_results.items():
            tier_metrics = {
                'read_bw': [],
                'write_bw': [],
                'read_iops': [],
                'write_iops': [],
                'read_lat': [],
                'write_lat': []
            }
            
            for test_name, metrics in tests.items():
                # Sequential bandwidth
                if 'read_' in test_name and '4k' not in test_name:
                    tier_metrics['read_bw'].append(metrics.get('bw_read_kb', 0) / 1024)
                if 'write_' in test_name and '4k' not in test_name:
                    tier_metrics['write_bw'].append(metrics.get('bw_write_kb', 0) / 1024)
                
                # Random IOPS (4k tests)
                if 'rand' in test_name and '4k' in test_name:
                    tier_metrics['read_iops'].append(metrics.get('iops_read', 0))
                    tier_metrics['write_iops'].append(metrics.get('iops_write', 0))
                
                # Latencies
                tier_metrics['read_lat'].append(metrics.get('lat_read_ns', 0) / 1000)
                tier_metrics['write_lat'].append(metrics.get('lat_write_ns', 0) / 1000)
            
            # Calculate tier aggregates
            tier_summary = {
                'seq_read_mb': statistics.mean(tier_metrics['read_bw']) if tier_metrics['read_bw'] else 0,
                'seq_write_mb': statistics.mean(tier_metrics['write_bw']) if tier_metrics['write_bw'] else 0,
                'rand_read_iops': statistics.mean(tier_metrics['read_iops']) if tier_metrics['read_iops'] else 0,
                'rand_write_iops': statistics.mean(tier_metrics['write_iops']) if tier_metrics['write_iops'] else 0,
                'avg_read_latency_us': statistics.mean(tier_metrics['read_lat']) if tier_metrics['read_lat'] else 0,
                'avg_write_latency_us': statistics.mean(tier_metrics['write_lat']) if tier_metrics['write_lat'] else 0
            }
            
            aggregated['tiers'][tier] = tier_summary
            
            # Update overall aggregates
            aggregated['seq_read_mb'] = max(aggregated['seq_read_mb'], tier_summary['seq_read_mb'])
            aggregated['seq_write_mb'] = max(aggregated['seq_write_mb'], tier_summary['seq_write_mb'])
            aggregated['rand_read_iops'] = max(aggregated['rand_read_iops'], tier_summary['rand_read_iops'])
            aggregated['rand_write_iops'] = max(aggregated['rand_write_iops'], tier_summary['rand_write_iops'])
        
        return aggregated
    
    def calculate_statistics(self, data: List[float]) -> Dict[str, float]:
        """Calculate statistical metrics for a dataset"""
        if not data:
            return {'min': 0, 'max': 0, 'avg': 0, 'median': 0, 'stddev': 0}
        
        return {
            'min': min(data),
            'max': max(data),
            'avg': statistics.mean(data),
            'median': statistics.median(data),
            'stddev': statistics.stdev(data) if len(data) > 1 else 0,
            'p95': self._percentile(data, 95),
            'p99': self._percentile(data, 99)
        }
    
    def format_results(self, results: Union[Dict, List], format: str = 'cli') -> str:
        """
        Format results for display
        
        Args:
            results: Benchmark or comparison results
            format: Output format (cli, json, markdown, html, csv)
            
        Returns:
            Formatted results string
        """
        if format not in self.formatters:
            raise ValueError(f"Unknown format: {format}")
        
        return self.formatters[format](results)
    
    def _format_cli(self, results: Dict[str, Any]) -> str:
        """Format results for CLI display"""
        output = []
        output.append("\n" + "="*60)
        output.append("BENCHMARK RESULTS SUMMARY")
        output.append("="*60)
        
        if 'results' in results:
            # Standard benchmark results
            for tier, tier_results in results['results'].items():
                output.append(f"\n{tier}:")
                
                # Aggregate metrics
                total_iops_read = 0
                total_iops_write = 0
                total_bw_read = 0
                total_bw_write = 0
                count = 0
                
                for test, metrics in tier_results.items():
                    total_iops_read += metrics.get('iops_read', 0)
                    total_iops_write += metrics.get('iops_write', 0)
                    total_bw_read += metrics.get('bw_read_kb', 0)
                    total_bw_write += metrics.get('bw_write_kb', 0)
                    count += 1
                
                if count > 0:
                    output.append(f"  Average IOPS - Read: {total_iops_read/count:,.0f}, "
                                f"Write: {total_iops_write/count:,.0f}")
                    output.append(f"  Average BW - Read: {total_bw_read/count/1024:.1f} MB/s, "
                                f"Write: {total_bw_write/count/1024:.1f} MB/s")
        
        elif 'benchmarks' in results:
            # Comparison results
            output.append("\nCOMPARISON RESULTS")
            for metric in results.get('metrics', []):
                output.append(f"\n{metric.upper()}:")
                if metric in results:
                    for tier, values in results[metric].items():
                        output.append(f"  {tier}:")
                        for bid, value in values.items():
                            output.append(f"    {bid}: {value:,.1f}")
        
        output.append("\n" + "="*60)
        return '\n'.join(output)
    
    def _format_json(self, results: Dict[str, Any]) -> str:
        """Format results as JSON"""
        return json.dumps(results, indent=2, default=str)
    
    def _format_markdown(self, results: Dict[str, Any]) -> str:
        """Format results as Markdown"""
        lines = []
        
        if 'results' in results:
            # Standard benchmark results
            lines.append("# Benchmark Results\n")
            lines.append(f"**Timestamp:** {results.get('timestamp', 'Unknown')}\n")
            lines.append(f"**Duration:** {results.get('duration', 0)}s\n")
            
            lines.append("## Results by Tier\n")
            
            for tier, tier_results in results.get('results', {}).items():
                lines.append(f"### {tier}\n")
                lines.append("| Test | Read IOPS | Write IOPS | Read BW (MB/s) | Write BW (MB/s) |")
                lines.append("|------|-----------|------------|----------------|-----------------|")
                
                for test, metrics in tier_results.items():
                    lines.append(f"| {test} | "
                               f"{metrics.get('iops_read', 0):,.0f} | "
                               f"{metrics.get('iops_write', 0):,.0f} | "
                               f"{metrics.get('bw_read_kb', 0)/1024:.1f} | "
                               f"{metrics.get('bw_write_kb', 0)/1024:.1f} |")
                lines.append("")
        
        elif 'benchmarks' in results:
            # Comparison results
            lines.append("# Benchmark Comparison\n")
            
            for metric in results.get('metrics', []):
                lines.append(f"## {metric.upper()}\n")
                
                if metric in results and results[metric]:
                    # Get all benchmark IDs
                    all_bids = set()
                    for tier_values in results[metric].values():
                        all_bids.update(tier_values.keys())
                    bid_list = sorted(all_bids)
                    
                    # Table header
                    lines.append("| Tier | " + " | ".join(bid_list) + " |")
                    lines.append("|------|" + "------|" * len(bid_list))
                    
                    # Table rows
                    for tier, values in results[metric].items():
                        row = f"| {tier} |"
                        for bid in bid_list:
                            value = values.get(bid, 'N/A')
                            if isinstance(value, (int, float)):
                                row += f" {value:,.1f} |"
                            else:
                                row += f" {value} |"
                        lines.append(row)
                    lines.append("")
        
        return '\n'.join(lines)
    
    def _format_html(self, results: Dict[str, Any]) -> str:
        """Format results as HTML"""
        html = ['<html><head><style>',
                'table { border-collapse: collapse; margin: 20px 0; }',
                'th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }',
                'th { background-color: #f2f2f2; }',
                '</style></head><body>']
        
        if 'results' in results:
            html.append('<h1>Benchmark Results</h1>')
            html.append(f'<p>Timestamp: {results.get("timestamp", "Unknown")}</p>')
            
            for tier, tier_results in results.get('results', {}).items():
                html.append(f'<h2>{tier}</h2>')
                html.append('<table>')
                html.append('<tr><th>Test</th><th>Read IOPS</th><th>Write IOPS</th>'
                          '<th>Read BW (MB/s)</th><th>Write BW (MB/s)</th></tr>')
                
                for test, metrics in tier_results.items():
                    html.append(f'<tr><td>{test}</td>'
                              f'<td>{metrics.get("iops_read", 0):,.0f}</td>'
                              f'<td>{metrics.get("iops_write", 0):,.0f}</td>'
                              f'<td>{metrics.get("bw_read_kb", 0)/1024:.1f}</td>'
                              f'<td>{metrics.get("bw_write_kb", 0)/1024:.1f}</td></tr>')
                
                html.append('</table>')
        
        html.append('</body></html>')
        return '\n'.join(html)
    
    def _format_csv(self, results: Dict[str, Any]) -> str:
        """Format results as CSV"""
        lines = []
        
        if 'results' in results:
            lines.append('Tier,Test,Read_IOPS,Write_IOPS,Read_BW_MB,Write_BW_MB')
            
            for tier, tier_results in results.get('results', {}).items():
                for test, metrics in tier_results.items():
                    lines.append(f'{tier},{test},'
                               f'{metrics.get("iops_read", 0):.0f},'
                               f'{metrics.get("iops_write", 0):.0f},'
                               f'{metrics.get("bw_read_kb", 0)/1024:.1f},'
                               f'{metrics.get("bw_write_kb", 0)/1024:.1f}')
        
        return '\n'.join(lines)
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
