#!/usr/bin/env python3
"""
Benchmark Analysis Result Structures (Tiered Storage I/O Benchmark)

This module provides standardized data structures for analysis results,
ensuring consistency across different analysis modules.

Author: Jack Ogaja
Date: 2025-06-26
"""

import json
import datetime
from typing import Dict, List, Any, Optional, Union

class AnalysisResult:
    """
    Standard container for analysis results.
    
    Provides a consistent structure for all analysis results,
    regardless of the specific analyzer that produced them.
    """
    
    def __init__(
        self,
        analysis_type: str,
        benchmark_id: str,
        timestamp: Optional[str] = None
    ):
        """
        Initialize analysis result.
        
        Args:
            analysis_type: Type of analysis (e.g., "statistics", "network", "time_series")
            benchmark_id: Benchmark identifier
            timestamp: Analysis timestamp (ISO format, default: current time)
        """
        self.analysis_type = analysis_type
        self.benchmark_id = benchmark_id
        self.timestamp = timestamp or datetime.datetime.utcnow().isoformat()
        self.tier_results = {}
        self.overall_results = {}
        self.recommendations = []
        self.metadata = {}
        self.severity = "info"  # Severity of findings: info, low, medium, high, critical
    
    def add_tier_result(self, tier: str, result: Dict[str, Any]) -> None:
        """
        Add analysis result for a specific tier.
        
        Args:
            tier: Storage tier path
            result: Analysis result for this tier
        """
        self.tier_results[tier] = result
    
    def add_overall_result(self, key: str, result: Any) -> None:
        """
        Add overall analysis result.
        
        Args:
            key: Result key
            result: Result value
        """
        self.overall_results[key] = result
    
    def add_recommendation(self, recommendation: str) -> None:
        """
        Add a recommendation based on the analysis.
        
        Args:
            recommendation: Recommendation text
        """
        if recommendation not in self.recommendations:
            self.recommendations.append(recommendation)
    
    def set_severity(self, severity: str) -> None:
        """
        Set severity level of analysis findings.
        
        Args:
            severity: Severity level (info, low, medium, high, critical)
        """
        valid_severities = ["info", "low", "medium", "high", "critical"]
        if severity not in valid_severities:
            raise ValueError(f"Invalid severity: {severity}. Must be one of {valid_severities}")
        
        self.severity = severity
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata value.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def get_tier_result(self, tier: str) -> Optional[Dict[str, Any]]:
        """
        Get analysis result for a specific tier.
        
        Args:
            tier: Storage tier path
            
        Returns:
            Analysis result for this tier, or None if not available
        """
        return self.tier_results.get(tier)
    
    def get_overall_result(self, key: str, default: Any = None) -> Any:
        """
        Get overall analysis result.
        
        Args:
            key: Result key
            default: Default value if key not found
            
        Returns:
            Result value, or default if not found
        """
        return self.overall_results.get(key, default)
    
    def has_findings(self) -> bool:
        """
        Check if analysis has any findings.
        
        Returns:
            True if analysis has any findings, False otherwise
        """
        return (
            bool(self.tier_results) or
            bool(self.overall_results) or
            bool(self.recommendations)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation of analysis result
        """
        return {
            "analysis_type": self.analysis_type,
            "benchmark_id": self.benchmark_id,
            "timestamp": self.timestamp,
            "tier_results": self.tier_results,
            "overall_results": self.overall_results,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
            "severity": self.severity
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert to JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON string representation of analysis result
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """
        Create analysis result from dictionary.
        
        Args:
            data: Dictionary representation of analysis result
            
        Returns:
            AnalysisResult instance
        """
        result = cls(
            analysis_type=data["analysis_type"],
            benchmark_id=data["benchmark_id"],
            timestamp=data["timestamp"]
        )
        
        # Set data from dictionary
        result.tier_results = data.get("tier_results", {})
        result.overall_results = data.get("overall_results", {})
        result.recommendations = data.get("recommendations", [])
        result.metadata = data.get("metadata", {})
        result.severity = data.get("severity", "info")
        
        return result
