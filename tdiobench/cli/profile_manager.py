"""
ProfileManager - Benchmark Profile Management for eTIOBench
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ProfileManager:
    """Manages benchmark profiles for different testing scenarios"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize ProfileManager with built-in and custom profiles"""
        self.config_path = config_path or Path.home() / '.etiobench' / 'profiles.json'
        self.profiles = self._load_default_profiles()
        self.custom_profiles = {}
        self._load_custom_profiles()
    
    def _load_default_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load built-in benchmark profiles"""
        return {
            'quick_scan': {
                'name': 'quick_scan',
                'description': 'Fast overview of performance characteristics',
                'duration': 30,
                'ramp_time': 5,
                'block_sizes': ['4k', '1m'],
                'patterns': ['read', 'write'],
                'io_depth': 32,
                'num_jobs': 1,
                'direct': True
            },
            'production_safe': {
                'name': 'production_safe',
                'description': 'Minimal impact benchmark for production hours',
                'duration': 60,
                'ramp_time': 10,
                'block_sizes': ['4k', '1m'],
                'patterns': ['read', 'randread'],
                'io_depth': 8,
                'num_jobs': 1,
                'rate_limit': '50m',
                'direct': True,
                'safety_override': {
                    'max_cpu_percent': 50,
                    'max_io_percent': 40
                }
            },
            'comprehensive': {
                'name': 'comprehensive',
                'description': 'Thorough performance characterization',
                'duration': 300,
                'ramp_time': 15,
                'block_sizes': ['512', '4k', '8k', '16k', '32k', '64k', '128k', '256k', '512k', '1m', '2m', '4m'],
                'patterns': ['read', 'write', 'randread', 'randwrite', 'readwrite', 'randrw'],
                'io_depth': 64,
                'num_jobs': 4,
                'direct': True,
                'mixed_workload': {
                    'read_percentage': 70,
                    'sequential_percentage': 80
                }
            },
            'latency_focused': {
                'name': 'latency_focused',
                'description': 'Detailed latency analysis',
                'duration': 120,
                'ramp_time': 10,
                'block_sizes': ['512', '4k', '8k'],
                'patterns': ['randread', 'randwrite'],
                'io_depth': 1,
                'num_jobs': 1,
                'direct': True,
                'latency_targets': [100, 500, 1000, 5000],
                'percentiles': [50.0, 90.0, 95.0, 99.0, 99.9, 99.99]
            }
        }
    
    def _load_custom_profiles(self) -> None:
        """Load custom profiles from configuration file"""
        config_file = Path(self.config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    self.custom_profiles = data.get('custom_profiles', {})
            except Exception as e:
                logger.warning(f"Failed to load custom profiles: {e}")
    
    def _save_custom_profiles(self) -> None:
        """Save custom profiles to configuration file"""
        config_file = Path(self.config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump({
                'custom_profiles': self.custom_profiles,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def get_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a profile configuration by name
        
        Args:
            name: Profile name
            
        Returns:
            Profile configuration dict or None if not found
        """
        # Check built-in profiles first
        if name in self.profiles:
            return self.profiles[name].copy()
        
        # Check custom profiles
        if name in self.custom_profiles:
            return self.custom_profiles[name].copy()
        
        return None
    
    def create_custom_profile(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a custom profile from provided configuration
        
        Args:
            config: Profile configuration from CLI options
            
        Returns:
            Validated profile configuration
        """
        # Start with defaults
        profile = {
            'name': 'custom',
            'description': 'Custom benchmark profile',
            'duration': 60,
            'ramp_time': 5,
            'block_sizes': ['4k', '1m'],
            'patterns': ['read', 'write'],
            'io_depth': 32,
            'num_jobs': 1,
            'direct': True
        }
        
        # Override with provided config
        profile.update(config)
        
        # Ensure lists are properly formatted
        if isinstance(profile.get('block_sizes'), str):
            profile['block_sizes'] = [s.strip() for s in profile['block_sizes'].split(',')]
        if isinstance(profile.get('patterns'), str):
            profile['patterns'] = [p.strip() for p in profile['patterns'].split(',')]
        
        # Validate the profile
        issues = self.validate_profile(profile)
        if issues:
            logger.warning(f"Custom profile has issues: {issues}")
        
        return profile
    
    def list_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available profiles (built-in and custom)
        
        Returns:
            Dictionary of all profiles
        """
        all_profiles = {}
        
        # Add built-in profiles
        all_profiles.update(self.profiles)
        
        # Add custom profiles
        for name, config in self.custom_profiles.items():
            all_profiles[name] = config
        
        return all_profiles
    
    def validate_profile(self, profile: Dict[str, Any]) -> List[str]:
        """
        Validate a profile configuration
        
        Args:
            profile: Profile configuration to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Required fields
        required = ['duration', 'block_sizes', 'patterns']
        for field in required:
            if field not in profile:
                issues.append(f"Missing required field: {field}")
        
        # Duration validation
        if 'duration' in profile:
            if not isinstance(profile['duration'], (int, float)) or profile['duration'] <= 0:
                issues.append("Duration must be a positive number")
            elif profile['duration'] > 86400:  # 24 hours
                issues.append("Duration exceeds maximum (24 hours)")
        
        # Block sizes validation
        if 'block_sizes' in profile:
            if not profile['block_sizes']:
                issues.append("At least one block size required")
            else:
                valid_sizes = self._validate_block_sizes(profile['block_sizes'])
                if not valid_sizes:
                    issues.append("Invalid block sizes format")
        
        # Pattern validation
        if 'patterns' in profile:
            valid_patterns = {'read', 'write', 'randread', 'randwrite', 'readwrite', 'randrw'}
            invalid = [p for p in profile['patterns'] if p not in valid_patterns]
            if invalid:
                issues.append(f"Invalid patterns: {invalid}")
        
        # IO depth validation
        if 'io_depth' in profile:
            if not isinstance(profile['io_depth'], int) or profile['io_depth'] < 1:
                issues.append("IO depth must be a positive integer")
            elif profile['io_depth'] > 256:
                issues.append("IO depth exceeds maximum (256)")
        
        # Number of jobs validation
        if 'num_jobs' in profile:
            if not isinstance(profile['num_jobs'], int) or profile['num_jobs'] < 1:
                issues.append("Number of jobs must be a positive integer")
            elif profile['num_jobs'] > 64:
                issues.append("Number of jobs exceeds maximum (64)")
        
        # Rate limit validation
        if 'rate_limit' in profile:
            if not self._validate_rate_limit(profile['rate_limit']):
                issues.append("Invalid rate limit format (use e.g., '50m' for 50MB/s)")
        
        return issues
    
    def save_custom_profile(self, name: str, profile: Dict[str, Any]) -> None:
        """Save a custom profile for future use"""
        if name in self.profiles:
            raise ValueError(f"Cannot override built-in profile '{name}'")
        
        # Validate before saving
        issues = self.validate_profile(profile)
        if issues:
            raise ValueError(f"Profile validation failed: {', '.join(issues)}")
        
        profile['name'] = name
        profile['created_date'] = datetime.now().isoformat()
        
        self.custom_profiles[name] = profile
        self._save_custom_profiles()
        logger.info(f"Saved custom profile '{name}'")
    
    def delete_custom_profile(self, name: str) -> None:
        """Delete a custom profile"""
        if name in self.profiles:
            raise ValueError(f"Cannot delete built-in profile '{name}'")
        
        if name not in self.custom_profiles:
            raise KeyError(f"Custom profile '{name}' not found")
        
        del self.custom_profiles[name]
        self._save_custom_profiles()
        logger.info(f"Deleted custom profile '{name}'")
    
    def _validate_block_sizes(self, block_sizes: List[str]) -> bool:
        """Validate block size format"""
        import re
        pattern = re.compile(r'^\d+[kmgKMG]?$')
        return all(pattern.match(size) for size in block_sizes)
    
    def _validate_rate_limit(self, rate_limit: str) -> bool:
        """Validate rate limit format"""
        import re
        pattern = re.compile(r'^\d+[kmgKMG]?$')
        return bool(pattern.match(rate_limit))
