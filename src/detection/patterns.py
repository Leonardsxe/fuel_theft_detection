"""
Detection pattern definitions and validation.
Defines the criteria for different fuel theft patterns.
"""

from dataclasses import dataclass
from typing import Optional, Dict
import logging

from src.config.settings import PatternConfig

logger = logging.getLogger(__name__)


@dataclass
class DetectionPattern:
    """
    Single detection pattern with validation criteria.
    """
    name: str
    min_duration_min: float
    max_duration_min: Optional[float]
    min_cumulative_gal: float
    min_negative_steps: Optional[int] = None
    min_points: Optional[int] = None
    min_median_dt_s: Optional[float] = None
    description: str = ""
    
    def validate_duration(self, duration_min: float) -> bool:
        """Check if duration meets pattern criteria"""
        if duration_min < self.min_duration_min:
            return False
        
        if self.max_duration_min is not None and duration_min > self.max_duration_min:
            return False
        
        return True
    
    def validate_cumulative(self, cumulative_drop: float, threshold_cum: float) -> bool:
        """Check if cumulative drop meets criteria"""
        return cumulative_drop >= max(threshold_cum, self.min_cumulative_gal)
    
    def validate_step(self, max_step: float, threshold_step: float) -> bool:
        """Check if single step meets criteria"""
        return max_step >= threshold_step
    
    def validate_points(self, n_points: int) -> bool:
        """Check if number of points meets criteria"""
        if self.min_points is None:
            return True
        return n_points >= self.min_points
    
    def validate_median_dt(self, median_dt_s: float) -> bool:
        """Check if median time delta meets criteria"""
        if self.min_median_dt_s is None:
            return True
        return median_dt_s >= self.min_median_dt_s


class PatternManager:
    """
    Manage detection patterns and their validation rules.
    """
    
    def __init__(self, patterns: Dict[str, PatternConfig]):
        """
        Initialize with pattern configurations.
        
        Args:
            patterns: Dictionary of pattern configurations from config
        """
        self.patterns = self._create_patterns(patterns)
    
    def _create_patterns(self, configs: Dict[str, PatternConfig]) -> Dict[str, DetectionPattern]:
        """
        Create DetectionPattern objects from configurations.
        
        Args:
            configs: Pattern configurations
        
        Returns:
            Dictionary of DetectionPattern objects
        """
        patterns = {}
        
        for key, cfg in configs.items():
            pattern = DetectionPattern(
                name=cfg.name,
                min_duration_min=cfg.min_duration_min,
                max_duration_min=cfg.max_duration_min,
                min_cumulative_gal=cfg.min_cumulative_gal or 0.0,
                min_negative_steps=cfg.min_negative_steps,
                min_points=cfg.min_points,
                min_median_dt_s=cfg.min_median_dt_s,
                description=cfg.description
            )
            patterns[key] = pattern
        
        logger.info(f"Loaded {len(patterns)} detection patterns:")
        for key, pattern in patterns.items():
            logger.info(f"  - {pattern.name}: {pattern.description}")
        
        return patterns
    
    def get_pattern(self, key: str) -> Optional[DetectionPattern]:
        """Get pattern by key"""
        return self.patterns.get(key)
    
    def get_patterns_for_state(self, state: str) -> Dict[str, DetectionPattern]:
        """
        Get all patterns applicable to a given state.
        
        Args:
            state: State name ("stationary_on" or "ign_off")
        
        Returns:
            Dictionary of applicable patterns
        """
        if state == "stationary_on":
            return {
                k: v for k, v in self.patterns.items()
                if k in ["short_drain", "extended_drain"]
            }
        elif state == "ign_off":
            return {
                k: v for k, v in self.patterns.items()
                if k == "postjourney_off"
            }
        else:
            return {}


def create_default_patterns() -> Dict[str, DetectionPattern]:
    """
    Create default detection patterns if configuration not available.
    
    Returns:
        Dictionary of default DetectionPattern objects
    """
    return {
        "short_drain": DetectionPattern(
            name="short_4_10m_3gal",
            min_duration_min=4.0,
            max_duration_min=10.0,
            min_cumulative_gal=3.0,
            min_negative_steps=2,
            description="Quick drains during stationary periods (4-10 min, ≥3 gal)"
        ),
        "extended_drain": DetectionPattern(
            name="extended_15m_6gal",
            min_duration_min=15.0,
            max_duration_min=None,
            min_cumulative_gal=6.0,
            description="Prolonged drains during stationary periods (≥15 min, ≥6 gal)"
        ),
        "postjourney_off": DetectionPattern(
            name="postjourney_off",
            min_duration_min=45.0,
            max_duration_min=None,
            min_cumulative_gal=6.0,
            min_points=2,
            min_median_dt_s=1800.0,
            description="Post-journey drains with ignition off (≥45 min, ≥6 gal)"
        )
    }


def validate_event_pattern(
    event: Dict,
    pattern: DetectionPattern,
    threshold_step: float,
    threshold_cum: float
) -> bool:
    """
    Validate if an event matches a pattern's criteria.
    
    Args:
        event: Event dictionary with duration, drop, etc.
        pattern: Pattern to validate against
        threshold_step: Step threshold
        threshold_cum: Cumulative threshold
    
    Returns:
        True if event matches pattern
    """
    # Duration check
    if not pattern.validate_duration(event.get("duration_min", 0)):
        return False
    
    # Cumulative drop check
    if not pattern.validate_cumulative(
        event.get("cum_drop", 0),
        threshold_cum
    ):
        return False
    
    # Step check (optional - can be OR with cumulative)
    # This is pattern-specific logic
    
    # Points check
    if not pattern.validate_points(event.get("n_points", 0)):
        return False
    
    # Median dt check
    if not pattern.validate_median_dt(event.get("median_dt_s", 0)):
        return False
    
    return True