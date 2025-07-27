"""
Framework Agents

This package contains the agent implementations for the dual-agent system.
"""

from .discovery_agent import DiscoveryAgent
from .critical_agent import CriticalAgent

__all__ = [
    'DiscoveryAgent',
    'CriticalAgent'
] 