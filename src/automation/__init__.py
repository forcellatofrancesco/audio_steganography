"""Automation framework for social media platforms.

This package provides an extensible architecture for automating voice message
recording and transmission across different social media platforms.

To add a new platform:
1. Create a new file inheriting from SocialMediaAutomation
2. Implement all abstract methods for that platform
3. Import and instantiate it in main_automate_test.py
"""

from .base import SocialMediaAutomation
from .whatsapp import WhatsAppAutomation

__all__ = ["SocialMediaAutomation", "WhatsAppAutomation"]
