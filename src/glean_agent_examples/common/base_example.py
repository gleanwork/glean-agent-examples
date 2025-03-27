"""
Base class for all Glean agent examples.
"""

from enum import Enum

class IconType(Enum):
    """
    Common icons that can be used with print_title.
    """
    SEARCH = "🔍"
    START = "🚀"
    SUCCESS = "✅"
    ERROR = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"
    QUESTION = "❓"
    CHAT = "💬"
    API = "🔌"
    DATABASE = "🗄️"
    DOCUMENT = "📄"
    CODE = "💻"
    GLEAN = "🔍"
    AGENT = "🤖"
    LOADING = "⏳"
    COMPLETE = "🏁"

class BaseExample:
    """
    Base class for all Glean agent examples.
    
    This class provides common functionality for all example classes.
    """
    
    def print_title(self, title: str, icon: str | IconType = IconType.SEARCH) -> None:
        """
        Print a section title in a consistent format.
        
        Args:
            title: The title text to print
            icon: An icon to display before the title. Can be a string or an IconType enum value.
        """

        icon_str = icon.value if isinstance(icon, IconType) else icon
        print(f"\n{icon_str} {title}\n")


    def print_message(self, message: str, icon: str | IconType = "") -> None:
        """
        Print a message in a consistent format.
        
        Args:
            message: The message text to print
            icon: An icon to display before the message. Can be a string or an IconType enum value.
        """
        icon_str = icon.value if isinstance(icon, IconType) else icon
        green_info = "\033[32mINFO\033[0m"
        
        if icon_str:
            prefix = f"{icon_str} "
            print(f"\n{green_info}:     {prefix}{message}\n")
        else:
            print(f"{green_info}:     {message}")
    
