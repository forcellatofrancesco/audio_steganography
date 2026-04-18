"""Abstract base class for social media automation implementations.

This module defines the interface that all social media automation
implementations must follow. This allows for a plugin-like architecture
where new platforms can be added without modifying existing code.
"""

from abc import ABC, abstractmethod

from playwright.sync_api import BrowserContext


class SocialMediaAutomation(ABC):
    """Abstract base class defining the interface for social media automation.

    Subclasses must implement all abstract methods to support a new social media platform.
    The workflow is:
    1. wait_for_main_ui() - Ensure the app is ready
    2. wait_and_start_voice_recording() - Initiate voice recording
    3. run_playback_script() - Feed audio to the microphone
    4. wait_and_send_voice_message() - Submit the recorded message
    5. wait_and_download_audio() - Retrieve the sent message
    """

    def __init__(self, context: BrowserContext):
        """Initialize automation with a Playwright browser context.

        Args:
            context: Active Playwright browser context
        """
        self.context = context
        self.page = context.pages[0] if context.pages else context.new_page()

    @abstractmethod
    def load_main_ui(self) -> None:
        """Wait until the main UI of the platform is ready.

        This should block until the user can interact with the core interface,
        typically after login/authentication.

        Raises:
            TimeoutError: If UI is not ready within timeout
        """
        pass

    @abstractmethod
    def start_recording(self) -> None:
        """Wait for and click the voice recording button.

        This should locate the microphone/voice recording button in the currently
        open chat/conversation and click it to start recording.

        Raises:
            RuntimeError: If recording button not found within timeout
        """
        pass

    @abstractmethod
    def send_recording(self) -> None:
        """Wait for and click the voice message send button.

        This should locate the send button after audio recording is complete
        and click it to submit the voice message.

        Raises:
            RuntimeError: If send button not found within timeout
        """
        pass

    @abstractmethod
    def download_audio(
        self,
        previous_visible_audio_count: int,
        previous_visible_audio_signatures: set[tuple[int, int]] | None = None,
    ) -> str:
        """Wait for the sent voice message and download it.

        Args:
            previous_visible_audio_count: Number of audio messages visible before send.
                Used to detect when a new message arrives.
            previous_visible_audio_signatures: Optional signatures of visible audio
                controls before send, used to avoid reselecting old messages.

        Returns:
            Path to the downloaded audio file.

        Raises:
            RuntimeError: If audio message not found or download fails
        """
        pass

    @abstractmethod
    def play_audio(self) -> None:
        """Execute the script that feeds audio to the virtual microphone.

        This should run an external script that simulates audio input,
        so the platform records the intended message.

        Raises:
            FileNotFoundError: If playback script not found
            subprocess.CalledProcessError: If script execution fails
        """
        pass

    @abstractmethod
    def run(self) -> str:
        """Execute the complete voice message recording workflow.

        Orchestrates the full sequence:
        1. Wait for main UI
        2. Start recording
        3. Play audio
        4. Send audio message
        5. Download audio

        Returns:
            Path to the downloaded audio file.
        """
        pass
