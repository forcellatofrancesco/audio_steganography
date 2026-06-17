"""WhatsApp-specific automation implementation.

Provides all WhatsApp-specific UI interaction logic. Can be swapped out
for other social media implementations without changing the main orchestration code.
"""

import datetime
import re
import subprocess
import time
from pathlib import Path

from playwright.sync_api import BrowserContext, TimeoutError as PlaywrightTimeoutError

from .base import SocialMediaAutomation
from .config import BASE_DIR, WhatsAppConfig


def _next_available_path(output_dir: Path, filename: str) -> Path:
    """Return a non-colliding file path in output_dir for filename."""
    candidate = output_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        unique_candidate = output_dir / f"{stem}_{counter:03d}{suffix}"
        if not unique_candidate.exists():
            return unique_candidate
        counter += 1


class WhatsAppAutomation(SocialMediaAutomation):
    """WhatsApp Web automation implementation."""

    # CSS selectors for WhatsApp UI elements
    VOICE_BUTTONS_SELECTOR = (
        "button[aria-label*='Riproduci'], "
        "button[aria-label*='Play'], "
        "button[aria-label*='Reproducir'], "
        "button[aria-label*='vocale'], "
        "button[aria-label*='voice'], "
        "button[aria-label*='mensaje'], "
        "button[aria-label*='audio'], "
        "button[aria-label*='ptt'], "
        "div[aria-label*='ptt'], "
        "span[aria-label*='ptt'], "
        "div[role='button']:has-text('ptt'), "
        "div:has-text('ptt'):has-text('ic-play-arrow-filled'), "
        "span:has-text('ptt'):has-text('ic-play-arrow-filled')"
    )

    READY_SELECTORS = [
        "div[aria-label='Chat list']",
        "div[role='grid']",
        "div[title='Search input textbox']",
    ]

    RECORDING_BUTTON_SELECTORS = [
        "button[data-tab='11'][aria-disabled='false']",
        "button[aria-label='Messaggio vocale'][aria-disabled='false']",
        "button[aria-label='Voice message'][aria-disabled='false']",
        "button[aria-label='Record'][aria-disabled='false']",
        "button:has(span[data-icon='mic-outlined'])",
        "button:has(span[data-icon='ptt'])",
        "span[data-icon='mic-outlined']",
        "span[data-icon='ptt']",
    ]

    SEND_BUTTON_SELECTORS = [
        "button[data-tab='11'][aria-label='Invia'][aria-disabled='false']",
        "button[aria-label='Send'][aria-disabled='false']",
        "button[aria-label='Invia'][aria-disabled='false']",
        "button:has(svg title:has-text('ic-send-filled'))",
    ]

    DOWNLOAD_MENU_ITEMS = [
        "Scarica",
    ]

    def __init__(self, context: BrowserContext, config: WhatsAppConfig | None = None):
        """Initialize WhatsApp automation.

        Args:
            context: Active Playwright browser context
            config: WhatsApp configuration. If None, loads from config.toml
        """
        super().__init__(context)
        self.config = config or WhatsAppConfig()
        self.context.set_default_timeout(self.config.default_timeout_ms)

    def navigate_to_whatsapp(self) -> None:
        """Navigate to WhatsApp Web and set permissions."""
        self.context.grant_permissions(["microphone"], origin=self.config.origin)
        self.page.goto(self.config.url, wait_until="domcontentloaded")

    def load_main_ui(self) -> None:
        """Wait until WhatsApp main UI is ready after login/QR scan."""
        for selector in self.READY_SELECTORS:
            try:
                self.page.locator(selector).first.wait_for(
                    state="visible", timeout=4_000
                )
                return
            except PlaywrightTimeoutError:
                continue

        # Keep waiting globally because first login can take longer while user scans QR.
        self.page.wait_for_timeout(1_000)
        self.page.wait_for_selector(
            "div[title='Search input textbox'], div[aria-label='Chat list']",
            timeout=120_000,
        )

    def start_recording(self) -> None:
        """Wait for and click the voice recording button."""
        deadline = time.monotonic() + (self.config.record_button_wait_timeout_ms / 1000)
        last_wait_log = 0.0

        while time.monotonic() < deadline:
            now = time.monotonic()
            if now - last_wait_log > 5:
                print("Waiting for recording button in the currently opened chat...")
                last_wait_log = now

            for selector in self.RECORDING_BUTTON_SELECTORS:
                try:
                    button = self.page.locator(selector).first
                    button.wait_for(state="visible", timeout=1_000)

                    # WhatsApp sometimes exposes only the icon span, so click its nearest button.
                    if selector.startswith("span["):
                        button = button.locator("xpath=ancestor::button[1]")

                    button.scroll_into_view_if_needed()
                    try:
                        button.click(timeout=1_000)
                    except Exception:
                        button.click(timeout=1_000, force=True)

                    print(f"Recording button detected with selector: {selector}")
                    return
                except Exception:
                    continue

            self.page.wait_for_timeout(1_000)

        raise RuntimeError(
            "Could not find WhatsApp recording button within the configured timeout. "
            "Open a chat and make sure the microphone button is visible."
        )

    def send_recording(self) -> None:
        """Wait for and click the voice message send button."""
        deadline = time.monotonic() + (self.config.send_button_wait_timeout_ms / 1000)

        while time.monotonic() < deadline:
            for selector in self.SEND_BUTTON_SELECTORS:
                try:
                    button = self.page.locator(selector).first
                    button.wait_for(state="visible", timeout=1_000)
                    button.scroll_into_view_if_needed()

                    try:
                        button.click(timeout=1_000)
                    except Exception:
                        button.click(timeout=1_000, force=True)

                    print(f"Send button detected with selector: {selector}")
                    return
                except Exception:
                    continue

            self.page.wait_for_timeout(250)

        raise RuntimeError("Could not find send button after playback finished.")

    def download_audio(
        self,
        previous_visible_audio_count: int,
        previous_visible_audio_signatures: set[tuple[int, int]] | None = None,
    ) -> str:
        """Wait for the sent audio message and download the one at the very bottom."""
        output_dir = self.config.download_output_path
        output_dir.mkdir(parents=True, exist_ok=True)

        self.page.wait_for_timeout(self.config.post_send_settle_delay_ms)

        deadline = time.monotonic() + (self.config.download_wait_timeout_ms / 1000)

        print("Waiting for sent audio message to appear at the bottom of the chat...")

        # Pattern from user recording: 'ic-play-arrow-filled' + timestamp + 'ptt'
        # We use regex to handle varying durations (e.g., 0:05, 0:10, etc.)
        AUDIO_PATTERN = re.compile(r"ic-play-arrow-filled.*ptt", re.IGNORECASE)

        while time.monotonic() < deadline:
            try:
                # 1. Primary Strategy: Find the LATEST outgoing message bubble.
                # Sent messages are in 'div.message-out'.
                latest_out_bubble = self.page.locator(
                    'div[role="row"]:has(button[aria-label="Riproduci messaggio vocale"])'
                ).last
                if latest_out_bubble.count() > 0:
                    audio_btn = latest_out_bubble.get_by_text(AUDIO_PATTERN)
                    if audio_btn.count() > 0:
                        print("Found audio control in latest outgoing message bubble.")
                        target = audio_btn.first
                        target.scroll_into_view_if_needed()

                        # Execute recorded sequence
                        target.click(button="right")
                        with self.page.expect_download(timeout=10000) as download_info:
                            self.page.get_by_role("menuitem", name="Scarica").click()

                        download = download_info.value
                        filename = download.suggested_filename
                        if not filename or filename in ["audio.ogg", "audio", "ptt"]:
                            filename = f"downloaded_audio_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.ogg"

                        output_path = _next_available_path(output_dir, filename)
                        download.save_as(str(output_path))
                        print(
                            f"Successfully downloaded latest sent audio: {output_path}"
                        )
                        return str(output_path)

                # 2. Fallback Strategy: Find the absolute bottom-most audio control on screen.
                # This handles cases where '.message-out' might have changed or isn't matching correctly.
                all_candidates = self.page.get_by_text(AUDIO_PATTERN).all()
                if not all_candidates:
                    # Final fallback to generic buttons
                    all_candidates = self.page.locator(
                        self.VOICE_BUTTONS_SELECTOR
                    ).all()

                if all_candidates:
                    # Calculate bottom Y for each and pick the highest
                    boxes = []
                    for cand in all_candidates:
                        try:
                            if not cand.is_visible():
                                continue
                            box = cand.bounding_box()
                            if box:
                                boxes.append((box["y"] + box["height"], cand))
                        except Exception:
                            continue

                    if boxes:
                        boxes.sort(key=lambda x: x[0], reverse=True)
                        target = boxes[0][1]
                        print(
                            f"Fallback: Selecting absolute bottom-most audio element (Y={boxes[0][0]:.1f})"
                        )

                        target.scroll_into_view_if_needed()
                        target.click(button="right")
                        with self.page.expect_download(timeout=10000) as download_info:
                            self.page.get_by_role("menuitem", name="Scarica").click()

                        download = download_info.value
                        filename = download.suggested_filename
                        if not filename or filename in ["audio.ogg", "audio", "ptt"]:
                            filename = f"downloaded_audio_fallback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.ogg"

                        output_path = _next_available_path(output_dir, filename)
                        download.save_as(str(output_path))
                        return str(output_path)

            except Exception as e:
                print(f"Retrying download... ({e})")
                self.page.wait_for_timeout(300)

        raise RuntimeError(
            "Could not find and download the audio message at the bottom of the chat."
        )

    def _find_and_click_download(
        self, audio_container, audio_button, output_dir: Path
    ) -> str:
        # Obsolete
        return ""

    def play_audio(self) -> None:
        """Execute the playback script that feeds audio to the virtual microphone."""
        script_path = self.config.playback_script_path

        if not script_path.exists():
            raise FileNotFoundError(f"Playback script not found: {script_path}")

        print(f"Running playback script: {script_path}")
        subprocess.run(["bash", str(script_path)], cwd=str(BASE_DIR), check=True)
        print("Playback script finished.")

    def _count_visible_voice_buttons(self) -> int:
        """Count the number of visible voice message buttons in the chat."""
        count = 0
        for button in self.page.locator(self.VOICE_BUTTONS_SELECTOR).all():
            try:
                if button.is_visible() and button.bounding_box():
                    count += 1
            except Exception:
                continue
        return count

    def _visible_audio_signatures(self) -> set[tuple[int, int]]:
        """Return coarse signatures for currently visible audio controls."""
        signatures: set[tuple[int, int]] = set()
        for button in self.page.locator(self.VOICE_BUTTONS_SELECTOR).all():
            try:
                if not button.is_visible():
                    continue
                box = button.bounding_box()
                if not box:
                    continue
                center_y = box["y"] + (box["height"] / 2)
                center_x = box["x"] + (box["width"] / 2)
                signatures.add((int(round(center_y)), int(round(center_x))))
            except Exception:
                continue
        return signatures

    def run(self) -> str:
        """Execute the complete WhatsApp voice message workflow.

        Returns:
            Path to the downloaded audio file.
        """
        self.start_recording()

        playback_error = None
        try:
            self.play_audio()
        except Exception as exc:
            playback_error = exc

        print("Trying to send the recorded audio...")
        self.send_recording()

        if playback_error is not None:
            raise playback_error

        print("Waiting for audio to be downloaded...")
        downloaded_path = self.download_audio(0, set())
        print(f"Audio file downloaded to: {downloaded_path}")

        return downloaded_path
