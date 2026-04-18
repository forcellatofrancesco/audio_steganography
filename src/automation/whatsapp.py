"""WhatsApp-specific automation implementation.

Provides all WhatsApp-specific UI interaction logic. Can be swapped out
for other social media implementations without changing the main orchestration code.
"""

import datetime
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
        "button[aria-label*='ptt']"
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
        "div[role='menuitem']:has-text('Scarica')",
        "div[role='menuitem']:has-text('Download')",
        "div[role='menuitem']:has-text('Télécharger')",
        "div[role='menuitem']:has-text('Descargar')",
        "div[role='menuitem']:has-text('Baixar')",
        "div[role='menuitem']:has-text('تحميل')",
        "text='Scarica'",
        "text='Download'",
        "text='Télécharger'",
        "text='Descargar'",
        "text='Baixar'",
        "text='تحميل'",
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

            self.page.wait_for_timeout(300)

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
        """Wait for the sent audio message and download it to the output folder."""
        output_dir = self.config.download_output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        previous_visible_audio_signatures = previous_visible_audio_signatures or set()

        if self.config.post_send_settle_delay_ms > 0:
            print(
                "Waiting for WhatsApp UI to render the sent audio bubble "
                f"({self.config.post_send_settle_delay_ms} ms)..."
            )
            self.page.wait_for_timeout(self.config.post_send_settle_delay_ms)

        deadline = time.monotonic() + (self.config.download_wait_timeout_ms / 1000)
        last_wait_log = 0.0

        print("Waiting for sent audio message to appear in chat...")

        while time.monotonic() < deadline:
            now = time.monotonic()
            if now - last_wait_log > 5:
                print("Searching for audio message in the chat...")
                last_wait_log = now

            try:
                voice_buttons = self.page.locator(self.VOICE_BUTTONS_SELECTOR).all()

                if voice_buttons:
                    latest_button = None
                    latest_container = None
                    latest_key = (float("-inf"), float("-inf"))
                    visible_count = 0
                    new_outgoing_candidates = []
                    new_fallback_candidates = []
                    outgoing_candidates = []
                    fallback_candidates = []
                    page_width = (self.page.viewport_size or {}).get("width", 1366)

                    for button in voice_buttons:
                        try:
                            if not button.is_visible():
                                continue

                            button_box = button.bounding_box()
                            if not button_box:
                                continue

                            visible_count += 1

                            container = button.locator(
                                "xpath=ancestor::div[contains(@class, '_ak4a')][1]"
                            )
                            if container.count() == 0:
                                container = button.locator(
                                    "xpath=ancestor::div[@role='gridcell'][1]"
                                )
                            if container.count() == 0:
                                continue

                            box = container.first.bounding_box()
                            if not box:
                                continue

                            button_center_y = button_box["y"] + (
                                button_box["height"] / 2
                            )
                            button_center_x = button_box["x"] + (
                                button_box["width"] / 2
                            )
                            signature = (
                                int(round(button_center_y)),
                                int(round(button_center_x)),
                            )

                            entry = (
                                (button_center_y, button_center_x),
                                button,
                                container.first,
                            )
                            fallback_candidates.append(entry)
                            if signature not in previous_visible_audio_signatures:
                                new_fallback_candidates.append(entry)

                            # Prefer outgoing (right side) messages, then latest by Y/X.
                            if button_center_x > (page_width * 0.55):
                                outgoing_candidates.append(entry)
                                if signature not in previous_visible_audio_signatures:
                                    new_outgoing_candidates.append(entry)
                        except Exception:
                            continue

                    # Wait until at least one new visible audio appears after send.
                    if visible_count <= previous_visible_audio_count:
                        self.page.wait_for_timeout(300)
                        continue

                    candidates = (
                        new_outgoing_candidates
                        or new_fallback_candidates
                        or outgoing_candidates
                        or fallback_candidates
                    )
                    if candidates:
                        latest_key, latest_button, latest_container = max(
                            candidates, key=lambda x: x[0]
                        )

                    if latest_container is not None and latest_button is not None:
                        print(
                            "Found new audio message; selecting latest "
                            f"{'new outgoing' if new_outgoing_candidates else ('new visible' if new_fallback_candidates else ('outgoing' if outgoing_candidates else 'visible'))} audio button "
                            f"(visible={visible_count}, previous={previous_visible_audio_count}, y={latest_key[0]:.1f})..."
                        )
                        latest_button.scroll_into_view_if_needed()
                        latest_container.scroll_into_view_if_needed()

                        return self._find_and_click_download(
                            latest_container, latest_button, output_dir
                        )

            except Exception as e:
                print(f"Error during search: {e}")

            self.page.wait_for_timeout(500)

        raise RuntimeError(
            "Could not find audio message within the configured timeout."
        )

    def _find_and_click_download(
        self, audio_container, audio_button, output_dir: Path
    ) -> str:
        """Right-click the center of latest audio bubble and click download."""
        deadline = time.monotonic() + 15
        print("Opening context menu on latest audio message...")

        download = None
        attempt = 0

        while time.monotonic() < deadline and download is None:
            attempt += 1

            try:
                audio_container.wait_for(state="visible", timeout=1_500)
                audio_container.scroll_into_view_if_needed()

                container_box = audio_container.bounding_box()
                if not container_box:
                    self.page.wait_for_timeout(250)
                    continue

                button_box = None
                try:
                    button_box = audio_button.bounding_box()
                except Exception:
                    button_box = None

                # Target bubble center. If container is too tall, anchor Y to audio controls.
                center_x = container_box["x"] + (container_box["width"] / 2)
                center_y = container_box["y"] + (container_box["height"] / 2)
                if button_box and container_box["height"] > (
                    button_box["height"] * 2.0
                ):
                    center_y = button_box["y"] + (button_box["height"] / 2)

                click_offsets = [
                    (0.0, 0.0),
                    (10.0, 0.0),
                    (-10.0, 0.0),
                    (0.0, 8.0),
                    (0.0, -8.0),
                ]
                offset_x, offset_y = click_offsets[(attempt - 1) % len(click_offsets)]

                click_x = center_x + offset_x
                click_y = center_y + offset_y

                # Clamp inside bubble bounds.
                click_x = max(
                    container_box["x"] + 2,
                    min(container_box["x"] + container_box["width"] - 2, click_x),
                )
                click_y = max(
                    container_box["y"] + 2,
                    min(container_box["y"] + container_box["height"] - 2, click_y),
                )

                self.page.mouse.click(click_x, click_y, button="right")
            except Exception:
                self.page.wait_for_timeout(250)
                continue

            self.page.wait_for_timeout(250)

            for selector in self.DOWNLOAD_MENU_ITEMS:
                try:
                    menu_item = self.page.locator(selector).first
                    menu_item.wait_for(state="visible", timeout=600)

                    with self.page.expect_download(timeout=3_000) as download_info:
                        menu_item.click(timeout=1_000)

                    download = download_info.value
                    print(f"Clicked menu item for download: {selector}")
                    break
                except PlaywrightTimeoutError:
                    continue
                except Exception:
                    continue

            if download is None:
                if attempt % 3 == 1:
                    print(f"Download menu item not captured yet (attempt {attempt})")
                try:
                    self.page.keyboard.press("Escape")
                except Exception:
                    pass
                self.page.wait_for_timeout(250)

        if download is None:
            raise RuntimeError(
                "Could not click download from context menu after right-clicking the latest audio."
            )

        filename = download.suggested_filename

        # Generate a timestamped filename if needed
        if (
            not filename
            or filename == "audio.ogg"
            or filename == "audio"
            or filename == "ptt"
        ):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"downloaded_audio_{timestamp}.ogg"

        output_path = _next_available_path(output_dir, filename)
        download.save_as(str(output_path))

        print(f"Audio downloaded successfully to: {output_path}")
        return str(output_path)

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

        previous_visible_audio_count = self._count_visible_voice_buttons()
        previous_visible_audio_signatures = self._visible_audio_signatures()

        print("Trying to send the recorded audio...")
        self.send_recording()

        if playback_error is not None:
            raise playback_error

        print("Waiting for audio to be downloaded...")
        downloaded_path = self.download_audio(
            previous_visible_audio_count,
            previous_visible_audio_signatures,
        )
        print(f"Audio file downloaded to: {downloaded_path}")

        return downloaded_path
