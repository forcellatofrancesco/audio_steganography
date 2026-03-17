import os
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

WHATSAPP_URL = (
    os.getenv("WHATSAPP_URL", "https://web.whatsapp.com/").strip()
    or "https://web.whatsapp.com/"
)
WHATSAPP_ORIGIN = "https://web.whatsapp.com"
PROFILE_DIR = Path(
    os.getenv(
        "WHATSAPP_PROFILE_DIR",
        str(BASE_DIR / ".playwright_whatsapp_profile"),
    )
).expanduser()
HEADLESS = _get_env_bool("WHATSAPP_HEADLESS", False)
KEEP_OPEN = _get_env_bool("WHATSAPP_KEEP_OPEN", True)
DEFAULT_TIMEOUT_MS = _get_env_int("WHATSAPP_TIMEOUT_MS", 15_000)
RECORD_BUTTON_WAIT_TIMEOUT_MS = _get_env_int("WHATSAPP_RECORD_WAIT_TIMEOUT_MS", 300_000)
SEND_BUTTON_WAIT_TIMEOUT_MS = _get_env_int("WHATSAPP_SEND_WAIT_TIMEOUT_MS", 20_000)
PLAYBACK_SCRIPT_PATH = os.getenv(
    "WHATSAPP_PLAYBACK_SCRIPT",
    "src/util/scripts/virtual_microphone/play_mic.sh",
).strip()


def _wait_for_main_ui(page) -> None:
    """Wait until WhatsApp main UI is ready after login/QR scan."""
    ready_selectors = [
        "div[aria-label='Chat list']",
        "div[role='grid']",
        "div[title='Search input textbox']",
    ]
    for selector in ready_selectors:
        try:
            page.locator(selector).first.wait_for(state="visible", timeout=4_000)
            return
        except PlaywrightTimeoutError:
            continue

    # Keep waiting globally because first login can take longer while user scans QR.
    page.wait_for_timeout(1_000)
    page.wait_for_selector(
        "div[title='Search input textbox'], div[aria-label='Chat list']",
        timeout=120_000,
    )


def _wait_and_start_voice_recording(page, wait_timeout_ms: int) -> None:
    recording_button_selectors = [
        "button[data-tab='11'][aria-disabled='false']",
        "button[aria-label='Messaggio vocale'][aria-disabled='false']",
        "button[aria-label='Voice message'][aria-disabled='false']",
        "button[aria-label='Record'][aria-disabled='false']",
        "button:has(span[data-icon='mic-outlined'])",
        "button:has(span[data-icon='ptt'])",
        "span[data-icon='mic-outlined']",
        "span[data-icon='ptt']",
    ]

    deadline = time.monotonic() + (wait_timeout_ms / 1000)
    last_wait_log = 0.0

    while time.monotonic() < deadline:
        now = time.monotonic()
        if now - last_wait_log > 5:
            print("Waiting for recording button in the currently opened chat...")
            last_wait_log = now

        for selector in recording_button_selectors:
            try:
                button = page.locator(selector).first
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

        page.wait_for_timeout(300)

    raise RuntimeError(
        "Could not find WhatsApp recording button within the configured timeout. "
        "Open a chat and make sure the microphone button is visible."
    )


def _wait_and_send_voice_message(page, wait_timeout_ms: int) -> None:
    send_button_selectors = [
        "button[data-tab='11'][aria-label='Invia'][aria-disabled='false']",
        "button[aria-label='Send'][aria-disabled='false']",
        "button[aria-label='Invia'][aria-disabled='false']",
        "button:has(svg title:has-text('ic-send-filled'))",
    ]

    deadline = time.monotonic() + (wait_timeout_ms / 1000)

    while time.monotonic() < deadline:
        for selector in send_button_selectors:
            try:
                button = page.locator(selector).first
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

        page.wait_for_timeout(250)

    raise RuntimeError("Could not find send button after playback finished.")


def _run_playback_script() -> None:
    script_path = Path(PLAYBACK_SCRIPT_PATH).expanduser()
    if not script_path.is_absolute():
        script_path = (BASE_DIR / script_path).resolve()

    if not script_path.exists():
        raise FileNotFoundError(f"Playback script not found: {script_path}")

    print(f"Running playback script: {script_path}")
    subprocess.run(["bash", str(script_path)], cwd=str(BASE_DIR), check=True)
    print("Playback script finished.")


def main():
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=HEADLESS,
            args=["--start-maximized"],
            viewport={"width": 1366, "height": 768},
        )
        context.set_default_timeout(DEFAULT_TIMEOUT_MS)
        context.grant_permissions(["microphone"], origin=WHATSAPP_ORIGIN)

        page = context.pages[0] if context.pages else context.new_page()
        page.goto(WHATSAPP_URL, wait_until="domcontentloaded")

        print("Waiting for WhatsApp Web. Scan QR code if prompted...")
        _wait_for_main_ui(page)

        print("Now open the target chat manually.")
        print("Waiting for the recording button to appear...")
        _wait_and_start_voice_recording(page, RECORD_BUTTON_WAIT_TIMEOUT_MS)

        playback_error = None
        try:
            _run_playback_script()
        except Exception as exc:
            playback_error = exc

        print("Trying to send the recorded audio...")
        _wait_and_send_voice_message(page, SEND_BUTTON_WAIT_TIMEOUT_MS)

        if playback_error is not None:
            raise playback_error

        if KEEP_OPEN:
            print("Recording completed. Browser will stay open; press Ctrl+C to stop.")
            try:
                while True:
                    page.wait_for_timeout(1_000)
            except KeyboardInterrupt:
                print("Stopping automation.")
            finally:
                context.close()
        else:
            print("Recording completed. Closing browser in 3 seconds.")
            page.wait_for_timeout(3_000)
            context.close()


if __name__ == "__main__":
    main()
