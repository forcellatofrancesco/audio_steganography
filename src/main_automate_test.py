import subprocess
import time
import tomllib
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parent.parent


def _read_whatsapp_automation_config() -> dict:
    config_path = BASE_DIR / "config.toml"
    with config_path.open("rb") as f:
        config = tomllib.load(f)
    section = config.get("whatsapp_automation", {})
    if not isinstance(section, dict):
        return {}
    return section


def _as_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _as_int(value, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


WHATSAPP_AUTOMATION_CONFIG = _read_whatsapp_automation_config()
WHATSAPP_URL = (
    str(WHATSAPP_AUTOMATION_CONFIG.get("url", "https://web.whatsapp.com/")).strip()
    or "https://web.whatsapp.com/"
)
WHATSAPP_ORIGIN = (
    str(WHATSAPP_AUTOMATION_CONFIG.get("origin", "https://web.whatsapp.com")).strip()
    or "https://web.whatsapp.com"
)

_profile_dir_value = str(
    WHATSAPP_AUTOMATION_CONFIG.get("profile_dir", ".playwright_whatsapp_profile")
).strip()
PROFILE_DIR = Path(_profile_dir_value).expanduser()
if not PROFILE_DIR.is_absolute():
    PROFILE_DIR = (BASE_DIR / PROFILE_DIR).resolve()

HEADLESS = _as_bool(WHATSAPP_AUTOMATION_CONFIG.get("headless"), False)
KEEP_OPEN = _as_bool(WHATSAPP_AUTOMATION_CONFIG.get("keep_open"), True)
DEFAULT_TIMEOUT_MS = _as_int(
    WHATSAPP_AUTOMATION_CONFIG.get("default_timeout_ms"), 15_000
)
RECORD_BUTTON_WAIT_TIMEOUT_MS = _as_int(
    WHATSAPP_AUTOMATION_CONFIG.get("record_button_wait_timeout_ms"), 300_000
)
SEND_BUTTON_WAIT_TIMEOUT_MS = _as_int(
    WHATSAPP_AUTOMATION_CONFIG.get("send_button_wait_timeout_ms"), 20_000
)
DOWNLOAD_WAIT_TIMEOUT_MS = _as_int(
    WHATSAPP_AUTOMATION_CONFIG.get("download_wait_timeout_ms"), 30_000
)
PLAYBACK_SCRIPT_PATH = str(
    WHATSAPP_AUTOMATION_CONFIG.get(
        "playback_script_path", "src/util/scripts/virtual_microphone/play_mic.sh"
    )
).strip()
DOWNLOAD_OUTPUT_PATH = str(
    WHATSAPP_AUTOMATION_CONFIG.get("download_output_path", "output")
).strip()

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


def _count_visible_voice_buttons(page) -> int:
    count = 0
    for button in page.locator(VOICE_BUTTONS_SELECTOR).all():
        try:
            if button.is_visible() and button.bounding_box():
                count += 1
        except Exception:
            continue
    return count


def _wait_and_download_audio(
    page, wait_timeout_ms: int, previous_visible_audio_count: int
) -> str:
    """Wait for the sent audio message and download it to the output folder.

    Returns the path to the downloaded file.
    """
    output_dir = Path(DOWNLOAD_OUTPUT_PATH).expanduser()
    if not output_dir.is_absolute():
        output_dir = (BASE_DIR / output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    deadline = time.monotonic() + (wait_timeout_ms / 1000)
    last_wait_log = 0.0

    print("Waiting for sent audio message to appear in chat...")

    while time.monotonic() < deadline:
        now = time.monotonic()
        if now - last_wait_log > 5:
            print("Searching for audio message in the chat...")
            last_wait_log = now

        try:
            voice_buttons = page.locator(VOICE_BUTTONS_SELECTOR).all()

            if voice_buttons:
                latest_button = None
                latest_container = None
                latest_key = (float("-inf"), float("-inf"))
                visible_count = 0
                outgoing_candidates = []
                fallback_candidates = []
                page_width = (page.viewport_size or {}).get("width", 1366)

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

                        button_center_y = button_box["y"] + (button_box["height"] / 2)
                        button_center_x = button_box["x"] + (button_box["width"] / 2)

                        entry = (
                            (button_center_y, button_center_x),
                            button,
                            container.first,
                        )
                        fallback_candidates.append(entry)

                        # Prefer outgoing (right side) messages, then latest by Y/X.
                        if button_center_x > (page_width * 0.55):
                            outgoing_candidates.append(entry)
                    except Exception:
                        continue

                # Wait until at least one new visible audio appears after send.
                if visible_count <= previous_visible_audio_count:
                    page.wait_for_timeout(300)
                    continue

                candidates = outgoing_candidates or fallback_candidates
                if candidates:
                    latest_key, latest_button, latest_container = max(
                        candidates, key=lambda x: x[0]
                    )

                if latest_container is not None and latest_button is not None:
                    print(
                        "Found new audio message; selecting latest "
                        f"{'outgoing' if outgoing_candidates else 'visible'} audio button "
                        f"(visible={visible_count}, previous={previous_visible_audio_count}, y={latest_key[0]:.1f})..."
                    )
                    latest_button.scroll_into_view_if_needed()
                    latest_container.scroll_into_view_if_needed()

                    return _find_and_click_download(
                        page, latest_container, latest_button, output_dir
                    )

        except Exception as e:
            print(f"Error during search: {e}")

        page.wait_for_timeout(500)

    raise RuntimeError("Could not find audio message within the configured timeout.")


def _find_and_click_download(
    page, audio_container, audio_button, output_dir: Path
) -> str:
    """Right-click the center of latest audio bubble and click download."""
    import datetime

    menu_item_selectors = [
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
                page.wait_for_timeout(250)
                continue

            button_box = None
            try:
                button_box = audio_button.bounding_box()
            except Exception:
                button_box = None

            # Target bubble center. If container is too tall, anchor Y to audio controls.
            center_x = container_box["x"] + (container_box["width"] / 2)
            center_y = container_box["y"] + (container_box["height"] / 2)
            if button_box and container_box["height"] > (button_box["height"] * 2.0):
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

            page.mouse.click(click_x, click_y, button="right")
        except Exception:
            page.wait_for_timeout(250)
            continue

        page.wait_for_timeout(250)

        for selector in menu_item_selectors:
            try:
                menu_item = page.locator(selector).first
                menu_item.wait_for(state="visible", timeout=600)

                with page.expect_download(timeout=3_000) as download_info:
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
                page.keyboard.press("Escape")
            except Exception:
                pass
            page.wait_for_timeout(250)

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

    output_path = output_dir / filename
    download.save_as(str(output_path))

    print(f"Audio downloaded successfully to: {output_path}")
    return str(output_path)


def main():
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=HEADLESS,
            args=["--start-maximized"],
            viewport={"width": 1270, "height": 720},
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

        previous_visible_audio_count = _count_visible_voice_buttons(page)

        print("Trying to send the recorded audio...")
        _wait_and_send_voice_message(page, SEND_BUTTON_WAIT_TIMEOUT_MS)

        if playback_error is not None:
            raise playback_error

        print("Waiting for audio to be downloaded...")
        downloaded_path = _wait_and_download_audio(
            page,
            DOWNLOAD_WAIT_TIMEOUT_MS,
            previous_visible_audio_count,
        )
        print(f"Audio file downloaded to: {downloaded_path}")

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
