"""Main orchestrator for social media voice message automation.

This module is the entry point for the voice message recording and transmission
automation framework. It handles browser setup and delegates to the appropriate
social media implementation.

To switch between social media platforms, change the import and instantiation
at the bottom of this file. Currently uses WhatsApp.
"""

from playwright.sync_api import sync_playwright

from automation.config import WhatsAppConfig
from automation.whatsapp import WhatsAppAutomation


def main():
    """Main orchestration function for voice message automation.

    Flow:
    1. Load configuration
    2. Launch browser
    3. Instantiate the social media automation (WhatsApp)
    4. Execute the recording workflow
    5. Handle post-completion behavior (keep browser open or close)

    To add a new platform (e.g., Instagram):
    1. Create src/automation/instagram.py with InstagramAutomation class
    2. Create an InstagramConfig class in config.py
    3. Change lines below:
       - from src.automation.instagram import InstagramAutomation
       - config = InstagramConfig()
       - automation = InstagramAutomation(context, config)

    That's it! The rest of the code remains unchanged.
    """
    # Load WhatsApp configuration (edit config.toml to customize)
    config = WhatsAppConfig()
    config.profile_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        # Launch persistent browser context to maintain login session
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(config.profile_dir),
            headless=config.headless,
            args=["--start-maximized"],
            viewport={"width": 1270, "height": 720},
        )

        try:
            # Instantiate WhatsApp automation
            # SWITCH PLATFORMS HERE: Replace with other social media class
            automation = WhatsAppAutomation(context, config)
            automation.navigate_to_whatsapp()

            # Execute the complete workflow
            downloaded_path = automation.run()

            # Handle post-completion behavior
            if config.keep_open:
                print(
                    "Recording completed. Browser will stay open; press Ctrl+C to stop."
                )
                try:
                    while True:
                        context.pages[0].wait_for_timeout(1_000)
                except KeyboardInterrupt:
                    print("Stopping automation.")
            else:
                print("Recording completed. Closing browser in 3 seconds.")
                context.pages[0].wait_for_timeout(3_000)

        finally:
            context.close()


if __name__ == "__main__":
    main()
