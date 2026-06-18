import asyncio
import os
from playwright.async_api import async_playwright
import score_api

async def run():
    # Insert test data using score_api
    import PIL.Image as Image
    # Create two dummy images
    img1 = Image.new('L', (100, 100), color=200)
    img2 = Image.new('L', (100, 100), color=100)

    saved_dir, score_id = score_api.save_and_register_score(
        [img1, img2], "2024", "Delete Test Event", "Delete Test Piece", "Delete Comp", "Delete Arr", "Delete Inst"
    )
    print(f"Test data created. ID: {score_id}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Setup context for video recording
        context = await browser.new_context(
            record_video_dir="videos_delete/",
            record_video_size={"width": 1280, "height": 720}
        )
        page = await context.new_page()

        # Handle JS confirm dialog automatically
        page.on("dialog", lambda dialog: dialog.accept())

        # Go to the newly created score's view page
        url = f"http://127.0.0.1:5000/view_score?id={score_id}&instrument=Delete+Inst"
        await page.goto(url)

        # Wait for the delete buttons to be visible
        await page.wait_for_selector('button:has-text("削除")')

        delete_buttons = await page.locator('button:has-text("削除")').all()
        initial_count = len(delete_buttons)
        print(f"Initial image count: {initial_count}")

        # Take screenshot before delete
        await page.screenshot(path="screenshot_before_delete.png")

        # Click the first delete button
        await delete_buttons[0].click()

        # Wait for the page to reload
        await page.wait_for_selector('button:has-text("削除")')

        # Take screenshot after delete
        await page.screenshot(path="screenshot_after_delete.png")

        # Verify the number of images decreased
        new_delete_buttons = await page.locator('button:has-text("削除")').all()
        new_count = len(new_delete_buttons)
        print(f"New image count: {new_count}")

        assert new_count == initial_count - 1, "Image was not deleted!"

        await context.close()
        await browser.close()
        print("Playwright delete script completed successfully.")

if __name__ == "__main__":
    asyncio.run(run())
