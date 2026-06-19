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
        [img1, img2], "2024", "Test Event", "Test Piece", "Test Comp", "Test Arr", "Test Inst"
    )
    print(f"Test data created. ID: {score_id}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Setup context for video recording
        context = await browser.new_context(
            record_video_dir="videos/",
            record_video_size={"width": 1280, "height": 720}
        )
        page = await context.new_page()

        # Go to the newly created score's view page
        url = f"http://127.0.0.1:5000/view_score?id={score_id}&instrument=Test+Inst"
        await page.goto(url)

        # Wait for the order inputs to be visible
        await page.wait_for_selector('input[name="orders[]"]')

        # Take screenshot before change
        await page.screenshot(path="screenshot_before_order_update.png")

        # Change the order
        inputs = await page.locator('input[name="orders[]"]').all()
        await inputs[0].fill("2")
        await inputs[1].fill("1")

        # Submit the form
        await page.click('button:has-text("✓ 表示順を更新する")')

        # Wait for the page to reload and inputs to be visible again
        await page.wait_for_selector('input[name="orders[]"]')

        # Take screenshot after change
        await page.screenshot(path="screenshot_after_order_update.png")

        # Verify the order visually/programmatically if needed (e.g., checking values)
        new_inputs = await page.locator('input[name="orders[]"]').all()
        val0 = await new_inputs[0].input_value()
        val1 = await new_inputs[1].input_value()
        print(f"New input values: {val0}, {val1}")

        await context.close()
        await browser.close()
        print("Playwright script completed successfully.")

if __name__ == "__main__":
    asyncio.run(run())
