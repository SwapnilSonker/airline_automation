import asyncio
from datetime import datetime
from playwright.async_api import async_playwright

def format_aria_label(date_obj):
    # Format date like "Wed May 28 2025"
    return date_obj.strftime("%a %b %d %Y")

async def fill_route_fields(page, from_city, to_city):
    # From City
    await page.click("input#fromCity")
    await page.fill("input.react-autosuggest__input", from_city)
    await page.wait_for_timeout(1000)
    await page.keyboard.press("ArrowDown")
    await page.keyboard.press("Enter")

    # To City
    await page.click("input#toCity")
    await page.fill("input.react-autosuggest__input", to_city)
    await page.wait_for_timeout(1000)
    await page.keyboard.press("ArrowDown")
    await page.keyboard.press("Enter")
    print("end here")

async def open_makemytrip_and_close_modal():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        # Open MakeMyTrip
        await page.goto("https://www.makemytrip.com")

        # Give time for modal to appear (you can tune this delay or use wait_for_selector)
        await page.wait_for_selector('span.commonModal__close', timeout=10000)

        # Click the close button
        await page.click('span.commonModal__close')

        from_city = "Delhi"
        to_city = "Mumbai"

        await fill_route_fields(page , from_city , to_city)

        print("filled the route")

        # await page.wait_for_selector("div.flt_fsw_inputBox dates inactiveWidget")
        # await page.click('div.flt_fsw_inputBox dates inactiveWidget')


        # 🗓️ Set the date you want to click
        target_date = datetime(2025, 10, 10)
        aria_label_value = format_aria_label(target_date)

        print(f"🗓️ Target date: {aria_label_value}")

        # 🧠 Decide if this is today
        is_today = target_date.date() == datetime.today().date()
        print(f"🧠 Is today: {is_today}")

        # 🧬 Build the final class filter
        target_class = "DayPicker-Day--today" if is_today else "DayPicker-Day--selected"

        # 🧪 CSS Selector: must match both class and aria-label
        # selector = f'div.DayPicker-Day.{target_class}[aria-label="{aria_label_value}"]'
        selector = f'div[aria-label="{aria_label_value}"]'

        print(f"🧪 CSS Selector: {selector}")
        # 🕒 Wait for the element
        # await page.wait_for_selector(selector, timeout=5000)

        print("waiting for the element")

        # 🎯 Click it
        max_tries = 12  # Limit to prevent infinite loop
        tries = 0

        while tries < max_tries:
                element = await page.query_selector(selector)
                if element:
                    await element.dblclick()
                    print(f"✅ Double-clicked on: {aria_label_value}")
                    await asyncio.sleep(20)
                    return
                else:
                    print(f"🔍 Not found: {aria_label_value}, clicking 'Next Month'...")

                    # Click Next Month
                    next_btn = page.locator('span[aria-label="Next Month"]')
                    print(f"🔍 Next Month button: {next_btn}")
                    if not next_btn:
                        print("❌ 'Next Month' button not found.")
                        break

                    await next_btn.click()
                    await page.wait_for_timeout(1000)  # wait for calendar to update

                tries += 1

        print(f"🚫 Failed to find: {aria_label_value} after {max_tries} attempts.")


        # Optional: keep browser open for a few seconds to observe
        # await asyncio.sleep(20)

        await browser.close()



# Run the function
asyncio.run(open_makemytrip_and_close_modal())
