import asyncio
from datetime import datetime
from playwright.async_api import async_playwright

async def open_booking_com():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        # ✅ Inject zoom styling into all pages opened in this context
        await context.add_init_script("""
            document.addEventListener('DOMContentLoaded', () => {
                document.body.style.transform = 'scale(0.8)';
                document.body.style.transformOrigin = '0 0';
            });
        """)
        page = await context.new_page()

        await page.goto("https://www.booking.com")

        await page.wait_for_selector("#flights")
        await page.click("#flights")

        await asyncio.sleep(3)
        await page.click('[data-ui-name="input_location_from_segment_0"]');

        try:
            span_selector = 'span.Icon-module__root___GDSyQ Icon-module__root--size-small___PDjre Icon-module__root--scale___SID92 svg'
            span = page.locator(span_selector)
            if await span.is_visible():
                print("in the if")
                await span.click()
                print("Clicked optional span.")
            else:
                print("Span not visible, skipping click.")
        except Exception as e:
            print("Error checking/clicking span:", e)


        # Fill the autocomplete input
        await asyncio.sleep(3)
        # input_selector = '[data-ui-name="input_text_autocomplete"]'
        # await page.wait_for_selector(input_selector)
        # await page.fill(input_selector, "Bengaluru")

        # # Trigger suggestion and select it
        # await asyncio.sleep(3)
        # await page.keyboard.press("ArrowDown")
        # await page.keyboard.press("Enter")

        # here for the destination airport
        await page.wait_for_selector('[data-ui-name="input_location_to_segment_0"]')
        await page.click('[data-ui-name="input_location_to_segment_0"]')
        print("🔍 Clicked on 'Where to?' span (XPath).")
        await page.fill('[data-ui-name="input_text_autocomplete"]', "Bengaluru")
        # await page.fill(output_selector,"Bengaluru")

        await asyncio.sleep(3)
        await page.keyboard.press("ArrowDown")
        await page.keyboard.press("Enter")

        await asyncio.sleep(3)

        await page.wait_for_selector('[data-ui-name="button_date_segment_0"]')
        await page.click('[data-ui-name="button_date_segment_0"]')
        print("✅ Opened date picker")

        await asyncio.sleep(3)

        # Format the target date
        target_date = "29 June 2025"
        aria_label = f'span[aria-label="{target_date}"]'

        # Wait and click the date
        await asyncio.sleep(3)
        from_date = page.locator(aria_label)
        await from_date.dblclick()
        print(f"📅 Selected date: {target_date}")

        await asyncio.sleep(3)

        # click on the search button
        await page.wait_for_selector('[data-ui-name="button_search_submit"]')
        await page.click('[data-ui-name="button_search_submit"]')

        await asyncio.sleep(15)

        # view details of the flight
        await page.wait_for_selector('button[aria-label="View details "]')
        await page.click('button[aria-label="View details "]')

        await asyncio.sleep(10)

        # Selector for the "Select" button inside a <span>
        select_button_selector = 'span.Button-module__text___9rBFs:text("Select")'

        # Try to find the "Select" button with scrolling
        max_scroll_attempts = 20
        scroll_attempt = 0
        select_element = None

        while scroll_attempt < max_scroll_attempts:
            select_element = await page.query_selector(select_button_selector)

            if select_element:
                print(f"✅ Found 'Select' button after {scroll_attempt} scrolls.")
                await select_element.click()
                print("✅ Clicked 'Select' button.")
                break
            else:
                print(f"🔍 'Select' button not found. Scrolling attempt {scroll_attempt + 1}...")
                await page.evaluate("window.scrollBy(0, window.innerHeight)")
                await page.wait_for_timeout(1000)  # wait a bit for content to load
                scroll_attempt += 1

        if not select_element:
            print("❌ Failed to find the 'Select' button after scrolling.")

        await asyncio.sleep(6)
        # print(await page.content())
        # locator1 = page.locator('div.Grid-module__column___5SxWE.Grid-module__column--size-6___FmzOC')
        # outer_html = await locator1.evaluate("el => el.outerHTML")
        # print("==== Element Outer HTML ====")
        # print(outer_html)
        await page.locator('span.Button-module__text___9rBFs').nth(0).click()
        print(await page.locator('span.Button-module__text___9rBFs').count())

        await asyncio.sleep(3) 

        # Fill First Name
        await page.click('input[name="passengers.0.firstName"]')
        await page.fill('input[name="passengers.0.firstName"]', 'John')
        await asyncio.sleep(3)

        # Fill Last Name
        await page.click('input[name="passengers.0.lastName"]')
        await page.fill('input[name="passengers.0.lastName"]', 'Doe')
        await asyncio.sleep(3)

        await page.evaluate("window.scrollBy(0, window.innerHeight)")

        await page.select_option('select[name="passengers.0.gender"]', value="male")
        await asyncio.sleep(3)

        await page.select_option('select[name="passengers.0.birthDate"]', label="December")
        await asyncio.sleep(3)

        await page.click('input[aria-label="Enter your birth date using two digits"]')
        await page.fill('input[aria-label="Enter your birth date using two digits"]' , "6")
        await asyncio.sleep(3)

        await page.click('input[placeholder="YYYY"]')
        await page.fill('input[placeholder="YYYY"]', '2000')
        await asyncio.sleep(3)

        done = 'span.Button-module__text___9rBFs:text("Done")'
        doneButton = await page.query_selector(done)
        await doneButton.click()

        await page.click('input[name="booker.email"]')
        await page.fill('input[name="booker.email"]' , 'abc123@gmail.com')
        await asyncio.sleep(3)

        await page.click('input[name="number"]')
        await page.fill('input[name="number"]' , '6392672698')
        await asyncio.sleep(3)

        next = 'span.Button-module__text___9rBFs:text("Next")'
        nextButton = await page.query_selector(next)
        await nextButton.click()

        await asyncio.sleep(10)
        skip = 'span.Button-module__text___9rBFs:text("Skip")'
        skipButton = await page.query_selector(skip)
        await skipButton.click()



        print("🔵 Press ENTER to close the browser.")
        await asyncio.get_event_loop().run_in_executor(None, input)

        await browser.close()

asyncio.run(open_booking_com())