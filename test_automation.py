import asyncio
from auto_browser import AutoBrowser , run_automation

async def main():
    print("Starting automation...")

    task = "Go to example.com and click on the login button"

    await run_automation(task)

if __name__ == "__main__":
    asyncio.run(main())    
