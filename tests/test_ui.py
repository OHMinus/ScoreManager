import pytest
from playwright.sync_api import Page, expect

def test_home_page(page: Page):
    page.goto("http://localhost:5000/")
    expect(page.locator("text=Score Processor")).to_be_visible()

    btn_process = page.locator("#btn-process")
    expect(btn_process).to_be_visible()

    btn_scan = page.locator("#btn-scan")
    expect(btn_scan).to_be_visible()

    btn_login = page.locator("text=Login to Google Drive")
    expect(btn_login).to_be_visible()

    btn_search = page.locator("button:has-text('DBから検索')")
    expect(btn_search).to_be_visible()

    btn_list = page.locator("text=登録されているすべての楽譜一覧を見る")
    expect(btn_list).to_be_visible()

def test_search_page(page: Page):
    page.goto("http://localhost:5000/")

    page.locator("#keyword").fill("test")
    page.locator("button:has-text('DBから検索')").click()

    expect(page.locator("text=の検索結果")).to_be_visible()

def test_list_page(page: Page):
    page.goto("http://localhost:5000/list")
    expect(page.locator("text=登録済みの楽譜一覧")).to_be_visible()
