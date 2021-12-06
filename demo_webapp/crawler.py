import time
import requests
import os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


configs = {
    "wait-time": 5,
    "browser": "chrome",
}


def create_browser(browser_type=configs["browser"]):
    if browser_type == "chrome":
        options = webdriver.ChromeOptions()
        options.add_argument('headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('disable-web-security')
        options.add_argument('allow-running-insecure-content')
        options.add_argument('disable-extensions')
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--ignore-ssl-errors')
        options.add_argument('--log-level=3')
        options.add_argument('--silent')
        browser = webdriver.Chrome(options=options)
    else:
        print("Browser is not supported!")
    return browser


def crawl_page(url, browser):
    print("Crawling %s..." % url)
    browser.get(url)
    try:
        title = browser.find_element(By.CSS_SELECTOR, "h1.title-detail").text
        see_more = WebDriverWait(browser, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR,
                                                 "a.txt_666"))
        )
        see_more = browser.find_elements(By.CSS_SELECTOR, "a.txt_666")
        for button in see_more:
            if button.text == "Xem thêm":
                browser.execute_script("arguments[0].click();", button)
                break
    except TimeoutException:
        raise TimeoutException
    return_comments = []
    first = True
    while True:
        if first:
            comments = WebDriverWait(browser, 10).until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, "full_content")))
            first = False
        else:
            time.sleep(5)
            comments = browser.find_elements(By.CLASS_NAME, "full_content")
        for comment in comments:
            try:
                username = comment.find_element(By.CLASS_NAME, "nickname").text
            except NoSuchElementException:
                continue
            comment_txt = comment.text[len(username):]
            comment_txt = comment_txt.replace("\n", " ")
            return_comments.append(comment_txt)
        try:
            next_page = browser.find_element(By.XPATH, "//a[contains(@class, 'btn-page next-page')]")
            if "disable" in next_page.get_attribute("class"):
                break
        except NoSuchElementException:
            break
        browser.execute_script("arguments[0].click();", next_page)
    return return_comments, title
