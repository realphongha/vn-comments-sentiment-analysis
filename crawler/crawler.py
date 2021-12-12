import requests
import os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


configs = {
    "wait-time": 5,
    "browser": "chrome",
    "limit": 1000,
    "save_path": "results/",
}


def create_browser(type):
    if type == "chrome":
        options = webdriver.ChromeOptions()
        options.add_argument('headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        # options.add_argument('disable-gpu')
        # options.add_argument('window-size=800,600')
        options.add_argument('disable-web-security')
        options.add_argument('allow-running-insecure-content')
        options.add_argument('disable-extensions')
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--ignore-ssl-errors')
        # options.add_argument('user-data-dir=%s' % configs["user_data_dir"])
        options.add_argument('--log-level=3')
        options.add_argument('--silent')
        browser = webdriver.Chrome(options=options)
    else:
        print("Browser is not supported!")
    return browser


class Enough(Exception):
    pass


def crawl_list(url, page_number=1):
    page = requests.get(url % page_number)
    if page.status_code != 200:
        print("Cannot get list page!")
        return
    soup = BeautifulSoup(page.text, 'html.parser')
    indices = soup.select("div.width_common.list-news-subfolder")
    pre_limit = limit
    # print(indices)
    for index in indices:
        articles = index.select("article.item-news.item-news-common")
        # print(articles)
        for article in articles:
            try:
                article_url = article.select("a")[0].get("href")
                crawl_page(article_url)
            except Enough:
                raise Enough()
            except IndexError:
                continue
            except Exception as e:
                print("Failed!")
                print("Exception:", e)
    if limit != pre_limit:
        crawl_list(url, page_number+1)
    print("Completed!")


def crawl_page(url):
    global browser, limit, result
    print("Crawling %s..." % url)
    browser.get(url)
    try:
        comments = WebDriverWait(browser, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, 
                "full_content"))
        )
        see_more = browser.find_elements(By.CSS_SELECTOR, "a.txt_666")
        for button in see_more:
            if button.text == "Xem thêm":
                # button.click()
                browser.execute_script("arguments[0].click();", button)
                comments = WebDriverWait(browser, 10).until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, 
                        "full_content"))
                )
                break
    except TimeoutException:
        return
    for comment in comments:
        username = comment.find_element(By.CLASS_NAME, "nickname").text
        comment_txt = comment.text[len(username):]
        comment_txt = comment_txt.replace("\n", " ")
        result.write(comment_txt + "\n")
        limit -= 1
        print("Crawled %d/%d comment(s)..." % 
            (configs["limit"]-limit, configs["limit"]))
        if limit == 0:
            raise Enough()


def main():
    global browser, limit, result
    print("Start crawling with config:", configs)
    browser = create_browser(configs["browser"])
    categories =   {'thoi-su': 'https://vnexpress.net/thoi-su-p%d',  
                    'the-gioi': 'https://vnexpress.net/the-gioi-p%d', 
                    'kinh-doanh': 'https://vnexpress.net/kinh-doanh-p%d', 
                    'giai-tri': 'https://vnexpress.net/giai-tri-p%d', 
                    'the-thao': 'https://vnexpress.net/the-thao-p%d', 
                    'phap-luat': 'https://vnexpress.net/phap-luat-p%d', 
                    'giao-duc': 'https://vnexpress.net/giao-duc-p%d', 
                    'suc-khoe': 'https://vnexpress.net/suc-khoe-p%d', 
                    'doi-song': 'https://vnexpress.net/doi-song-p%d', 
                    'du-lich': 'https://vnexpress.net/du-lich-p%d', 
                    'khoa-hoc': 'https://vnexpress.net/khoa-hoc-p%d', 
                    'so-hoa': 'https://vnexpress.net/so-hoa-p%d', 
                    'oto-xe-may': 'https://vnexpress.net/oto-xe-may-p%d', 
                    'y-kien': 'https://vnexpress.net/y-kien-p%d',  
                    'hai': 'https://vnexpress.net/hai-p%d',}
    for cat in categories:
        print("Start crawl '%s'" % cat)
        limit = configs["limit"]
        result = open(os.path.join(configs["save_path"], cat + ".txt"), 
            "w", encoding="utf8")
        try:
            crawl_list(categories[cat], page_number=1)
        except Enough:
            print("Completed!")
        except Exception as e:
            print("Failed to get %s!" % cat)
            print("Exception:", e)
        finally:
            result.close()


if __name__ == "__main__":
    main()