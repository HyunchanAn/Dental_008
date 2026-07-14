from icrawler.builtin import GoogleImageCrawler

def test():
    google_crawler = GoogleImageCrawler(storage={'root_dir': 'scratch/test_icrawler'})
    google_crawler.crawl(keyword='edentulous panoramic radiograph', max_num=20)

if __name__ == "__main__":
    test()
