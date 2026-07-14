from bing_image_downloader import downloader

def test():
    downloader.download("edentulous panoramic x-ray", limit=10, output_dir='scratch/test_crawled', adult_filter_off=True, force_replace=True, timeout=60, verbose=True)

if __name__ == "__main__":
    test()
