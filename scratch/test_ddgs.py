from duckduckgo_search import DDGS
import json

def test():
    results = DDGS().images("edentulous panoramic radiograph", max_results=5)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    test()
