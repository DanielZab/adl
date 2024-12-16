import serpapi

params = {
  "engine": "google_finance",
  "q": "GOOG:NASDAQ",
  "api_key": "",
  "window": "5D"
}

search = serpapi.search(params)
results = search.as_dict()
print(results)