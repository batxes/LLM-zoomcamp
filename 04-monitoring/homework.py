import pandas as pd

github_url = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-monitoring/data/results-gpt4o-mini.csv"

url = f'{github_url}?raw=1'
df = pd.read_csv(url)

