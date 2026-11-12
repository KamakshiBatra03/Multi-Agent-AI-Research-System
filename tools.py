from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchResults
from rich import print
import os
from dotenv import load_dotenv

load_dotenv()

@tool
def web_search(query:str)->str:
    """search the web for recent and reliable information on a topic. returns titles, URLs and snippets."""
    search=DuckDuckGoSearchResults(max_results=5,output_format="list")
    result=search.invoke(query)
    output=[]
    for r in result:
        output.append(
            f"""
                title:{r.get('title')}
                URLs:{r.get('link')}
                snippet:{r.get('snippet')}
                """
        )
    return "\n---\n".join(output)
#print(web_search.invoke("what is the latest news about war"))

  
@tool
def web_scraper(url:str)->str:
    """scrape and return clean text content from the given URL for deeper reading"""
    try:
        resp=requests.get(url,timeout=10,headers={"User-Agent":"Mozilla/5.0"})
        soup=BeautifulSoup(resp.text,"html.parser")
        for tag in soup(["script","style","nav","footer"]):
            tag.decompose()
        return soup.get_text(separator=" ",strip=True)[:3000]
    except Exception as e:
        return f"could not scrape URL : {str(e)}"

print(web_scraper.invoke("https://www.hindustantimes.com/photos/cricket/ind-vs-nz-2nd-odi-gurnoor-brar-stars-with-four-wicket-haul-as-india-edge-new-zealand-in-rain-hit-match-action-in-images-101794038032182.html"))
