from agents import build_search_agent,build_reader_agent,writer_chain,critic_chain

def caller_func(topic:str)->dict:
    state={}

    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    search_agent=build_search_agent()
    search_result=search_agent.invoke({
        "messages":[("user", f"""
            Search for recent and reliable information about {topic}.

            IMPORTANT:
            Return at least 5 search results.

            For EACH result provide:
            1. Title
            2. Actual URL (complete https:// link)
            3. Snippet

            Do not provide only a summary.
            Do not omit URLs.
            """)]
    })

    #storing result in state so that it can be used by next func
    state["search_results"]=search_result["messages"][-1].content #-1 bec last ai msg is only needed as and

    print("\n search result ",state['search_results'])

    #step 2 - reader agent 
    print("\n"+" ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    reader_agent=build_reader_agent()
    reader_result=reader_agent.invoke({"messages":[("user", f"""Based on the following search results about {topic},
         pick the most relevant URL and scrape it for deeper content.
         Search results:
         {state['search_results'][:800]}
        Instructions:
            1. Extract the actual URLs from the search results.
            2. Select the most relevant and reliable URL.
            3. Call the web_scraper tool using that URL.
            4. Return the scraped content.
            5. Do not ask the user for a URL.""")]
    })

    state["scraped_result"]=reader_result["messages"][-1].content

    print("\nscraped content: \n", state['scraped_result'])

    #step 3 - writer chain 

    print("\n"+" ="*50)
    print("step 3 - Writer is drafting the report ...")
    print("="*50)

    research_combined=(
       f"SEARCH RESULTS:\n{state['search_results']}\n\n"
       f"DETAILED SCRAPED RESULTS:\n{state['scraped_result']}"
    )

    state["report"]=writer_chain.invoke({
        "topic":topic,
        "research":research_combined
    })

    print("\n Final Report\n",state['report'])
    #critic report 

    print("\n"+" ="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)

    state["feedback"]=critic_chain.invoke({
        "report":state["report"]
    })

    print("\n",state["feedback"])

    return state

if __name__=="__main__":
    topic=input("enter research topic")
    caller_func(topic)