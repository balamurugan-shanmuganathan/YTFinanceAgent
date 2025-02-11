import streamlit as st
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

load_dotenv()

#model_name=Groq(id="llama-3.3-70b-versatile")
model_name=Gemini(id="gemini-2.0-flash-exp")

finance_agent = Agent(
    name="Finance Agent",
    model=model_name,
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    instructions=["Use table to display data."],
    show_tool_calls=True,
    markdowon=True,    
    debug_mode=True
)

web_agent = Agent(
    name="Web Agent",
    model=model_name,
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
)

agent_team=Agent(
    name="Agent Team",
    model=model_name,
    team=[web_agent,finance_agent],
    instructions=["Always include sources","Use table to display data."],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True

)

# Streamlit App
def main():
    st.set_page_config(page_title="StockIntel AI", page_icon=":bar_chart:", layout="wide")
    st.markdown("""
        <h1 style="text-align: center; color: #4CAF50;">StockIntel AI</h1>
        <p style="text-align: center; font-size: 20px; color: #777;">AI-powered assistant for analyzing your stocks.</p>
    """, unsafe_allow_html=True)

    user_input = st.text_input("Enter your query (e.g., 'Tell me about Apple stock')")
    submit = st.button("Get Stock Data")

    if submit and user_input.strip():
        # Extract the stock ticker using LLM
        result = agent_team.run(user_input)
        st.markdown(result.content, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
