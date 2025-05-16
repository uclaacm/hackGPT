import os
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.types import Command
from typing import Literal
from langgraph.graph import MessagesState, StateGraph, START, END


# create title and description
st.title("Startup Evaluator")
st.write("Enter your startup/business idea below to view insights.")

# create text input to enter idea
user_prompt = st.text_input("Enter your business or startup idea:", "")
submit = st.button("Submit", disabled=(not user_prompt))

# create sidebar for API keys
st.sidebar.header("API Keys")
openai_api_key = st.sidebar.text_input("Google API Key", type="password")
tavily_api_key = st.sidebar.text_input("Tavily API Key", type="password")

# takes the keys the user typed in and sets them as environment variables
# necessary when using langchain and tavily because they read keys from the environment variables
if openai_api_key:
    os.environ["GOOGLE_API_KEY"] = openai_api_key
if tavily_api_key:
    os.environ["TAVILY_API_KEY"] = tavily_api_key

# check if API keys are set, if not, show warning message and stop execution
if not openai_api_key or not tavily_api_key:
    st.warning("Please enter your API keys in the sidebar to continue.")
    st.stop()

# initialize models
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17")
tavily_tool = TavilySearchResults(max_results=5)

# create common system prompt for both agents
def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )

# create agents
research_agent = create_react_agent(
    llm,
    tools=[tavily_tool],
    prompt=make_system_prompt(
        "You are a research assistant helping evaluate startup ideas."
        " Your job is to gather relevant market data, consumer trends, and competitor insights from the past year."
        " You must not make recommendations or evaluations—leave that to your colleague."
        " Use the available tools to collect data that your advisor teammate can use to assess viability."
    ),
)


advisor_agent = create_react_agent(
    llm,
    tools=[],
    prompt=make_system_prompt(
        "You are a startup advisor tasked with evaluating the viability of business ideas."
        " Base your assessment solely on the research data provided by your researcher colleague."
        " Your job is to deliver a clear recommendation: pursue or do not pursue the startup idea."
        " Support your decision with market size, competitive landscape, consumer demand, and any other relevant quantitative or qualitative data."
        " If an idea makes no sense logically then say the idea is not viable. DO NOT BE OVERLY OPTIMISTIC."
    ),
)

def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        return END
    return goto

# Define functionresearch_node that takes in the current state (a dictionary with message history)
# Returns Command that either sends us to the "startup_advisor" node or the END node
def research_node(state: MessagesState) -> Command[Literal["startup_advisor", END]]:    # type: ignore
    
    # Call research agent with current state and store result. The result will contain updated messages
    result = research_agent.invoke(state)
    
    # Determine which node to go to next based on the content of the last message in the result
    goto = get_next_node(result["messages"][-1], "startup_advisor")
    
    # Rename the author of the last message to "researcher" by wrapping it in a new HumanMessage
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content,  # Keep the message content the same
        name="researcher"  # Set the name to "researcher" for traceability in the conversation
    )
    
    # Return a Command object which updates the messages in state and tells the graph which node to go to
    return Command(
        update={"messages": result["messages"]},  # Update the state with the modified message list
        goto=goto,  # Set the next node to transition to
    )

# same as research_node but for the advisor agent
def advisor_node(state: MessagesState) -> Command[Literal["researcher", END]]:  # type: ignore
    result = advisor_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "researcher")
    result["messages"][-1] = HumanMessage(content=result["messages"][-1].content, name="startup_advisor")
    return Command(
        update ={ "messages": result["messages"]},
        goto = goto,
    )

# initialize a LangGraph StateGraph with MessagesState as the shared state for all nodes
workflow = StateGraph(MessagesState)
# add "researcher" node to the graph, which executes the research_node function
workflow.add_node("researcher", research_node)
# add "startup_advisor" node to the graph, which executes the advisor_node function
workflow.add_node("startup_advisor", advisor_node)
# define the starting point of the graph to be the "researcher" node
workflow.add_edge(START, "researcher")

# compile the graph into an executable version that can be invoked
graph = workflow.compile()


# initialize data variable to store the final answer
data = None
flow = []

# run the graph if user submits
if submit:
    # create a temporary UI placeholder to show status while processing
    status_placeholder = st.empty()
    status_placeholder.info("Processing your request...")

    events = graph.stream(
        {"messages": [("user", user_prompt)]},
        {"recursion_limit": 150},
    )
    
    # iterate through the events produced by the graph
    for event in events:
        if event.get("startup_advisor", {}) or event.get("researcher", {}):
            if event.get("startup_advisor", {}):
                data = event["startup_advisor"]
                flow.append("Advisor")
            elif event.get("researcher", {}):
                data = event["researcher"]
                flow.append("Researcher")

    # clear the status placeholder
    status_placeholder.empty()

    # if we have a final answer, extract it from the last message and display it
    if data:
        final_answer = data["messages"][-1].content.strip()
        st.subheader("Final Answer")
        st.markdown(final_answer)
    else:
        st.write("No final answer was produced.")

# display the flow of the conversation
with st.expander("Response Flow", expanded=True):
    depiction = ""
    for i, step in enumerate(flow):
        if i < len(flow)-1:
            depiction += step + " -> "
        else:
            depiction += step
            st.write(depiction)