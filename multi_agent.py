import operator
from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

class Route(BaseModel):
    next: Literal["LocalGuideAgent", "LogisticsAgent", "BudgetAgent", "FINISH"] = Field(
        description="The next agent to route to, or FINISH if the user's request is fully answered."
    )

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def build_multi_agent_graph(llm, tools_dict, checkpointer):
    """
    Builds a Hierarchical Multi-Agent System using LangGraph.
    tools_dict expects: local_guide_tools, logistics_tools, budget_tools, planner_tools
    """
    
    # 1. Create Sub-Agents
    local_guide_agent = create_react_agent(
        llm, 
        tools=tools_dict.get("local_guide_tools", []), 
        state_modifier="You are a Local Guide Agent. Use your tools to find top attractions and search the travel knowledge base."
    )
    
    logistics_agent = create_react_agent(
        llm, 
        tools=tools_dict.get("logistics_tools", []),
        state_modifier="You are a Logistics Agent. Your job is to fetch the current weather and generate map coordinates for requested places."
    )
    
    budget_agent = create_react_agent(
        llm, 
        tools=tools_dict.get("budget_tools", []),
        state_modifier="You are a Budget Agent. Your job is to search the web for cheap flights, hotel deals, and budget tips."
    )
    
    planner_synthesizer = create_react_agent(
        llm,
        tools=tools_dict.get("planner_tools", []),
        state_modifier=(
            "You are Maya, the Chief Planner Agent. Synthesize the findings from the sub-agents into a highly-detailed, beautifully formatted day-by-day itinerary. "
            "IMPORTANT: If the user stated a preference, use the SavePreferenceTool to record it in Semantic Memory. "
            "You MUST present real image URLs retrieved by the budget agent (web search) side by side using HTML as per the critical image instructions."
        )
    )

    # 2. Define Node Functions
    def local_guide_node(state: AgentState):
        result = local_guide_agent.invoke({"messages": state["messages"]})
        last_message = result["messages"][-1].content
        return {"messages": [AIMessage(content=f"LocalGuide Findings: {last_message}", name="LocalGuideAgent")]}
        
    def logistics_node(state: AgentState):
        result = logistics_agent.invoke({"messages": state["messages"]})
        last_message = result["messages"][-1].content
        return {"messages": [AIMessage(content=f"Logistics Findings: {last_message}", name="LogisticsAgent")]}
        
    def budget_node(state: AgentState):
        result = budget_agent.invoke({"messages": state["messages"]})
        last_message = result["messages"][-1].content
        return {"messages": [AIMessage(content=f"Budget Findings: {last_message}", name="BudgetAgent")]}

    # 3. Define the Supervisor (Planner Node)
    system_prompt = (
        "You are the Supervisor Planner Agent coordinating a travel itinerary.\n"
        "You have access to the following sub-agents:\n"
        "- LocalGuideAgent: For finding top attractions and curated knowledge base info.\n"
        "- LogisticsAgent: For weather updates and location maps.\n"
        "- BudgetAgent: For web search, cost analysis, flights, hotels, and finding real image URLs.\n\n"
        "Read the conversation below and decide who should act next. "
        "If you need to search the web, route to BudgetAgent. If you need attractions, route to LocalGuideAgent. "
        "If you have gathered enough information to fully satisfy the user's request, respond with FINISH."
    )
    
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Given the conversation above, who should act next? Or should we FINISH?")
    ])
    
    supervisor_chain = supervisor_prompt | llm.with_structured_output(Route)
    
    def planner_node(state: AgentState):
        # 1. Decide routing
        route = supervisor_chain.invoke({"messages": state["messages"]})
        
        # 2. If FINISH, synthesize the final response
        if route.next == "FINISH":
            final_out = planner_synthesizer.invoke({"messages": state["messages"]})
            final_message = final_out["messages"][-1].content
            return {"messages": [AIMessage(content=final_message, name="Planner")], "next": "FINISH"}
        
        # 3. Otherwise, just route to the sub-agent
        return {"next": route.next}

    # 4. Construct StateGraph
    workflow = StateGraph(AgentState)
    
    workflow.add_node("Planner", planner_node)
    workflow.add_node("LocalGuideAgent", local_guide_node)
    workflow.add_node("LogisticsAgent", logistics_node)
    workflow.add_node("BudgetAgent", budget_node)
    
    # Sub-agents always report back to Planner
    workflow.add_edge("LocalGuideAgent", "Planner")
    workflow.add_edge("LogisticsAgent", "Planner")
    workflow.add_edge("BudgetAgent", "Planner")
    
    # Planner decides where to go
    workflow.add_conditional_edges(
        "Planner",
        lambda x: x["next"],
        {
            "LocalGuideAgent": "LocalGuideAgent",
            "LogisticsAgent": "LogisticsAgent",
            "BudgetAgent": "BudgetAgent",
            "FINISH": END
        }
    )
    
    workflow.add_edge(START, "Planner")
    
    # Compile the graph
    return workflow.compile(checkpointer=checkpointer)
