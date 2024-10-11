from langchain_groq import ChatGroq
#--------LLM
GROQ_API_KEY = "<API>"
llm = ChatGroq(model = "mixtral-8x7b-32768",
        temperature = 0,
        max_tokens = 100,
        max_retries = 2,
        api_key = GROQ_API_KEY
        )

# Defining the tools that our LLM can use

def multiply(a: float, b: float) -> float:
  """
  Multiply a and b.
  Args:
      a: first number
      b: second number
  """
  return a*b

def add(a: int, b: int) -> float:
  """
  Add a and b.
  Args:
      a: float
      b: float
  """
  return a+b

def divide(a: float, b: float):
  """
  Divide a by b.
  Args:
      a : float
      b: float
  """
  return a/b

tools = [add, multiply, divide]
llm_with_tools = llm.bind_tools(tools= tools, parallel_tool_calls = False)

# -------Defining MessagesState ---------
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# --- System messages----
sys_message = SystemMessage(content = "You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node (A function)
def assistant(state: MessagesState):
  return {"messages": [llm_with_tools.invoke([sys_message] + state["messages"])]}


#------- Graph Building-------

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode

builder = StateGraph(MessagesState)

#---Defining the node, These are the functions.

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools = tools))

#---Define Edges: MessagesState changes when it will passes through them.
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # Assistannt ---> Tools, When tool is called.
    #Or, Assistant --> End
    tools_condition,
)

builder.add_edge("tools", "assistant")
react_graph = builder.compile()


# ---- Associating Memory to our agent
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

# This agent will remember the past.--->
react_graph_memory = builder.compile(checkpointer = memory)

# To use memory we need to specify thread_id
# Thread_id wikk store collection of graph State.
config_past = {"configurable": {"thread_id": "2"}}


# #------To View The Graph
# from IPython.display import Image, display
# display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))

#-----Real Time conversation.
while True:
  query = input("You: ")
  if query.lower() == "exit":
    break
  result = react_graph_memory.invoke({"messages": query}, config_past)
  response = result["messages"][-1].content
  print(response)


