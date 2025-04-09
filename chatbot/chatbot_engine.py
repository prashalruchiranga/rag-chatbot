from langgraph.graph import MessagesState, StateGraph
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
import uuid


class ChatSession:
    def __init__(self, model, vector_store):
        self.model = model
        self.vector_store = vector_store
        self.memory = MemorySaver()
        self.retrieve = self._make_retrieve_tool(self.vector_store)
        self.graph = self._build_graph()
        self.thread_id = self._create_thread()


    def _create_thread(self):
        thread_id = str(uuid.uuid4())
        return thread_id


    def _make_retrieve_tool(self, vector_store):
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""
            retrieved_docs = vector_store.similarity_search(query, k=5)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        return retrieve
    

    def _query_or_respond(self, state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = self.model.bind_tools([self.retrieve])

        # Add explicit message to prefer using tools
        guidelines = [
            "You are a RAG chatbot. Your task is to respond to questions regarding the provided text.",
            "You can access the provided text using the 'retrive' tool.",
            "Always use the 'retrieve' tool unless the user is greeting.",
            "Don't respond to anything not related to the provided text."
            ]
        guidelines = " ".join(guidelines)
        init_message = SystemMessage(guidelines)
        # prepend the initialization message
        messages = [init_message] + state["messages"]
        
        response = llm_with_tools.invoke(messages)
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}
    

    def _generate(self, state: MessagesState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]


        # format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        guidelines = [
            "Use the following pieces of retrieved context to answer the question.",
            "If you don't know the answer, say that you don't know.",
            "Use 10 sentences maximum and keep the answer concise.",
            "If the user is asking for a list of items, present them as a numbered or bulleted list, as appropriate to the scenario."
            ]
        guidelines = " ".join(guidelines)
        system_message_content = f"{guidelines}\n\n{docs_content}"
        conversation_messages = [
            message for message in state["messages"]
            if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = self.model.invoke(prompt)
        return {"messages": [response]}


    def _build_graph(self):
        graph_builder = StateGraph(MessagesState)
        # Step 1: Generate an AIMessage that may include a tool-call to be sent
        graph_builder.add_node("query_or_respond", self._query_or_respond)
        # Step 2: Execute the retrieval.
        graph_builder.add_node("tools", ToolNode([self.retrieve])) 
        # Step 3: Generate a response using the retrieved content.
        graph_builder.add_node("generate", self._generate)

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond", tools_condition, {END: END, "tools": "tools"}
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)
        return graph_builder.compile(checkpointer=self.memory)
    
    
    def stream_values(self, message: str):
        for step in self.graph.stream(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode="values",
            config = {"configurable": {"thread_id": self.thread_id}}
        ):
            # step["messages"][-1].pretty_print()
            yield step["messages"][-1]

    
    def send_message(self, message: str):
        response =  self.graph.invoke(
             {"messages": [{"role": "user", "content": message}]},
             config={"configurable": {"thread_id": self.thread_id}}
        )
        return response["messages"][-1]

