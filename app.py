import streamlit as st
from gtts import gTTS
from typing import List
from langchain.llms import OpenAI
import langchain_community
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import (
    tool,
    AgentExecutor,
    OpenAIFunctionsAgent,
)
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun


# Setup the Streamlit page
st.set_page_config(page_title="PhramaConnect", page_icon="🤖", layout="wide")
st.header("🤖 PhramaConnect : Medicines made simple!", divider="rainbow")

OPENAI_API_KEY = "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"
resultant_response = ""


# Define a function to call DuckDuckGo search engine with a query
@tool
def duckduckgo_search(query: str):
    """Call DuckDuckGo search engine with a query on medicines and medication."""
    return DuckDuckGoSearchRun().run(query)


# Setup the first chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Get the input
text_input = st.chat_input("Type the name of the medication")
if text_input:
    st.session_state.messages.append({"role": "user", "content": text_input})

    # Initialize the input prompts
    system_message = SystemMessage(
        content="You are a helpful pharmaceutical cum medical assistant that provides accurate and reliable information about medicines from trusted sources. You clearly explain the up-to-date information of medicines that the user asks about. Give detailed answers with visually-appealing titles and section divisions. Use these websites for more details on medications: Centers for Disease Control and Prevention, ClinicalTrials.gov, Food and Drug Administration, National Cancer Institute, National Institutes of Health, National Library of Medicine, World Health Organization, Indian Medical Association, VoxHealth, Truven Health Analytics, Harvard Medical School, https://www.drugs.com/fda-consumer, https://www.webmd.com/drugs/2/index, https://medlineplus.gov/"
    )
    human_message = HumanMessage(
        content="Give a detailed description about the name, brand names, active ingredients, uses, side-effects, adverse reactions, potential drug interactions with other medications, foods, or substances, precautions and dosage forms of the medicine: "
        + text_input
    )

    st.chat_message("user").write(text_input)

    # Setup closed-source OpenAI API. You can replace this with any other chat-based LLM
    llm = ChatOpenAI(
        temperature=0,
        streaming=True,
        openai_api_key=OPENAI_API_KEY,
    )

    tools = [
        duckduckgo_search,
    ]
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

    # Initialize the response agent with OpenAI
    response_agent = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=True,
    )

    # Initialize the text to speech agent
    output_filename = "Output_Audio.wav"

    # Setup the chat for response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        with st.spinner("Loading... Please wait"):
            resultant_response = response_agent.run(
                st.session_state.messages, callbacks=[st_cb]
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": resultant_response}
            )
        st.write(resultant_response)

        # Convert the result into a speech
        with st.spinner("Generating voice output... Please wait"):
            speech = gTTS(text=resultant_response, lang="en", slow=False)
            speech_response = speech.save(output_filename)
            st.session_state.messages.append(
                {"role": "assistant", "content": speech_response}
            )
        st.audio(output_filename)

        # Convert the result into a document
        with st.spinner("Generating document... Please wait"):
            st.download_button(
                label="Download Conversation as a TXT file",
                data=str(st.session_state.messages),
            )
