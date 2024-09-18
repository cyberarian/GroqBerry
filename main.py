import streamlit as st
import requests
from pocketgroq import GroqProvider
import base64

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'available_models' not in st.session_state:
    st.session_state.available_models = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "llama2-70b-4096"  # Default model

def get_groq_provider():
    api_key = st.secrets["GROQ_API_KEY"]
    return GroqProvider(api_key=api_key)

def fetch_available_models():
    api_key = st.secrets["GROQ_API_KEY"]
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        models_data = response.json()
        st.session_state.available_models = [model['id'] for model in models_data['data']]
        if st.session_state.selected_model not in st.session_state.available_models:
            st.session_state.selected_model = st.session_state.available_models[0]
    except requests.RequestException as e:
        st.error(f"Error fetching models: {str(e)}")

def generate_response(prompt: str, use_cot: bool, model: str) -> str:
    groq = get_groq_provider()
    
    # Include chat history in the prompt
    history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
    full_prompt = f"{history}\nUser: {prompt}"
    
    if use_cot:
        cot_prompt = f"You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.\n\n{full_prompt}\n\nSolution:"
        return groq.generate(cot_prompt, max_tokens=1000, temperature=0, model=model)
    else:
        return groq.generate(full_prompt, temperature=0, model=model)

def on_model_change():
    st.session_state.selected_model = st.session_state.model_selectbox

def clear_chat():
    st.session_state.messages = []

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide", page_title="Strawberry Groq DEMO")
    
    # Set background image
    set_png_as_page_bg('groqberry.jpg')
    
    # Main container styling
    st.markdown(
        """
        <style>
        .main-container {
            background-color: rgba(255, 255, 255, 0.5);
            padding: 2rem;
            border-radius: 10px;
        }
        .main-container * {
            color: black !important;
            font-weight: bold !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Main container
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.title("Strawberry Groq DEMO")
        st.write("This is a simple demo of the PocketGroq library's new 'Chain of Thought' functionality.")
        st.write("<a href='https://github.com/jgravelle/pocketgroq'>https://github.com/jgravelle/pocketgroq</a> | <a href='https://www.youtube.com/watch?v=S5dY0DG-q-U'>https://www.youtube.com/watch?v=S5dY0DG-q-U</a>", unsafe_allow_html=True)

        # Fetch available models on startup
        if not st.session_state.available_models:
            fetch_available_models()

        # Model selection dropdown at the top
        if st.session_state.available_models:
            st.selectbox(
                "Select a model:", 
                st.session_state.available_models, 
                index=st.session_state.available_models.index(st.session_state.selected_model),
                key="model_selectbox",
                on_change=on_model_change
            )
        
        # Create three tabs
        tab1, tab2, tab3 = st.tabs(["Chat", "Settings", "About"])
        
        with tab1:
            # Google-like search box
            prompt = st.text_input("Ask me anything:", key="chat_input")
            
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("Send"):
                    if prompt:
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        with st.chat_message("user"):
                            st.write(prompt)
                        
                        with st.chat_message("assistant"):
                            response = generate_response(prompt, st.session_state.use_cot, st.session_state.selected_model)
                            st.write(response)
                        
                        st.session_state.messages.append({"role": "assistant", "content": response})
            
            with col2:
                if st.button("Clear Chat"):
                    clear_chat()
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
        with tab2:
            # CoT toggle
            st.session_state.use_cot = st.checkbox("Use Chain of Thought")
        
        with tab3:
            st.write("About this demo:")
            st.write("This is a Streamlit-based demo showcasing the capabilities of the PocketGroq library.")
            st.write("It allows you to interact with various Groq models and experiment with Chain of Thought reasoning.")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()