import streamlit as st
from api_service import send_message, get_conversation_messages
import io

st.set_page_config(page_title="Cal.com Scheduler Chatbot", page_icon="🤖", layout="centered")

# --- Welcome Message ---
st.title("Cal.com Scheduler Chatbot 🤖")
st.markdown("""
Welcome! This chatbot helps you manage your Cal.com meetings.\
Enter your email to start a new session. Your chat history will be saved for this session only.
""")

# --- User Email Input ---
if "user_email" not in st.session_state:
    st.session_state.user_email = st.text_input("Enter your email to start:")
    st.stop()
else:
    st.session_state.user_email = st.text_input("Enter your email to start:", value=st.session_state.user_email)
    if not st.session_state.user_email:
        st.stop()

# --- Conversation State ---
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Chat Controls ---
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🧹 Clear Conversation"):
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.experimental_rerun()
with col2:
    if st.session_state.messages:
        chat_text = io.StringIO()
        for msg in st.session_state.messages:
            role = "You" if msg.get("role") == "user" else "Bot"
            chat_text.write(f"{role}: {msg['content']}\n")
        st.download_button("⬇️ Export Chat History", chat_text.getvalue(), file_name="chat_history.txt")

# --- Chat History Display ---
st.markdown("---")
st.subheader("Chat History")
if st.session_state.conversation_id:
    history = get_conversation_messages(st.session_state.conversation_id)
    if isinstance(history, list):
        st.session_state.messages = history
    elif isinstance(history, dict) and "error" in history:
        st.warning(f"Could not load conversation: {history['error']}")

for msg in st.session_state.messages:
    if msg.get("role") == "user":
        st.markdown(f"<div style='text-align: right; color: #2563eb;'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left; color: #059669;'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)

st.markdown("---")

# --- Message Input ---
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message:", key="user_input")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    with st.spinner("Bot is thinking..."):
        response = send_message(user_input, st.session_state.user_email, st.session_state.conversation_id)
        if "error" in response:
            st.error(f"Error: {response['error']}")
        else:
            st.session_state.conversation_id = response.get("conversation_id", st.session_state.conversation_id)
            # Add user and bot messages to local state
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response.get("response", "")})
            st.experimental_rerun()

# --- Footer ---
st.markdown("<hr style='margin-top:2em;'>", unsafe_allow_html=True)
st.caption("Built with Streamlit • Cal.com Scheduler Agent • Session chat only (refresh to reset)") 