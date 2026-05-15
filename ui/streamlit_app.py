import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation import repondre

st.set_page_config(page_title="KOVA IA", page_icon="✦", layout="centered")


def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


ui_dir = os.path.dirname(os.path.abspath(__file__))
css_path = os.path.join(ui_dir, "style.css")
local_css(css_path)

load_dotenv()

st.markdown(
    """
    <div class="fixed-header">
        <div class="header-content">
            <h1 class="main-title">KOVA <span class="accent">IA</span></h1>
            <p class="sub-title">Assistant d'orientation client expert</p>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="chat-spacer-top"></div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

st.markdown('<div class="chat-spacer-bottom"></div>', unsafe_allow_html=True)

if prompt := st.chat_input("Demander à Kova..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("KOVA réfléchit..."):
            try:
                # Construire l'historique LangChain depuis la session Streamlit
                
                historique = [
                    (
                        HumanMessage(content=m["content"])
                        if m["role"] == "user"
                        else AIMessage(content=m["content"])
                    )
                    for m in st.session_state.messages[:-1]
                ]

                resultat = repondre(prompt, historique=historique)
                response = resultat.get("reponse", "")
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

                if resultat.get("nb_sources", 0):
                    with st.expander("📚 Sources utilisées"):
                        sources_uniques = {}
                        for source in resultat.get("sources", []):
                            fichier_complet = source.get("fichier", "Inconnu")
                            nom_fichier = os.path.basename(fichier_complet)

                            if nom_fichier not in sources_uniques:
                                sources_uniques[nom_fichier] = source

                        for nom_fichier, source in sources_uniques.items():
                            score = source.get("score", 0)

                            if score >= 70:
                                score_color = "🟢"
                                confidence_label = "Très pertinent"
                            elif score >= 40:
                                score_color = "🟡"
                                confidence_label = "Pertinent"
                            else:
                                score_color = "🔴"
                                confidence_label = "Peu pertinent"

                            st.markdown(
                                f"{score_color} **{nom_fichier}**\n"
                                f"- Confiance: {score}% ({confidence_label})"
                            )

            except Exception as e:
                st.error(f"Erreur : {e}")
