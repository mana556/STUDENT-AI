import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

NOTES_PATH = Path("notes_agent_tp.txt")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
BASE_URL = os.getenv("GROQ_API_BASE", "https://api.groq.com/v1")


@tool
def calculer_moyenne(notes_json: str) -> str:
    """Calcule la moyenne, la note minimale et la note maximale à partir d'une liste JSON de notes."""
    try:
        notes = json.loads(notes_json)
        if not isinstance(notes, list) or not notes:
            return "Erreur : fournissez une liste JSON non vide, par exemple [12, 15, 9]."

        notes = [float(note) for note in notes]
        moyenne = sum(notes) / len(notes)
        minimum = min(notes)
        maximum = max(notes)

        return (
            f"Moyenne : {moyenne:.2f}\n"
            f"Minimum : {minimum:.2f}\n"
            f"Maximum : {maximum:.2f}"
        )
    except Exception as exc:
        return f"Erreur pendant le calcul : {exc}"


@tool
def chercher_dans_les_notes(mot_cle: str) -> str:
    """Cherche un mot-clé simple dans le fichier de notes local et retourne les lignes pertinentes."""
    if not NOTES_PATH.exists():
        return "Erreur : le fichier notes_agent_tp.txt est introuvable."

    lines = NOTES_PATH.read_text(encoding="utf-8").splitlines()
    mot_cle = mot_cle.strip().lower()

    if not mot_cle:
        return "Erreur : fournissez un mot-clé non vide."

    matches = [line for line in lines if mot_cle in line.lower()]

    if not matches:
        return f"Aucune ligne trouvée pour le mot-clé : {mot_cle}"

    return "\n".join(matches[:5])


@tool
def generer_plan_revision(sujet: str) -> str:
    """Génère un mini plan de révision en 5 points à partir d'un sujet simple."""
    sujet = sujet.strip()
    if not sujet:
        return "Erreur : fournissez un sujet."

    return (
        f"Plan de révision pour : {sujet}\n"
        "1. Relire les définitions et notions de base.\n"
        "2. Identifier 5 mots-clés essentiels.\n"
        "3. Refaire un exemple concret ou un mini exercice.\n"
        "4. Résumer le sujet en 5 lignes maximum.\n"
        "5. Se tester avec 3 questions de révision."
    )


TOOLS = [calculer_moyenne, chercher_dans_les_notes, generer_plan_revision]
TOOLS_BY_NAME = {tool_.name: tool_ for tool_ in TOOLS}


def _parse_tool_call(tool_call, index: int):
    if isinstance(tool_call, dict):
        name = tool_call.get("name") or tool_call.get("tool_name")
        args = tool_call.get("args") or tool_call.get("input") or ""
        tool_call_id = tool_call.get("id") or tool_call.get("tool_call_id")
    else:
        name = getattr(tool_call, "name", None) or getattr(tool_call, "tool_name", None)
        args = getattr(tool_call, "args", None) or getattr(tool_call, "input", None) or ""
        tool_call_id = getattr(tool_call, "id", None) or getattr(tool_call, "tool_call_id", None)

    return name, args, tool_call_id or f"tool_call_{index}"


def run_agent(question: str) -> str:
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "La variable GROQ_API_KEY est absente. Ajoutez-la dans un fichier .env."
        )

    model = os.getenv("GROQ_MODEL", MODEL_NAME)
    base_url = os.getenv("GROQ_API_BASE", BASE_URL)
    
    # ChatGroq adds /openai/v1 internally, so strip it if present
    if base_url.endswith("/openai/v1"):
        base_url = base_url[:-len("/openai/v1")]

    try:
        llm = ChatGroq(
            model=model,
            temperature=0,
            api_key=api_key,
            base_url=base_url,
        )
        llm_with_tools = llm.bind_tools(TOOLS)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ChatGroq: {e}")

    messages = [
        SystemMessage(
            content=(
                "Tu es un agent d'assistance pour étudiant. "
                "Tu peux utiliser des outils pour calculer, chercher dans des notes locales "
                "et préparer un plan de révision. "
                "Utilise un outil seulement si c'est utile. "
                "Quand un outil est utilisé, base ta réponse finale sur son résultat."
            )
        ),
        HumanMessage(content=question),
    ]

    try:
        first_response = llm_with_tools.invoke(messages)
    except Exception as e:
        raise RuntimeError(f"LLM invocation failed: {e}")

    # Check if tool calls were made
    tool_calls = getattr(first_response, "tool_calls", None)
    if tool_calls:
        try:
            messages.append(first_response)
            for idx, tool_call in enumerate(tool_calls, start=1):
                tool_name, tool_args, tool_call_id = _parse_tool_call(tool_call, idx)
                if not tool_name:
                    continue

                selected_tool = TOOLS_BY_NAME.get(tool_name)
                if not selected_tool:
                    continue

                if hasattr(selected_tool, "invoke"):
                    tool_output = selected_tool.invoke(tool_args)
                else:
                    tool_output = selected_tool(tool_args)

                messages.append(
                    ToolMessage(
                        content=tool_output,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )

            # Use base LLM (without tools) for final response after tool execution
            final_response = llm.invoke(messages)
            return final_response.content
        except Exception as e:
            raise RuntimeError(f"Tool execution failed: {e}")

    return first_response.content


def main():
    print("Agent d'étude prêt.")
    print("Exemples :")
    print("- Calcule la moyenne de [12, 15, 9].")
    print("- Cherche le mot RAG dans mes notes.")
    print("- Prépare-moi un plan de révision sur les agents IA.")
    print("Tapez 'quit' pour quitter.\n")

    while True:
        question = input("Question > ").strip()
        if not question:
            print("Veuillez saisir une question.\n")
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Fin du programme.")
            break

        try:
            answer = run_agent(question)
            print("\n--- Réponse de l'agent ---")
            print(answer)
            print()
        except Exception as exc:
            print(f"\nErreur : {exc}\n")


if __name__ == "__main__":
    main()
