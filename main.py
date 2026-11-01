import os
import io
import json
import time
import uuid
import random
import logging
import threading
from datetime import datetime
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict

from dotenv import load_dotenv
from PIL import Image

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode

from google import genai
from google.genai import types

try:
    from google.cloud import firestore
    FIRESTORE_AVAILABLE = True
except Exception:
    FIRESTORE_AVAILABLE = False
    firestore = None

try:
    from google.cloud import logging as cloud_logging
    CLOUD_LOGGING_AVAILABLE = True
except Exception:
    CLOUD_LOGGING_AVAILABLE = False
    cloud_logging = None

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "fifa_fc_decisions")
PORT = int(os.getenv("PORT", "8080"))

def validate_keys():
    errors = []
    if not TELEGRAM_TOKEN or len(TELEGRAM_TOKEN.strip()) < 20:
        errors.append("TELEGRAM_BOT_TOKEN manquant ou invalide.")
    if not GEMINI_API_KEY or len(GEMINI_API_KEY.strip()) < 20:
        errors.append("GEMINI_API_KEY manquant ou invalide.")
    if not GOOGLE_CLOUD_PROJECT or len(GOOGLE_CLOUD_PROJECT.strip()) < 3:
        errors.append("GOOGLE_CLOUD_PROJECT manquant ou invalide.")
    if errors:
        raise ValueError("Erreurs de configuration: " + " | ".join(errors))

validate_keys()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

if CLOUD_LOGGING_AVAILABLE and GOOGLE_CLOUD_PROJECT:
    try:
        cloud_client = cloud_logging.Client(project=GOOGLE_CLOUD_PROJECT)
        cloud_client.setup_logging()
        logger.info("Google Cloud Logging initialisé.")
    except Exception as e:
        logger.warning(f"Cloud Logging unavailable: {e}")

client = genai.Client(api_key=GEMINI_API_KEY)

DB = None
if FIRESTORE_AVAILABLE and GOOGLE_CLOUD_PROJECT:
    try:
        DB = firestore.Client(project=GOOGLE_CLOUD_PROJECT)
        logger.info("Firestore initialisé.")
    except Exception as e:
        logger.warning(f"Firestore initialisation failed: {e}")
        DB = None

DECISION_STORE = {}

def save_decision(decision_id: str, payload: Dict[str, Any]):
    payload["updated_at"] = datetime.utcnow().isoformat()
    if DB is not None:
        try:
            DB.collection(FIRESTORE_COLLECTION).document(decision_id).set(payload, merge=True)
            logger.info(f"Decision saved to Firestore: {decision_id}")
        except Exception as e:
            logger.warning(f"Firestore save failed: {e}")
    DECISION_STORE[decision_id] = payload

CHAMPIONNATS_FIFA = [
    "FC 25. 3x3. Ligue de conférence",
    "FC 26. 5x5 Rush. Superligue",
    "FC 26. Championnat du monde",
    "FC 26. Ligue des champions",
    "FC 24. Championnat d'Angleterre",
]

PREDICTION_HISTORY = []
USER_RATE_LIMIT = defaultdict(list)

SYSTEM_PROMPT = """
Tu es un agent IA de décision sportive pour FIFA FC.

RÈGLES ABSOLUES :
1. Tu ne dois JAMAIS inventer des données statistiques.
2. Tu ne dois JAMAIS te baser sur le nom des équipes pour faire la prédiction.
3. Tu ne dois utiliser que les statistiques visibles dans l'image ou dans le texte fourni.
4. Si une donnée est absente, marque-la comme "missing".
5. Si elle est illisible, marque-la comme "unreadable".
6. Si les données sont insuffisantes, tu dois dire :
   <b>Données insuffisantes pour prédire.</b>
7. Réponds exclusivement en HTML Telegram valide.
8. N'utilise que les balises : <b>, <i>, <code>, <s>.
9. N'utilise JAMAIS Markdown.
"""

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"FIFA FC Predictor Bot est en ligne et fonctionnel !")

    def log_message(self, format, *args):
        return

def run_web_server():
    server_address = ("0.0.0.0", PORT)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    logger.info(f"Serveur HTTP de maintien actif démarré sur le port {PORT}")
    httpd.serve_forever()

def now_iso():
    return datetime.utcnow().isoformat()

def sanitize_html(text: str) -> str:
    if not text:
        return "Erreur de génération de réponse."
    text = text.replace("```", "")
    text = text.replace("[", "").replace("]", "")
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\r\t")
    return text.strip()

def rate_limit_ok(user_id: int, max_per_minute: int = 5) -> bool:
    now = time.time()
    history = USER_RATE_LIMIT.get(user_id, [])
    history = [ts for ts in history if now - ts < 60]
    history.append(now)
    USER_RATE_LIMIT[user_id] = history
    return len(history) <= max_per_minute

def is_match_like_text(raw: str) -> bool:
    if not raw or len(raw.strip()) < 8:
        return False
    lower = raw.lower()
    markers = [
        "match", "score", "team", "teams", "possession",
        "xg", "shots", "fifa", "prediction", "vs", "versus",
        "goals", "passes", "tirs", "stats"
    ]
    return any(marker in lower for marker in markers)

def safe_json_load(raw_text: str) -> Dict[str, Any]:
    try:
        if not raw_text:
            return {}
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw_text[start:end+1])
        return json.loads(raw_text)
    except Exception:
        return {}

def tool_extract_stats_from_text(raw_text: str) -> Dict[str, Any]:
    prompt = f"""
    Tu es un extracteur de statistiques de football FIFA FC.

    RÈGLES :
    - Ne jamais inventer de données.
    - Ne jamais utiliser le nom des équipes comme base de décision.
    - Utiliser uniquement les statistiques visibles dans le texte.
    - Si une donnée manque, mets "missing".
    - Si elle est illisible, mets "unreadable".
    - Retourne un JSON valide uniquement.

    Texte :
    {raw_text}

    Format JSON attendu :
    {{
      "team_1": {{"label": "Team A", "stats": {{"possession": 52, "shots": 9, "shots_on_target": 4, "xg": 1.5, "goals": 1}}}},
      "team_2": {{"label": "Team B", "stats": {{"possession": 48, "shots": 7, "shots_on_target": 2, "xg": 1.0, "goals": 0}}}},
      "missing_fields": [],
      "unreadable_fields": [],
      "confidence": 0.0,
      "notes": ""
    }}
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=1600,
            ),
        )
        text = getattr(response, "text", "") or ""
        data = safe_json_load(text)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.exception("tool_extract_stats_from_text failed")
        return {
            "team_1": {"label": "N/A", "stats": {}},
            "team_2": {"label": "N/A", "stats": {}},
            "missing_fields": ["all_stats"],
            "unreadable_fields": ["all_stats"],
            "confidence": 0.0,
            "notes": f"Erreur extraction: {str(e)}"
        }

def tool_extract_stats_from_image(image: Image.Image) -> Dict[str, Any]:
    prompt = """
    Tu es un extracteur de statistiques de football FIFA FC.

    RÈGLES :
    - Ne jamais inventer.
    - Utiliser seulement les valeurs visibles sur l'image.
    - Le nom des équipes ne doit pas servir à la décision.
    - Si absent, mettre "missing".
    - Si illisible, mettre "unreadable".
    - Retourne un JSON valide strictement.

    Format JSON attendu :
    {
      "team_1": {"label": "Team A", "stats": {"possession": 52, "shots": 9, "shots_on_target": 4, "xg": 1.5, "goals": 1}},
      "team_2": {"label": "Team B", "stats": {"possession": 48, "shots": 7, "shots_on_target": 2, "xg": 1.0, "goals": 0}},
      "missing_fields": [],
      "unreadable_fields": [],
      "confidence": 0.0,
      "notes": ""
    }
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=1800,
            ),
        )
        text = getattr(response, "text", "") or ""
        data = safe_json_load(text)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.exception("tool_extract_stats_from_image failed")
        return {
            "team_1": {"label": "N/A", "stats": {}},
            "team_2": {"label": "N/A", "stats": {}},
            "missing_fields": ["all_stats"],
            "unreadable_fields": ["all_stats"],
            "confidence": 0.0,
            "notes": f"Erreur extraction: {str(e)}"
        }

def tool_validate_stats(stats_data: Dict[str, Any]) -> Dict[str, Any]:
    team1 = stats_data.get("team_1", {}).get("stats", {})
    team2 = stats_data.get("team_2", {}).get("stats", {})

    required = ("possession", "shots", "shots_on_target", "xg", "goals")
    team1_has_data = any(key in team1 for key in required)
    team2_has_data = any(key in team2 for key in required)

    missing = stats_data.get("missing_fields", []) or []
    unreadable = stats_data.get("unreadable_fields", []) or []

    if not team1_has_data or not team2_has_data:
        return {"status": "rejected", "reason": "Données insuffisantes ou trop incomplètes."}

    if missing or unreadable:
        return {"status": "approved_with_warnings", "reason": "Base de données suffisante, mais des champs manquent."}

    return {"status": "approved", "reason": "Base de données suffisante."}

def tool_compute_prediction(stats_data: Dict[str, Any]) -> Dict[str, Any]:
    team1 = stats_data.get("team_1", {}).get("stats", {})
    team2 = stats_data.get("team_2", {}).get("stats", {})

    xg1 = team1.get("xg", 0) or team1.get("expected_goals", 0) or 0
    xg2 = team2.get("xg", 0) or team2.get("expected_goals", 0) or 0

    shots1 = team1.get("shots", 0) or 0
    shots2 = team2.get("shots", 0) or 0

    if xg1 > xg2:
        winner = "V1"
    elif xg2 > xg1:
        winner = "V2"
    else:
        winner = "Nul"

    score1 = int(round(max(0, xg1)))
    score2 = int(round(max(0, xg2)))

    if score1 == 0 and score2 == 0:
        score1, score2 = 1, 1

    if shots1 > shots2 * 1.4 and xg1 > xg2:
        score1 += 1
    if shots2 > shots1 * 1.4 and xg2 > xg1:
        score2 += 1

    return {
        "status": "approved",
        "decision": {
            "mt1": {"winner": winner, "score_estimate": f"{score1}-{score2}"},
            "mt2": {"winner": winner, "score_estimate": f"{score1}-{score2}"},
            "full": {"winner": winner, "score_estimate": f"{score1 + 1}-{score2}"},
        },
    }

def run_agent_from_text(raw_text: str, user_id: int) -> Dict[str, Any]:
    decision_id = str(uuid.uuid4())
    decision = {
        "decision_id": decision_id,
        "status": "queued",
        "source": "telegram_text",
        "user_id": user_id,
        "created_at": now_iso(),
        "stats_data": {},
        "validation": {},
        "decision": {
            "mt1": {"winner": "N/A", "score_estimate": "N/A"},
            "mt2": {"winner": "N/A", "score_estimate": "N/A"},
            "full": {"winner": "N/A", "score_estimate": "N/A"},
        },
    }

    try:
        stats_data = tool_extract_stats_from_text(raw_text)
        validation = tool_validate_stats(stats_data)

        decision["stats_data"] = stats_data
        decision["validation"] = validation

        if validation["status"] == "approved":
            prediction = tool_compute_prediction(stats_data)
            decision["decision"] = prediction["decision"]
            decision["status"] = "approved"
        elif validation["status"] == "approved_with_warnings":
            prediction = tool_compute_prediction(stats_data)
            decision["decision"] = prediction["decision"]
            decision["status"] = "approved_with_warnings"
        else:
            decision["status"] = "rejected"
    except Exception as e:
        logger.exception("run_agent_from_text failed")
        decision["status"] = "error"
        decision["error"] = str(e)

    save_decision(decision_id, decision)
    return decision

def run_agent_from_image(image: Image.Image, user_id: int) -> Dict[str, Any]:
    decision_id = str(uuid.uuid4())
    decision = {
        "decision_id": decision_id,
        "status": "queued",
        "source": "telegram_photo",
        "user_id": user_id,
        "created_at": now_iso(),
        "stats_data": {},
        "validation": {},
        "decision": {
            "mt1": {"winner": "N/A", "score_estimate": "N/A"},
            "mt2": {"winner": "N/A", "score_estimate": "N/A"},
            "full": {"winner": "N/A", "score_estimate": "N/A"},
        },
    }

    try:
        stats_data = tool_extract_stats_from_image(image)
        validation = tool_validate_stats(stats_data)

        decision["stats_data"] = stats_data
        decision["validation"] = validation

        if validation["status"] == "approved":
            prediction = tool_compute_prediction(stats_data)
            decision["decision"] = prediction["decision"]
            decision["status"] = "approved"
        elif validation["status"] == "approved_with_warnings":
            prediction = tool_compute_prediction(stats_data)
            decision["decision"] = prediction["decision"]
            decision["status"] = "approved_with_warnings"
        else:
            decision["status"] = "rejected"
    except Exception as e:
        logger.exception("run_agent_from_image failed")
        decision["status"] = "error"
        decision["error"] = str(e)

    save_decision(decision_id, decision)
    return decision

def format_decision_for_telegram(result: Dict[str, Any]) -> str:
    status = result.get("status", "unknown")
    stats_data = result.get("stats_data", {})
    decision = result.get("decision", {})
    missing = stats_data.get("missing_fields", []) or []
    unreadable = stats_data.get("unreadable_fields", []) or []

    if status in ("rejected", "error"):
        return (
            "<b>Décision de l'agent IA</b>\n"
            "<i>Validation refusée.</i>\n"
            "<b>Raison :</b> Données insuffisantes, illisibles ou incomplètes.\n"
            f"<b>Stats manquantes :</b> {', '.join(missing) if missing else 'Aucune'}\n"
            f"<b>Stats illisibles :</b> {', '.join(unreadable) if unreadable else 'Aucune'}\n"
            "<b>Prédiction :</b> <b>Refusée</b>\n"
            "<b>Données insuffisantes pour prédire.</b>"
        )

    mt1 = decision.get("mt1", {})
    mt2 = decision.get("mt2", {})
    full = decision.get("full", {})

    return (
        "<b>Décision IA validée</b>\n"
        "<i>Basée uniquement sur les statistiques visibles.</i>\n"
        f"<b>MT1 :</b> {mt1.get('winner', 'N/A')} • <b>Score Estimé :</b> {mt1.get('score_estimate', 'N/A')}\n"
        f"<b>MT2 :</b> {mt2.get('winner', 'N/A')} • <b>Score Estimé :</b> {mt2.get('score_estimate', 'N/A')}\n"
        f"<b>FULL :</b> {full.get('winner', 'N/A')} • <b>Score Estimé Final :</b> {full.get('score_estimate', 'N/A')}\n"
        f"<b>Stats manquantes :</b> {', '.join(missing) if missing else 'Aucune'}\n"
        f"<b>Stats illisibles :</b> {', '.join(unreadable) if unreadable else 'Aucune'}\n"
        "<b>Validation :</b> <i>Les données visibles ont été utilisées sans invention.</i>"
    )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    championnats = "\n".join([f"• {c}" for c in CHAMPIONNATS_FIFA])
    message = (
        "🤖 <b>FIFA FC Predictor IA</b>\n\n"
        "• Envoie une <b>photo</b> ou un <b>texte</b> contenant les statistiques visibles.\n"
        "• Le système ne se base pas sur le nom des équipes.\n"
        "• Il refuse la prédiction si les données sont insuffisantes.\n"
        "• Tape <code>/aleatoire</code> pour une génération aléatoire.\n"
        "• Tape <code>/autoeval</code> pour un rapport d'optimisation.\n\n"
        "<b>Championnats supportés :</b>\n"
        f"{championnats}"
    )
    await update.message.reply_text(message, parse_mode=ParseMode.HTML)

async def random_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🎲 1 Match", callback_data="rand_1"), InlineKeyboardButton("🎲 3 Matchs", callback_data="rand_3")],
        [InlineKeyboardButton("🎲 5 Matchs", callback_data="rand_5"), InlineKeyboardButton("🎲 10 Matchs", callback_data="rand_10")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "🎯 <b>Menu de sélection aléatoire</b>\n\nChoisis la taille de la série de matchs :",
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML,
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if not data or not data.startswith("rand_"):
        return

    try:
        count = int(data.split("_")[1])
    except Exception:
        await query.edit_message_text("❌ Paramètre invalide.", parse_mode=ParseMode.HTML)
        return

    if count <= 0 or count > 10:
        count = 3

    leagues = random.sample(CHAMPIONNATS_FIFA, min(count, len(CHAMPIONNATS_FIFA)))
    leagues_str = "\n".join([f"- {league}" for league in leagues])

    await query.edit_message_text(
        f"⏳ <i>Simulation aléatoire en cours...</i>\n\n<b>Championnats sélectionnés :</b>\n{leagues_str}",
        parse_mode=ParseMode.HTML,
    )

    prompt = f"""
    Génère une série de {count} simulations de matchs FIFA FC, basées uniquement sur des statistiques visibles et cohérentes.
    N'invente rien. Les statistiques doivent rester plausibles et lisibles.
    Championnats :
    {leagues_str}

    Réponds en HTML Telegram uniquement, avec :
    - <b>Stats visibles</b>
    - <b>Stats manquantes</b>
    - <b>Stats illisibles</b>
    - <b>MT1</b>, <b>MT2</b>, <b>FULL</b>
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=1200,
            ),
        )
        text = sanitize_html(getattr(response, "text", "") or "Aucune réponse.")
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"📊 <b>Résultat aléatoire</b>\n\n{text}",
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"❌ Erreur génération aléatoire : {str(e)}",
            parse_mode=ParseMode.HTML,
        )

async def self_evaluation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not PREDICTION_HISTORY:
        await update.message.reply_text("📊 Aucun historique disponible.", parse_mode=ParseMode.HTML)
        return

    recent = PREDICTION_HISTORY[-5:]
    payload = json.dumps(recent, ensure_ascii=False, default=str)

    prompt = f"""
    Tu es le module d'auto-évaluation de ce bot FIFA FC.

    Historique :
    {payload}

    Analyse :
    - les erreurs de prise de décision
    - les causes d'hallucination
    - les champs manquants
    - la qualité des données visibles
    - les améliorations à faire pour un système plus sûr

    Réponds en HTML Telegram uniquement.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=1200,
            ),
        )
        text = sanitize_html(getattr(response, "text", "") or "Aucune réponse.")
        await update.message.reply_text(f"📈 <b>Rapport d'auto-évaluation</b>\n\n{text}", parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur auto-évaluation : {str(e)}", parse_mode=ParseMode.HTML)

async def analyze_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user is None:
        return

    user_id = user.id

    if not rate_limit_ok(user_id, 5):
        await update.message.reply_text(
            "⏳ Trop de requêtes. Patiente quelques secondes avant une nouvelle analyse.",
            parse_mode=ParseMode.HTML,
        )
        return

    if update.message.text and update.message.text.startswith("/"):
        return

    if update.message.photo:
        status_msg = await update.message.reply_text(
            "⏳ <i>Analyse IA des statistiques visibles en cours...</i>",
            parse_mode=ParseMode.HTML,
        )
        try:
            photo_file = await update.message.photo[-1].get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            image = Image.open(io.BytesIO(photo_bytes))
            decision = run_agent_from_image(image, user_id)
            formatted = format_decision_for_telegram(decision)
            PREDICTION_HISTORY.append({"kind": "photo", "decision_id": decision["decision_id"], "payload": decision})
            await status_msg.edit_text(formatted, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.exception("Erreur analyze_input photo")
            await status_msg.edit_text(f"❌ Erreur de l'agent sur image : {str(e)}", parse_mode=ParseMode.HTML)
    elif update.message.text:
        text = update.message.text.strip()
        if not is_match_like_text(text):
            await update.message.reply_text(
                "📌 Donne un texte ou une image contenant des statistiques visibles. "
                "Je refuse toute prédiction basée sur le nom des équipes ou sur des hypothèses.",
                parse_mode=ParseMode.HTML,
            )
            return

        status_msg = await update.message.reply_text(
            "⏳ <i>Analyse IA des statistiques en cours...</i>",
            parse_mode=ParseMode.HTML,
        )
        try:
            decision = run_agent_from_text(text, user_id)
            formatted = format_decision_for_telegram(decision)
            PREDICTION_HISTORY.append({"kind": "text", "decision_id": decision["decision_id"], "payload": decision})
            await status_msg.edit_text(formatted, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.exception("Erreur analyze_input text")
            await status_msg.edit_text(f"❌ Erreur de l'agent sur texte : {str(e)}", parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text("📌 Je n'analyse que les photos ou les textes contenant des stats visibles.", parse_mode=ParseMode.HTML)

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()
    logger.info("Serveur HTTP keepalive initialisé.")

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("aleatoire", random_menu))
    app.add_handler(CommandHandler("autoeval", self_evaluation))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.TEXT | filters.PHOTO, analyze_input))

    print("🤖 FIFA FC Predictor IA prêt.")
    app.run_polling()

if __name__ == "__main__":
    main()
