# slack_bot.py (With Background Tasks)
import os
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.signature import SignatureVerifier
from dotenv import load_dotenv
import asyncio

# Import our existing RAG logic from the backend
from backend import get_chains

# --- INITIALIZATION ---
load_dotenv()
app = FastAPI()
slack_bot_token = os.environ["SLACK_BOT_TOKEN"]
slack_signing_secret = os.environ["SLACK_SIGNING_SECRET"]
slack_client = AsyncWebClient(token=slack_bot_token)
signature_verifier = SignatureVerifier(signing_secret=slack_signing_secret)
retriever, generation_chain = None, None

@app.on_event("startup")
async def startup_event():
    global retriever, generation_chain
    print("Initializing RAG chain for Slack bot...")
    try:
        retriever, generation_chain = await asyncio.to_thread(get_chains)
        print("RAG chain initialized successfully.")
    except Exception as e:
        print(f"FATAL: Could not initialize RAG chain on startup. Error: {e}")

# --- NEW: The Background Task Function ---
# This function contains our slow RAG logic.
async def run_rag_and_reply(channel_id: str, thread_ts: str, user_question: str):
    print(f"Background task started for question: {user_question}")
    if not generation_chain or not retriever:
        await slack_client.chat_postMessage(channel=channel_id, text="Sorry, my brain (RAG chain) is not initialized.", thread_ts=thread_ts)
        return

    try:
        source_docs = retriever.invoke(user_question)
        context_text = "\n\n".join([doc.page_content for doc in source_docs])
        chain_input = {"context": context_text, "question": user_question}
        answer = await generation_chain.ainvoke(chain_input)
        
        final_text = answer
        if source_docs and "enough information" not in answer.lower():
            sources_text = "\n\n*Sources used for this answer:*\n"
            seen_sources = set()
            for doc in source_docs:
                source_name = doc.metadata.get('source_doc_name')
                if source_name and source_name not in seen_sources:
                    if source_name.startswith("http"):
                        sources_text += f"• <{source_name}|{source_name.split('//')[-1]}>\n"
                    else:
                        sources_text += f"• {source_name}\n"
                    seen_sources.add(source_name)
            final_text += sources_text

        await slack_client.chat_postMessage(
            channel=channel_id,
            text=final_text,
            thread_ts=thread_ts
        )
        print("Background task finished, reply sent.")
    except Exception as e:
        print(f"Error in background task: {e}")
        await slack_client.chat_postMessage(channel=channel_id, text=f"Sorry, I encountered an error: {e}", thread_ts=thread_ts)

# --- API ENDPOINT (Now much faster) ---
@app.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    headers = request.headers
    if not signature_verifier.is_valid_request(body, headers):
        raise HTTPException(status_code=403, detail="Invalid Slack signature")

    event_data = await request.json()

    if "challenge" in event_data:
        return {"challenge": event_data["challenge"]}

    if "event" in event_data:
        event = event_data["event"]
        
        if event.get("type") == "app_mention":
            channel_id = event["channel"]
            user_question = event["text"]
            thread_ts = event.get("ts")
            
            # --- THIS IS THE CRITICAL CHANGE ---
            # Instead of running the slow logic here, we add it as a background task.
            # This function call returns immediately.
            background_tasks.add_task(run_rag_and_reply, channel_id, thread_ts, user_question)

    # We immediately return a 200 OK to Slack to prevent a timeout.
    print("Acknowledged Slack event. Processing in background.")
    return {"status": "ok"}