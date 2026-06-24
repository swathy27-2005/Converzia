# app.py
"""
Meeting Assistant Pro — Mobile-style Enhanced Single-File Streamlit App
- Handles transcription, summary, actions, QA, analytics
- Fixed menu_choice, duplicate buttons, PDF errors
"""

import streamlit as st
import os, tempfile, json, re
from datetime import datetime, timedelta
from collections import Counter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# ---------------- Load external CSS ----------------
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")
# Optional libraries
try: import whisper
except: whisper = None
try: from transformers import pipeline
except: pipeline = None
try: import spacy
except: spacy = None
try: from streamlit_webrtc import webrtc_streamer, WebRtcMode
except: WEBSOCKET_AVAILABLE = False
else: WEBSOCKET_AVAILABLE = True

# ---------------- Config ----------------
STORAGE_FILE = "meetings.json"
MAX_SUMMARY_CHUNK = 1500
VERBS_REGEX = r"(prepare|draft|create|send|submit|update|review|finalize|organize|book|test|collect|assign|follow up|follow-up|followup|action)"
AVG_WPM_DEFAULT = 130

# ---------------- Session Defaults ----------------
if "theme" not in st.session_state: st.session_state.theme="light"
if "logged_in" not in st.session_state: st.session_state.logged_in=False
if "current" not in st.session_state: st.session_state.current={}
if "menu_choice" not in st.session_state: st.session_state.menu_choice="Dashboard"
if "show_toast" not in st.session_state: st.session_state.show_toast=False
if "toast_message" not in st.session_state: st.session_state.toast_message=""

def toggle_theme(): st.session_state.theme="dark" if st.session_state.theme=="light" else "light"

# ---------------- Model loaders ----------------
@st.cache_resource
def get_whisper_model():
    if whisper is None: return None
    try: return whisper.load_model("base")
    except: return None

@st.cache_resource
def get_summarizer():
    if pipeline is None: return None
    try: return pipeline("summarization", model="facebook/bart-large-cnn")
    except:
        try: return pipeline("summarization")
        except: return None

@st.cache_resource
def get_sentiment():
    if pipeline is None: return None
    try: return pipeline("sentiment-analysis")
    except: return None

@st.cache_resource
def get_spacy():
    if spacy is None: return None
    try: return spacy.load("en_core_web_sm")
    except: return None

@st.cache_resource
def get_qa():
    if pipeline is None: return None
    try: return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    except:
        try: return pipeline("question-answering")
        except: return None

# ---------------- Persistence ----------------
def load_history():
    if os.path.exists(STORAGE_FILE):
        try:
            with open(STORAGE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return []
    return []

def save_history(history):
    with open(STORAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def append_meeting(meeting):
    hist = load_history()
    hist.append(meeting)
    save_history(hist)

def overwrite_meeting_at(index, meeting):
    hist = load_history()
    if 0 <= index < len(hist):
        hist[index] = meeting
        save_history(hist)
        return True
    return False

# ---------------- Utilities ----------------
def chunked_summarize(text, summarizer, max_chunk_chars=MAX_SUMMARY_CHUNK):
    text = (text or "").strip()
    if not text: return ""
    if summarizer is None: return text[:800] + ("..." if len(text) > 800 else "")
    if len(text) <= max_chunk_chars:
        out = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return out[0].get("summary_text","")
    chunks=[]
    start=0; L=len(text)
    while start<L:
        end=min(start+max_chunk_chars,L)
        if end<L:
            cut=text.rfind(".", start,end)
            if cut>start: end=cut+1
        chunk=text[start:end].strip()
        if chunk:
            s=summarizer(chunk,max_length=120,min_length=20,do_sample=False)[0].get("summary_text","")
            chunks.append(s)
        start=end
    combined=" ".join(chunks)
    final=summarizer(combined,max_length=150,min_length=30,do_sample=False)[0].get("summary_text","")
    return final

def extract_action_items(text,nlp):
    tasks=[]
    if not text: return tasks
    if nlp is None:
        sentences=re.split(r'(?<=[.?!])\s+',text)
        for s in sentences:
            if re.search(VERBS_REGEX,s,re.IGNORECASE):
                tasks.append({"Task":s.strip(),"Assignee":"Unassigned","Deadline":"Not specified","Priority":"Normal","Status":"Pending"})
        return tasks
    doc=nlp(text)
    for sent in doc.sents:
        sentence=sent.text.strip()
        if re.search(VERBS_REGEX,sentence,re.IGNORECASE):
            sdoc=nlp(sentence)
            assignees=[ent.text for ent in sdoc.ents if ent.label_=="PERSON"]
            deadlines=[ent.text for ent in sdoc.ents if ent.label_ in ("DATE","TIME")]
            priority="High" if re.search(r"\b(urgent|asap|immediately|priority|today|by end of day)\b", sentence,re.IGNORECASE) else "Normal"
            tasks.append({"Task":sentence,"Assignee":", ".join(assignees) if assignees else "Unassigned","Deadline":deadlines[0] if deadlines else "Not specified","Priority":priority,"Status":"Pending"})
    return tasks

def simulate_speaker_diarization(transcript):
    sentences=[s.strip() for s in re.split(r'(?<=[.?!])\s+',transcript) if s.strip()]
    return [{"Speaker":f"Speaker {i%3+1}","Sentence":s} for i,s in enumerate(sentences)]

def generate_wordcloud_figure(text):
    if not text:
        fig=plt.figure(figsize=(8,4))
        plt.text(0.5,0.5,"No text",ha='center')
        plt.axis('off')
        return fig
    wc=WordCloud(width=800,height=400,background_color="white").generate(text)
    fig=plt.figure(figsize=(9,4.5))
    plt.imshow(wc,interpolation="bilinear")
    plt.axis("off")
    return fig

def create_task_timeline(actions):
    events=[]
    base_dt=datetime.now()
    for i,a in enumerate(actions):
        dl=a.get("Deadline","Not specified")
        if dl and re.search(r"\btomorrow\b",str(dl),re.IGNORECASE): d=base_dt+timedelta(days=1)
        elif dl and re.search(r"\bnext week\b",str(dl),re.IGNORECASE): d=base_dt+timedelta(days=7)
        else: d=base_dt+timedelta(days=i*2)
        events.append({"Task":a.get("Task","")[:80],"Start":d,"End":d+timedelta(hours=1)})
    if not events: return None
    tdf=pd.DataFrame(events)
    fig=px.timeline(tdf,x_start="Start",x_end="End",y="Task",color="Task")
    fig.update_layout(showlegend=False)
    fig.update_yaxes(autorange="reversed")
    return fig

def estimate_duration(transcript,wpm=AVG_WPM_DEFAULT):
    words=re.findall(r"\b[a-zA-Z']+\b", transcript or "")
    minutes=len(words)/max(1,wpm)
    return round(minutes,2),len(words)

# Safe FPDF multi-cell to handle long words
def safe_multicell(pdf, text, w=0, h=6):
    words = text.split(" ")
    lines = []
    current_line = ""
    for word in words:
        while len(word) > 80:
            lines.append(word[:80])
            word = word[80:]
        if len(current_line) + len(word) + 1 <= 200:
            current_line += (" " if current_line else "") + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    for l in lines:
        pdf.multi_cell(w, h, l)

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def export_meeting_pdf(meeting, out_path):
    doc = SimpleDocTemplate(out_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>Meeting Report: {meeting.get('file')}</b>", styles['Title']))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Timestamp: {meeting.get('timestamp','')}", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Summary:</b>", styles['Heading2']))
    story.append(Paragraph(meeting.get("summary","No summary available."), styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Action Items:</b>", styles['Heading2']))
    if meeting.get("actions"):
        for a in meeting.get("actions", []):
            line = f"- {a.get('Task')} | Assignee: {a.get('Assignee')} | Deadline: {a.get('Deadline')} | Priority: {a.get('Priority')} | Status: {a.get('Status')}"
            story.append(Paragraph(line, styles['Normal']))
    else:
        story.append(Paragraph("No action items.", styles['Normal']))

    doc.build(story)


# ---------------- App UI ----------------
st.set_page_config(page_title="CONVERZIA - From Talk to Action", page_icon="✨", layout="wide")

# ---------------- Load models ----------------
with st.spinner("Loading models..."):
    whisper_model=get_whisper_model()
    summarizer=get_summarizer()
    sentiment_model=get_sentiment()
    nlp=get_spacy()
    qa_pipeline=get_qa()

# ---------------- Simple Auth ----------------
if not st.session_state.logged_in:
    st.markdown("<div style='max-width:480px;margin:30px auto;'>", unsafe_allow_html=True)
    st.markdown("<div style='padding:16px;background:#fff;border-radius:12px;text-align:center'><h3>CONVERZIA ✨</h3><p class='small-muted'>Sign in to access your meetings</p></div>", unsafe_allow_html=True)
    username=st.text_input("Username")
    password=st.text_input("Password", type="password")
    col1,col2=st.columns(2)
    with col1:
        if st.button("Sign in", key="login_btn"):
            if username=="admin" and password=="1234":
                st.session_state.logged_in=True; st.rerun()
            else: st.error("Invalid credentials")
    with col2:
        if st.button("Demo login", key="demo_login_btn"): st.session_state.logged_in=True; st.rerun()
    st.stop()

# -------------------------------
# Sidebar Navigation (Mobile App Style)
# -------------------------------

# Define menu items
menu_items = [
    "Dashboard", "Upload", "Transcript", "Summary",
    "Action Items", "Analytics", "Q&A", "Insights",
    "Search", "History", "Profile"
]

# Initialize menu choice if not already set
if "menu_choice" not in st.session_state:
    st.session_state.menu_choice = "Dashboard"

# Sidebar UI
with st.sidebar:
    st.title("CONVERZIA ✨")
    st.markdown("---")
    menu_choice = st.radio(
        "Navigate",
        menu_items,
        index=menu_items.index(st.session_state.menu_choice),
        key="sidebar_menu"
    )

# Save selection to session state
st.session_state.menu_choice = menu_choice


# ---------------- Load meeting history ----------------
hist = load_history()

# ---------------- Dashboard / Pages ----------------
# Paste your existing backend code here exactly for each page
# For example:
if menu_choice=="Dashboard":
    st.header("Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Meetings", len(hist))
    recent_actions = sum(len(m.get("actions",[])) for m in hist[-5:]) if hist else 0
    c2.metric("Recent actions", recent_actions)
    pending = sum(1 for m in hist for a in m.get("actions",[]) if a.get("Status")!="Done")
    c3.metric("Pending tasks", pending)
    st.markdown("---")
    st.subheader("Recent meetings")
    if hist:
        grid = st.container()
        for m in reversed(hist[-6:]):
            st.markdown(f"<div class='card'><b>{m.get('file')}</b> — {m.get('timestamp')}<div class='small-muted'>{(m.get('summary') or '')[:180]}</div></div>", unsafe_allow_html=True)
    else:
        st.info("No meetings yet. Upload & process one to get started.")
    pass

elif menu_choice=="Upload":
    st.header("Upload & Transcribe")
    colA, colB = st.columns([2,1])
    with colA:
        audio_file = st.file_uploader("Upload audio file (mp3/wav/m4a)", type=["mp3","wav","m4a"])
        if audio_file:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1])
            tmp.write(audio_file.read())
            tmp.flush()
            tmp.close()
            st.session_state.current["audio_path"] = tmp.name
            st.success(f"Saved: {tmp.name}")
    with colB:
        st.markdown("**Record in browser**")
        if WEBSOCKET_AVAILABLE:
            st.info("Browser recording available")
            webrtc_streamer(key="rec", mode=WebRtcMode.SENDONLY)
            st.caption("Recording support is experimental. Upload recommended.")
        else:
            st.info("Browser recording not installed.")
    st.markdown("---")
    wpm = st.number_input("Estimated speaking speed (words per minute)", min_value=80, max_value=300, value=AVG_WPM_DEFAULT)
    summarizer_max = st.slider("Summary chunk size (chars)", 800, 4000, MAX_SUMMARY_CHUNK)
    if st.button("Transcribe & Analyze", key="transcribe"):
        audio_path = st.session_state.current.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            st.error("Please upload an audio file first.")
        else:
            transcript = ""
            if whisper_model is not None:
                with st.spinner("Running Whisper transcription..."):
                    try:
                        res = whisper_model.transcribe(audio_path)
                        transcript = res.get("text","").strip()
                    except Exception as e:
                        st.error(f"Whisper failed: {e}")
                        transcript = ""
            else:
                st.warning("Whisper not available — transcription skipped. Paste transcript manually in Transcript page.")
                transcript = ""
            st.session_state.current["transcript"] = transcript

            with st.spinner("Generating summary..."):
                summary = chunked_summarize(transcript, summarizer, max_chunk_chars=summarizer_max)
            st.session_state.current["summary"] = summary

            with st.spinner("Extracting action items..."):
                actions = extract_action_items(transcript, nlp)
            st.session_state.current["actions"] = actions

            if sentiment_model is not None:
                try:
                    sent = sentiment_model((transcript or "")[:1000])[0]
                except Exception:
                    sent = {"label":"N/A","score":0.0}
            else:
                sent = {"label":"N/A","score":0.0}
            st.session_state.current["sentiment"] = sent

            diar = simulate_speaker_diarization(transcript)
            st.session_state.current["diarization"] = diar

            minutes, words = estimate_duration(transcript, wpm=wpm)
            st.session_state.current["est_minutes"] = minutes
            st.session_state.current["word_count"] = words

            record = {
                "file": os.path.basename(audio_path),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "transcript": transcript,
                "summary": summary,
                "actions": st.session_state.current["actions"],
                "sentiment": sent,
                "diarization": diar,
                "est_minutes": minutes,
                "word_count": words
            }
            append_meeting(record)
            st.success("Processed and saved to meetings.json")
            st.session_state.show_toast = True
            st.experimental_set_query_params(page="Summary")
    pass

elif menu_choice=="Transcript":
    st.header("Transcript")
    transcript = st.session_state.current.get("transcript","")
    if transcript:
        ta = st.text_area("Transcript (editable)", transcript, height=360, key="transcript_area")
        if st.button("Save transcript edits"):
            st.session_state.current["transcript"] = st.session_state.get("transcript_area","")
            st.success("Saved to session. Re-run summary or extract actions if needed.")
        if st.session_state.current.get("diarization"):
            st.subheader("Speaker view (simulated)")
            for row in st.session_state.current["diarization"]:
                st.markdown(f"**{row['Speaker']}**: {row['Sentence']}")
    else:
        st.info("No transcript in session. Upload & transcribe or load from History.")
    pass

elif menu_choice=="Summary":
    st.header("Summary")
    summary = st.session_state.current.get("summary","")
    if summary:
        left, right = st.columns([3,1])
        with left:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write(summary)
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.metric("Estimated Minutes", st.session_state.current.get("est_minutes","N/A"))
            st.metric("Word count", st.session_state.current.get("word_count",0))
            if st.button("Download TXT"):
                txt = f"SUMMARY:\n\n{summary}\n\nActions:\n"
                for a in st.session_state.current.get("actions",[]):
                    txt += f"- {a.get('Task')} | {a.get('Assignee')} | {a.get('Deadline')} | {a.get('Priority')} | {a.get('Status')}\n"
                st.download_button("Download summary", data=txt, file_name="meeting_summary.txt", mime="text/plain")
    pass
elif menu_choice=="Action Items":
    st.header("Action Items")
    actions = st.session_state.current.get("actions", [])
    if not actions:
        st.info("No action items in current session. Process a meeting or load from History.")
    else:
        df = pd.DataFrame(actions)
        st.write("Edit action item details below. Click Save to persist to meetings.json.")
        updated = []
        for i, row in df.iterrows():
            status_class = "task-done" if row.get("Status")=="Done" else ("task-progress" if row.get("Status")=="In Progress" else "task-pending")
            st.markdown(f"<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"**{i+1}.** {row.get('Task')}")
            cols = st.columns([3,1,1,1])
            assignee = cols[0].text_input(f"Assignee_{i}", value=row.get("Assignee","Unassigned"), key=f"a_{i}")
            deadline = cols[1].text_input(f"Deadline_{i}", value=row.get("Deadline","Not specified"), key=f"d_{i}")
            priority = cols[2].selectbox(f"Priority_{i}", ["Normal","High"], index=0 if row.get("Priority","Normal")=="Normal" else 1, key=f"p_{i}")
            status = cols[3].selectbox(f"Status_{i}", ["Pending","In Progress","Done"], index=["Pending","In Progress","Done"].index(row.get("Status","Pending")), key=f"s_{i}")
            updated.append({"Task": row.get("Task"), "Assignee": assignee, "Deadline": deadline, "Priority": priority, "Status": status})
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
        if st.button("Save actions & persist"):
            st.session_state.current["actions"] = updated
            hist = load_history()
            idx_to_update = None
            for idx in range(len(hist)-1, -1, -1):
                if hist[idx].get("transcript","") == st.session_state.current.get("transcript",""):
                    idx_to_update = idx
                    break
            if idx_to_update is None and hist:
                idx_to_update = len(hist)-1
            if idx_to_update is not None:
                hist[idx_to_update]["actions"] = updated
                overwrite_meeting_at(idx_to_update, hist[idx_to_update])
                st.success("Saved to meetings.json")
            else:
                st.error("No matching meeting found to update.")

        # --- CSV Export Option ---
        import csv
        if st.button("Export Action Items as CSV", key="export_actions_csv"):
            csv_path = "action_items.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["Task", "Assignee", "Deadline", "Priority", "Status"])
                writer.writeheader()
                writer.writerows(updated)
            with open(csv_path, "rb") as f:
                st.download_button(
                    "Download CSV",
                    f,
                    file_name="action_items.csv",
                    mime="text/csv",
                    key="download_actions_csv"
                )

    pass
elif menu_choice=="Analytics":
    st.header("Analytics")
    transcript = st.session_state.current.get("transcript","")
    if not transcript:
        st.info("No transcript in session.")
    else:
        a1, a2 = st.columns([1,1])
        with a1:
            words = [w.lower() for w in re.findall(r"\b[a-zA-Z']+\b", transcript) if len(w)>2]
            if nlp is None:
                stop_words = set()
            else:
                stop_words = set([w.lower() for w in nlp.Defaults.stop_words])
            freq = Counter([w for w in words if w not in stop_words])
            top = freq.most_common(20)
            if top:
                df_freq = pd.DataFrame(top, columns=["Word","Count"])
                fig = px.bar(df_freq, x="Word", y="Count", title="Top words")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No frequent words found.")
            diar = st.session_state.current.get("diarization",[])
            if diar:
                ddf = pd.DataFrame(diar)
                cnt = ddf["Speaker"].value_counts().reset_index()
                cnt.columns = ["Speaker","Count"]
                st.subheader("Speaker sentence counts")
                st.table(cnt)
        with a2:
            st.write("Word Cloud")
            wc_fig = generate_wordcloud_figure(transcript)
            st.pyplot(wc_fig)
        st.markdown("---")
        c1,c2,c3 = st.columns(3)
        minutes = st.session_state.current.get("est_minutes","N/A")
        c1.metric("Estimated duration (mins)", minutes)
        c2.metric("Word count", st.session_state.current.get("word_count",0))
        sent = st.session_state.current.get("sentiment",{})
        sent_label = sent.get("label","N/A")
        sent_score = round(sent.get("score",0)*100,1) if isinstance(sent.get("score",0),(float,int)) else 0
        c3.metric("Sentiment", sent_label, delta=f"{sent_score}%")
        actions = st.session_state.current.get("actions",[])
        if actions:
            done = sum(1 for a in actions if a.get("Status")=="Done")
            total = len(actions)
            fig = go.Figure(go.Pie(values=[done, total-done], labels=["Completed","Pending"], hole=0.6))
            st.plotly_chart(fig, use_container_width=True)
        timeline_fig = create_task_timeline(actions)
        if timeline_fig is not None:
            st.subheader("Task timeline (simulated)")
            st.plotly_chart(timeline_fig, use_container_width=True)
    pass
elif menu_choice=="Q&A":
    st.header("Q&A over transcript")
    transcript = st.session_state.current.get("transcript","")
    if not transcript:
        st.info("No transcript in session.")
    else:
        q = st.text_input("Ask a question about the transcript")
        if st.button("Ask"):
            if not q:
                st.warning("Please enter a question.")
            else:
                if qa_pipeline is None:
                    st.error("QA model not available (transformers not installed).")
                else:
                    with st.spinner("Running QA..."):
                        try:
                            res = qa_pipeline(question=q, context=transcript)
                            st.write("**Answer:**", res.get("answer",""))
                            st.write("Score:", round(res.get("score",0),3))
                        except Exception as e:
                            st.error("QA failed: "+str(e))
    pass
elif menu_choice=="Insights":
    st.header("Insights")
    sent = st.session_state.current.get("sentiment",{})
    sent_score = 0
    try:
        sent_score = round(sent.get('score',0)*100,1)
    except Exception:
        sent_score = 0
    st.metric("Sentiment", sent.get("label","N/A"), delta=f"{sent_score}%")
    actions = st.session_state.current.get("actions",[])
    if actions:
        urgent = [a for a in actions if a.get("Priority")=="High"]
        if urgent:
            st.error(f"🚨 {len(urgent)} urgent tasks")
            for u in urgent:
                st.write(f"- {u.get('Task')} (Assignee: {u.get('Assignee')})")
        else:
            st.success("No urgent tasks.")
    pass
elif menu_choice=="Search":
    st.header("Search saved meetings")
    history = load_history()
    if not history:
        st.info("No saved meetings.")
    else:
        q = st.text_input("Keyword (transcript/summary/task)")
        assignee_q = st.text_input("Assignee filter (partial)")
        deadline_q = st.text_input("Deadline filter")
        priority_filter = st.selectbox("Priority", ["Any","High","Normal"])
        if st.button("Search"):
            results = []
            for i,m in enumerate(history):
                score = 0
                if q and (q.lower() in m.get("transcript","").lower() or q.lower() in m.get("summary","").lower() or any(q.lower() in a.get("Task","").lower() for a in m.get("actions",[]))):
                    score += 1
                if assignee_q and any(assignee_q.lower() in (a.get("Assignee","") or "").lower() for a in m.get("actions",[])):
                    score += 1
                if deadline_q and any(deadline_q.lower() in (a.get("Deadline","") or "").lower() for a in m.get("actions",[])):
                    score += 1
                if priority_filter!="Any" and any(a.get("Priority")==priority_filter for a in m.get("actions",[])):
                    score += 1
                if score>0:
                    results.append((i,score,m))
            results.sort(key=lambda x:-x[1])
            st.write(f"Found {len(results)} results")
            for idx,sc,m in results:
                with st.expander(f"{m.get('file')} — {m.get('timestamp')} (score {sc})"):
                    st.write("Summary:", m.get("summary",""))
                    st.table(pd.DataFrame(m.get("actions",[])))
                    if st.button("Load into session", key=f"load_{idx}"):
                        st.session_state.current = {
                            "transcript": m.get("transcript",""),
                            "summary": m.get("summary",""),
                            "actions": m.get("actions",[]),
                            "sentiment": m.get("sentiment",{}),
                            "diarization": m.get("diarization",[]),
                            "est_minutes": m.get("est_minutes"),
                            "word_count": m.get("word_count")
                        }
                        st.success("Loaded meeting into session.")
    pass
elif menu_choice=="History":
    st.header("Saved Meetings")
    history = load_history()
    if not history:
        st.info("No saved meetings.")
    else:
        for i, m in enumerate(reversed(history)):
            with st.expander(f"{m.get('file')} — {m.get('timestamp')}"):
                st.write("**Summary:**")
                st.write(m.get("summary", ""))
                if m.get("actions"):
                    st.write("**Action Items:**")
                    st.table(pd.DataFrame(m.get("actions", [])))
                col1, col2, col3 = st.columns([1,1,1])
                with col1:
                    if st.button("Load into session", key=f"load_hist_{i}"):
                        st.session_state.current = {
                            "transcript": m.get("transcript", ""),
                            "summary": m.get("summary", ""),
                            "actions": m.get("actions", []),
                            "sentiment": m.get("sentiment", {}),
                            "est_minutes": m.get("est_minutes"),
                            "word_count": m.get("word_count"),
                        }
                        st.success("Loaded meeting into session.")
                with col2:
                    if st.button("Export PDF", key=f"pdf_{i}"):
                        out_file = f"meeting_{i+1}.pdf"
                        export_meeting_pdf(m, out_file)
                        with open(out_file, "rb") as f:
                            st.download_button("Download PDF", f, file_name=out_file, mime="application/pdf")
                with col3:
                    if st.button("Delete", key=f"delete_{i}"):
                        hist = load_history()
                        idx = len(hist) - 1 - i
                        if 0 <= idx < len(hist):
                            hist.pop(idx)
                            save_history(hist)
                            st.success("Meeting deleted. Please refresh.")
                            st.rerun()
    pass
elif menu_choice=="Profile":
    st.header("Profile & Settings")
    st.text_input("Name","Admin")
    st.text_input("Email","admin@example.com")
    st.number_input("Speaking WPM", min_value=80, max_value=300, value=AVG_WPM_DEFAULT)
    st.selectbox("Theme", ["light","dark"], index=0 if st.session_state.theme=="light" else 1, on_change=toggle_theme)
    if st.button("Logout"): st.session_state.logged_in=False; st.rerun()
# ---------------- Pages ----------------
# (Keep the rest of the pages exactly as you implemented, just ensure each st.button has a unique key using key=f"{name}_{i}" or key="...")

# ---------------- FAB ----------------
st.markdown("""
<style>
.fab {position:fixed;bottom:24px;right:24px;background:#2563eb;color:white;border-radius:50%;width:56px;height:56px;font-size:32px;text-align:center;line-height:56px;cursor:pointer;box-shadow:0 2px 6px rgba(0,0,0,0.3);}
</style>
<div class="fab" onclick="document.getElementById('upload_file').click()">+</div>
""",unsafe_allow_html=True)
st.file_uploader("Upload", key="upload_file", label_visibility="collapsed")

