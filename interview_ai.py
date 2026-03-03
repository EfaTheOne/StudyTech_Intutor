#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║                       INTERVIEW  AI                              ║
║  Practice job interviews with a realistic AI interviewer.        ║
║  Animated eyes help you train eye-contact confidence.            ║
╚══════════════════════════════════════════════════════════════════╝

Requirements:
    pip install openai pyttsx3 SpeechRecognition pyaudio

Usage:
    python interview_ai.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import time
import math
import random
import json
from pathlib import Path
from datetime import datetime

# ── Optional dependencies ──────────────────────────────────────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    import speech_recognition as sr
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False

# ── Config path ────────────────────────────────────────────────────────────────
CONFIG_FILE = Path.home() / ".interview_ai_config.json"

# ── Tunable constants ──────────────────────────────────────────────────────────
OPENAI_MODEL       = "gpt-4o"
TTS_SPEECH_RATE    = 158    # words per minute for pyttsx3
STT_LISTEN_TIMEOUT = 10     # seconds to wait for speech to start
STT_PHRASE_LIMIT   = 60     # max seconds for a single phrase

# ── Color palette ──────────────────────────────────────────────────────────────
C = {
    "bg":           "#0d1117",
    "bg_card":      "#161b22",
    "bg_input":     "#21262d",
    "border":       "#30363d",
    "accent":       "#2ea043",
    "accent_blue":  "#388bfd",
    "accent_red":   "#f85149",
    "accent_gold":  "#d29922",
    "text":         "#c9d1d9",
    "text_dim":     "#8b949e",
    "text_bright":  "#f0f6fc",
    "interviewer":  "#79c0ff",
    "user":         "#56d364",
    "system":       "#d29922",
    "error":        "#f85149",
}

# Face / eye palette
SKIN       = "#d4956a"
SKIN_DARK  = "#b07248"
SKIN_SHADE = "#9a6240"
EYE_WHITE  = "#f5f5f0"
IRIS_DARK  = "#1a3556"
IRIS_MID   = "#2a5c96"
IRIS_LITE  = "#4a8ac4"
IRIS_RING  = "#6aaee8"
PUPIL_COL  = "#090909"
HILITE_COL = "#ffffff"
BROW_COL   = "#3d2210"
LIP_COL    = "#c06858"
LIP_DARK   = "#a04040"

# ── Interview tones ────────────────────────────────────────────────────────────
TONES = {
    "Professional":  "formal and professional, maintaining a structured and respectful pace",
    "Technical":     "highly technical, probing deep expertise, architecture decisions, and edge cases",
    "Friendly":      "warm and encouraging while still being thorough and professional",
    "Strict":        "demanding and rigorous — you push back on vague answers and ask tough follow-ups",
    "Casual":        "relaxed and conversational, like a laid-back startup-culture chat",
    "Behavioral":    "entirely STAR-method focused: Situation, Task, Action, Result",
}

# ══════════════════════════════════════════════════════════════════════════════
# AI Interviewer
# ══════════════════════════════════════════════════════════════════════════════

class AIInterviewer:
    """Manages the GPT-4o-powered interview conversation."""

    def __init__(self, api_key: str, job_title: str, tone: str,
                 custom_questions: str, instructions: str):
        self.job_title = job_title
        self.tone_desc = TONES.get(tone, TONES["Professional"])
        self.history: list[dict] = []
        self.client = None

        if OPENAI_AVAILABLE and api_key.strip():
            try:
                self.client = OpenAI(api_key=api_key.strip())
            except Exception:
                pass

        self._build_system(custom_questions.strip(), instructions.strip())

    def _build_system(self, custom_q: str, extra: str):
        lines = [
            f"You are a seasoned recruiter conducting a job interview for the role of {self.job_title}.",
            f"Your style: {self.tone_desc}.",
            "",
            "Rules you MUST follow:",
            "- Open by briefly introducing yourself (use a realistic fictional name, e.g. 'Hi, I'm Sarah from HR') "
            "and immediately ask your first question — all in ONE short message.",
            "- Ask exactly ONE question per message. Never ask two questions at once.",
            "- After the candidate answers, respond with a brief acknowledgment (one sentence), "
            "then ask the next question or a follow-up.",
            "- Mix question types: behavioral (STAR), situational, technical, and motivational.",
            "- Keep your messages SHORT — real interviewers don't monologue.",
            "- After 6–8 questions, close the interview gracefully and give 2–3 sentences of "
            "honest, constructive feedback.",
            "- NEVER break character or reveal you are an AI.",
            "- Do NOT use markdown bullet lists or headers in your responses — speak naturally.",
        ]
        if custom_q:
            lines += ["", f"You MUST ask these specific questions at some point: {custom_q}"]
        if extra:
            lines += ["", f"Additional instructions: {extra}"]

        self.history = [{"role": "system", "content": "\n".join(lines)}]

    def start(self) -> str:
        return self._call(inject="Begin the interview now.")

    def reply(self, candidate_text: str) -> str:
        self.history.append({"role": "user", "content": candidate_text})
        return self._call()

    def _call(self, inject: str | None = None) -> str:
        if not self.client:
            if not OPENAI_AVAILABLE:
                return "⚠ OpenAI package not installed.  Run:  pip install openai"
            return "⚠ Please enter a valid OpenAI API key in the setup screen."

        messages = list(self.history)
        if inject:
            messages.append({"role": "user", "content": inject})

        try:
            resp = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=350,
                temperature=0.75,
            )
            content = resp.choices[0].message.content.strip()
            self.history.append({"role": "assistant", "content": content})
            return content
        except Exception as exc:
            err = str(exc)
            if "api_key" in err.lower() or "authentication" in err.lower() or "401" in err:
                return "⚠ Invalid API key. Please check your OpenAI key in settings."
            if "quota" in err.lower() or "429" in err:
                return "⚠ API quota exceeded. Please check your OpenAI account."
            return f"⚠ API Error: {err}"


# ══════════════════════════════════════════════════════════════════════════════
# Text-to-Speech Engine  (queue-based, thread-safe)
# ══════════════════════════════════════════════════════════════════════════════

class TTSEngine:
    """Thread-safe TTS wrapper around pyttsx3."""

    def __init__(self, enabled: bool = True):
        self.available = False
        self._q: queue.Queue = queue.Queue()
        if enabled and TTS_AVAILABLE:
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()

    def _worker(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", TTS_SPEECH_RATE)
            # Prefer a female voice
            for v in engine.getProperty("voices"):
                name = v.name.lower()
                if any(w in name for w in ("female", "zira", "hazel", "kate", "samantha")):
                    engine.setProperty("voice", v.id)
                    break
            self.available = True
            while True:
                text = self._q.get()
                if text is None:
                    break
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    pass
        except Exception:
            pass

    def speak(self, text: str):
        if self.available:
            self._q.put(text)

    def stop(self):
        self._q.put(None)


# ══════════════════════════════════════════════════════════════════════════════
# Face / Eye Animator
# ══════════════════════════════════════════════════════════════════════════════

class FaceAnimator:
    """
    Draws a realistic human face with animated, blinking eyes on a tk.Canvas.

    Pupils respond to interview state:
      - 'idle'      : gentle random drift, occasional glances
      - 'thinking'  : eyes look upward / to the side
      - 'speaking'  : eyes look directly at the viewer
      - 'listening' : eyes look attentively forward and slightly down
    """

    W, H = 340, 400   # canvas dimensions

    # Eye geometry
    LX, RX = 112, 228   # left/right eye centre x
    EY      = 160        # eye centre y
    EW      = 50         # eye semi-width
    EH      = 31         # eye semi-height
    IR      = 20         # iris radius
    PR      = 11         # pupil radius

    def __init__(self, parent: tk.Widget):
        self.canvas = tk.Canvas(
            parent,
            width=self.W, height=self.H,
            bg=C["bg_card"], highlightthickness=0,
        )
        self.canvas.pack(expand=True, fill="both")

        # Pupil state  [dx, dy]  (current and target offsets)
        self._lp = [0.0, 0.0]; self._lt = [0.0, 0.0]
        self._rp = [0.0, 0.0]; self._rt = [0.0, 0.0]

        # Blink state
        self._blink_t   = 0.0   # 0 = open, 1 = closed
        self._blink_dir = 0     # 1 = closing, -1 = opening
        self._next_blink = time.time() + random.uniform(1.5, 3.5)

        # Mode
        self._mode = "idle"

        self._build_face()
        self._tick()          # start animation loop

    # ── Geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _smoothstep(t: float) -> float:
        return t * t * (3 - 2 * t)

    def _top_lid_pts(self, cx, cy, p):
        """Top eyelid polygon.  p=0 → open (thin strip), p=1 → eye closed."""
        ew, eh = self.EW, self.EH
        lid_y = (cy - eh + 3) + p * (2 * eh - 5)
        crv   = int(eh * 0.22)
        return [
            cx - ew - 10, cy - eh - 20,
            cx,           cy - eh - 26,
            cx + ew + 10, cy - eh - 20,
            cx + ew + 2,  lid_y,
            cx,           lid_y + crv,
            cx - ew - 2,  lid_y,
        ]

    def _bot_lid_pts(self, cx, cy, p):
        """Bottom eyelid polygon.  p=0 → thin lower edge, p=1 → rises up."""
        ew, eh = self.EW, self.EH
        rise  = int(p * eh * 0.40)
        bot_y = cy + eh - 1 - rise
        crv   = int(eh * 0.12)
        return [
            cx - ew - 10, cy + eh + 20,
            cx,           cy + eh + 22,
            cx + ew + 10, cy + eh + 20,
            cx + ew + 2,  bot_y,
            cx,           bot_y - crv,
            cx - ew - 2,  bot_y,
        ]

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _build_face(self):
        c = self.canvas
        W, H = self.W, self.H
        cx = W // 2

        # ── Background subtle radial glow behind face ──
        for r, alpha in [(130, "#1c2940"), (115, "#162035"), (100, "#10192c")]:
            c.create_oval(cx - r, 28, cx + r, H - 25,
                          fill=alpha, outline="", tags="glow")

        # ── Head shadow / depth ──
        c.create_oval(cx - 88, 54, cx + 88, H - 30,
                      fill=SKIN_SHADE, outline="", tags="head_shadow")
        # ── Head ──
        c.create_oval(cx - 84, 50, cx + 84, H - 34,
                      fill=SKIN, outline=SKIN_DARK, width=2, tags="head")

        # Cheek highlights
        c.create_oval(cx - 74, 170, cx - 40, 220,
                      fill="#e0a87c", outline="", tags="cheek_l")
        c.create_oval(cx + 40, 170, cx + 74, 220,
                      fill="#e0a87c", outline="", tags="cheek_r")

        # ── Ears ──
        c.create_oval(cx - 100, 158, cx - 82, 215,
                      fill=SKIN_DARK, outline=SKIN_SHADE, width=1, tags="ear_l")
        c.create_oval(cx + 82,  158, cx + 100, 215,
                      fill=SKIN_DARK, outline=SKIN_SHADE, width=1, tags="ear_r")
        # Inner ear detail
        c.create_oval(cx - 97,  166, cx - 87,  204,
                      fill=SKIN_SHADE, outline="", tags="ear_l_inner")
        c.create_oval(cx + 87,  166, cx + 97,  204,
                      fill=SKIN_SHADE, outline="", tags="ear_r_inner")

        # ── Neck ──
        c.create_rectangle(cx - 20, H - 52, cx + 20, H - 18,
                           fill=SKIN_DARK, outline="", tags="neck")

        # ── Eye whites ──
        lx, rx = self.LX, self.RX
        ey, ew, eh = self.EY, self.EW, self.EH

        for tag, ox in (("white_l", lx), ("white_r", rx)):
            # Shadow ring
            c.create_oval(ox - ew - 1, ey - eh - 1,
                          ox + ew + 1, ey + eh + 1,
                          fill="#d8d0c0", outline="", tags=tag + "_shadow")
            c.create_oval(ox - ew, ey - eh, ox + ew, ey + eh,
                          fill=EYE_WHITE, outline="#c8c0b0", width=1, tags=tag)

        # ── Irises ── (three concentric rings for depth)
        ir = self.IR
        for side, ox in (("l", lx), ("r", rx)):
            c.create_oval(ox - ir - 2, ey - ir - 2, ox + ir + 2, ey + ir + 2,
                          fill=IRIS_DARK, outline="", tags=f"iris_o_{side}")
            c.create_oval(ox - ir, ey - ir, ox + ir, ey + ir,
                          fill=IRIS_MID, outline="", tags=f"iris_{side}")
            ri = int(ir * 0.62)
            c.create_oval(ox - ri, ey - ri, ox + ri, ey + ri,
                          fill=IRIS_LITE, outline="", tags=f"iris_i_{side}")
            # Subtle radial lines (drawn as a lighter inner ring)
            rr = int(ir * 0.38)
            c.create_oval(ox - rr, ey - rr, ox + rr, ey + rr,
                          fill=IRIS_RING, outline="", tags=f"iris_c_{side}")

        # ── Pupils ──
        pr = self.PR
        for side, ox in (("l", lx), ("r", rx)):
            c.create_oval(ox - pr, ey - pr, ox + pr, ey + pr,
                          fill=PUPIL_COL, outline="", tags=f"pupil_{side}")

        # ── Catchlights (highlights) ──
        for side, ox in (("l", lx), ("r", rx)):
            hx, hy = ox - pr // 3, ey - pr // 3
            c.create_oval(hx - 4, hy - 4, hx + 4, hy + 4,
                          fill=HILITE_COL, outline="", tags=f"hl1_{side}")
            hx2, hy2 = ox + pr // 3, ey - pr // 2
            c.create_oval(hx2 - 2, hy2 - 2, hx2 + 2, hy2 + 2,
                          fill=HILITE_COL, outline="", tags=f"hl2_{side}")

        # ── Eyelid overlays (ON TOP of iris/pupil) ──
        for side, ox in (("l", lx), ("r", rx)):
            pts_t = self._top_lid_pts(ox, ey, 0.0)
            c.create_polygon(*pts_t, fill=SKIN, outline=SKIN,
                             smooth=True, tags=f"lid_top_{side}")
            pts_b = self._bot_lid_pts(ox, ey, 0.0)
            c.create_polygon(*pts_b, fill=SKIN, outline=SKIN,
                             smooth=True, tags=f"lid_bot_{side}")

        # ── Eyelashes (top) ──
        for side, ox in (("l", lx), ("r", rx)):
            for i, deg in enumerate(range(-70, 75, 14)):
                rad = math.radians(deg - 90)
                cx2 = ox + (ew - 3) * math.cos(rad)
                cy2 = ey + (eh - 2) * math.sin(rad)
                if cy2 > ey + 2:
                    continue   # skip bottom lashes
                length = random.uniform(5, 9)
                curl   = math.radians(random.uniform(-15, 5))
                tx = cx2 + length * math.cos(rad + curl)
                ty = cy2 + length * math.sin(rad + curl)
                c.create_line(cx2, cy2, tx, ty,
                              fill="#111111", width=1.5,
                              capstyle="round", tags=f"lash_{side}_{i}")

        # ── Eyebrows ──
        brow_y = ey - eh - 17
        # Left brow (natural arch)
        c.create_line(lx - 30, brow_y + 5,
                      lx - 12, brow_y,
                      lx + 20, brow_y + 6,
                      fill=BROW_COL, width=5,
                      capstyle="round", joinstyle="round",
                      smooth=True, tags="brow_l")
        # Right brow
        c.create_line(rx - 20, brow_y + 6,
                      rx + 12, brow_y,
                      rx + 30, brow_y + 5,
                      fill=BROW_COL, width=5,
                      capstyle="round", joinstyle="round",
                      smooth=True, tags="brow_r")
        # Brow highlight
        c.create_line(lx - 28, brow_y + 6,
                      lx + 18, brow_y + 7,
                      fill="#c08060", width=1,
                      smooth=True, tags="brow_hl_l")
        c.create_line(rx - 18, brow_y + 7,
                      rx + 28, brow_y + 6,
                      fill="#c08060", width=1,
                      smooth=True, tags="brow_hl_r")

        # ── Nose ──
        ny = ey + 58
        nx = cx
        # Bridge
        c.create_line(nx, ey + 14, nx - 3, ny - 4, nx + 3, ny - 4,
                      fill=SKIN_DARK, width=1, smooth=True, tags="nose_bridge")
        # Tip
        c.create_oval(nx - 10, ny - 8, nx + 10, ny + 4,
                      fill=SKIN_DARK, outline="", tags="nose_tip")
        # Nostrils
        c.create_oval(nx - 15, ny - 4, nx - 5, ny + 5,
                      fill=SKIN_SHADE, outline="", tags="nostril_l")
        c.create_oval(nx + 5,  ny - 4, nx + 15, ny + 5,
                      fill=SKIN_SHADE, outline="", tags="nostril_r")

        # ── Mouth / lips ──
        my = ny + 34
        # Upper lip bow
        c.create_line(nx - 18, my,
                      nx - 8,  my - 5,
                      nx,      my - 6,
                      nx + 8,  my - 5,
                      nx + 18, my,
                      fill=LIP_DARK, width=2,
                      smooth=True, capstyle="round", tags="lip_top")
        # Lower lip
        c.create_line(nx - 16, my,
                      nx,      my + 9,
                      nx + 16, my,
                      fill=LIP_COL, width=3,
                      smooth=True, capstyle="round", tags="lip_bot")
        # Lip shine
        c.create_line(nx - 6, my + 4, nx + 6, my + 4,
                      fill="#e09080", width=1,
                      smooth=True, tags="lip_shine")
        # Mouth line
        c.create_line(nx - 16, my, nx + 16, my,
                      fill=LIP_DARK, width=1,
                      smooth=True, tags="mouth_line")

        # ── Professional collar / suit ──
        shirt_y = H - 34
        c.create_polygon(
            cx - 84, H,
            cx - 84, shirt_y,
            cx - 28, shirt_y - 28,
            cx,      shirt_y - 12,
            cx + 28, shirt_y - 28,
            cx + 84, shirt_y,
            cx + 84, H,
            fill="#1a2b45", outline="#0f1e33", width=1, tags="shirt"
        )
        # White shirt / collar
        c.create_polygon(
            cx - 18, shirt_y - 20,
            cx,      shirt_y - 10,
            cx + 18, shirt_y - 20,
            cx + 10, shirt_y + 5,
            cx - 10, shirt_y + 5,
            fill="#e8e8e8", outline="#cccccc", tags="collar"
        )
        # Tie
        c.create_polygon(
            cx - 7,  shirt_y - 18,
            cx + 7,  shirt_y - 18,
            cx + 5,  shirt_y + 8,
            cx,      shirt_y + 26,
            cx - 5,  shirt_y + 8,
            fill="#8b1a1a", outline="#6b1010", tags="tie"
        )
        c.create_oval(cx - 5, shirt_y - 20, cx + 5, shirt_y - 10,
                      fill="#a02020", outline="", tags="tie_knot")

    # ── Iris / pupil repositioning ────────────────────────────────────────────

    def _move_iris(self, side: str, cx: int, cy: int, dx: float, dy: float):
        c  = self.canvas
        ir = self.IR
        pr = self.PR
        x, y = cx + dx, cy + dy

        c.coords(f"iris_o_{side}", x - ir - 2, y - ir - 2, x + ir + 2, y + ir + 2)
        c.coords(f"iris_{side}",   x - ir,     y - ir,     x + ir,     y + ir)
        ri = int(ir * 0.62)
        c.coords(f"iris_i_{side}", x - ri, y - ri, x + ri, y + ri)
        rr = int(ir * 0.38)
        c.coords(f"iris_c_{side}", x - rr, y - rr, x + rr, y + rr)

        c.coords(f"pupil_{side}",  x - pr, y - pr, x + pr, y + pr)

        hx, hy = x - pr // 3, y - pr // 3
        c.coords(f"hl1_{side}", hx - 4, hy - 4, hx + 4, hy + 4)
        hx2, hy2 = x + pr // 3, y - pr // 2
        c.coords(f"hl2_{side}", hx2 - 2, hy2 - 2, hx2 + 2, hy2 + 2)

    # ── Animation loop ────────────────────────────────────────────────────────

    def _tick(self):
        now = time.time()

        # ── Blink ──
        if self._blink_dir == 0 and now >= self._next_blink:
            self._blink_dir = 1
        if self._blink_dir == 1:
            self._blink_t = min(1.0, self._blink_t + 0.28)
            if self._blink_t >= 1.0:
                self._blink_dir = -1
        elif self._blink_dir == -1:
            self._blink_t = max(0.0, self._blink_t - 0.22)
            if self._blink_t <= 0.0:
                self._blink_dir = 0
                self._next_blink = now + random.uniform(2.5, 6.5)

        sbp = self._smoothstep(self._blink_t)

        # ── Pupil target logic ──
        m = self._mode
        if m == "thinking":
            if random.random() < 0.04:
                dx = random.uniform(-8, 8)
                dy = random.uniform(-10, -2)
                self._lt = [dx, dy]
                self._rt = [dx + random.uniform(-1.5, 1.5), dy]
        elif m == "speaking":
            # Look at the viewer; gentle micro-saccades
            if random.random() < 0.018:
                d = random.uniform(-2.5, 2.5)
                self._lt = [d, 0.0]
                self._rt = [d, 0.0]
            else:
                self._lt[0] *= 0.92; self._lt[1] *= 0.92
                self._rt[0] *= 0.92; self._rt[1] *= 0.92
        elif m == "listening":
            # Attentive, slightly downward gaze
            self._lt = [0.5, 2.5]
            self._rt = [0.5, 2.5]
        else:
            # Idle: occasional wandering
            if random.random() < 0.012:
                d = random.uniform(-5, 5)
                self._lt = [d, random.uniform(-3, 3)]
                self._rt = [d, random.uniform(-3, 3)]

        # Smooth lerp
        spd = 0.18
        for i in range(2):
            self._lp[i] += (self._lt[i] - self._lp[i]) * spd
            self._rp[i] += (self._rt[i] - self._rp[i]) * spd

        # ── Apply updates ──
        self._move_iris("l", self.LX, self.EY, self._lp[0], self._lp[1])
        self._move_iris("r", self.RX, self.EY, self._rp[0], self._rp[1])

        c = self.canvas
        lx, rx, ey = self.LX, self.RX, self.EY
        c.coords("lid_top_l", *self._top_lid_pts(lx, ey, sbp))
        c.coords("lid_top_r", *self._top_lid_pts(rx, ey, sbp))
        c.coords("lid_bot_l", *self._bot_lid_pts(lx, ey, sbp))
        c.coords("lid_bot_r", *self._bot_lid_pts(rx, ey, sbp))

        self.canvas.after(48, self._tick)   # ~20 fps

    # ── Public API ────────────────────────────────────────────────────────────

    def set_mode(self, mode: str):
        """
        Set expression mode:
          'idle'      – relaxed random drift
          'thinking'  – eyes look up/aside (AI is processing)
          'speaking'  – eyes look at viewer (AI is talking)
          'listening' – attentive forward gaze (user is typing/speaking)
        """
        self._mode = mode
        if mode == "thinking":
            self._lt = [-4.0, -7.0]
            self._rt = [4.0, -7.0]
        elif mode in ("speaking", "listening"):
            self._lt = [0.0, 0.0]
            self._rt = [0.0, 0.0]

    @property
    def widget(self) -> tk.Canvas:
        return self.canvas


# ══════════════════════════════════════════════════════════════════════════════
# Setup Screen
# ══════════════════════════════════════════════════════════════════════════════

class SetupScreen:
    """
    Configuration screen where the user enters their API key, job title,
    desired tone, custom questions, and extra instructions.
    """

    def __init__(self, app: "App"):
        self.app = app
        self.frame = tk.Frame(app.root, bg=C["bg"])
        self._build()
        self._load_config()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _lbl(self, parent, text, **kw):
        defaults = dict(bg=C["bg_card"], fg=C["text"], font=("Helvetica", 11),
                        anchor="w")
        defaults.update(kw)
        return tk.Label(parent, text=text, **defaults)

    def _entry(self, parent, textvariable=None, show=None, **kw):
        defaults = dict(bg=C["bg_input"], fg=C["text_bright"], relief="flat",
                        font=("Helvetica", 11), insertbackground=C["text_bright"],
                        highlightthickness=1, highlightbackground=C["border"],
                        highlightcolor=C["accent_blue"])
        defaults.update(kw)
        e = tk.Entry(parent, textvariable=textvariable, show=show, **defaults)
        return e

    def _textarea(self, parent, height=4, **kw):
        defaults = dict(bg=C["bg_input"], fg=C["text_bright"], relief="flat",
                        font=("Helvetica", 11), insertbackground=C["text_bright"],
                        wrap="word", highlightthickness=1,
                        highlightbackground=C["border"],
                        highlightcolor=C["accent_blue"])
        defaults.update(kw)
        t = tk.Text(parent, height=height, **defaults)
        return t

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build(self):
        # Scrollable outer container
        outer = tk.Frame(self.frame, bg=C["bg"])
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, bg=C["bg"], highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical",
                                command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=C["bg"])
        canvas_window = canvas.create_window(0, 0, window=inner, anchor="nw")

        def _resize(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_window, width=event.width)

        inner.bind("<Configure>", _resize)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(
            canvas_window, width=e.width))

        # ── Header ──
        hdr = tk.Frame(inner, bg="#0a0f1a", pady=20)
        hdr.pack(fill="x")

        tk.Label(hdr, text="🎙  Interview AI",
                 bg="#0a0f1a", fg=C["text_bright"],
                 font=("Helvetica", 28, "bold")).pack()
        tk.Label(hdr,
                 text="Practice job interviews with a realistic AI interviewer.\n"
                      "Animated eyes help you build eye-contact confidence.",
                 bg="#0a0f1a", fg=C["text_dim"],
                 font=("Helvetica", 12)).pack(pady=(4, 0))

        # ── Card ──
        card = tk.Frame(inner, bg=C["bg_card"],
                        padx=36, pady=28, relief="flat")
        card.pack(fill="both", expand=True, padx=60, pady=24)

        def section_header(text):
            tk.Label(card, text=text, bg=C["bg_card"],
                     fg=C["accent_blue"], font=("Helvetica", 12, "bold"),
                     anchor="w").pack(fill="x", pady=(14, 2))
            tk.Frame(card, bg=C["border"], height=1).pack(fill="x")

        def field_row(label_text, widget):
            row = tk.Frame(card, bg=C["bg_card"])
            row.pack(fill="x", pady=4)
            lbl = self._lbl(row, label_text)
            lbl.config(width=18)
            lbl.pack(side="left")
            widget.pack(side="left", fill="x", expand=True)

        # ── API Key ──
        section_header("  🔑  OpenAI API Key")
        self._lbl(card,
                  "Your key is kept in memory for this session only and sent exclusively to OpenAI.",
                  fg=C["text_dim"], font=("Helvetica", 10)).pack(fill="x",
                                                                  pady=(2, 6))
        self._api_key_var = tk.StringVar()
        api_row = tk.Frame(card, bg=C["bg_card"])
        api_row.pack(fill="x", pady=(0, 4))
        self._api_entry = self._entry(api_row, textvariable=self._api_key_var,
                                      show="•")
        self._api_entry.pack(side="left", fill="x", expand=True)
        self._show_key_var = tk.BooleanVar(value=False)

        def toggle_show():
            self._api_entry.config(
                show="" if self._show_key_var.get() else "•")

        tk.Checkbutton(api_row, text="Show", variable=self._show_key_var,
                       command=toggle_show,
                       bg=C["bg_card"], fg=C["text_dim"],
                       selectcolor=C["bg_input"],
                       activebackground=C["bg_card"],
                       font=("Helvetica", 10)).pack(side="left", padx=(8, 0))

        # ── Job Details ──
        section_header("  💼  Interview Setup")

        self._job_var = tk.StringVar()
        field_row("Job title / role:", self._entry(card, textvariable=self._job_var))

        self._tone_var = tk.StringVar(value="Professional")
        tone_lbl_row = tk.Frame(card, bg=C["bg_card"])
        tone_lbl_row.pack(fill="x", pady=4)
        lbl_style = self._lbl(tone_lbl_row, "Interview style:")
        lbl_style.config(width=18)
        lbl_style.pack(side="left")
        style_frame = tk.Frame(tone_lbl_row, bg=C["bg_card"])
        style_frame.pack(side="left")
        for tone in TONES:
            rb = tk.Radiobutton(style_frame, text=tone,
                                variable=self._tone_var, value=tone,
                                bg=C["bg_card"], fg=C["text"],
                                selectcolor=C["bg_input"],
                                activebackground=C["bg_card"],
                                font=("Helvetica", 10))
            rb.pack(side="left", padx=4)

        # ── Custom Questions ──
        section_header("  ❓  Custom Questions  (optional)")
        self._lbl(card,
                  "Type any specific questions you want the interviewer to ask you.",
                  fg=C["text_dim"], font=("Helvetica", 10)).pack(fill="x",
                                                                   pady=(2, 4))
        self._questions_txt = self._textarea(card, height=4)
        self._questions_txt.pack(fill="x")

        # ── Extra Instructions ──
        section_header("  📋  Extra Instructions  (optional)")
        self._lbl(card,
                  "E.g. 'Ask about my leadership experience' or 'Be very challenging'.",
                  fg=C["text_dim"], font=("Helvetica", 10)).pack(fill="x",
                                                                   pady=(2, 4))
        self._instructions_txt = self._textarea(card, height=3)
        self._instructions_txt.pack(fill="x")

        # ── Options ──
        section_header("  ⚙️   Options")
        opts_row = tk.Frame(card, bg=C["bg_card"])
        opts_row.pack(fill="x", pady=6)

        self._tts_var = tk.BooleanVar(value=TTS_AVAILABLE)
        tk.Checkbutton(opts_row,
                       text=f"Text-to-speech  {'✓ available' if TTS_AVAILABLE else '✗ install pyttsx3'}",
                       variable=self._tts_var,
                       state="normal" if TTS_AVAILABLE else "disabled",
                       bg=C["bg_card"], fg=C["text"],
                       selectcolor=C["bg_input"],
                       activebackground=C["bg_card"],
                       font=("Helvetica", 10)).pack(side="left", padx=(0, 20))

        self._stt_var = tk.BooleanVar(value=STT_AVAILABLE)
        tk.Checkbutton(opts_row,
                       text=f"Voice input  {'✓ available' if STT_AVAILABLE else '✗ install pyaudio'}",
                       variable=self._stt_var,
                       state="normal" if STT_AVAILABLE else "disabled",
                       bg=C["bg_card"], fg=C["text"],
                       selectcolor=C["bg_input"],
                       activebackground=C["bg_card"],
                       font=("Helvetica", 10)).pack(side="left")

        # ── Eye-contact tip ──
        tip_frame = tk.Frame(card, bg="#0f2035", padx=14, pady=10)
        tip_frame.pack(fill="x", pady=(18, 4))
        tk.Label(tip_frame,
                 text="👁  Eye-Contact Tip",
                 bg="#0f2035", fg=C["accent_gold"],
                 font=("Helvetica", 11, "bold"),
                 anchor="w").pack(fill="x")
        tk.Label(tip_frame,
                 text="During the interview, look at the interviewer's eyes on screen — "
                      "not at the chat text. This simulates real eye contact and "
                      "builds the habit before your actual interview.",
                 bg="#0f2035", fg=C["text"],
                 font=("Helvetica", 10),
                 wraplength=600, justify="left", anchor="w").pack(fill="x")

        # ── Start button ──
        btn_frame = tk.Frame(card, bg=C["bg_card"])
        btn_frame.pack(fill="x", pady=(24, 4))
        tk.Button(btn_frame,
                  text="  Start Interview  ▶",
                  command=self._start,
                  bg=C["accent"], fg="white",
                  font=("Helvetica", 14, "bold"),
                  relief="flat", padx=24, pady=10,
                  cursor="hand2",
                  activebackground="#3cb550",
                  activeforeground="white").pack(side="left")

        # Status label
        self._status_lbl = tk.Label(btn_frame, text="",
                                    bg=C["bg_card"], fg=C["text_dim"],
                                    font=("Helvetica", 10))
        self._status_lbl.pack(side="left", padx=16)

    def _start(self):
        api_key = self._api_key_var.get().strip()
        job     = self._job_var.get().strip()

        if not job:
            messagebox.showwarning("Missing Info",
                                   "Please enter the job title you are applying for.",
                                   parent=self.app.root)
            return

        # Warn (but don't block) if no API key
        if not api_key:
            if not messagebox.askyesno(
                    "No API Key",
                    "No OpenAI API key entered. The interviewer won't be able to respond.\n\n"
                    "Continue anyway? (You can enter your key later in settings.)",
                    parent=self.app.root):
                return

        config = {
            "api_key":          api_key,
            "job_title":        job,
            "tone":             self._tone_var.get(),
            "custom_questions": self._questions_txt.get("1.0", "end-1c").strip(),
            "instructions":     self._instructions_txt.get("1.0", "end-1c").strip(),
            "tts_enabled":      self._tts_var.get(),
            "stt_enabled":      self._stt_var.get(),
        }
        self._save_config(config)
        self.app.start_interview(config)

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save_config(self, config: dict):
        try:
            # Never save the API key to disk in plain text — user re-enters it
            safe = {k: v for k, v in config.items() if k != "api_key"}
            CONFIG_FILE.write_text(json.dumps(safe, indent=2))
        except Exception:
            pass

    def _load_config(self):
        try:
            data = json.loads(CONFIG_FILE.read_text())
            self._job_var.set(data.get("job_title", ""))
            self._tone_var.set(data.get("tone", "Professional"))
            if data.get("custom_questions"):
                self._questions_txt.insert("1.0", data["custom_questions"])
            if data.get("instructions"):
                self._instructions_txt.insert("1.0", data["instructions"])
        except Exception:
            pass

    def show(self):
        self.frame.pack(fill="both", expand=True)

    def hide(self):
        self.frame.pack_forget()


# ══════════════════════════════════════════════════════════════════════════════
# Interview Screen
# ══════════════════════════════════════════════════════════════════════════════

class InterviewScreen:
    """
    The main interview view.  Left panel shows the animated face;
    right panel shows the conversation transcript and input controls.
    """

    INTERVIEWER_TAG = "iv"
    USER_TAG        = "usr"
    SYSTEM_TAG      = "sys"
    ERROR_TAG       = "err"

    def __init__(self, app: "App"):
        self.app  = app
        self.root = app.root
        self.frame = tk.Frame(app.root, bg=C["bg"])

        self._interviewer: AIInterviewer | None = None
        self._tts:         TTSEngine | None     = None
        self._stt_engine: sr.Recognizer | None  = None
        self._listening   = False
        self._busy        = False    # waiting for AI response
        self._face: FaceAnimator | None = None

        self._build()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build(self):
        # ── Top bar ──
        bar = tk.Frame(self.frame, bg="#0a0f1a", padx=16, pady=10)
        bar.pack(fill="x", side="top")

        self._title_lbl = tk.Label(bar, text="Interview AI",
                                   bg="#0a0f1a", fg=C["text_bright"],
                                   font=("Helvetica", 15, "bold"))
        self._title_lbl.pack(side="left")

        self._mode_lbl = tk.Label(bar, text="",
                                  bg="#0a0f1a", fg=C["accent_gold"],
                                  font=("Helvetica", 11, "italic"))
        self._mode_lbl.pack(side="left", padx=20)

        # Navigation buttons on the right
        btn_kw = dict(relief="flat", font=("Helvetica", 10),
                      cursor="hand2", padx=10, pady=4)
        tk.Button(bar, text="⚙ Settings",
                  command=self._go_settings,
                  bg=C["bg_input"], fg=C["text"],
                  activebackground=C["border"], **btn_kw).pack(side="right",
                                                                padx=4)
        tk.Button(bar, text="↩ New Interview",
                  command=self._new_interview,
                  bg=C["bg_input"], fg=C["text"],
                  activebackground=C["border"], **btn_kw).pack(side="right",
                                                                padx=4)

        # ── Main content  (left face pane + right chat pane) ──
        content = tk.Frame(self.frame, bg=C["bg"])
        content.pack(fill="both", expand=True)

        # Left panel — face
        left = tk.Frame(content, bg=C["bg_card"], width=360)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        self._face_container = tk.Frame(left, bg=C["bg_card"])
        self._face_container.pack(expand=True, fill="both", pady=(20, 0))

        # Status beneath the face
        self._status_lbl = tk.Label(left, text="● Waiting to start…",
                                    bg=C["bg_card"], fg=C["text_dim"],
                                    font=("Helvetica", 10, "italic"))
        self._status_lbl.pack(pady=(6, 4))

        # Eye-contact reminder
        tk.Label(left,
                 text="👁  Look at the eyes, not the text",
                 bg=C["bg_card"], fg=C["accent_gold"],
                 font=("Helvetica", 9, "italic"),
                 wraplength=300).pack(pady=(0, 12))

        # Right panel — chat
        right = tk.Frame(content, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True)

        # Transcript
        chat_frame = tk.Frame(right, bg=C["bg"])
        chat_frame.pack(fill="both", expand=True, padx=8, pady=(8, 4))

        self._chat = tk.Text(
            chat_frame,
            bg=C["bg_card"], fg=C["text"],
            font=("Helvetica", 11),
            relief="flat", padx=14, pady=12,
            wrap="word", state="disabled",
            highlightthickness=0,
            spacing1=4, spacing3=4,
        )
        scrollbar = ttk.Scrollbar(chat_frame, orient="vertical",
                                  command=self._chat.yview)
        self._chat.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self._chat.pack(side="left", fill="both", expand=True)

        # Configure tags
        self._chat.tag_config(self.INTERVIEWER_TAG,
                              foreground=C["interviewer"],
                              font=("Helvetica", 11, "bold"),
                              spacing1=6)
        self._chat.tag_config("iv_body",
                              foreground=C["text"],
                              font=("Helvetica", 11),
                              lmargin1=10, lmargin2=10,
                              spacing3=10)
        self._chat.tag_config(self.USER_TAG,
                              foreground=C["user"],
                              font=("Helvetica", 11, "bold"),
                              spacing1=6)
        self._chat.tag_config("usr_body",
                              foreground=C["text"],
                              font=("Helvetica", 11),
                              lmargin1=10, lmargin2=10,
                              spacing3=10)
        self._chat.tag_config(self.SYSTEM_TAG,
                              foreground=C["system"],
                              font=("Helvetica", 10, "italic"),
                              justify="center", spacing1=6, spacing3=6)
        self._chat.tag_config(self.ERROR_TAG,
                              foreground=C["error"],
                              font=("Helvetica", 10),
                              spacing1=4, spacing3=4)

        # ── Thinking indicator (animated dots) ──
        self._thinking_visible = False
        self._dot_count = 0

        # ── Input bar ──
        input_bar = tk.Frame(right, bg=C["bg_input"], pady=8, padx=8)
        input_bar.pack(fill="x", side="bottom")

        self._input_var = tk.StringVar()
        self._input_entry = tk.Entry(
            input_bar,
            textvariable=self._input_var,
            bg=C["bg_input"], fg=C["text_bright"],
            font=("Helvetica", 12),
            relief="flat",
            insertbackground=C["text_bright"],
            highlightthickness=1,
            highlightbackground=C["border"],
            highlightcolor=C["accent_blue"],
        )
        self._input_entry.pack(side="left", fill="x", expand=True, ipady=6,
                               padx=(0, 8))
        self._input_entry.bind("<Return>", lambda _: self._send())

        btn_kw2 = dict(relief="flat", font=("Helvetica", 11, "bold"),
                       padx=14, pady=6, cursor="hand2")
        self._send_btn = tk.Button(input_bar, text="Send ▶",
                                   command=self._send,
                                   bg=C["accent"], fg="white",
                                   activebackground="#3cb550", **btn_kw2)
        self._send_btn.pack(side="left", padx=(0, 6))

        mic_state = "normal" if STT_AVAILABLE else "disabled"
        self._mic_btn = tk.Button(input_bar, text="🎤",
                                  command=self._toggle_mic,
                                  bg=C["bg_card"], fg=C["text"],
                                  activebackground=C["border"],
                                  state=mic_state, **btn_kw2)
        self._mic_btn.pack(side="left")

        self._char_lbl = tk.Label(input_bar, text="",
                                  bg=C["bg_input"], fg=C["text_dim"],
                                  font=("Helvetica", 9))
        self._char_lbl.pack(side="right", padx=4)
        self._input_var.trace_add("write",
                                  lambda *_: self._char_lbl.config(
                                      text=f"{len(self._input_var.get())} chars"))

    # ── Start interview ───────────────────────────────────────────────────────

    def start(self, config: dict):
        self._config = config

        # Title
        job   = config["job_title"]
        tone  = config["tone"]
        self._title_lbl.config(
            text=f"🎙  Interview — {job}  |  {tone} style")

        # Build AI
        self._interviewer = AIInterviewer(
            api_key         = config["api_key"],
            job_title       = job,
            tone            = tone,
            custom_questions= config["custom_questions"],
            instructions    = config["instructions"],
        )

        # TTS
        self._tts = TTSEngine(enabled=config.get("tts_enabled", False))

        # STT
        if config.get("stt_enabled") and STT_AVAILABLE:
            self._stt_engine = sr.Recognizer()

        # Face animator
        for w in self._face_container.winfo_children():
            w.destroy()
        self._face = FaceAnimator(self._face_container)

        # Clear chat
        self._set_chat_editable(True)
        self._chat.delete("1.0", "end")
        self._set_chat_editable(False)

        self._append_system(
            f"Interview for: {job}  |  Style: {tone}\n"
            f"{'TTS ON' if self._tts and self._tts.available else 'TTS OFF'}  ·  "
            f"{'Mic ON' if self._stt_engine else 'Mic OFF'}  ·  "
            f"{datetime.now().strftime('%H:%M %d %b %Y')}"
        )

        self._set_status("● Connecting…", C["accent_gold"])
        self._set_mode_label("Thinking…")
        self._set_input_enabled(False)
        if self._face:
            self._face.set_mode("thinking")

        threading.Thread(target=self._fetch_opening, daemon=True).start()

    def _fetch_opening(self):
        if not self._interviewer:
            return
        text = self._interviewer.start()
        self.root.after(0, lambda t=text: self._on_ai_message(t))

    # ── Sending / receiving ───────────────────────────────────────────────────

    def _send(self):
        if self._busy or self._listening:
            return
        text = self._input_var.get().strip()
        if not text:
            return

        self._input_var.set("")
        self._append_message("You", text, self.USER_TAG, "usr_body")
        self._set_input_enabled(False)
        self._set_status("● Thinking…", C["accent_gold"])
        self._set_mode_label("Thinking…")
        self._busy = True
        if self._face:
            self._face.set_mode("thinking")
        self._start_thinking_dots()

        def worker():
            if self._interviewer:
                response = self._interviewer.reply(text)
            else:
                response = "⚠ Interviewer not initialised."
            self.root.after(0, lambda r=response: self._on_ai_message(r))

        threading.Thread(target=worker, daemon=True).start()

    def _on_ai_message(self, text: str):
        self._stop_thinking_dots()
        self._busy = False

        if text.startswith("⚠"):
            self._append_message("System", text, self.ERROR_TAG, self.ERROR_TAG)
            self._set_status("● Error — check your API key", C["accent_red"])
            self._set_mode_label("")
        else:
            self._append_message("Interviewer", text, self.INTERVIEWER_TAG, "iv_body")
            self._set_status("● Listening…", C["accent"])
            self._set_mode_label("Your turn")
            if self._tts:
                self._tts.speak(text)
            if self._face:
                self._face.set_mode("speaking")
                # Switch to listening mode after ~3 s
                self.root.after(3000, lambda: self._face and
                                self._face.set_mode("listening"))

        self._set_input_enabled(True)
        self._input_entry.focus_set()

    # ── Thinking dots animation ───────────────────────────────────────────────

    def _start_thinking_dots(self):
        self._thinking_visible = True
        self._dot_count = 0
        self._animate_dots()

    def _animate_dots(self):
        if not self._thinking_visible:
            return
        self._dot_count = (self._dot_count % 3) + 1
        dots = "●" * self._dot_count + "○" * (3 - self._dot_count)
        self._mode_lbl.config(text=f"Thinking {dots}")
        self.root.after(400, self._animate_dots)

    def _stop_thinking_dots(self):
        self._thinking_visible = False

    # ── Microphone input ──────────────────────────────────────────────────────

    def _toggle_mic(self):
        if self._busy:
            return
        if self._listening:
            self._listening = False
            self._mic_btn.config(bg=C["bg_card"], text="🎤")
            self._set_status("● Ready", C["accent"])
            return

        self._listening = True
        self._mic_btn.config(bg=C["accent_red"], text="⏹")
        self._set_status("● Listening (speak now)…", C["accent_blue"])
        if self._face:
            self._face.set_mode("listening")
        self._set_input_enabled(False)

        threading.Thread(target=self._do_listen, daemon=True).start()

    def _do_listen(self):
        if not self._stt_engine or not STT_AVAILABLE:
            self.root.after(0, lambda: self._on_stt_done(None, "STT not available"))
            return
        try:
            with sr.Microphone() as source:
                self._stt_engine.adjust_for_ambient_noise(source, duration=0.4)
                audio = self._stt_engine.listen(source, timeout=STT_LISTEN_TIMEOUT,
                                                phrase_time_limit=STT_PHRASE_LIMIT)
            text = self._stt_engine.recognize_google(audio)
            self.root.after(0, lambda t=text: self._on_stt_done(t, None))
        except sr.WaitTimeoutError:
            self.root.after(0, lambda: self._on_stt_done(None, "No speech detected"))
        except sr.UnknownValueError:
            self.root.after(0, lambda: self._on_stt_done(None, "Could not understand speech"))
        except Exception as e:
            self.root.after(0, lambda err=str(e): self._on_stt_done(None, err))

    def _on_stt_done(self, text: str | None, error: str | None):
        self._listening = False
        self._mic_btn.config(bg=C["bg_card"], text="🎤")
        self._set_input_enabled(True)
        if error:
            self._append_system(f"Mic: {error}")
            self._set_status("● Ready", C["accent"])
        else:
            self._input_var.set(text or "")
            self._set_status("● Transcript ready — press Send", C["accent"])
        if self._face:
            self._face.set_mode("listening")

    # ── Chat helpers ──────────────────────────────────────────────────────────

    def _set_chat_editable(self, editable: bool):
        self._chat.config(state="normal" if editable else "disabled")

    def _append_message(self, speaker: str, body: str,
                        speaker_tag: str, body_tag: str):
        self._set_chat_editable(True)
        self._chat.insert("end", f"{speaker}:\n", speaker_tag)
        self._chat.insert("end", f"{body}\n\n", body_tag)
        self._chat.see("end")
        self._set_chat_editable(False)

    def _append_system(self, text: str):
        self._set_chat_editable(True)
        self._chat.insert("end", f"— {text} —\n\n", self.SYSTEM_TAG)
        self._chat.see("end")
        self._set_chat_editable(False)

    # ── UI state helpers ──────────────────────────────────────────────────────

    def _set_input_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self._input_entry.config(state=state)
        self._send_btn.config(state=state)

    def _set_status(self, text: str, color: str = ""):
        self._status_lbl.config(text=text,
                                fg=color or C["text_dim"])

    def _set_mode_label(self, text: str):
        self._mode_lbl.config(text=text)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _go_settings(self):
        if not messagebox.askyesno(
                "Return to Settings?",
                "Go back to the settings screen?\n"
                "The current interview session will end.",
                parent=self.root):
            return
        self._cleanup()
        self.app.show_setup()

    def _new_interview(self):
        if not messagebox.askyesno(
                "New Interview?",
                "Start a new interview?\nThis session will end.",
                parent=self.root):
            return
        self._cleanup()
        self.app.show_setup()

    def _cleanup(self):
        if self._tts:
            self._tts.stop()
        self._listening = False
        self._busy      = False

    def show(self):
        self.frame.pack(fill="both", expand=True)

    def hide(self):
        self.frame.pack_forget()


# ══════════════════════════════════════════════════════════════════════════════
# Main Application
# ══════════════════════════════════════════════════════════════════════════════

class App:
    """Top-level application controller."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Interview AI")
        self.root.geometry("1160x780")
        self.root.minsize(900, 640)
        self.root.configure(bg=C["bg"])

        # Window icon (built from canvas to avoid needing an external file)
        self._set_icon()

        # Apply ttk theme adjustments
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Vertical.TScrollbar",
                        background=C["bg_input"],
                        troughcolor=C["bg"],
                        arrowcolor=C["text_dim"])

        self._setup   = SetupScreen(self)
        self._interview = InterviewScreen(self)

        self._setup.show()
        self.root.mainloop()

    def _set_icon(self):
        """Create a simple emoji-style icon without external files."""
        try:
            icon = tk.PhotoImage(width=32, height=32)
            self.root.iconphoto(True, icon)
        except Exception:
            pass

    def start_interview(self, config: dict):
        self._setup.hide()
        self._interview.start(config)
        self._interview.show()

    def show_setup(self):
        self._interview.hide()
        self._setup.show()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not OPENAI_AVAILABLE:
        print("⚠  openai package not found.  Install with:  pip install openai")
    if not TTS_AVAILABLE:
        print("ℹ  pyttsx3 not found — text-to-speech disabled.  "
              "Install with:  pip install pyttsx3")
    if not STT_AVAILABLE:
        print("ℹ  SpeechRecognition/pyaudio not found — microphone disabled.  "
              "Install with:  pip install SpeechRecognition pyaudio")

    App()
