# SOOCHAK Implementation Plan
## Beginner-Friendly | Interview-Safe | Placement Preparation
**Version 1.0 | Explainable Road Accident Severity Predictor**

**Who is this for?** You are a student or early-career developer preparing for campus placements or junior developer interviews. This document uses only concepts you can explain in an interview — no buzzwords you don't understand.

## Table of Contents
1. What is SOOCHAK? (The Elevator Pitch)
2. Tech Stack (And Why Each One)
3. Pages & Features
4. Design System (Colors, Fonts, Spacing)
5. Components (Reusable UI Pieces)
6. Forms & Validation
7. API Endpoints (What the Frontend Calls)
8. State Management (Where Data Lives)
9. Animations & Interactions
10. Error Handling
11. Performance Targets
12. Testing Strategy
13. Deployment
14. Interview Cheat Sheet

---

### 1. WHAT IS SOOCHAK? (The Elevator Pitch)
SOOCHAK is a web application that predicts how severe a road accident will be (fatal vs. non-fatal) based on 11 input factors like road type, speed limit, weather, vehicle type, etc.

**What makes it special?**
- It explains WHY it made a prediction using SHAP values (like "Speed Limit contributed +45% to the fatal risk")
- It lets you simulate interventions ("What if we reduced the speed limit from 80 to 40?")
- It uses a real machine learning model (XGBoost → ONNX) deployed on a free cloud server

**One-line pitch for interviews:**
"I built a system that predicts fatal road accidents using real FIR data, explains why two-wheelers on undivided highways are highest risk, and simulates what happens if you add a divider — deployed live with zero infrastructure cost."

---

### 2. TECH STACK (And Why Each One)

| Technology | What It Does | Why I Chose It | Interview Answer |
|------------|--------------|----------------|------------------|
| **Python + FastAPI** | Backend web framework | FastAPI is modern, auto-generates API docs, async support | "FastAPI gives automatic Swagger docs and handles async requests efficiently" |
| **ONNX Runtime** | Runs the ML model for predictions | Faster than running raw XGBoost, language-agnostic | "ONNX makes inference 2x faster and works across languages" |
| **XGBoost** | The actual ML algorithm | Industry standard for tabular data, great performance | "XGBoost is the go-to for structured data and handles missing values well" |
| **SHAP** | Explains model predictions | Shows which features drove the prediction | "SHAP tells us WHY the model predicted fatal — e.g., speed limit was the top factor" |
| **SQLite** | Lightweight database | Zero setup, single file, perfect for demos | "SQLite needs no server — it's just a file, ideal for portfolio projects" |
| **aiosqlite** | Async SQLite wrapper | Non-blocking database calls | "aiosqlite lets the server handle other requests while waiting for DB" |
| **Tailwind CSS** | Utility CSS framework | Rapid styling without writing custom CSS | "Tailwind speeds up UI development with pre-built utility classes" |
| **Chart.js** | Draws charts (SHAP bars, gauges) | Simple to use, canvas-based, good performance | "Chart.js renders charts on canvas, which is faster than DOM manipulation" |
| **Jinja2** | HTML templating in Python | Server-side rendering for the dashboard | "Jinja2 lets me render HTML directly from Python without a separate frontend build" |
| **Docker** | Packages the app for deployment | Same environment everywhere | "Docker ensures my app runs identically on my laptop and the cloud server" |
| **Render** | Free cloud hosting | Zero cost, auto-deploys from GitHub | "Render's free tier hosts my app with zero cost — perfect for portfolios" |

**What I DON'T use (and why):**
- **React/Vue/Angular** — Not needed. This is a single-page dashboard rendered server-side. Adding a JS framework would be overkill for a demo.
- **PostgreSQL/MySQL** — Overkill. SQLite handles <10K records perfectly.
- **Redis** — Overkill. In-memory rate limiting is sufficient for a demo.
- **AWS/GCP** — Overkill. Render free tier is enough.

---

### 3. PAGES & FEATURES

#### 3.1 Dashboard (/)
**What it is:** The main page. Everything happens here.

| Section | What It Shows | User Action |
|---------|---------------|-------------|
| **Header** | Logo + "System Online" status dot | Click logo to refresh |
| **Metrics Bar** | 4 cards: Predictions Today, Avg Inference Time, Cache Hit Rate, Model Version | Auto-refreshes every minute |
| **Prediction Form** | 11 dropdowns/text fields for accident details | Fill and click "Predict Severity" |
| **Results Panel** | Severity badge (FATAL/NON-FATAL), probability %, inference time | View result |
| **SHAP Panel** | Horizontal bar chart showing which features pushed the prediction | Hover bars for details |
| **AI Explanation Button** | "Get AI Explanation" — calls Groq LLM | Click to see plain-English explanation |
| **What-If Simulator** | Change one feature, see how probability changes | Select feature, enter new value, simulate |
| **Incident History** | Table of recent predictions with severity | Scroll, paginate |
| **Footer** | Version info, GitHub link, disclaimer | — |

#### 3.2 API Docs (/docs)
**What it is:** Auto-generated by FastAPI. Shows all endpoints, request/response formats, lets you test APIs directly.
**Why it's useful:** Recruiters can see your API design without reading code.

---

### 4. DESIGN SYSTEM

#### 4.1 Colors (Why These?)
| Color | Hex | Used For | Why This Color |
|-------|-----|----------|----------------|
| **Deep Navy** | `#0F172A` | Background, primary buttons | Conveys trust, authority, government/enterprise feel |
| **Slate** | `#1E293B` | Cards, panels | Slightly lighter than navy for depth |
| **Amber** | `#F59E0B` | Warnings, medium risk | Road safety standard — yellow/amber signs |
| **Emerald** | `#10B981` | Success, non-fatal | Green = safe, positive |
| **Rose** | `#F43F5E` | Errors, fatal predictions | Red = danger, urgent |
| **Blue** | `#3B82F6` | Links, info | Standard link color, trustworthy |

*Interview tip: "I chose a dark navy theme because it conveys authority and reduces eye strain during monitoring. Amber is the standard color for road safety warnings."*

#### 4.2 Fonts
| Font | Used For | Why |
|------|----------|-----|
| **Inter** | All text (headings, body, labels) | Clean, modern, highly readable on screens |
| **JetBrains Mono**| Numbers, probabilities, code | Monospace makes numbers align and easier to scan |

#### 4.3 Spacing
Everything uses multiples of 4px:
- Small gap: 8px
- Medium gap: 16px
- Large gap: 24px
- Section gap: 32px

*Why? Consistent spacing makes the UI look polished. No random pixel values.*

---

### 5. COMPONENTS (Reusable UI Pieces)

Think of components like LEGO blocks — build once, use everywhere.

#### 5.1 Button
**What it does:** Clickable action element.
**Variants:**
- Primary (dark navy) — Main actions like "Predict Severity"
- Success (green) — Positive actions
- Danger (red) — Destructive actions
- Ghost (transparent) — Secondary actions like "Load Demo"
- Loading — Shows spinner instead of text during API call
**States:** Default → Hover (slightly lighter) → Click (slightly smaller) → Loading (spinner)

*Interview question you might get: "Why different button colors?"*
*Answer: "Color communicates intent. Green = safe action, red = dangerous action, primary = main flow. This follows Nielsen's usability heuristics."*

#### 5.2 Input / Select
**What it does:** Lets users enter or choose values.
**Features:**
- Label above every field (accessibility)
- Placeholder text showing examples
- Red border + error message if invalid
- Disabled state when form is submitting
**Validation rules:**
- Speed Limit: must be a number between 0 and 200
- All text fields: max length limits (prevents abuse)
- Required fields: cannot be empty

#### 5.3 Severity Badge
**What it does:** Shows prediction result visually.
| Severity | Color | Shape | Animation |
|----------|-------|-------|-----------|
| **FATAL** | Red background, red text | Rounded pill | Pulse glow on appear |
| **NON-FATAL**| Green background, green text| Rounded pill | Fade in |
| **Unknown** | Gray | Rounded pill | None |

*Why a pill shape? Pills feel medical/clinical — appropriate for a severity indicator.*

#### 5.4 Probability Gauge
**What it does:** Doughnut chart showing probability as a percentage.
**Color zones:**
- 0-30%: Green (low risk)
- 30-70%: Amber (medium risk)
- 70-100%: Red (high risk)
*Why a doughnut? Shows the full 0-100% range while highlighting where the current value falls.*

#### 5.5 SHAP Bar Chart
**What it does:** Horizontal bars showing each feature's contribution to the prediction.
**Colors:**
- Red bar = This feature INCREASED fatal risk
- Green bar = This feature DECREASED fatal risk

*Interview explanation: "SHAP values show how each feature pushed the prediction. Positive values (red) increase fatal probability, negative (green) decrease it. Speed Limit of 80 km/h added 45% to the risk."*

#### 5.6 Metric Card
**What it does:** Shows a single number with context.
**Features:** Big number (JetBrains Mono), Label above, Trend indicator.

#### 5.7 Incident Table
**What it does:** Lists recent predictions in a table.
**Features:** Click row to expand details, Pagination, Colored badges.

#### 5.8 Toast Notification
**What it does:** Brief message that appears and disappears automatically.
**Types:** Success, Error, Warning, Info.
**Behavior:** Slides in from right, stays 5 seconds, slides out.

---

### 6. FORMS & VALIDATION

#### 6.1 Prediction Form Fields
| Field | Type | Required? | Validation |
|-------|------|-----------|------------|
| Road Type | Dropdown | Yes | Must select one |
| Speed Limit | Dropdown | Yes | 0-200 only |
| Weather | Dropdown | Yes | Must select one |
| Lighting | Dropdown | Yes | Must select one |
| Junction | Dropdown | Yes | Must select one |
| Junction Control| Dropdown | Yes | Must select one |
| Vehicle Type | Text | Yes | Max 50 chars |
| Driver Age | Dropdown | Yes | Must select one |
| Urban/Rural | Dropdown | Yes | Must select one |
| State | Text | Yes | Max 100 chars |
| City | Text | Yes | Max 50 chars |

#### 6.2 Validation Flow
```
User clicks "Predict"
    ↓
Check all fields filled?
    ↓ NO → Show red border + error message on empty fields
    ↓ YES
Check Speed Limit is 0-200?
    ↓ NO → Show "Speed limit must be 0-200"
    ↓ YES
Strip whitespace from text fields (prevents encoding issues)
    ↓
Send to API
    ↓
Show loading spinner on button
    ↓
Receive response
    ↓ SUCCESS → Show results, SHAP chart, enable What-If
    ↓ ERROR → Show toast with error message
```

#### 6.3 Counterfactual Form
**What it does:** After a prediction, let the user change ONE feature and see how the probability changes.
*Why only one feature at a time? Keeps the explanation simple. Changing multiple features makes it hard to attribute the effect.*

---

### 7. API ENDPOINTS (What the Frontend Calls)

#### 7.1 POST /v1/predict
**What it does:** Takes accident details, returns severity prediction.
*Interview explanation: "The frontend sends 11 features to the backend. The ONNX model runs inference, returns a probability (0-1), and SHAP explains which 3 features contributed most. The result is saved to SQLite for history."*

#### 7.2 POST /v1/counterfactual
**What it does:** Simulates changing one feature.

#### 7.3 POST /v1/explain
**What it does:** Generates plain-English explanation using Groq LLM.
*Interview question: "What if the LLM API fails?"*
*Answer: "I built a fallback that generates explanations using predefined rules based on the top SHAP feature. The user never sees an error — they always get an explanation."*

#### 7.4 GET /v1/health
**What it does:** Checks if the system is running.
*Why it matters: Render uses this to know if your app is healthy. If it fails 3 times, Render restarts the container.*

#### 7.5 GET /v1/incidents
**What it does:** Returns paginated list of past predictions.

---

### 8. STATE MANAGEMENT (Where Data Lives)

#### 8.1 What is State?
State = "What the app knows right now."

#### 8.2 Where State Lives
| State | Location | Why |
|-------|----------|-----|
| Form values | Browser memory (JS variables) | Temporary, lost on refresh |
| Prediction result| Browser memory | Needed by multiple sections |
| Theme | localStorage | Persists across visits |
| Incident list | Fetched from server | Source of truth is DB |
| System status | Fetched from /v1/health | Must be current |

#### 8.3 No React/Redux? How?
Since this is server-rendered HTML, state is simpler:
*Interview explanation: "I used vanilla JavaScript for state because this is a single-page dashboard. Adding React would add 100KB+ bundle size and build complexity for no benefit. The state is simple enough to manage with plain JS."*

---

### 9. ANIMATIONS & INTERACTIONS

#### 9.1 Why Animations Matter
Animations aren't just decoration — they guide attention, show state changes, and reduce perceived wait time.

#### 9.2 Key Animations
| Animation | When It Happens | What It Does | Duration |
|-----------|-----------------|--------------|----------|
| Stagger fade-in | Page loads | Metrics cards appear one by one | 50ms delay each |
| Button loading | Click "Predict" | Text changes to spinner | Instant |
| Results slide-in| Prediction succeeds | Results panel slides from right | 300ms |
| Badge pulse | Severity appears | Red badge pulses with glow | 400ms |
| SHAP bars grow | Chart renders | Bars grow from zero width | 500ms staggered |
| Toast slide | Error/success occurs| Message slides in from right | 300ms |
| Table row hover | Mouse over row | Row background lightens | 150ms |
| Card lift | Hover metric card | Card moves up slightly | 150ms |

*Interview tip: "I used CSS animations (not JavaScript) because they're GPU-accelerated and don't block the main thread."*

---

### 10. ERROR HANDLING

#### 10.1 Types of Errors & How We Handle Them
| Error | What Happens | What User Sees | How We Recover |
|-------|--------------|----------------|----------------|
| No internet | fetch() fails | "You're offline" banner | Auto-retry when connection returns |
| Invalid input | Speed Limit = 250 | "Speed limit must be 0-200" | User fixes input |
| Rate limit (429)| Too many requests | "Please wait 30 seconds" | Countdown timer, then retry |
| Server error (500)| Backend crashes | "Service unavailable" | Auto-retry once, then manual retry button |
| Timeout (>10s) | API too slow | "Request timed out" | Retry button |
| Model not loaded| ONNX file missing | "Model loading..." | Poll health endpoint every 5s |
| LLM fails | Groq API error | Still shows explanation | Fallback rule-based explanation |
| Chart fails | Chart.js error | Shows text list instead | Graceful degradation |

*Interview question: "How do you handle API failures?"*
*Answer: "I wrap all API calls in try-catch. For HTTP errors, I parse the error message from the backend. For network failures, I show a user-friendly message. I also auto-retry once for 5xx errors."*

---

### 11. PERFORMANCE TARGETS

#### 11.1 What We're Measuring
| Metric | Target | What It Means |
|--------|--------|---------------|
| First Contentful Paint | < 1.5s | User sees something on screen |
| Largest Contentful Paint | < 2.5s | Main content is visible |
| Time to Interactive | < 3.5s | User can click buttons |
| API Response (p50) | < 50ms | Half of predictions finish in 50ms |
| API Response (p95) | < 200ms | 95% of predictions finish in 200ms |

#### 11.2 How We Achieve This
| Technique | What It Does | Benefit |
|-----------|--------------|---------|
| Inline critical CSS | Put essential styles in `<head>` | Page renders before CSS downloads |
| Preload fonts | `<link rel="preload">` | Fonts ready when needed |
| Defer Chart.js | Load chart library after page renders | Faster initial paint |
| ONNX Runtime | Optimized C++ inference engine | ~2x faster than Python XGBoost |
| SQLite WAL mode | Write-Ahead Logging | Reads don't block writes |
| In-memory cache | SHA256-keyed cache | Skip model inference for repeats |
| Async database | aiosqlite | Handle multiple requests simultaneously |

*Interview question: "How did you make it fast?"*
*Answer: "Three things: ONNX Runtime for fast inference, async SQLite so requests don't wait for DB, and client-side caching so repeated scenarios don't recompute."*

---

### 12. TESTING STRATEGY

#### 12.1 Types of Tests
| Test Type | What It Tests | Example | Tool |
|-----------|---------------|---------|------|
| Unit | Single function | Does predict() return prob 0-1? | pytest |
| Integration | Multiple parts | Form → API → DB → response? | httpx + TestClient |
| Security | Vulnerabilities | Are secrets committed? | Bandit, Safety |
| Accessibility | Can everyone use it?| Keyboard nav, screen readers | axe-core |

*Interview question: "Did you write tests?"*
*Answer: "Yes — unit tests for the ML service, integration tests for the API endpoints, and security scans with Bandit. I aim for 70%+ coverage."*

---

### 13. DEPLOYMENT

#### 13.1 Environments
| Environment | Where | Purpose |
|-------------|-------|---------|
| Local | `localhost:8000` | Development, testing |
| Production | `soochak.onrender.com` | Live demo for recruiters |

#### 13.2 Deployment Flow
1. Write code locally
2. Run tests: `pytest`
3. Commit to Git: `git commit -m "feat: new feature"`
4. Push to GitHub: `git push origin main`
5. Render auto-detects push
6. Render builds Docker image
7. Render deploys new container
8. Health check passes → Live!

#### 13.3 Docker (Simplified)
**What Docker does:** Packages your app + all dependencies into a container that runs the same everywhere.
**Key points:** Multi-stage build, Non-root user, Health check.

*Interview question: "Why Docker?"*
*Answer: "Docker ensures my app runs identically on my laptop and the cloud. I used a multi-stage build to keep the image small and a non-root user for security."*

#### 13.4 Free Tier Limitations (Be Honest!)
| Limitation | Why It Happens | How I Handle It |
|------------|----------------|-----------------|
| SQLite resets on sleep | Render free tier stops container after 15 min idle | Documented in README; predictions are stateless |
| Rate limit 30/min | In-memory limiter, no Redis | Sufficient for demo; documented |
| No persistent storage | Free tier has ephemeral filesystem | All predictions are stateless; history is nice-to-have |

*CRITICAL interview tip: "I chose zero-cost deployment with documented trade-offs rather than hiding limitations. This shows systems thinking — I understand the constraints and made informed decisions."*

---

### 14. INTERVIEW CHEAT SHEET

#### 14.1 The 30-Second Pitch
"SOOCHAK predicts fatal road accidents using an XGBoost model trained on Indian FIR data. It explains predictions with SHAP values — like 'speed limit contributed 45% to the risk' — and simulates interventions like adding road dividers. It's built with FastAPI, ONNX Runtime, and SQLite, deployed free on Render."

#### 14.2 Common Questions & Answers

**Q: Why XGBoost?**
A: XGBoost is industry-standard for tabular data. It handles missing values well, trains fast, and has built-in regularization to prevent overfitting.

**Q: Why ONNX?**
A: ONNX separates training from inference. I train in Python with XGBoost, export to ONNX, and get 2x faster inference with a smaller runtime footprint.

**Q: Why SHAP?**
A: SHAP tells us WHY the model made a prediction. In safety-critical applications, black-box models aren't acceptable — you need explainability.

**Q: Why SQLite?**
A: For a demo handling <10K predictions, SQLite is perfect. Zero setup, single file, no server needed. I documented the trade-off: no concurrent writes, ephemeral on free tier.

**Q: Why no React/Vue?**
A: This is a single-page dashboard. Adding a JS framework would add 100KB+ bundle size and build complexity. Server-rendered HTML with vanilla JS is simpler and faster for this use case.

**Q: How do you handle unknown categories?**
A: The OrdinalEncoder maps unknown categories to -1. The model was trained with this fallback, so it handles unseen values gracefully.

**Q: What if the LLM API fails?**
A: I built a rule-based fallback. If Groq is down, the app generates an explanation using predefined rules based on the top SHAP feature. The user never sees an error.

**Q: How do you prevent abuse?**
A: In-memory rate limiting (30 req/min per IP) and an API key gate on POST endpoints. It's documented as a demo limitation — production would use Redis.

**Q: What's your model's accuracy?**
A: F1 score of 0.65 on the fatal class, which is reasonable given the class imbalance (~15% fatal). I prioritized recall for fatal cases — better to flag a non-fatal as risky than miss a fatal.

**Q: How did you deploy?**
A: Docker multi-stage build on Render's free tier. Auto-deploys from GitHub. Health checks ensure the container stays running.

#### 14.3 Architecture Diagram (Draw This in Interviews)
```text
┌─────────────────┐     ┌──────────────────┐
│   Browser       │────▶│   FastAPI        │
│   (Dashboard)   │◀────│   (Python)       │
└─────────────────┘     └────────┬─────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌─────────┐ ┌──────────┐ ┌──────────┐
              │  ONNX   │ │  SQLite  │ │  Groq    │
              │ Runtime │ │  (WAL)   │ │  API     │
              │ (2MB)   │ │          │ │ (LLM)    │
              └─────────┘ └──────────┘ └──────────┘
```
**Explain:** "The browser sends accident details to FastAPI. FastAPI runs the ONNX model for prediction, saves to SQLite, and optionally calls Groq for an explanation. Everything is async so the server handles multiple requests."

#### 14.4 If They Ask About Scaling
"For scaling, I'd add Redis for rate limiting and caching, PostgreSQL for persistent storage, and horizontal scaling with multiple containers behind a load balancer. The ONNX model is stateless, so it scales perfectly."

#### 14.5 If They Ask About Improvements
"Future improvements: geographic heatmap of risk areas, PDF report generation, multi-language support, and a feedback loop where users correct predictions to improve the model."

---

*Document Version: 1.0 (Beginner Edition)*
*Last Updated: 2026-06-12*
*Purpose: Placement preparation — every decision explainable in an interview.*
