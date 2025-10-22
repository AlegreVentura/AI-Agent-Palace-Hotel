# 🎯 AI Sales Copilot for Call Centers

![Status](https://img.shields.io/badge/status-active-success.svg)
![Difficulty](https://img.shields.io/badge/difficulty-balanced-blue.svg)
![Duration](https://img.shields.io/badge/duration-10%20hours-orange.svg)

> An intelligent AI-powered web interface that assists call center agents in real-time during hotel reservation calls, capturing customer information, providing contextual tips, and retrieving relevant data from the company database.

## 📋 Overview

Call center agents handling hotel reservations at **Le Blanc Spa Resorts** need instant access to customer information, conversation guidance, and company data to maximize conversions. **AI Sales Copilot** is a real-time web application that transcribes calls, extracts key information, guides agents through the reservation process, and automatically retrieves relevant pricing, availability, and property details from the company database.

## 🎯 The Challenge

**Build the Brain Behind Every Conversion**

Design an AI-powered copilot that seamlessly supports the human agent — not replaces them — throughout the entire sales journey.

## ✨ Key Features

### 1. 📊 Real-Time Customer Information Panel
Automatically captures and displays critical customer data during the call:

**Example Display:**
```
┌─────────────────────────────────────────┐
│ 👤 CUSTOMER INFORMATION                 │
├─────────────────────────────────────────┤
│ Name: Maria Rodriguez                   │
│ Age: 35 years                           │
│ Phone: +52 998 123 4567                 │
│ Email: maria.rodriguez@email.com        │
│ Travel Purpose: Anniversary celebration │
│ Trip Type: Romantic getaway             │
│ Party Size: 2 adults                    │
│ Check-in: Oct 21, 2025                  │
│ Check-out: Oct 25, 2025                 │
│ Nights: 4                               │
│ Budget Range: $2,000 - $3,000           │
│ Special Requests: Ocean view preferred  │
│ Dietary Restrictions: Vegetarian        │
└─────────────────────────────────────────┘
```

### 2. 💡 Conversation Tips & Next Questions
AI-powered suggestions for what to ask next based on conversation context:

**Example Tips:**
```
💬 SUGGESTED NEXT QUESTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ "Have you celebrated any special occasions 
   with us before?"

✅ "Would you be interested in our Romance 
   Package which includes couples massage and 
   champagne?"

✅ "Since it's your anniversary, would you like 
   to arrange a private beachfront dinner?"

✅ "Do you have any preferred room amenities, 
   such as a jacuzzi or private balcony?"

⚠️  DETECTED: Customer mentioned budget concerns
    → Suggest: Value-added packages instead of 
              premium suites
```

### 3. ✅ Reservation Checklist
Interactive checklist ensuring no critical information is missed:

**Example Checklist:**
```
📋 RESERVATION REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Customer full name
✅ Contact phone number
✅ Contact email
✅ Check-in date
✅ Check-out date
✅ Number of adults
⬜ Number of children (if applicable)
⬜ Room type preference
✅ Special occasions/celebrations
✅ Dietary restrictions/allergies
⬜ Airport transfer needed
⬜ Spa services interest
⬜ Payment method preference
⬜ Loyalty program member
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Progress: 8/13 required fields completed
```

### 4. 🗄️ Intelligent Database Retrieval
Automatically queries company database and displays relevant information:

**Example Database Responses:**
```
🏨 AVAILABLE ROOMS (Oct 21-25, 2025)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💎 Deluxe Oceanfront Junior Suite
   • $2,450 total (4 nights)
   • $612.50/night
   • Ocean view, King bed, Balcony
   • 450 sq ft
   • ⭐ 3 rooms available

💎 Luxury Swim-Up Suite
   • $3,200 total (4 nights)
   • $800/night
   • Direct pool access, King bed
   • 520 sq ft
   • ⭐ 2 rooms available

📦 ROMANCE PACKAGE UPGRADE: +$350
   • Couples massage (80 min)
   • Champagne & strawberries
   • Rose petal turndown service
   • Private beach dinner setup

🚗 AIRPORT TRANSFER: $120 roundtrip
   (Cancun Airport ↔ Le Blanc Spa Resort)

🍽️ DINING OPTIONS INCLUDED:
   • 5 gourmet restaurants
   • 24-hour room service
   • Premium bar selections
   • Vegetarian menu available ✓
```

## 👥 Team

### Project Owner
**Rolando Cesai Romero Díaz**

### Impact Builders

| Name | Role | Contribution |
|------|------|--------------|
| **Ricardo** | Active Participant | Active participant in development |
| **Jonathan Palma** | Technical Lead | Experience with MCP, MCP-UI, STT-TTS |

## 📅 Project Timeline

| Milestone | Date | Time |
|-----------|------|------|
| **Start Date** | Tuesday, October 21, 2025 | 08:00 |
| **End Date** | Wednesday, October 22, 2025 | 18:00 |
| **Total Duration** | 10 hours | - |

## 🏆 Recognition

**Award**: 4 Double Nights at Palace Company Hotels

## 🤖 System Prompt

```markdown
# AI Sales Copilot - System Instructions

You are an intelligent assistant supporting call center agents at Le Blanc Spa Resorts 
during live reservation calls. Your role is to help agents maximize conversions by 
providing real-time intelligence and actionable guidance.

## PRIMARY OBJECTIVES:

1. **Extract Customer Information**: Listen to the call transcription and identify:
   - Full name
   - Age (if mentioned)
   - Contact information (phone, email)
   - Travel purpose (honeymoon, anniversary, vacation, business, family trip)
   - Trip type (romantic, family, solo, group)
   - Party composition (number of adults, children, ages)
   - Check-in and check-out dates
   - Budget range or price sensitivity
   - Special requests or preferences
   - Dietary restrictions or allergies
   - Accessibility needs
   - Previous stays or loyalty status

2. **Provide Contextual Conversation Tips**: Based on what has been discussed, suggest:
   - Specific follow-up questions to gather missing information
   - Upsell opportunities aligned with customer needs
   - Empathy statements for objection handling
   - Closing techniques when all requirements are met
   
   Example scenarios:
   - If customer mentions "anniversary" → Suggest romance package and private dinner
   - If budget concerns detected → Highlight included amenities and value
   - If comparing properties → Emphasize unique differentiators
   - If hesitation detected → Ask about specific concerns proactively

3. **Maintain Reservation Checklist**: Track in real-time which required fields have been collected:
   - MUST HAVE: Name, contact, dates, guest count
   - SHOULD HAVE: Room preference, special occasions, dietary needs
   - NICE TO HAVE: Airport transfer, spa interest, activity preferences
   
   Alert agent when critical information is still missing.

4. **Query Database Intelligently**: When customer asks about:
   - Pricing → Return rates for check-in/out dates, room categories, packages
   - Availability → Check room inventory for specific dates
   - Amenities → List included services, restaurants, activities
   - Location → Provide property details, nearby attractions, distance from airport
   - Policies → Retrieve cancellation policy, payment terms, check-in/out times
   - Packages → Show available upgrades (romance, spa, adventure, family)
   
   Format database responses clearly with pricing, availability status, and key features.

## CONVERSATION FLOW STAGES:

### Stage 1: Information Gathering
- Greet and build rapport
- Identify travel purpose and needs
- Collect essential guest information
- Understand budget and expectations

**Agent Tips:**
- "What brings you to Cancun? Special occasion?"
- "How many guests will be traveling?"
- "What dates work best for you?"
- "Have you stayed with Palace Resorts before?"

### Stage 2: Solution Presentation
- Present appropriate room categories
- Highlight relevant packages and upgrades
- Address specific needs (dietary, accessibility, etc.)
- Compare options if customer is deciding

**Agent Tips:**
- "Based on your anniversary, our Deluxe Oceanfront Suite would be perfect..."
- "This includes all meals, premium drinks, and activities..."
- "For vegetarian options, all our restaurants accommodate..."

### Stage 3: Objection Handling
- Address price concerns with value emphasis
- Handle comparison with competitors
- Resolve availability or timing issues
- Clarify policies or restrictions

**Agent Tips:**
- "Compared to other properties, we include..."
- "Our all-inclusive rate actually saves you..."
- "We can offer a payment plan if that helps..."

### Stage 4: Closing
- Summarize selected options
- Confirm all details
- Explain next steps (payment, confirmation)
- Set expectations for pre-arrival communication

**Agent Tips:**
- "Let me confirm: Deluxe Suite, Oct 21-25, 2 adults..."
- "I'll send you a secure payment link via email..."
- "You'll receive confirmation within 15 minutes..."

## OUTPUT FORMAT:

Always structure your response in 4 sections:

```
[CUSTOMER INFORMATION]
{Display extracted fields with confidence indicators}

[CONVERSATION TIPS]
{3-5 specific next questions or actions}
{Include urgency/priority flags if needed}

[CHECKLIST STATUS]
{Show completion percentage and missing items}

[DATABASE RESULTS]
{Relevant pricing, availability, or information}
{Format as clean, scannable data}
```

## BEHAVIORAL RULES:

✅ DO:
- Update information in real-time as conversation progresses
- Prioritize questions that unblock the reservation process
- Highlight upsell opportunities naturally aligned with customer needs
- Flag when customer seems ready to close
- Provide specific numbers (prices, availability count, distances)

❌ DON'T:
- Suggest questions already answered
- Recommend products that don't match budget or needs
- Display overwhelming amounts of data
- Use technical jargon or internal codes
- Make assumptions about incomplete information

## INTEGRATION POINTS:

- **Booking System**: https://bookingscancunpr.leblancsparesorts.com/rooms
  Query parameters: skd-total-rooms, adult_room1, skd-checkin, skd-checkout, skd-language-code

- **Database Schema**:
  - rooms(id, name, category, price_per_night, max_occupancy, amenities, availability)
  - packages(id, name, price, inclusions, valid_for_properties)
  - reservations(id, customer_id, room_id, check_in, check_out, status)
  - customers(id, name, email, phone, loyalty_tier, previous_stays)

Remember: You are supporting the human agent, not replacing them. Your goal is to make 
them more efficient, informed, and successful at closing reservations.
```

## 🛠️ Technical Stack

### Frontend
- **React** or **Next.js**: Web interface for agent dashboard
- **WebSocket**: Real-time communication between transcription and UI
- **Tailwind CSS**: Responsive, modern UI design

### Backend
- **Node.js/Express** or **Python/FastAPI**: API server
- **MCP (Model Context Protocol)**: AI model integration and context management
- **PostgreSQL**: Company database for properties, pricing, availability

### AI/ML Components
- **STT (Speech-to-Text)**: Real-time call transcription (Whisper, Deepgram, or Assembly AI)
- **LLM**: GPT-4 or Claude for information extraction and conversation analysis
- **NLP Pipeline**: Named Entity Recognition (NER) for customer data extraction
- **Intent Detection**: Identify customer needs and conversation stage
- **MCP-UI**: Custom UI components for agent interface

## 🚀 Getting Started

### Prerequisites
```bash
# Node.js 18+ and npm
node --version
npm --version

# Python 3.10+ (for AI services)
python3 --version

# PostgreSQL 14+
psql --version
```

### Installation
```bash
# Clone the repository
git clone https://github.com/Vania-Janet/AI-Agent-Call-Center-Assistant.git
cd summit2025

# Install frontend dependencies
cd frontend
npm install

# Install backend dependencies
cd ../backend
npm install

# Set up Python environment for AI services
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### Configuration
```bash
# Copy environment template
cp .env.example .env

# Configure your environment variables
nano .env
```

**Required Environment Variables:**
```env
# AI Services
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPGRAM_API_KEY=...  # or other STT service

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/leblanc_db

# Le Blanc Booking System
LEBLANC_API_URL=https://bookingscancunpr.leblancsparesorts.com
LEBLANC_API_KEY=...

# WebSocket
WS_PORT=8080

# Server
PORT=3000
NODE_ENV=development
```

### Database Setup
```bash
# Create database
createdb leblanc_db

# Run migrations
cd database
psql leblanc_db < schema.sql
psql leblanc_db < seed_data.sql
```

### Running the Application

**Development Mode:**
```bash
# Terminal 1: Start backend API
cd backend
npm run dev

# Terminal 2: Start frontend
cd frontend
npm run dev

# Terminal 3: Start AI assistant service
cd backend
source ../venv/bin/activate
python ai_service.py

# Access the application at http://localhost:3000
```

**Production Mode:**
```bash
# Build frontend
cd frontend
npm run build

# Start all services
npm run start:all
```

## 📁 Project Structure

```
summit2025/
├── README.md
├── .env
├── frontend/
│   ├── components/
│   │   ├── CustomerInfoPanel.jsx      # Real-time customer data display
│   │   ├── ConversationTips.jsx       # AI-suggested next questions
│   │   ├── ReservationChecklist.jsx   # Progress tracking
│   │   └── DatabaseResults.jsx        # Dynamic pricing/availability
│   ├── hooks/
│   │   ├── useWebSocket.js            # Real-time transcription feed
│   │   └── useAIAssistant.js          # AI copilot integration
│   └── pages/
│       └── AgentDashboard.jsx         # Main agent interface
├── backend/
│   ├── api/
│   │   ├── transcription.js           # STT service integration
│   │   ├── ai_assistant.js            # LLM processing logic
│   │   └── database_query.js          # Company DB queries
│   ├── services/
│   │   ├── information_extractor.js   # NER for customer data
│   │   ├── conversation_analyzer.js   # Stage detection & tips
│   │   └── booking_integration.js     # Le Blanc API integration
│   └── models/
│       ├── Customer.js
│       ├── Reservation.js
│       └── Property.js
├── database/
│   ├── schema.sql                     # DB schema for properties
│   └── seed_data.sql                  # Sample Le Blanc data
├── config/
│   ├── mcp_config.json                # Model Context Protocol setup
│   └── system_prompt.txt              # AI assistant instructions
├── tests/
└── docs/
```

## 💡 Usage Example

### Scenario: Agent receives incoming call

1. **Agent opens dashboard** → System starts transcribing call in real-time

2. **Customer speaks**: *"Hi, I'm looking to book a room for my wife and I. We're celebrating our 10th anniversary next month."*

3. **System instantly displays**:
   ```
   [CUSTOMER INFORMATION]
   👤 Party Size: 2 adults
   💝 Travel Purpose: 10th Anniversary celebration
   🎯 Trip Type: Romantic getaway
   
   [CONVERSATION TIPS]
   ✅ Ask: "Congratulations! What dates are you considering?"
   ✅ Ask: "Have you stayed with Palace Resorts before?"
   💡 Upsell opportunity: Romance Package detected
   
   [CHECKLIST STATUS]
   ⬜ Customer name
   ⬜ Contact phone/email
   ⬜ Check-in date
   ⬜ Check-out date
   Progress: 2/13 fields (15%)
   ```

4. **Agent asks suggested questions**, system continues extracting data

5. **Customer provides dates** → System automatically queries database:
   ```
   [DATABASE RESULTS]
   🏨 AVAILABILITY: Nov 15-19, 2025
   
   💎 Deluxe Oceanfront Junior Suite - $2,800 total
   💎 Luxury Swim-Up Suite - $3,600 total
   
   📦 RECOMMENDED ADD-ON:
   Anniversary Package (+$350)
   - Couples massage, champagne, private dinner
   ```

6. **Agent presents options** with confidence, closes reservation efficiently

## 🎯 Success Metrics

- **Conversion Rate Increase**: Target +25% improvement in closed sales
- **Average Handling Time**: Target -30% reduction (faster data access)
- **Information Completeness**: Target 95% of required fields captured
- **Agent Satisfaction**: Post-implementation survey score
- **Response Time**: AI suggestions delivered in <2 seconds
- **Upsell Success Rate**: Track package/upgrade conversion
- **First-Call Resolution**: % of reservations closed without follow-up

## 🔐 Security & Compliance

- Secure payment link generation
- Customer data encryption
- Call recording compliance (GDPR, local regulations)
- Access control and audit logs

## 📝 Roadmap

### Day 1 (Oct 21, 2025)
- [x] Project kickoff and team formation
- [ ] Design agent dashboard UI/UX mockups
- [ ] Set up frontend React application
- [ ] Implement real-time transcription (STT integration)
- [ ] Build Customer Information Panel component
- [ ] Create database schema for Le Blanc properties
- [ ] Develop system prompt and AI assistant logic

### Day 2 (Oct 22, 2025)
- [ ] Implement Conversation Tips engine
- [ ] Build Reservation Checklist component
- [ ] Integrate Le Blanc booking system API
- [ ] Develop database query service
- [ ] Connect WebSocket for real-time updates
- [ ] Test end-to-end flow with sample calls
- [ ] Deploy demo version
- [ ] Prepare presentation and documentation

## 🧪 Testing Strategy

### Unit Tests
- Information extraction accuracy (NER)
- Database query correctness
- Checklist logic validation

### Integration Tests
- STT → AI Assistant → UI pipeline
- Le Blanc API integration
- WebSocket real-time communication

### User Acceptance Testing
- Agent feedback sessions
- Call simulation scenarios
- Performance under load

## 🎥 Demo Scenarios

### Test Case 1: Anniversary Booking
**Input**: "We want to book for our anniversary, Oct 21-25, two people"
**Expected Output**:
- Extract: 2 adults, anniversary, 4 nights, Oct 21-25
- Suggest: Romance package, ocean view rooms
- Query: Available suites in date range

### Test Case 2: Budget-Conscious Customer
**Input**: "What's your cheapest room for 3 nights?"
**Expected Output**:
- Detect: Price sensitivity
- Tips: Emphasize all-inclusive value
- Query: Entry-level room categories with inclusions

### Test Case 3: Family Vacation
**Input**: "Family of 4, two kids ages 8 and 10, need connecting rooms"
**Expected Output**:
- Extract: 2 adults, 2 children (ages 8, 10)
- Suggest: Family packages, kids club
- Query: Connecting room availability

## 🤝 Contributing

This is a closed hackathon project (Summit 2025). For questions or collaboration inquiries, please contact the project owner.

## 📄 License

[To be defined]

## 📧 Contact

**Project Owner**: Rolando Cesai Romero Díaz

---

**Built with ❤️ at Summit 2025 Hackathon**

*Transforming every call into a conversion opportunity*
