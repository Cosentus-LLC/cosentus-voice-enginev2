# How to Run Real Test Calls (Anant's guide)

This walks you through making the AI voice agent **call your own phone**, talking to it, changing what it says, and reading what happened afterward. No RCM knowledge needed — there's a cheat sheet at the bottom for what to say.

Everything here is copy-paste. You only need a terminal.

---

## The 2 things you need (set once)

```bash
# 1. The voice engine that places calls
ENGINE="https://api.cosentusaibackend.com"
API_KEY="<ask Alex for the X-API-Key — not stored here because this repo is public>"

# 2. The data/admin API (edit agents, read call results)
API="https://2srzoajmyd.execute-api.us-east-1.amazonaws.com/dev/voice"

# 3. Your phone number in E.164 format (country code, no spaces/dashes)
MY_PHONE="+1XXXXXXXXXX"      # <-- put YOUR number here
FROM="+12098075018"          # the number the agent calls FROM (leave as-is)
```

**Two agents you'll use:**
- `chris-claim-status` — the real Cosentus agent. It calls an insurance company to check a claim. **You play the insurance rep.** (Cheat sheet at the bottom.)
- `v2-tools-test` — a throwaway sandbox agent. **Safe to edit and break.** Use this one when you want to experiment with prompts.

---

## Part 1 — Make the agent call your phone

This tells the engine: "call my phone, using the Chris agent."

```bash
curl -s -X POST "$ENGINE/start" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "direction": "outbound",
    "agent_id": "chris-claim-status",
    "from_number": "'"$FROM"'",
    "target_number": "'"$MY_PHONE"'",
    "case_data": {}
  }'
```

Your phone rings in ~5-10 seconds. Answer it and talk. The response gives you a `call_id` — **save it**, you'll use it in Part 5 to read the results.

To test the **sandbox** agent instead, change `"agent_id": "chris-claim-status"` to `"agent_id": "v2-tools-test"`.

---

## Part 2 — Set the pre-call data (`case_data`)

`case_data` is the info the agent knows *before* it dials — like the patient and claim it's calling about. The agent reads these values out loud when you (the insurance rep) ask for them.

For Chris, fill in a fake patient/claim like this:

```bash
curl -s -X POST "$ENGINE/start" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "direction": "outbound",
    "agent_id": "chris-claim-status",
    "from_number": "'"$FROM"'",
    "target_number": "'"$MY_PHONE"'",
    "case_data": {
      "Patient_Name": "John Smith",
      "Patient_Birth_Date": "March 4 1985",
      "Practice_Name": "Valley Medical Group",
      "Provider": "Dr. Kim",
      "NPI": "1234567890",
      "Tax_ID": "12-3456789",
      "Claim#": "CLM12345",
      "Service_Date": "January 12 2026",
      "Primary_Carrier_Name": "Blue Cross",
      "Total_Charge": "450.00"
    }
  }'
```

When you (as the rep) ask "what's the patient name and date of birth?", Chris will answer "John Smith, March fourth nineteen eighty-five" — straight from this data. Change any values you want; they're all fake.

---

## Part 3 — Talking to Chris (you = the insurance rep)

**What's happening:** Chris is a billing person at a doctor's office. He's calling YOU — the insurance company — to find out what happened to a bill (a "claim") for a patient. Your job is just to act like a normal insurance phone rep.

**The flow is simple:**
1. **You answer:** *"Thanks for calling Blue Cross, this is Sarah, how can I help?"*
2. **Chris says** he's checking a claim status.
3. **You verify who he is** — ask for a couple of these (he'll read them from the case data): patient name + date of birth, the provider/doctor name, NPI or Tax ID, the claim number, or the date of service.
4. **You give him a claim status.** Pick one of the scripts below.
5. **Chris will ask follow-up questions.** Just answer naturally. Give a reference number if he asks (make one up, like "REF8842").
6. When he has what he needs, he says goodbye and hangs up.

You don't need to be perfect — just keep it conversational. Pick one of these to read:

**Script A — Claim was PAID**
> "That one was paid. Paid on February tenth, check number 4-4-7-8-2, for one hundred forty two dollars and thirty seven cents."

**Script B — Claim was DENIED**
> "That claim was denied. Denial code C-O sixteen — missing information. You'll need to resubmit with the correct provider NPI. You can fax it to 8-0-0, 5-5-5, 1-2-1-2."

**Script C — Claim is still PROCESSING**
> "It's still in process. Give it another thirty days from the date of service. Nothing needed on your end right now."

That's it. Any of those is a complete, useful test call.

---

## Part 4 — Edit a test agent's prompt

Use the **sandbox agent** (`v2-tools-test`) so you never change the real Chris agent.

**See the current prompt:**
```bash
curl -s "$API/api/agents/v2-tools-test/prompt"
```

**Change the prompt** (this becomes what the agent "is" and how it behaves):
```bash
curl -s -X PUT "$API/api/agents/v2-tools-test/prompt" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "You are a friendly test assistant. Greet the caller, ask how their day is going, and keep replies short and warm."
  }'
```

Then make a call to it (Part 1, with `"agent_id": "v2-tools-test"`) and you'll hear your new prompt in action. Edit, call, repeat.

> Tip: you can also change the voice, model, etc. via `PUT $API/api/agents/v2-tools-test` — ask if you want those fields.

---

## Part 5 — Review the call afterward (transcript + AI analysis)

After you hang up, the system saves the full conversation and an AI-generated summary.

**See your recent calls** (newest first):
```bash
curl -s "$API/api/calls?page=1&page_size=5"
```

**See one call in full** (use the `call_id` from Part 1, or an `id` from the list above):
```bash
curl -s "$API/api/calls/PASTE_CALL_ID_HERE"
```

In that response, look at:
- **`transcript`** — the word-for-word back-and-forth (what you said, what the agent said).
- **`post_call_analyses`** — the AI's summary of the call (for Chris, a "call summary" field describing the claim outcome and next steps).
- **`recording_path`** — the audio recording (ask if you want a playable link).

That's the whole loop: **call → talk → read results → tweak prompt → call again.**

---

## Quick reference

| Thing | Value |
|---|---|
| Place a call | `POST $ENGINE/start` (with `X-API-Key`) |
| Edit agents / read calls | `$API/api/...` |
| Real agent (you = insurance rep) | `chris-claim-status` |
| Sandbox agent (safe to edit) | `v2-tools-test` |
| Phone format | `+1` then 10 digits, no spaces |

**Glossary (RCM terms you'll hear):**
- **Claim** — a bill the doctor's office sent to insurance for a patient.
- **Payer / carrier** — the insurance company (that's you, on these calls).
- **NPI** — the provider's national ID number.
- **Denial / denial code** — insurance refused to pay, with a reason code.
- **Timely filing** — the bill was sent too late, so it's denied.
- **EOB / remit** — the insurance's explanation of what they paid.
