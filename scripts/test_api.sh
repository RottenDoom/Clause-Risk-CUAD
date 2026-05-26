#!/usr/bin/env bash
# scripts/test_api.sh
#
# End-to-end API test using curl.
# Works against local dev server or a deployed EC2 instance.
#
# Usage:
#   # Local
#   ./scripts/test_api.sh
#
#   # Against EC2 / deployed URL
#   BASE_URL=http://YOUR_IP ./scripts/test_api.sh
#
#   # Single family, specific model
#   BASE_URL=http://localhost:8000 FAMILIES=termination MODEL=claude-haiku-4-5-20251001 ./scripts/test_api.sh
#
#   # From a .txt file instead of inline text
#   CONTRACT_FILE=data/test/SomeContract.txt ./scripts/test_api.sh

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
FAMILIES="${FAMILIES:-all}"
MODEL="${MODEL:-claude-sonnet-4-6}"
CONTRACT_FILE="${CONTRACT_FILE:-}"
POLL_INTERVAL=5    # seconds between status polls
MAX_POLLS=60       # give up after 5 minutes

BOLD="\033[1m"
GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
RESET="\033[0m"

ok()   { echo -e "${GREEN}✓${RESET} $*"; }
fail() { echo -e "${RED}✗${RESET} $*"; exit 1; }
info() { echo -e "${YELLOW}→${RESET} $*"; }

# ── Sample contract used when no CONTRACT_FILE is given ───────────────────────
SAMPLE_CONTRACT='MASTER SERVICES AGREEMENT

This Master Services Agreement ("Agreement") is entered into as of January 1, 2024,
by and between Acme Corp ("Client") and Vendor Inc ("Vendor").

1. TERM AND TERMINATION
The Agreement shall commence on the Effective Date and continue for one (1) year
unless earlier terminated. Either party may terminate this Agreement for convenience
upon sixty (60) days written notice to the other party. Client may terminate immediately
upon written notice if Vendor materially breaches any obligation herein and fails to
cure such breach within thirty (30) days of receiving written notice.

2. ASSIGNMENT
Neither party shall assign this Agreement or any rights hereunder without the prior
written consent of the other party, which shall not be unreasonably withheld. Any
attempted assignment in violation of this Section shall be null and void. Notwithstanding
the foregoing, either party may assign this Agreement to an affiliate without consent.

3. CHANGE OF CONTROL
If Vendor undergoes a Change of Control, Client shall have the right to terminate
this Agreement upon thirty (30) days written notice. For purposes hereof, "Change of
Control" means any transaction resulting in a change of more than fifty percent of
the voting securities of Vendor.

4. EXCLUSIVITY AND NON-COMPETE
During the Term and for a period of twelve (12) months thereafter, Vendor shall not
provide services substantially similar to those provided hereunder to any direct
competitor of Client operating in the same geographic market without Client'\''s
prior written consent.

5. GOVERNING LAW
This Agreement shall be governed by the laws of the State of Delaware.'

# ── 1. Health check ───────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}=== Contract Review API Test ===${RESET}"
echo "Target: $BASE_URL"
echo ""

info "1/6  Health check..."
HTTP=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health")
[ "$HTTP" = "200" ] && ok "Health OK" || fail "Health returned HTTP $HTTP — is the server running?"

# ── 2. Catalogue endpoints ────────────────────────────────────────────────────
info "2/6  GET /models..."
MODELS=$(curl -s "$BASE_URL/models")
echo "$MODELS" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for m in d['models']:
    print(f\"    {m['id']} — {m['display_name']}\")
"
ok "Models OK"

info "2/6  GET /families..."
FAMILIES_RESP=$(curl -s "$BASE_URL/families")
echo "$FAMILIES_RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for f in d['families']:
    print(f\"    {f['id']} — {f['display_name']}\")
"
ok "Families OK"

# ── 3. Submit review ──────────────────────────────────────────────────────────
info "3/6  POST /review (families=$FAMILIES, model=$MODEL)..."

if [ -n "$CONTRACT_FILE" ]; then
    echo "     Using file: $CONTRACT_FILE"
    SUBMIT=$(curl -s -X POST "$BASE_URL/review" \
        -F "file=@${CONTRACT_FILE};type=text/plain" \
        -F "families=$FAMILIES" \
        -F "model=$MODEL")
else
    echo "     Using inline sample contract text."
    SUBMIT=$(curl -s -X POST "$BASE_URL/review" \
        -F "contract_text=$SAMPLE_CONTRACT" \
        -F "families=$FAMILIES" \
        -F "model=$MODEL")
fi

JOB_ID=$(echo "$SUBMIT" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])" 2>/dev/null || true)
[ -n "$JOB_ID" ] || fail "Submission failed. Response: $SUBMIT"
ok "Submitted — job_id: $JOB_ID"

# ── 4. Poll until done ────────────────────────────────────────────────────────
info "4/6  Polling GET /review/$JOB_ID ..."
POLLS=0
while true; do
    STATUS_RESP=$(curl -s "$BASE_URL/review/$JOB_ID")
    STATUS=$(echo "$STATUS_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "unknown")

    case "$STATUS" in
        done)
            ok "Job complete."
            break
            ;;
        failed)
            ERR=$(echo "$STATUS_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error','unknown'))" 2>/dev/null)
            fail "Job failed: $ERR"
            ;;
        pending|running)
            echo -n "     [$STATUS] waiting ${POLL_INTERVAL}s..."
            sleep "$POLL_INTERVAL"
            echo " (poll $((++POLLS)))"
            ;;
        *)
            fail "Unexpected status: $STATUS. Response: $STATUS_RESP"
            ;;
    esac

    if [ "$POLLS" -ge "$MAX_POLLS" ]; then
        fail "Timed out after $((MAX_POLLS * POLL_INTERVAL))s. Last status: $STATUS"
    fi
done

# ── 5. Print results ─────────────────────────────────────────────────────────
info "5/6  Parsing results..."
echo "$STATUS_RESP" | python3 -c "
import sys, json

d = json.load(sys.stdin)
r = d.get('result', {})

print()
print(f\"  Contract ID   : {r.get('contract_id', 'n/a')}\")
print(f\"  Overall Risk  : {r.get('overall_risk_rating', 'n/a')}\")
print(f\"  Summary       : {r.get('overall_summary', '(not generated — call /summarize)')}\")

flags = r.get('top_red_flags', [])
if flags:
    print('  Red Flags:')
    for f in flags:
        print(f'    - {f}')

cards = r.get('clause_cards', [])
print()
print('  Clause Cards:')
for card in cards:
    rating = card.get('llm_generated_risk_rating') or 'N/A'
    found  = 'found' if card.get('clause_found') else 'not found'
    family = card.get('clause_family', '?')
    rationale = (card.get('risk_rationale') or '')[:120]
    print(f\"    [{family}]  risk={rating}  ({found})\")
    if rationale:
        print(f\"      {rationale}...\")
print()
"
ok "Results parsed."

# ── 6. Optional: request summary ─────────────────────────────────────────────
info "6/6  POST /review/$JOB_ID/summarize (optional LLM call)..."
read -r -p "     Run summarize? [y/N] " CONFIRM
if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    SUM=$(curl -s -X POST "$BASE_URL/review/$JOB_ID/summarize")
    echo "$SUM" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print()
print('  Summary:', d.get('overall_summary', ''))
print('  Risk   :', d.get('overall_risk_rating', ''))
for f in d.get('top_red_flags', []):
    print(f'  Flag   : {f}')
print()
"
    ok "Summary done."
else
    echo "     Skipped."
fi

# ── 7. HTML report URL ───────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Report URL:${RESET} $BASE_URL/review/$JOB_ID/report"
echo ""
echo -e "${GREEN}All tests passed.${RESET}"
