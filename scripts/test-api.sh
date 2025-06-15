#!/bin/bash

# DexBrain API Testing Script
# Usage: ./test-api.sh [API_URL]

API_URL="${1:-http://157.90.230.148}"

echo "🧪 Testing DexBrain API at: $API_URL"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

test_endpoint() {
    local endpoint="$1"
    local method="${2:-GET}"
    local data="$3"
    local expected="$4"
    
    echo -n "Testing $method $endpoint... "
    
    if [[ "$method" == "POST" && -n "$data" ]]; then
        response=$(curl -s -w "\n%{http_code}" -X POST \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$API_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" "$API_URL$endpoint")
    fi
    
    http_code=$(echo "$response" | tail -1)
    body=$(echo "$response" | head -n -1)
    
    if [[ "$http_code" == "200" ]]; then
        if [[ -n "$expected" ]]; then
            if echo "$body" | grep -q "$expected"; then
                echo -e "${GREEN}✅ PASS${NC}"
                return 0
            else
                echo -e "${YELLOW}⚠️ PARTIAL${NC} (wrong response content)"
                echo "   Expected: $expected"
                echo "   Got: $body"
                return 1
            fi
        else
            echo -e "${GREEN}✅ PASS${NC}"
            return 0
        fi
    else
        echo -e "${RED}❌ FAIL${NC} (HTTP $http_code)"
        if [[ -n "$body" ]]; then
            echo "   Response: $body"
        fi
        return 1
    fi
}

# Test 1: Health Check
echo ""
echo "1️⃣ Health Check"
test_endpoint "/health" "GET" "" "healthy"

# Test 2: Agent Registration
echo ""
echo "2️⃣ Agent Registration"
REGISTER_DATA='{"agent_id": "test-agent-'$(date +%s)'", "metadata": {"test": true, "version": "1.0.0"}}'
if test_endpoint "/api/register" "POST" "$REGISTER_DATA" "api_key"; then
    # Extract API key for further tests
    API_KEY=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$REGISTER_DATA" \
        "$API_URL/api/register" | \
        grep -o '"api_key":"[^"]*"' | \
        cut -d'"' -f4)
    
    if [[ -n "$API_KEY" ]]; then
        echo "   📝 Generated API Key: ${API_KEY:0:20}..."
    fi
fi

# Test 3: Intelligence Feed (requires API key)
echo ""
echo "3️⃣ Intelligence Feed"
if [[ -n "$API_KEY" ]]; then
    INTEL_RESPONSE=$(curl -s -H "Authorization: Bearer $API_KEY" "$API_URL/api/intelligence")
    if echo "$INTEL_RESPONSE" | grep -q "insights"; then
        echo -e "   ${GREEN}✅ PASS${NC} (with API key)"
    else
        echo -e "   ${YELLOW}⚠️ PARTIAL${NC} (unexpected response)"
        echo "   Response: $INTEL_RESPONSE"
    fi
else
    echo -e "   ${YELLOW}⚠️ SKIPPED${NC} (no API key available)"
fi

# Test 4: Data Submission (requires API key)
echo ""
echo "4️⃣ Data Submission"
if [[ -n "$API_KEY" ]]; then
    SUBMIT_DATA='{
        "blockchain": "base",
        "dex_protocol": "uniswap_v3",
        "performance_data": {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "position_id": "test_pos_'$(date +%s)'",
            "total_return_usd": 125.50,
            "total_return_percent": 12.5,
            "fees_earned_usd": 25.30,
            "impermanent_loss_usd": -5.20,
            "net_profit_usd": 120.30,
            "apr": 15.2,
            "duration_hours": 168,
            "gas_costs_usd": 4.70,
            "slippage_percent": 0.15,
            "win": true
        }
    }'
    
    SUBMIT_RESPONSE=$(curl -s -X POST \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "$SUBMIT_DATA" \
        "$API_URL/api/submit-data")
    
    if echo "$SUBMIT_RESPONSE" | grep -q "success\|accepted"; then
        echo -e "   ${GREEN}✅ PASS${NC} (data accepted)"
    else
        echo -e "   ${YELLOW}⚠️ PARTIAL${NC} (unexpected response)"
        echo "   Response: $SUBMIT_RESPONSE"
    fi
else
    echo -e "   ${YELLOW}⚠️ SKIPPED${NC} (no API key available)"
fi

# Test 5: Network Statistics
echo ""
echo "5️⃣ Network Statistics"
test_endpoint "/api/stats" "GET" "" "network_stats"

# Test 6: Invalid endpoints
echo ""
echo "6️⃣ Error Handling"
echo -n "Testing invalid endpoint... "
INVALID_RESPONSE=$(curl -s -w "%{http_code}" "$API_URL/api/invalid-endpoint")
if [[ "$INVALID_RESPONSE" == *"404"* ]]; then
    echo -e "${GREEN}✅ PASS${NC} (returns 404)"
else
    echo -e "${YELLOW}⚠️ PARTIAL${NC} (unexpected error handling)"
fi

# Performance test
echo ""
echo "7️⃣ Performance Test"
echo "Testing response times..."

for i in {1..5}; do
    start_time=$(date +%s%N)
    curl -s "$API_URL/health" > /dev/null
    end_time=$(date +%s%N)
    response_time=$(( (end_time - start_time) / 1000000 ))
    echo "   Request $i: ${response_time}ms"
done

# Summary
echo ""
echo "🏁 Test Summary"
echo "==============="
echo "API URL: $API_URL"
echo "Server Status: $(curl -s "$API_URL/health" | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo "Unknown")"

if [[ -n "$API_KEY" ]]; then
    echo "Test API Key: ${API_KEY:0:30}..."
    echo ""
    echo "🔗 Integration Example:"
    echo "curl -H 'Authorization: Bearer $API_KEY' \\"
    echo "     '$API_URL/api/intelligence?blockchain=base&limit=10'"
fi

echo ""
echo "📚 Full API Documentation:"
echo "https://github.com/MeltedMindz/Dexter/blob/main/backend/API_DOCUMENTATION.md"