#!/bin/bash
# Monitor cluster jobs, report failures
while true; do
    echo "=== $(date) ==="
    
    # Queue summary
    ssh -o ConnectTimeout=10 yizhouc3@mind.cs.cmu.edu '
    echo "Queue:"
    squeue -u yizhouc3 -o "%.10T" 2>/dev/null | sort | uniq -c
    echo ""
    echo "Failed (last 15min):"
    sacct -u yizhouc3 --starttime $(date -u -d "15 minutes ago" +%Y-%m-%dT%H:%M 2>/dev/null || date -u -v-15M +%Y-%m-%dT%H:%M) --format=JobName%30,State -n 2>/dev/null | grep -v batch | grep -v extern | grep FAILED | sort -u
    ' 2>/dev/null
    
    echo ""
    sleep 600
done
