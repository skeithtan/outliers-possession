#!/bin/bash
# Battery guard v2, round 3 copy: pattern narrowed to the learned-stage training processes (tools.learned_spike);
# handover off by default (HANDOVER=0). (the user cannot reach the Mac before submission; it must never die).
# Pause heavy jobs at LOW% or when on battery power; resume only at HIGH% on AC while charging.
# At CRIT% and not charging, put the Mac to sleep (sleep draws almost nothing; the charger recovers it).
echo $$ > "$1.pid"; LOG="$1"; LOW=${LOW:-45}; HIGH=${HIGH:-60}; CRIT=${CRIT:-12}
pids() { pgrep -f 'MacOS/Python -m tools\.learned_spike\.' | grep -v "^$$\$"; }
read_batt() {
  b=$(pmset -g batt); pct=$(echo "$b" | grep -Eo '[0-9]+%' | head -1 | tr -d %)
  state=$(echo "$b" | grep -Eo '[0-9]+%; [a-zA-Z ]+' | head -1 | sed 's/^[0-9]*%; //')
  src=$(echo "$b" | head -1 | grep -o "'.*'")
}
read_batt
if { [ "$pct" -ge "$HIGH" ] || { [ -n "$START_RUNNING" ] && [ "$pct" -gt "$LOW" ]; }; } && [ "$src" = "'AC Power'" ]; then paused=0; p=$(pids|tr '\n' ' '); [ -n "$p" ] && kill -CONT $p 2>/dev/null
else paused=1; p=$(pids|tr '\n' ' '); [ -n "$p" ] && kill -STOP $p 2>/dev/null; fi
echo "$(date '+%m-%d %H:%M:%S') guard v2 start at ${pct}% ($state, $src) paused=$paused LOW=$LOW HIGH=$HIGH CRIT=$CRIT" >> "$LOG"
while true; do
  read_batt; now=$(date '+%m-%d %H:%M:%S')
  if [ "$paused" = 0 ] && { [ "$pct" -le "$LOW" ] || [ "$src" != "'AC Power'" ]; }; then
    p=$(pids | tr '\n' ' '); [ -n "$p" ] && kill -STOP $p 2>/dev/null
    paused=1; echo "$now PAUSED at ${pct}% ($state, $src)" >> "$LOG"
  elif [ "$paused" = 1 ] && [ "$pct" -ge "$HIGH" ] && [ "$src" = "'AC Power'" ] && [ "$state" = "charging" -o "$state" = "charged" -o "$state" = "finishing charge" ]; then
    p=$(pids | tr '\n' ' '); [ -n "$p" ] && kill -CONT $p 2>/dev/null
    paused=0; echo "$now RESUMED at ${pct}% ($state, $src)" >> "$LOG"
  elif [ "$paused" = 1 ]; then
    p=$(pids | tr '\n' ' '); [ -n "$p" ] && kill -STOP $p 2>/dev/null
  else
    # Running: at most MAXRUN heavy python workers at a time (the oldest keep going), the rest held.
    # Other compute first (corrupt, engine.run, sweeps: short jobs), then training (kfold, final). Drivers that only
    # wait on children (evaluate, lomo drivers) are not counted, so they never take a slot.
    wt=$(pgrep -f 'MacOS/Python -m tools\.learned_spike\.(kfold|final)' | sort -n)
    wo=$(pgrep -f 'MacOS/Python -m tools\.learned_spike\.(corrupt|holdout|r3_nested)' | sort -n)
    w=$(printf '%s\n%s\n' "$wo" "$wt" | grep -v '^$')   # short jobs (smoke tests, scoring) first
    mr=${MAXRUN:-1}; [ "$pct" -ge "${BOOST_AT:-50}" ] && mr=${MAXRUN_BOOST:-$mr}
    run=$(echo "$w" | head -n "$mr"); hold=$(echo "$w" | tail -n +"$(( mr + 1 ))")
    [ -n "$run" ] && kill -CONT $run 2>/dev/null; [ -n "$hold" ] && kill -STOP $hold 2>/dev/null
  fi
  if [ "$pct" -le "${HANDOVER:-0}" ] && [ ! -f "$LOG.handover_done" ]; then
    touch "$LOG.handover_done"; echo "$now HANDOVER triggered at ${pct}%" >> "$LOG"
    nohup "$(dirname "$0")/handover.sh" >> "$LOG.handover.out" 2>&1 &
  fi
  if [ "$pct" -le "$CRIT" ] && [ "$state" != "charging" ]; then
    echo "$now CRITICAL ${pct}% ($state, $src): sleeping the Mac" >> "$LOG"
    pmset sleepnow >/dev/null 2>&1 || osascript -e 'tell application "System Events" to sleep' >/dev/null 2>&1
    sleep 300
  fi
  echo "$now ${pct}% $state $src paused=$paused" > "${LOG}.last"
  sleep 60
done
