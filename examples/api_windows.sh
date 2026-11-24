#!/usr/bin/env bash
# 两种窗口的端到端 API 调用示例（curl + jq），同时作为可执行自测。
#
# 前置：
#   docker compose up --build          # 或本地 uvicorn app.main:app --port 8000
#
# 用法：
#   bash examples/api_windows.sh
#   BASE_URL=http://127.0.0.1:8000 bash examples/api_windows.sh
#
# 窗口一：12000x12000 扫描图的右下角 1x1 响应（本次内存修复的场景）。
#         /required-coordinates 只给 2 个必读点；/evaluate 额外接收一个
#         远在窗口外的合法图内像素，响应值不变；临时内存与 H*W 无关。
# 窗口二：全 a=0 模型的 3x3 全图窗口 —— 必读清单为空，pixels={} 可直接
#         求值；零缩放通道 b 折叠出的常量仍按图内 tap 数逐点计入
#         （角点 4 个 tap=4.5，边中 6 个=6.5，中心 9 个=9.5）。
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
if ! command -v jq >/dev/null 2>&1; then
  echo "本脚本需要 jq（也可照抄其中的 curl 命令）" >&2
  exit 2
fi

echo "== GET /health =="
curl -sf "$BASE_URL/health" | jq -c .

# ===========================================================================
# 窗口一：12000x12000 大图，右下角 1x1
# ===========================================================================
echo
echo "========== 窗口一：12000x12000 右下角 1x1 =========="

# 与 README / examples/corner_demo.py 相同的模型：C=2, O=3，通道 1 的 a=0
CONV1="$(curl -sf -X POST "$BASE_URL/convert" \
  -H 'Content-Type: application/json' \
  -d '{
    "a": [1.5, 0.0], "b": [-2.0, 0.75],
    "kernel": [
      [[[0,0,0],[0,1,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]],
      [[[0.25,0,-0.5],[0,1,0],[0.5,0,-0.25]], [[0,0,0],[0,0,0],[0,0,0]]],
      [[[0,0,0],[0,0,0],[0,0,0]], [[-1.25,0,0],[0,0,0],[0,0,0]]]
    ],
    "bias": [0.5, -1.0, 2.0]
  }')"
PACK1="$(jq -r '.pack' <<<"$CONV1")"
echo "[convert] $(jq -c '{pack_id,C,O}' <<<"$CONV1") pack长度=${#PACK1}"

WIN1='{"r0":11999,"c0":11999,"r1":12000,"c1":12000}'

echo "-- POST /required-coordinates --"
REQ1="$(curl -sf -X POST "$BASE_URL/required-coordinates" \
  -H 'Content-Type: application/json' \
  --data "$(jq -nc --arg p "$PACK1" '{pack:$p,H:12000,W:12000,window:'"$WIN1"'}')")"
jq -c '{count,coords,active_channels}' <<<"$REQ1"
jq -e '.coords == [[11998,11998],[11999,11999]]' >/dev/null <<<"$REQ1"

echo "-- POST /evaluate（2 个必读像素 + (12,34) 远处合法额外像素）--"
EVAL1="$(curl -sf -X POST "$BASE_URL/evaluate" \
  -H 'Content-Type: application/json' \
  --data "$(jq -nc --arg p "$PACK1" '{
      pack:$p,H:12000,W:12000,window:'"$WIN1"',
      pixels:{
        "11998,11998":[2.0,-2.0],
        "11999,11999":[3.0,-4.0],
        "12,34":[9.0,9.0]
      }}')")"
jq -c '{shape,output,pixels_used}' <<<"$EVAL1"
# 与 4x3 图右下角同一补零几何，期望值同 README：[3.0, 1.75, 1.0625]
jq -e '.output == [[[3.0,1.75,1.0625]]] and .pixels_used == 3' >/dev/null <<<"$EVAL1"
echo "窗口一核对通过 ✓"

echo "-- 缺任一必读点（删掉 11998,11998）必须 422 --"
CODE="$(curl -s -o /tmp/api_win1_missing.json -w '%{http_code}' \
  -X POST "$BASE_URL/evaluate" -H 'Content-Type: application/json' \
  --data "$(jq -nc --arg p "$PACK1" '{
      pack:$p,H:12000,W:12000,window:'"$WIN1"',
      pixels:{"11999,11999":[3.0,-4.0]}}')")"
echo "HTTP $CODE: $(jq -r '.detail' /tmp/api_win1_missing.json)"
[[ "$CODE" == "422" ]] || { echo "应当 422"; exit 1; }

# ===========================================================================
# 窗口二：空必读清单，pixels={}，零缩放通道的图内 tap 常量仍计入
# ===========================================================================
echo
echo "========== 窗口二：3x3 全图窗口，空清单 + pixels={} =========="

# a 全为 0 -> 不读任何像素；Y=b=[2,-1]，K 全 1 -> 每 tap 常量 2-1=1
CONV2="$(curl -sf -X POST "$BASE_URL/convert" \
  -H 'Content-Type: application/json' \
  -d '{
    "a": [0.0, 0.0], "b": [2.0, -1.0],
    "kernel": [[
      [[1,1,1],[1,1,1],[1,1,1]],
      [[1,1,1],[1,1,1],[1,1,1]]
    ]],
    "bias": [0.5]
  }')"
PACK2="$(jq -r '.pack' <<<"$CONV2")"

WIN2='{"r0":0,"c0":0,"r1":3,"c1":3}'
echo "-- POST /required-coordinates --"
REQ2="$(curl -sf -X POST "$BASE_URL/required-coordinates" \
  -H 'Content-Type: application/json' \
  --data "$(jq -nc --arg p "$PACK2" '{pack:$p,H:3,W:3,window:'"$WIN2"'}')")"
jq -c '{count,coords,active_channels}' <<<"$REQ2"
jq -e '.coords == [] and .active_channels == []' >/dev/null <<<"$REQ2"

echo "-- POST /evaluate with pixels={} --"
EVAL2="$(curl -sf -X POST "$BASE_URL/evaluate" \
  -H 'Content-Type: application/json' \
  --data "$(jq -nc --arg p "$PACK2" '{
      pack:$p,H:3,W:3,window:'"$WIN2"',pixels:{}}')")"
jq -c '{shape,output}' <<<"$EVAL2"
jq -e '.output == [
          [[4.5],[6.5],[4.5]],
          [[6.5],[9.5],[6.5]],
          [[4.5],[6.5],[4.5]]
        ]' >/dev/null <<<"$EVAL2"
echo "窗口二核对通过 ✓（角点/边中/中心分别计入 4/6/9 个图内 tap）"

echo
echo "全部 API 示例与断言通过 ✔"
