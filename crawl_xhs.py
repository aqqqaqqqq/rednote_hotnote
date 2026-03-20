import os
import json
import time
import asyncio
import http.client
import schedule
from datetime import datetime, date
from threading import Thread
from dotenv import load_dotenv

from flask import Flask, render_template, jsonify, request, Response, stream_with_context
from openai import OpenAI

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 假设你的 main.py 中有异步爬虫入口
from main import main as crawl_main  

# ==========================================
# 1. 初始化与配置 (Config & Init)
# ==========================================
load_dotenv()

app = Flask(__name__)

# 目录配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DAILY_REPORT_DIR = os.path.join(DATA_DIR, "daily_reports")
AI_SUMMARIES_DIR = os.path.join(DATA_DIR, "ai_summaries")

# 初始化目录
for directory in [DATA_DIR, DAILY_REPORT_DIR, AI_SUMMARIES_DIR]:
    os.makedirs(directory, exist_ok=True)

# OpenAI 配置
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 全局状态
LAST_SCHEDULE_STATUS = {"success": False, "time": None}
sse_clients =[] # 用于SSE通知客户端
USE_LOCAL_MODEL = True # 是否使用本地模型
LOCAL_MODEL_CONFIG = {
    "active": "qwen",   # "qwen" or "minimind"
    "models": {
        "minimind": "./models/MiniMind2",
        "qwen": "./models/Qwen3.5_0.8B",
    }
}

# 本地大模型加载
MODEL = None
TOKENIZER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_local_model_settings():
    model_type = LOCAL_MODEL_CONFIG["active"]
    model_path = LOCAL_MODEL_CONFIG["models"][model_type]
    return model_type, model_path

def init_local_model():
    global MODEL, TOKENIZER
    model_type, model_path = get_local_model_settings()
    print("🔄 正在加载本地模型...")
    TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    MODEL.eval()
    print(f"✅ 本地模型加载完成: type={model_type}, path={model_path}, device={DEVICE}")

# ==========================================
# 2. 通用工具函数 (Utils)
# ==========================================
def save_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_from_json(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_date_str(date_input=None, format="%Y%m%d"):
    """统一日期格式化工具"""
    if date_input is None:
        return date.today().strftime(format)
    if isinstance(date_input, str):
        return datetime.strptime(date_input, "%Y-%m-%d").strftime(format)
    return date_input.strftime(format)

# 文件读写封装
def load_ai_summaries(date_str=None):
    return load_from_json(os.path.join(AI_SUMMARIES_DIR, f"ai_summaries_{get_date_str(date_str)}.json")) or {}

def save_ai_summaries(data, date_str=None):
    save_to_json(data, os.path.join(AI_SUMMARIES_DIR, f"ai_summaries_{get_date_str(date_str)}.json"))

def load_report(date_str=None):
    return load_from_json(os.path.join(DAILY_REPORT_DIR, f"report_{get_date_str(date_str)}.json"))

def local_llm_generate(prompt, max_new_tokens=512):
    global MODEL, TOKENIZER
    messages = [
        {"role": "user", "content": prompt}
    ]
    inputs = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = TOKENIZER(
        inputs,
        return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        outputs = MODEL.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
    response = TOKENIZER.decode(
        outputs[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True
    )
    return response.strip()

# ==========================================
# 3. 核心服务：数据抓取 (Crawler Services)
# ==========================================
def crawl_xhs_hot_topics():
    """从 60s.viki.moe 获取小红书热榜"""
    try:
        conn = http.client.HTTPSConnection("60s.viki.moe", timeout=10)
        conn.request("GET", "/v2/rednote")
        res = conn.getresponse()
        raw = res.read().decode("utf-8")
        conn.close()

        resp = json.loads(raw)
        if resp.get("code") != 200:
            raise ValueError("接口返回异常")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        topics =[{
            "rank": item.get("rank"),
            "title": item.get("title"),
            "score": item.get("score"),
            "word_type": item.get("word_type"),
            "link": item.get("link"),
            "crawl_time": now_str
        } for item in resp.get("data", [])]

        hot_data = {"update_time": now_str, "source": "60s.viki.moe", "total": len(topics), "topics": topics}
        save_to_json(hot_data, os.path.join(DATA_DIR, "hot_topics.json"))
        print(f"✅ 热榜抓取成功，共 {len(topics)} 条")
        return hot_data
    except Exception as e:
        print(f"❌ 热榜抓取失败：{e}")
        return {"update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "topics":[]}

def run_crawler(keyword=None):
    """调用本地爬虫 (main.py) 生成 JSON"""
    try:
        asyncio.run(crawl_main(keywords=keyword))
        print("✅ 爬虫执行完成，JSON 文件已生成")
    except Exception as e:
        print(f"❌ 爬虫执行失败：{e}")

# ==========================================
# 4. 核心服务：AI 分析 (AI Services)
# ==========================================
def ai_analyze_content(content_list, topic_title):
    if not content_list: return "暂无足够内容进行分析。"
    prompt = f"""
    请你作为小红书热点分析师，分析以下「{topic_title}」话题下的内容共性，总结要点（不超过300字）：
    内容列表：{json.dumps([c['content_summary'] for c in content_list], ensure_ascii=False)}
    分析要求：1. 提炼核心观点和用户偏好 2. 总结内容呈现形式的共性 3. 语言简洁，符合小红书平台调性
    """
    try:
        if USE_LOCAL_MODEL:
            return local_llm_generate(prompt, 256)
        else:
            res = client.chat.completions.create(
                model="qwen3-max",
                messages=[{"role": "system", "content": "你是专业的小红书热点内容分析师"}, {"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=500
            )
            return res.choices[0].message.content.strip()
    except Exception as e:
        return f"AI分析异常：{str(e)}"

def generate_daily_report_from_ai(ai_summaries):
    prompt = f"""
    基于以下各热点话题的 AI 总结，生成一份【当日热点分析报告】：
    数据：{json.dumps(ai_summaries, ensure_ascii=False)}
    要求：1. 总结核心方向 2. 分析用户关注点 3. 提炼3-5条运营建议 4. 500字内。
    """
    try:
        if USE_LOCAL_MODEL:
            result = local_llm_generate(prompt, 512)
        else:
            res = client.chat.completions.create(
                model="qwen3-max",
                messages=[{"role": "system", "content": "你是专业分析师"}, {"role": "user", "content": prompt}],
                temperature=0.4, max_tokens=800
            )
            result = res.choices[0].message.content.strip()
        report_data = {
            "report_date": date.today().strftime("%Y-%m-%d"),
            "generate_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall_analysis": result,
            "topic_count": len(ai_summaries)
        }
        save_to_json(report_data, os.path.join(DAILY_REPORT_DIR, f"report_{get_date_str()}.json"))
        return report_data
    except Exception as e:
        print("❌ 日报生成失败：", e)
        return None

# ==========================================
# 5. 定时任务系统 (Scheduler)
# ==========================================
def scheduled_task():
    global LAST_SCHEDULE_STATUS
    try:
        crawl_xhs_hot_topics()
        LAST_SCHEDULE_STATUS = {"success": True, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        print("⏰ 自动抓取完成")
        for q in sse_clients: q.append(LAST_SCHEDULE_STATUS['time']) # 通知前端
    except Exception as e:
        LAST_SCHEDULE_STATUS = {"success": False, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        print("❌ 自动抓取失败：", e)

def start_schedule():
    print("🕒 定时任务线程已启动")
    schedule.every().day.at("10:00").do(scheduled_task)
    schedule.every().day.at("10:03").do(scheduled_task)
    scheduled_task() # 启动时立刻跑一次
    while True:
        schedule.run_pending()
        time.sleep(1)

# ==========================================
# 6. Web 页面路由 (HTML Views)
# ==========================================
@app.route('/')
def index():
    hot_data = load_from_json(os.path.join(DATA_DIR, "hot_topics.json")) or {}
    return render_template("index.html", hot_data=hot_data)

@app.route('/report')
def report():
    query_date = request.args.get("date")
    display_date = query_date or date.today().strftime("%Y-%m-%d")
    report_data = load_report(query_date)
    return render_template("report.html", report_data=report_data, today=display_date, is_history=bool(query_date))

# ==========================================
# 7. API 路由 (JSON Endpoints)
# ==========================================
@app.route('/events')
def sse_events():
    def event_stream():
        q =[]
        sse_clients.append(q)
        try:
            while True:
                if q: yield f"data: {q.pop(0)}\n\n"
                time.sleep(0.5)
        finally:
            sse_clients.remove(q)
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

@app.route('/api/schedule-status')
def schedule_status():
    global LAST_SCHEDULE_STATUS
    status = LAST_SCHEDULE_STATUS.copy()
    LAST_SCHEDULE_STATUS = {"success": False, "time": None} # 读取即重置
    return jsonify(status)

@app.route('/api/force-crawl', methods=['POST'])
def force_crawl():
    try:
        crawl_xhs_hot_topics()
        return jsonify({"code": 200, "msg": "手动抓取成功"})
    except Exception as e:
        return jsonify({"code": 500, "msg": f"手动抓取失败：{e}"})
    
@app.route('/api/view-topic', methods=['GET'])
def view_topic():
    keyword = request.args.get('keyword', '').strip()
    if not keyword:
        return jsonify({"code": 400, "msg": "未提供关键词", "data":[]})

    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        json_file = os.path.join(DATA_DIR, f"xhs/json/search_contents_{today_str}.json")

        if not os.path.exists(json_file):
            return jsonify({"code": 404, "msg": "今天暂未检索过任何话题，请先点击【检索话题】抓取数据", "data":[]})

        with open(json_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        results =[]
        for note in data_list:
            source_kw = note.get("source_keyword", "").strip().lower()
            if keyword.lower() == source_kw:
                results.append({
                    "title": note.get("title", ""),
                    "desc": note.get("desc", "")
                })
        if results:
            return jsonify({"code": 200, "msg": "获取成功", "data": results})
        else:
            return jsonify({"code": 404, "msg": "本地暂无该话题数据，请先点击【检索话题】", "data":[]})

    except json.JSONDecodeError:
        return jsonify({"code": 500, "msg": "本地数据文件格式异常，建议重新检索话题", "data":[]})
    except Exception as e:
        return jsonify({"code": 500, "msg": f"查看话题时发生异常：{str(e)}", "data":[]})

@app.route('/api/search-topic', methods=['GET'])
def search_topic():
    keyword = request.args.get('keyword', '').strip()
    if not keyword: return jsonify({"code": 400, "msg": "未提供关键词", "data":[]})
    
    run_crawler(keyword)
    
    json_file = os.path.join(DATA_DIR, f"xhs/json/search_contents_{get_date_str(format='%Y-%m-%d')}.json")
    data_list = load_from_json(json_file) or[]
    
    results =[{"title": n.get("title"), "desc": n.get("desc")} for n in data_list if keyword.lower() in n.get("source_keyword", "").lower()]
    return jsonify({"code": 200, "msg": "成功", "data": results})

@app.route("/api/ai-summary", methods=["POST"])
def ai_summary():
    data = request.json
    content_text, topic, report_date = data.get("content", "").strip(), data.get("topic", "").strip(), data.get("date")
    if not content_text or not topic: return jsonify({"code": 400, "msg": "内容或关键词为空"})

    content_list =[{"content_summary": line} for line in content_text.split("\n")[:20] if line.strip()]
    summary = ai_analyze_content(content_list, topic)

    summaries = load_ai_summaries(report_date)
    summaries[topic] = {"summary": summary, "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    save_ai_summaries(summaries, report_date)

    return jsonify({"code": 200, "msg": "AI总结成功", "data": {"topic": topic, "summary": summary}})

@app.route('/api/generate-daily-report', methods=['POST'])
def generate_daily_report_api():
    report_date = request.json.get("date")
    ai_summaries = load_ai_summaries(report_date)
    if not ai_summaries: return jsonify({"code": 400, "msg": "暂无话题总结"})
    
    report = generate_daily_report_from_ai(ai_summaries)
    return jsonify({"code": 200, "msg": "报告生成成功", "data": report}) if report else jsonify({"code": 500, "msg": "生成失败"})

# ==========================================
# 8. 启动服务 (Entry Point)
# ==========================================
if __name__ == '__main__':
    if USE_LOCAL_MODEL:
        init_local_model()
    Thread(target=start_schedule, daemon=True).start()
    print("🚀 小红书热点监控工具Web服务已启动：http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False) 
    # 注意：启用了后台线程，建议设置 use_reloader=False，防止线程被启动两次
