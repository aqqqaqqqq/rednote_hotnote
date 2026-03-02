import os
import json
import time
import schedule
from datetime import datetime, date
from threading import Thread
from dotenv import load_dotenv

from flask import Flask, render_template, jsonify
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# 加载环境变量
app = Flask(__name__)
# 定时抓取状态
LAST_SCHEDULE_STATUS = {
    "success": False,
    "time": None
}

# 配置项
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DAILY_REPORT_DIR = os.path.join(DATA_DIR, "daily_reports")
# 小红书热点榜单地址（演示用，仅作参考，可能失效）
XHS_HOT_URL = "https://www.xiaohongshu.com/explore"

# 初始化目录
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DAILY_REPORT_DIR, exist_ok=True)

# ---------------------- 工具函数 ----------------------
def save_to_json(data, file_path):
    """保存数据到JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_from_json(file_path):
    """从JSON文件加载数据"""
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

AI_SUMMARIES_DIR = os.path.join(DATA_DIR, "ai_summaries")
os.makedirs(AI_SUMMARIES_DIR, exist_ok=True)

def get_ai_summary_file(date_str=None):
    """
    date_str: YYYY-MM-DD
    """
    if date_str is None:
        date_str = date.today().strftime("%Y-%m-%d")
    file_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
    return os.path.join(AI_SUMMARIES_DIR, f"ai_summaries_{file_date}.json")

def load_ai_summaries(date_str=None):
    file_path = get_ai_summary_file(date_str)
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_ai_summaries(data, date_str=None):
    file_path = get_ai_summary_file(date_str)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


DAILY_REPORT_DIR = os.path.join(DATA_DIR, "daily_reports")

def load_today_report():
    today_str = date.today().strftime("%Y%m%d")
    report_file = os.path.join(
        DAILY_REPORT_DIR, f"report_{today_str}.json"
    )

    if not os.path.exists(report_file):
        return None

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# ---------------------- 路由 ----------------------

import http.client

def crawl_xhs_hot_topics():
    """从 60s.viki.moe 获取小红书热榜（真实结构适配版）"""
    try:
        conn = http.client.HTTPSConnection("60s.viki.moe", timeout=10)
        conn.request("GET", "/v2/rednote")
        res = conn.getresponse()
        raw = res.read().decode("utf-8")
        conn.close()

        resp = json.loads(raw)

        # 基本校验
        if resp.get("code") != 200:
            raise ValueError("接口返回异常")

        today_str = datetime.now().strftime("%Y%m%d")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        topics = []
        for item in resp.get("data", []):
            topics.append({
                "rank": item.get("rank"),
                "title": item.get("title"),
                "score": item.get("score"),
                "word_type": item.get("word_type"),
                "link": item.get("link"),
                "crawl_time": now_str
            })

        hot_data = {
            "update_time": now_str,
            "source": "60s.viki.moe",
            "total": len(topics),
            "topics": topics
        }

        hot_file = os.path.join(DATA_DIR, "hot_topics.json")
        save_to_json(hot_data, hot_file)

        print(f"✅ 热榜抓取成功，共 {len(topics)} 条")
        return hot_data

    except Exception as e:
        print(f"❌ 热榜抓取失败：{e}")
        return {
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "topics": []
        }
    
@app.route('/api/view-topic', methods=['GET'])
def view_topic():
    """根据 source_keyword 查看 JSON 中的现有话题"""
    keyword = request.args.get('keyword', '').strip()
    if not keyword:
        return jsonify({"code": 400, "msg": "未提供关键词", "data": []})

    # JSON 文件路径（按日期生成）
    today_str = datetime.now().strftime("%Y-%m-%d")
    json_file = os.path.join(DATA_DIR, f"xhs/json/search_contents_{today_str}.json")

    if not os.path.exists(json_file):
        return jsonify({"code": 404, "msg": "数据文件不存在", "data": []})

    with open(json_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # 查找 source_keyword
    results = []
    for note in data_list:
        source_kw = note.get("source_keyword", "").strip().lower()
        if keyword.lower() == source_kw:  # 精确匹配
            results.append({
                "title": note.get("title"),
                "desc": note.get("desc")
            })

    if results:
        return jsonify({"code": 200, "msg": "成功", "data": results})
    else:
        return jsonify({"code": 404, "msg": "没有该话题", "data": []})
    
from flask import request  # 在文件开头导入 request
import asyncio
from main import main as crawl_main  # 假设你贴的爬虫主函数在 main.py

def run_crawler(keyword=None):
    """同步函数，调用异步爬虫生成 JSON"""
    try:
        keywords = keyword if keyword else None  # 单个关键词传给 main()
        asyncio.run(crawl_main(keywords=keywords))
        print("✅ 爬虫执行完成，JSON 文件已生成")
    except Exception as e:
        print(f"❌ 爬虫执行失败：{e}")

@app.route('/api/search-topic', methods=['GET'])
def search_topic():
    """根据 keyword 检索话题内容，并返回 title + desc"""
    keyword = request.args.get('keyword', '').strip()
    if not keyword:
        return jsonify({"code": 400, "msg": "未提供关键词", "data": []})

    # 1️⃣ 先运行爬虫生成最新 JSON
    run_crawler(keyword)

    # 2️⃣ JSON 文件路径（按日期生成）
    today_str = datetime.now().strftime("%Y-%m-%d")
    json_file = os.path.join(DATA_DIR, f"xhs/json/search_contents_{today_str}.json")
    
    if not os.path.exists(json_file):
        return jsonify({"code": 404, "msg": "数据文件不存在", "data": []})

    # 3️⃣ 读取 JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    # 4️⃣ 匹配 source_keyword
    results = []
    for note in data_list:
        source_kw = note.get("source_keyword", "").strip().lower()
        if keyword.lower() in source_kw:
            results.append({
                "title": note.get("title"),
                "desc": note.get("desc")
            })

    print(f"keyword: {keyword}, matches: {len(results)}")

    return jsonify({"code": 200, "msg": "成功", "data": results})

# OpenAI 配置
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# ---------------------- AI分析功能 ----------------------
def ai_analyze_content(content_list, topic_title):
    """AI分析热点内容共性（使用OpenAI API）"""
    if not content_list:
        return "暂无足够内容进行分析。"
    
    # 构造分析提示词
    prompt = f"""
    请你作为小红书热点分析师，分析以下「{topic_title}」话题下的内容共性，总结要点（不超过300字）：
    内容列表：
    {json.dumps([content['content_summary'] for content in content_list], ensure_ascii=False, indent=2)}
    分析要求：
    1. 提炼核心观点和用户偏好
    2. 总结内容呈现形式的共性
    3. 语言简洁，符合小红书平台调性
    """
    
    try:
        # 调用OpenAI Chat Completions API
        completion = client.chat.completions.create(
            model="qwen3-max",
            messages=[
                {"role": "system", "content": "你是专业的小红书热点内容分析师，擅长总结共性和趋势。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # 降低随机性，保证分析准确性
            max_tokens=500
        )
        return completion.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"❌ AI分析失败：{str(e)}")
        return f"AI分析异常：{str(e)}"

@app.route("/api/ai-summary", methods=["POST"])
def ai_summary():
    data = request.json
    content_text = data.get("content", "").strip()
    topic = data.get("topic", "").strip()
    report_date = data.get("date")  # 可选参数 YYYY-MM-DD

    if not content_text or not topic:
        return jsonify({"code": 400, "msg": "内容或关键词为空"})

    content_list = [
        {"content_summary": line}
        for line in content_text.split("\n")
        if line.strip()
    ]

    summary = ai_analyze_content(content_list, topic)

    summaries = load_ai_summaries(report_date)
    summaries[topic] = {
        "summary": summary,
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    save_ai_summaries(summaries, report_date)

    return jsonify({
        "code": 200,
        "msg": "AI 总结成功",
        "data": {"topic": topic, "summary": summary}
    })


@app.route('/api/generate-daily-report', methods=['POST'])
def generate_daily_report_api():
    report_date = request.json.get("date")  # 可选
    ai_summaries = load_ai_summaries(report_date)

    if not ai_summaries:
        return jsonify({"code": 400, "msg": "暂无话题总结，请先执行话题总结"})

    report = generate_daily_report_from_ai(ai_summaries)
    if not report:
        return jsonify({"code": 500, "msg": "日报生成失败"})

    return jsonify({"code": 200, "msg": "每日报告生成成功", "data": report})


def generate_daily_report_from_ai(ai_summaries):
    today = date.today()
    today_str = today.strftime("%Y%m%d")

    summaries_text = json.dumps(
        ai_summaries, ensure_ascii=False, indent=2
    )

    prompt = f"""
    请你作为小红书内容策略分析师，
    基于以下各热点话题的 AI 总结，生成一份【当日热点分析报告】：

    话题总结数据：
    {summaries_text}

    报告要求：
    1. 总结当日热点核心方向
    2. 分析用户关注点与内容偏好
    3. 提炼 3-5 条可执行运营建议
    4. 字数 500 字以内，语言专业清晰
    """

    try:
        response = client.chat.completions.create(
            model="qwen3-max",
            messages=[
                {"role": "system", "content": "你是专业的小红书热点与趋势分析师"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=800
        )

        report_data = {
            "report_date": today.strftime("%Y-%m-%d"),
            "generate_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall_analysis": response.choices[0].message.content.strip(),
            "topic_count": len(ai_summaries)
        }

        os.makedirs(DAILY_REPORT_DIR, exist_ok=True)
        report_file = os.path.join(
            DAILY_REPORT_DIR, f"report_{today_str}.json"
        )

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        return report_data

    except Exception as e:
        print("❌ 日报生成失败：", e)
        return None

def load_report_by_date(date_str):
    """
    date_str: YYYY-MM-DD
    """
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

    file_date = date_obj.strftime("%Y%m%d")
    report_file = os.path.join(
        DAILY_REPORT_DIR, f"report_{file_date}.json"
    )

    if not os.path.exists(report_file):
        return None

    with open(report_file, "r", encoding="utf-8") as f:
        return json.load(f)
    
@app.route('/api/get-today-report', methods=['GET'])
def get_today_report_api():
    """获取今天的日报"""
    report = load_today_report()
    if report:
        return jsonify({"code": 200, "msg": "成功", "data": report})
    else:
        return jsonify({"code": 404, "msg": "今天的日报还未生成", "data": None})

@app.route('/api/compare-reports', methods=['POST'])
def compare_reports():
    """
    请求体：
    {
        "today_report": "今天的日报文本",
        "history_report": "历史日报文本"
    }
    """
    data = request.json
    today_report = data.get("today_report", "").strip()
    history_report = data.get("history_report", "").strip()

    if not today_report or not history_report:
        return jsonify({"code": 400, "msg": "今天或历史日报为空"})

    prompt = f"""
    你是小红书内容策略分析师，请对比两份每日热点报告：
    1️⃣ 历史日报：
    {history_report}

    2️⃣ 今日日报：
    {today_report}

    请总结两者的主要差异与趋势变化（核心方向、用户关注点、内容偏好），不超过300字。
    """

    try:
        response = client.chat.completions.create(
            model="qwen3-max",
            messages=[
                {"role": "system", "content": "你是专业的小红书热点与趋势分析师"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=500
        )

        comparison_text = response.choices[0].message.content.strip()

        return jsonify({
            "code": 200,
            "msg": "对比分析完成",
            "data": {"comparison_analysis": comparison_text}
        })

    except Exception as e:
        return jsonify({
            "code": 500,
            "msg": f"AI对比分析失败：{e}"
        })

# ---------------------- 定时任务 ----------------------

from flask import Response, stream_with_context

# 用于存储抓取状态的事件
clients = []

@app.route('/events')
def sse_events():
    def event_stream():
        q = []
        clients.append(q)
        try:
            while True:
                if q:
                    msg = q.pop(0)
                    yield f"data: {msg}\n\n"
                time.sleep(0.5)
        finally:
            clients.remove(q)
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


def scheduled_task():
    global LAST_SCHEDULE_STATUS
    try:
        do_crawl_job()
        LAST_SCHEDULE_STATUS = {
            "success": True,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        print("⏰ 自动抓取完成")

        # ⭐ 通知所有前端
        for q in clients:
            q.append(LAST_SCHEDULE_STATUS['time'])

    except Exception as e:
        LAST_SCHEDULE_STATUS = {
            "success": False,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        print("❌ 自动抓取失败：", e)

@app.route('/api/schedule-status')
def schedule_status():
    global LAST_SCHEDULE_STATUS

    status = LAST_SCHEDULE_STATUS.copy()

    # ⭐ 读取一次就重置
    LAST_SCHEDULE_STATUS = {
        "success": False,
        "time": None
    }

    return jsonify(status)

def start_schedule():
    print("🕒 定时任务线程已启动")

    schedule.every().day.at("10:00").do(scheduled_task)
    schedule.every().day.at("10:03").do(scheduled_task)

    # ⭐ 启动时立刻跑一次（非常关键）
    scheduled_task()

    while True:
        schedule.run_pending()
        time.sleep(1)

def do_crawl_job():
    """真正执行抓取任务（不关心是手动还是自动）"""
    crawl_xhs_hot_topics()

# ---------------------- Flask Web路由 ----------------------
@app.route('/')
def index():
    """主页：展示当前热点榜单"""
    hot_file = os.path.join(DATA_DIR, "hot_topics.json")
    hot_data = load_from_json(hot_file)
    return render_template("index.html", hot_data=hot_data)

@app.route('/report')
def report():
    """
    日报页面：
    - 默认显示今天
    - ?date=YYYY-MM-DD 显示历史日报
    """
    query_date = request.args.get("date")

    today_str = date.today().strftime("%Y-%m-%d")

    if query_date:
        report_data = load_report_by_date(query_date)
        display_date = query_date
    else:
        report_data = load_today_report()
        display_date = today_str

    return render_template(
        "report.html",
        report_data=report_data,
        today=display_date,
        is_history=bool(query_date)
    )


@app.route('/api/force-crawl', methods=['POST'])
def force_crawl():
    try:
        do_crawl_job()
        return jsonify({
            "code": 200,
            "msg": "手动抓取成功（不影响自动任务）"
        })
    except Exception as e:
        return jsonify({
            "code": 500,
            "msg": f"手动抓取失败：{e}"
        })

# ---------------------- 启动服务 ----------------------
if __name__ == '__main__':
    # 启动定时任务线程（后台运行，不阻塞Flask服务）
    schedule_thread = Thread(target=start_schedule, daemon=True)
    schedule_thread.start()
    
    # 启动Flask Web服务
    print("🚀 小红书热点监控工具Web服务已启动：http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
    # app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
