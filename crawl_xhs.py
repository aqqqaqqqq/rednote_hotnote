import json
import os
import time
from datetime import date, datetime
from threading import Thread

import schedule
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from xhs_services import (
    DATA_DIR,
    USE_LOCAL_MODEL,
    compare_reports_with_ai,
    crawl_xhs_hot_topics,
    generate_daily_report_from_ai,
    generate_topic_summary,
    init_local_model,
    load_ai_summaries,
    load_from_json,
    load_report,
    load_topic_search_data,
    run_crawler,
    save_ai_summaries,
)


app = Flask(__name__)

LAST_SCHEDULE_STATUS = {"success": False, "time": None}
sse_clients = []


def scheduled_task():
    # 执行一次热榜抓取并更新调度状态。
    global LAST_SCHEDULE_STATUS

    try:
        crawl_xhs_hot_topics()
        LAST_SCHEDULE_STATUS = {
            "success": True,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        print("自动抓取完成")
        for queue in sse_clients:
            queue.append(LAST_SCHEDULE_STATUS["time"])
    except Exception as exc:
        LAST_SCHEDULE_STATUS = {
            "success": False,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        print(f"自动抓取失败: {exc}")


def start_schedule():
    # 启动定时任务并持续轮询待执行任务。
    print("定时任务线程已启动")
    schedule.every().day.at("10:00").do(scheduled_task)
    schedule.every().day.at("10:03").do(scheduled_task)
    scheduled_task()
    while True:
        schedule.run_pending()
        time.sleep(1)


@app.route("/")
def index():
    # 渲染首页并展示当前热榜数据。
    hot_data = load_from_json(os.path.join(DATA_DIR, "hot_topics.json")) or {}
    return render_template("index.html", hot_data=hot_data)


@app.route("/report")
def report():
    # 渲染日报页面并支持按日期查看历史报告。
    query_date = request.args.get("date")
    display_date = query_date or date.today().strftime("%Y-%m-%d")
    report_data = load_report(query_date)
    return render_template(
        "report.html",
        report_data=report_data,
        today=display_date,
        is_history=bool(query_date),
    )


@app.route("/events")
def sse_events():
    # 提供 SSE 接口向前端推送抓取完成时间。
    def event_stream():
        # 为单个客户端持续输出事件流数据。
        queue = []
        sse_clients.append(queue)
        try:
            while True:
                if queue:
                    yield f"data: {queue.pop(0)}\n\n"
                time.sleep(0.5)
        finally:
            sse_clients.remove(queue)

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


@app.route("/api/schedule-status")
def schedule_status():
    # 返回最近一次定时任务状态并在读取后重置。
    global LAST_SCHEDULE_STATUS

    status = LAST_SCHEDULE_STATUS.copy()
    LAST_SCHEDULE_STATUS = {"success": False, "time": None}
    return jsonify(status)


@app.route("/api/force-crawl", methods=["POST"])
def force_crawl():
    # 手动触发一次热榜抓取。
    try:
        crawl_xhs_hot_topics()
        return jsonify({"code": 200, "msg": "手动抓取成功"})
    except Exception as exc:
        return jsonify({"code": 500, "msg": f"手动抓取失败：{exc}"})


@app.route("/api/get-today-report", methods=["GET"])
def get_today_report():
    # 获取今日日报，供历史日报对比前校验使用。
    today_report = load_report()
    if not today_report:
        return jsonify(
            {"code": 404, "msg": "今日未生成日报，请先生成日报。", "data": None}
        )
    return jsonify({"code": 200, "msg": "获取成功", "data": today_report})


@app.route("/api/view-topic", methods=["GET"])
def view_topic():
    # 查看本地缓存中某个话题对应的笔记内容。
    keyword = request.args.get("keyword", "").strip()
    if not keyword:
        return jsonify({"code": 400, "msg": "未提供关键词", "data": []})

    try:
        data_list = load_topic_search_data()
        if not data_list:
            return jsonify(
                {
                    "code": 404,
                    "msg": "今天暂未检索过任何话题，请先点击“检索话题”抓取数据",
                    "data": [],
                }
            )

        results = []
        for note in data_list:
            source_kw = note.get("source_keyword", "").strip().lower()
            if keyword.lower() == source_kw:
                results.append(
                    {
                        "title": note.get("title", ""),
                        "desc": note.get("desc", ""),
                    }
                )

        if results:
            return jsonify({"code": 200, "msg": "获取成功", "data": results})
        return jsonify({"code": 404, "msg": "本地暂无该话题数据，请先点击“检索话题”", "data": []})
    except json.JSONDecodeError:
        return jsonify({"code": 500, "msg": "本地数据文件格式异常，建议重新检索话题", "data": []})
    except Exception as exc:
        return jsonify({"code": 500, "msg": f"查看话题时发生异常：{exc}", "data": []})


@app.route("/api/search-topic", methods=["GET"])
def search_topic():
    # 调用爬虫搜索指定话题并返回结果。
    keyword = request.args.get("keyword", "").strip()
    if not keyword:
        return jsonify({"code": 400, "msg": "未提供关键词", "data": []})

    run_crawler(keyword)
    data_list = load_topic_search_data()
    results = [
        {"title": note.get("title"), "desc": note.get("desc")}
        for note in data_list
        if keyword.lower() in note.get("source_keyword", "").lower()
    ]
    return jsonify({"code": 200, "msg": "成功", "data": results})


@app.route("/api/ai-summary", methods=["POST"])
def ai_summary():
    # 生成单个话题的 AI 总结并写入本地缓存。
    data = request.json or {}
    content_text = data.get("content", "").strip()
    topic = data.get("topic", "").strip()
    report_date = data.get("date")
    enable_thinking = bool(data.get("enable_thinking", False))
    max_images = int(data.get("max_images", 6) or 6)

    if not topic:
        return jsonify({"code": 400, "msg": "关键词为空"})

    summary_data = generate_topic_summary(
        topic=topic,
        report_date=report_date,
        content_text=content_text,
        enable_thinking=enable_thinking,
        max_images=max_images,
    )
    if not summary_data:
        return jsonify({"code": 404, "msg": "未找到可用于总结的话题内容，请先检索话题"})

    summaries = load_ai_summaries(report_date)
    summaries[topic] = {
        "summary": summary_data["summary"],
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": summary_data["model"],
        "mode": summary_data["mode"],
        "image_count": summary_data["image_count"],
    }
    save_ai_summaries(summaries, report_date)

    return jsonify({"code": 200, "msg": "AI总结成功", "data": summary_data})


@app.route("/api/generate-daily-report", methods=["POST"])
def generate_daily_report_api():
    # 基于已有话题总结生成整日报告。
    data = request.json or {}
    report_date = data.get("date")
    ai_summaries = load_ai_summaries(report_date)
    if not ai_summaries:
        return jsonify({"code": 400, "msg": "暂无话题总结"})

    report = generate_daily_report_from_ai(ai_summaries)
    if report:
        return jsonify({"code": 200, "msg": "报告生成成功", "data": report})
    return jsonify({"code": 500, "msg": "生成失败"})


@app.route("/api/compare-reports", methods=["POST"])
def compare_reports_api():
    # 调用大模型对比今日日报和历史日报的趋势变化。
    data = request.json or {}
    today_report = (data.get("today_report") or "").strip()
    history_report = (data.get("history_report") or "").strip()

    if not today_report or not history_report:
        return jsonify({"code": 400, "msg": "缺少用于对比的日报内容"})

    comparison_analysis = compare_reports_with_ai(today_report, history_report)
    if not comparison_analysis:
        return jsonify({"code": 500, "msg": "日报对比分析失败"})

    return jsonify(
        {
            "code": 200,
            "msg": "对比成功",
            "data": {"comparison_analysis": comparison_analysis},
        }
    )


if __name__ == "__main__":
    if USE_LOCAL_MODEL:
        init_local_model()
    Thread(target=start_schedule, daemon=True).start()
    print("小红书热点监控 Web 服务已启动：http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
