import json
import re
from datetime import datetime
from typing import Any

from xhs_services import (
    compare_reports_with_ai,
    crawl_xhs_hot_topics,
    generate_daily_report_from_ai,
    generate_topic_summary,
    load_ai_summaries,
    load_report,
    load_topic_search_data,
    run_crawler,
    save_ai_summaries,
)


TOOL_DESCRIPTIONS = """
You are an agent for a RedNote hot-topic analysis system.
Choose the single best next tool based on the user instruction and previous observations.

Available tools:
1. get_hot_topics()
   Use when the user asks to crawl, refresh, or view today's hot topics.
2. search_topic(keyword: str)
   Use when the user asks to search notes/content for one topic.
3. summarize_topic(topic: str)
   Use when the user asks to summarize one topic.
4. generate_daily_report()
   Use when the user asks to generate today's daily report from existing summaries.
5. compare_reports(history_date: str)
   Use when the user asks to compare today's report with a historical report.
6. finish(answer: str)
   Use when you can directly answer the user.

Return JSON only:
{"tool": "...", "args": {...}, "reason": "..."}
"""


def _extract_date(text: str) -> str | None:
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", text)
    if match:
        return match.group(1)
    return None


def _extract_topic(text: str) -> str | None:
    patterns = [
        r"[\"'“”](.+?)[\"'“”]",
        r"话题[:：]\s*(.+)",
        r"主题[:：]\s*(.+)",
        r"分析(.+)",
        r"总结(.+)",
        r"搜索(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            topic = match.group(1).strip(" ，。！？,.!?\n")
            if topic:
                return topic
    return None


def _fallback_plan(instruction: str) -> dict[str, Any]:
    text = instruction.strip()
    if not text:
        return {
            "tool": "finish",
            "args": {"answer": "请输入你想执行的热点分析任务。"},
            "reason": "empty instruction",
        }

    lower_text = text.lower()
    history_date = _extract_date(text)
    topic = _extract_topic(text)

    if "对比" in text or "比较" in text:
        return {
            "tool": "compare_reports",
            "args": {"history_date": history_date},
            "reason": "user asks for report comparison",
        }
    if "日报" in text or "报告" in text:
        return {
            "tool": "generate_daily_report",
            "args": {},
            "reason": "user asks for daily report",
        }
    if any(keyword in text for keyword in ["总结", "概括", "摘要", "分析"]) and topic:
        return {
            "tool": "summarize_topic",
            "args": {"topic": topic},
            "reason": "user asks for topic summary",
        }
    if any(keyword in text for keyword in ["搜索", "检索", "查找", "看看"]) and topic:
        return {
            "tool": "search_topic",
            "args": {"keyword": topic},
            "reason": "user asks to search a topic",
        }
    if any(keyword in lower_text for keyword in ["crawl", "refresh", "hot"]) or any(
        keyword in text for keyword in ["热榜", "热点", "抓取", "刷新"]
    ):
        return {
            "tool": "get_hot_topics",
            "args": {},
            "reason": "user asks for hot topics",
        }

    return {
        "tool": "finish",
        "args": {
            "answer": "我可以帮你抓取热榜、搜索话题、总结热点、生成日报，或对比历史报告。"
        },
        "reason": "fallback direct answer",
    }


def _plan_next_action(instruction: str) -> dict[str, Any]:
    try:
        from xhs_services import client

        completion = client.chat.completions.create(
            model="qwen3-max",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": TOOL_DESCRIPTIONS},
                {"role": "user", "content": instruction},
            ],
        )
        content = completion.choices[0].message.content or "{}"
        payload = json.loads(content)
        if "tool" in payload:
            return payload
    except Exception:
        pass
    return _fallback_plan(instruction)


def _normalize_tool_name(tool_name: str) -> str:
    normalized = (tool_name or "").strip()
    if normalized.endswith("()"):
        normalized = normalized[:-2]
    normalized = normalized.split("(", 1)[0].strip()
    return normalized


def _tool_get_hot_topics() -> dict[str, Any]:
    hot_data = crawl_xhs_hot_topics()
    topics = hot_data.get("topics", [])[:10]
    lines = [
        f"{item.get('rank')}. {item.get('title')} (热度: {item.get('score')})"
        for item in topics
    ]
    return {
        "tool": "get_hot_topics",
        "observation": hot_data,
        "answer": "已抓取今日热点。\n" + ("\n".join(lines) if lines else "暂无热点数据。"),
    }


def _tool_search_topic(keyword: str) -> dict[str, Any]:
    run_crawler(keyword)
    notes = [
        note
        for note in load_topic_search_data()
        if keyword.lower() in (note.get("source_keyword") or "").lower()
    ]
    preview = []
    for index, note in enumerate(notes[:5], start=1):
        title = note.get("title", "").strip()
        desc = note.get("desc", "").strip()
        preview.append(f"{index}. {title}\n{desc}")
    return {
        "tool": "search_topic",
        "observation": {"keyword": keyword, "count": len(notes)},
        "answer": (
            f"已搜索话题“{keyword}”，共找到 {len(notes)} 条内容。\n\n"
            + ("\n\n".join(preview) if preview else "暂无可用内容。")
        ),
    }


def _tool_summarize_topic(topic: str) -> dict[str, Any]:
    summary_data = generate_topic_summary(topic=topic)
    if not summary_data:
        run_crawler(topic)
        summary_data = generate_topic_summary(topic=topic)
    if not summary_data:
        return {
            "tool": "summarize_topic",
            "observation": {"topic": topic, "success": False},
            "answer": f"未找到可用于总结的话题“{topic}”内容，建议先执行检索。",
        }

    summaries = load_ai_summaries()
    summaries[topic] = {
        "summary": summary_data.get("summary", ""),
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": summary_data.get("model"),
        "mode": summary_data.get("mode"),
        "image_count": summary_data.get("image_count", 0),
    }
    save_ai_summaries(summaries)

    return {
        "tool": "summarize_topic",
        "observation": {
            "topic": topic,
            "saved": True,
            "model": summary_data.get("model"),
            "image_count": summary_data.get("image_count"),
        },
        "answer": f"话题“{topic}”的总结如下：\n\n{summary_data.get('summary', '')}",
    }


def _tool_generate_daily_report() -> dict[str, Any]:
    ai_summaries = load_ai_summaries()
    if not ai_summaries:
        return {
            "tool": "generate_daily_report",
            "observation": {"success": False},
            "answer": "当前还没有可用的话题总结，建议先搜索并总结若干热点话题，再生成日报。",
        }
    report = generate_daily_report_from_ai(ai_summaries)
    if not report:
        return {
            "tool": "generate_daily_report",
            "observation": {"success": False},
            "answer": "日报生成失败，请检查模型或数据状态。",
        }
    return {
        "tool": "generate_daily_report",
        "observation": {"success": True, "topic_count": report.get("topic_count")},
        "answer": f"今日日报已生成：\n\n{report.get('overall_analysis', '')}",
    }


def _tool_compare_reports(history_date: str | None) -> dict[str, Any]:
    today_report = load_report()
    if not today_report:
        return {
            "tool": "compare_reports",
            "observation": {"success": False},
            "answer": "今日报告还不存在，建议先生成今日日报。",
        }

    if not history_date:
        return {
            "tool": "compare_reports",
            "observation": {"success": False},
            "answer": "请在指令中提供要对比的历史日期，例如 2026-04-03。",
        }

    history_report = load_report(history_date)
    if not history_report:
        return {
            "tool": "compare_reports",
            "observation": {"success": False, "history_date": history_date},
            "answer": f"未找到 {history_date} 的历史报告。",
        }

    result = compare_reports_with_ai(
        today_report.get("overall_analysis", ""),
        history_report.get("overall_analysis", ""),
    )
    if not result:
        return {
            "tool": "compare_reports",
            "observation": {"success": False, "history_date": history_date},
            "answer": "报告对比失败，请稍后重试。",
        }

    return {
        "tool": "compare_reports",
        "observation": {"success": True, "history_date": history_date},
        "answer": f"今日与 {history_date} 的热点趋势对比如下：\n\n{result}",
    }


def run_agent_instruction(instruction: str) -> dict[str, Any]:
    action = _plan_next_action(instruction)
    raw_tool_name = action.get("tool", "finish")
    tool_name = _normalize_tool_name(raw_tool_name)
    args = action.get("args") or {}
    steps = [
        {
            "stage": "plan",
            "tool": tool_name,
            "raw_tool": raw_tool_name,
            "reason": action.get("reason", ""),
            "args": args,
        }
    ]

    if tool_name == "get_hot_topics":
        result = _tool_get_hot_topics()
    elif tool_name == "search_topic":
        result = _tool_search_topic((args.get("keyword") or "").strip())
    elif tool_name == "summarize_topic":
        result = _tool_summarize_topic((args.get("topic") or "").strip())
    elif tool_name == "generate_daily_report":
        result = _tool_generate_daily_report()
    elif tool_name == "compare_reports":
        result = _tool_compare_reports(args.get("history_date"))
    else:
        result = {
            "tool": "finish",
            "observation": {},
            "answer": args.get("answer") or "任务已结束。",
        }

    steps.append(
        {
            "stage": "act",
            "tool": result.get("tool"),
            "observation": result.get("observation"),
        }
    )
    return {
        "success": True,
        "plan": action,
        "steps": steps,
        "answer": result.get("answer", ""),
    }
