import asyncio
import http.client
import json
import os
import sys
from datetime import date, datetime

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DAILY_REPORT_DIR = os.path.join(DATA_DIR, "daily_reports")
AI_SUMMARIES_DIR = os.path.join(DATA_DIR, "ai_summaries")
HOT_TOPIC_DIR = os.path.join(DATA_DIR, "hot_topic")

for directory in [DATA_DIR, DAILY_REPORT_DIR, AI_SUMMARIES_DIR, HOT_TOPIC_DIR]:
    os.makedirs(directory, exist_ok=True)

FRAMEWORK_DIR = os.path.join(BASE_DIR, "MediaCrawler")
if FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, FRAMEWORK_DIR)

from MediaCrawler.main import main as crawl_main


client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 是否使用本地模型
USE_LOCAL_MODEL = False
# 是否使用 VLM
USE_VLM_FOR_SUMMARY = True
# 选择本地模型
LOCAL_MODEL_CONFIG = {
    "active": "qwen",
    "models": {
        "minimind": "./models/MiniMind2",
        "qwen": "./models/Qwen3.5_0.8B",
    },
}

MODEL = None
TOKENIZER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_local_model_settings():
    # 读取当前启用的本地模型类型和路径。
    model_type = LOCAL_MODEL_CONFIG["active"]
    model_path = LOCAL_MODEL_CONFIG["models"][model_type]
    return model_type, model_path


def init_local_model():
    # 加载本地模型和分词器到指定设备。
    global MODEL, TOKENIZER

    model_type, model_path = get_local_model_settings()
    print("正在加载本地模型...")
    TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)
    MODEL.eval()
    print(f"本地模型加载完成: type={model_type}, path={model_path}, device={DEVICE}")


def save_to_json(data, file_path):
    # 将数据保存为 JSON 文件。
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def load_from_json(file_path):
    # 从 JSON 文件读取数据。
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_date_str(date_input=None, fmt="%Y%m%d"):
    # 统一格式化日期字符串。
    if date_input is None:
        return date.today().strftime(fmt)
    if isinstance(date_input, str):
        return datetime.strptime(date_input, "%Y-%m-%d").strftime(fmt)
    return date_input.strftime(fmt)


def load_ai_summaries(date_str=None):
    # 读取指定日期的话题总结缓存。
    file_path = os.path.join(
        AI_SUMMARIES_DIR, f"ai_summaries_{get_date_str(date_str)}.json"
    )
    return load_from_json(file_path) or {}


def save_ai_summaries(data, date_str=None):
    # 保存指定日期的话题总结缓存。
    file_path = os.path.join(
        AI_SUMMARIES_DIR, f"ai_summaries_{get_date_str(date_str)}.json"
    )
    save_to_json(data, file_path)


def load_report(date_str=None):
    # 读取指定日期的日报数据。
    file_path = os.path.join(
        DAILY_REPORT_DIR, f"report_{get_date_str(date_str)}.json"
    )
    return load_from_json(file_path)


def get_hot_topic_archive_path(date_str=None):
    # 返回按日期归档的热榜文件路径。
    return os.path.join(
        HOT_TOPIC_DIR, f"hot_topics_{get_date_str(date_str)}.json"
    )


def load_topic_search_data(date_str=None):
    # 读取指定日期的话题检索结果。
    target_date = get_date_str(date_str, fmt="%Y-%m-%d")
    json_file = os.path.join(DATA_DIR, f"xhs/json/search_contents_{target_date}.json")
    return load_from_json(json_file) or []


def get_topic_notes(topic, date_str=None):
    # 按关键词筛选匹配的话题笔记。
    keyword = (topic or "").strip().lower()
    if not keyword:
        return []

    notes = []
    for note in load_topic_search_data(date_str):
        source_kw = (note.get("source_keyword") or "").strip().lower()
        if source_kw == keyword:
            notes.append(note)
    return notes


def extract_image_urls_from_notes(notes, max_images=6):
    # 从笔记列表中提取去重后的图片链接。
    image_urls = []
    for note in notes:
        raw_images = note.get("image_list") or note.get("images_list") or ""
        candidates = raw_images if isinstance(raw_images, list) else str(raw_images).split(",")

        for image_url in candidates:
            clean_url = str(image_url).strip()
            if (
                clean_url
                and clean_url.startswith(("http://", "https://"))
                and clean_url not in image_urls
            ):
                image_urls.append(clean_url)
            if len(image_urls) >= max_images:
                return image_urls
    return image_urls


def local_llm_generate(prompt, max_new_tokens=512):
    # 使用本地模型生成文本回复。
    messages = [{"role": "user", "content": prompt}]
    inputs = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = TOKENIZER(inputs, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = MODEL.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
    response = TOKENIZER.decode(
        outputs[0][len(inputs["input_ids"][0]) :],
        skip_special_tokens=True,
    )
    return response.strip()


def crawl_xhs_hot_topics():
    # 从接口抓取小红书热榜并写入本地文件。
    try:
        conn = http.client.HTTPSConnection("60s.viki.moe", timeout=10)
        conn.request("GET", "/v2/rednote")
        response = conn.getresponse()
        raw = response.read().decode("utf-8")
        conn.close()

        payload = json.loads(raw)
        if payload.get("code") != 200:
            raise ValueError("接口返回异常")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        topics = [
            {
                "rank": item.get("rank"),
                "title": item.get("title"),
                "score": item.get("score"),
                "word_type": item.get("word_type"),
                "link": item.get("link"),
                "crawl_time": now_str,
            }
            for item in payload.get("data", [])
        ]

        hot_data = {
            "update_time": now_str,
            "source": "60s.viki.moe",
            "total": len(topics),
            "topics": topics,
        }
        save_to_json(hot_data, os.path.join(DATA_DIR, "hot_topics.json"))
        save_to_json(hot_data, get_hot_topic_archive_path())
        print(f"热榜抓取成功，共 {len(topics)} 条")
        return hot_data
    except Exception as exc:
        print(f"热榜抓取失败: {exc}")
        return {
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "topics": [],
        }


def run_crawler(keyword=None):
    # 调用 MediaCrawler 执行话题爬取。
    try:
        asyncio.run(crawl_main(keywords=keyword))
        print("爬虫执行完成，JSON 文件已生成")
    except Exception as exc:
        print(f"爬虫执行失败: {exc}")


def ai_analyze_content(content_list, topic_title):
    # 使用文本模型对话题内容做总结分析。
    if not content_list:
        return "暂无足够内容进行分析。"

    prompt = f"""
请你作为小红书热点分析师，分析以下“{topic_title}”话题下的内容共性，总结要点（不超过300字）。
内容列表：{json.dumps([c["content_summary"] for c in content_list], ensure_ascii=False)}
分析要求：1. 提炼核心观点和用户偏好 2. 总结内容呈现形式的共性 3. 语言简洁，符合小红书平台调性
"""
    try:
        if USE_LOCAL_MODEL:
            return local_llm_generate(prompt, 256)

        response = client.chat.completions.create(
            model="qwen3-max",
            messages=[
                {"role": "system", "content": "你是专业的小红书热点内容分析师"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"AI分析异常: {exc}"


def ai_analyze_content_with_qwen_vl(
    content_list, topic_title, image_urls=None, enable_thinking=False
):
    # 使用多模态模型结合图文内容生成总结。
    if not content_list:
        return "暂无足够内容进行分析。"

    image_urls = image_urls or []
    text_blocks = []
    for idx, item in enumerate(content_list[:12], start=1):
        title = (item.get("title") or "").strip()
        desc = (item.get("content_summary") or "").strip()
        text_blocks.append(f"{idx}. 标题：{title}\n内容：{desc}".strip())

    prompt = (
        f"请你作为小红书热点分析师，结合给定图片和文本内容，对“{topic_title}”做话题总结。\n"
        "输出要求：\n"
        "1. 总结该话题的核心内容方向。\n"
        "2. 提炼用户关注点、视觉风格或内容表达特点。\n"
        "3. 给出 2-3 条适合运营或选题的启发。\n"
        "4. 输出控制在 300 字内，语言自然、简洁。\n\n"
        f"文本素材：\n{chr(10).join(text_blocks)}"
    )

    message_content = [{"type": "text", "text": prompt}]
    for image_url in image_urls[:6]:
        message_content.append({"type": "image_url", "image_url": {"url": image_url}})

    try:
        completion = client.chat.completions.create(
            model="qwen3-vl-plus",
            messages=[{"role": "user", "content": message_content}],
            extra_body={
                "enable_thinking": enable_thinking,
                "thinking_budget": 8192,
            },
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        return f"Qwen_VL 分析异常: {exc}"


def build_summary_content_list(topic, report_date=None, content_text=""):
    # 整理生成总结所需的文本素材列表。
    topic_notes = get_topic_notes(topic, report_date)
    content_list = []

    if topic_notes:
        for note in topic_notes[:12]:
            content_list.append(
                {
                    "title": note.get("title", ""),
                    "content_summary": note.get("desc", ""),
                }
            )
    elif content_text:
        content_list = [
            {"title": "", "content_summary": line}
            for line in content_text.split("\n")[:20]
            if line.strip()
        ]

    return topic_notes, content_list


def generate_topic_summary(
    topic, report_date=None, content_text="", enable_thinking=False, max_images=6
):
    # 汇总素材并生成单个话题的总结结果。
    topic_notes, content_list = build_summary_content_list(topic, report_date, content_text)
    if not content_list:
        return None

    image_urls = extract_image_urls_from_notes(topic_notes, max_images=max_images)

    if USE_VLM_FOR_SUMMARY:
        summary = ai_analyze_content_with_qwen_vl(
            content_list=content_list,
            topic_title=topic,
            image_urls=image_urls,
            enable_thinking=enable_thinking,
        )
        model_name = "qwen3-vl-plus"
    else:
        summary = ai_analyze_content(content_list, topic)
        model_name = LOCAL_MODEL_CONFIG["active"] if USE_LOCAL_MODEL else "qwen3-max"

    return {
        "topic": topic,
        "summary": summary,
        "model": model_name,
        "image_urls": image_urls if USE_VLM_FOR_SUMMARY else [],
        "image_count": len(image_urls) if USE_VLM_FOR_SUMMARY else 0,
        "mode": "vlm" if USE_VLM_FOR_SUMMARY else "llm",
    }


def generate_daily_report_from_ai(ai_summaries):
    # 基于多个话题总结生成整日报告。
    prompt = f"""
基于以下各热点话题的 AI 总结，生成一份【当日热点分析报告】：
数据：{json.dumps(ai_summaries, ensure_ascii=False)}
要求：1. 总结核心方向 2. 分析用户关注点 3. 提炼3-5条运营建议 4. 500字内。
"""
    try:
        if USE_LOCAL_MODEL:
            result = local_llm_generate(prompt, 512)
        else:
            response = client.chat.completions.create(
                model="qwen3-max",
                messages=[
                    {"role": "system", "content": "你是专业分析师"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=800,
            )
            result = response.choices[0].message.content.strip()

        report_data = {
            "report_date": date.today().strftime("%Y-%m-%d"),
            "generate_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall_analysis": result,
            "topic_count": len(ai_summaries),
        }
        save_to_json(
            report_data,
            os.path.join(DAILY_REPORT_DIR, f"report_{get_date_str()}.json"),
        )
        return report_data
    except Exception as exc:
        print(f"日报生成失败: {exc}")
        return None


def compare_reports_with_ai(today_report, history_report):
    # 基于今日日报和历史日报生成趋势变化分析。
    if not today_report or not history_report:
        return None

    prompt = f"""
你是一名小红书热点趋势分析师，请对比下面两份日报内容，分析热点变化与趋势。

今日日报：
{today_report}

历史日报：
{history_report}

输出要求：
1. 概括两天热点关注方向的主要变化。
2. 分析有哪些内容热度上升、下降或发生迁移。
3. 总结用户兴趣和内容趋势的变化特征。
4. 给出 2-3 条适合运营选题的建议。
5. 输出控制在 500 字内，语言清晰、自然。
"""
    try:
        if USE_LOCAL_MODEL:
            return local_llm_generate(prompt, 512)

        response = client.chat.completions.create(
            model="qwen3-max",
            messages=[
                {"role": "system", "content": "你是专业的热点趋势分析师"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"日报对比分析失败: {exc}")
        return None
