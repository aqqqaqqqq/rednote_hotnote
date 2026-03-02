# rednote_hotnote

#### 环境配置

```bash
conda create -n xhs python=3.11
conda activate xhs
pip install -r requirements.txt
playwright install
pip install schedule/bs4/openai
```

#### 函数执行

```python
python -m crawl_xhs
```

#### 功能包含

1. 抓取小红书热点榜单
2. 定时抓取
3. 抓取热点下的话题图文
4. AI分析抓取的话题
5. 查看每日报告与历史报告
6. 生成当日报告
7. 对比每日报告

#### 参考接口

[【介绍】⏰ 60s API - 60s API](https://docs.60s-api.viki.moe/)

[MediaCrawler使用方法 | MediaCrawler自媒体爬虫](https://nanmicoder.github.io/MediaCrawler/)

[大模型服务平台百炼控制台](https://bailian.console.aliyun.com/cn-beijing/?msctype=pmsg&mscareaid=cn&mscsiteid=cn&mscmsgid=4430126011500654759&yunge_info=pmsg___4430126011500654759&tab=model#/model-market/detail/qwen3-max)