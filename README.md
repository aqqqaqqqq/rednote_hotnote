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
4. AI总结抓取的热点话题
   1. 本地模型（minimind / Qwen3.5）
   2. API 接口（Qwen3-max / Qwen3-VL）

5. 查看每日报告与历史报告
6. 生成当日报告
7. 对比历史报告变化趋势

#### 参考接口

[【介绍】⏰ 60s API - 60s API](https://docs.60s-api.viki.moe/)

[MediaCrawler 自媒体爬虫](https://nanmicoder.github.io/MediaCrawler/)

[大模型服务平台百炼控制台](https://bailian.console.aliyun.com/cn-beijing/?msctype=pmsg&mscareaid=cn&mscsiteid=cn&mscmsgid=4430126011500654759&yunge_info=pmsg___4430126011500654759&tab=model#/model-market/detail/qwen3-max)

[minimind 自己训练大模型](https://github.com/jingyaogong/minimind)