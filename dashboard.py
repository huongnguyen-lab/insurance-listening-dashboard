import os
import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.analytics import (
    get_metrics, get_sentiment_trend, get_top_topics,
    get_intent_distribution, get_crisis_alerts, get_groups,
    get_brand_mentions, get_thread_analysis,
    get_pain_clusters, get_pain_cluster_trend, get_comments, get_emerging_topics,
    get_brands, get_comment_timeline, get_sentiment_comparison, get_sentiment_score_distribution,
    get_campaign_impact,
    get_overview_spec, get_trends_spec, get_brand_intelligence_spec, get_deep_insight_spec,
    get_crisis_management_spec, get_intent_topic_showcase, get_community_report_spec, get_community_table_report, update_crisis_status,
)

BASE_DIR = os.getenv("INSURANCE_DATA_DIR", "./data_snapshot")

app = FastAPI(title="Insurance Listening Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/health")
def health():
    return {"status": "ok", "data_dir": BASE_DIR}


class CrisisStatusUpdate(BaseModel):
    comment_id: str
    status: str


@app.get("/api/groups")
def api_groups():
    return get_groups(BASE_DIR)


@app.get("/api/brand_options")
def api_brand_options():
    return get_brands(BASE_DIR)


@app.get("/api/metrics")
def api_metrics(start: str = Query(...), end: str = Query(...), group_id: str = Query("all"), advisor_filter: str = Query("all")):
    return get_metrics(start, end, group_id, BASE_DIR, advisor_filter)


@app.get("/api/sentiment_weekly")
def api_sentiment_weekly(start: str = Query(...), end: str = Query(...), group_id: str = Query("all"), granularity: str = Query("week"), advisor_filter: str = Query("all")):
    return get_sentiment_trend(start, end, group_id, granularity, BASE_DIR, advisor_filter)


@app.get("/api/topics")
def api_topics(start: str = Query(...), end: str = Query(...), group_id: str = Query("all"), advisor_filter: str = Query("all")):
    return get_top_topics(start, end, group_id, BASE_DIR, advisor_filter=advisor_filter)


@app.get("/api/intents")
def api_intents(start: str = Query(...), end: str = Query(...), group_id: str = Query("all"), advisor_filter: str = Query("all")):
    return get_intent_distribution(start, end, group_id, BASE_DIR, advisor_filter)


@app.get("/api/crises")
def api_crises(start: str = Query(...), end: str = Query(...), group_id: str = Query("all"), advisor_filter: str = Query("all")):
    return get_crisis_alerts(start, end, group_id, BASE_DIR, advisor_filter)


@app.get("/api/brands")
def api_brands(start: str = Query(...), end: str = Query(...), group_id: str = Query("all"), advisor_filter: str = Query("all")):
    return get_brand_mentions(start, end, group_id, BASE_DIR, advisor_filter)


@app.get("/api/threads")
def api_threads(start: str = Query(...), end: str = Query(...), group_id: str = Query("all"), advisor_filter: str = Query("all")):
    return get_thread_analysis(start, end, group_id, BASE_DIR, advisor_filter=advisor_filter)


@app.get("/api/pain_clusters")
def api_pain_clusters(start: str = Query(...), end: str = Query(...), group_id: str = Query("all"), advisor_filter: str = Query("all")):
    return get_pain_clusters(start, end, group_id, BASE_DIR, advisor_filter)


@app.get("/api/pain_cluster_trend")
def api_pain_cluster_trend(start: str = Query(...), end: str = Query(...), group_id: str = Query("all"), granularity: str = Query("week"), advisor_filter: str = Query("all")):
    return get_pain_cluster_trend(start, end, group_id, granularity, BASE_DIR, advisor_filter)


@app.get("/api/comments")
def api_comments(start: str = Query(...), end: str = Query(...), group_id: str = Query("all"), advisor_filter: str = Query("all"), limit: int = Query(100)):
    return get_comments(start, end, group_id, BASE_DIR, advisor_filter, limit)


@app.get("/api/emerging_topics")
def api_emerging_topics(start: str = Query(...), end: str = Query(...), group_id: str = Query("all"), advisor_filter: str = Query("all")):
    return get_emerging_topics(start, end, group_id, BASE_DIR, advisor_filter)


@app.get("/api/comment_timeline")
def api_comment_timeline(start: str = Query(...), end: str = Query(...), group_id: str = Query("all"), advisor_filter: str = Query("all"), granularity: str = Query("week")):
    return get_comment_timeline(start, end, group_id, BASE_DIR, advisor_filter, granularity)


@app.get("/api/sentiment_comparison")
def api_sentiment_comparison(start: str = Query(...), end: str = Query(...), group_id: str = Query("all"), advisor_filter: str = Query("all"), granularity: str = Query("week")):
    return get_sentiment_comparison(start, end, group_id, BASE_DIR, advisor_filter, granularity)


@app.get("/api/score_distribution")
def api_score_distribution(start: str = Query(...), end: str = Query(...), group_id: str = Query("all"), advisor_filter: str = Query("all"),
                           brand_id: str = Query("all"), insurance_lines: str = Query("all")):
    return get_sentiment_score_distribution(start, end, group_id, BASE_DIR, advisor_filter, brand_id, insurance_lines)


@app.get("/api/campaign_impact")
def api_campaign_impact(start: str | None = Query(None), end: str | None = Query(None), group_id: str = Query("all"), advisor_filter: str = Query("all"),
                        granularity: str = Query("week"), brand_id: str = Query("all"), insurance_lines: str = Query("all")):
    return get_campaign_impact(BASE_DIR, start, end, group_id, advisor_filter, granularity, brand_id, insurance_lines)


@app.get("/api/spec/overview")
def api_spec_overview(start: str = Query(...), end: str = Query(...), compare_start: str | None = Query(None), compare_end: str | None = Query(None),
                      groups: str = Query("all"), brands: str = Query("all"), sentiments: str = Query("all"),
                      advisor_filter: str = Query("all"), insurance_lines: str = Query("all")):
    return get_overview_spec(start, end, compare_start, compare_end, groups, brands, sentiments, advisor_filter, BASE_DIR, insurance_lines)


@app.get("/api/spec/trends")
def api_spec_trends(start: str = Query(...), end: str = Query(...), compare_start: str | None = Query(None), compare_end: str | None = Query(None),
                    groups: str = Query("all"), brands: str = Query("all"), sentiments: str = Query("all"), advisor_filter: str = Query("all"), granularity: str = Query("week"),
                    insurance_lines: str = Query("all")):
    return get_trends_spec(start, end, compare_start, compare_end, groups, brands, sentiments, advisor_filter, granularity, BASE_DIR, insurance_lines)


@app.get("/api/spec/brand_intelligence")
def api_spec_brand(start: str = Query(...), end: str = Query(...), groups: str = Query("all"), brands: str = Query("all"), sentiments: str = Query("all"), advisor_filter: str = Query("all"),
                   insurance_lines: str = Query("all")):
    return get_brand_intelligence_spec(start, end, groups, brands, sentiments, advisor_filter, BASE_DIR, insurance_lines)


@app.get("/api/spec/deep_insight")
def api_spec_deep(start: str = Query(...), end: str = Query(...), groups: str = Query("all"), brands: str = Query("all"), sentiments: str = Query("all"), advisor_filter: str = Query("all"),
                  insurance_lines: str = Query("all")):
    return get_deep_insight_spec(start, end, groups, brands, sentiments, advisor_filter, BASE_DIR, insurance_lines)


@app.get("/api/spec/crisis")
def api_spec_crisis(start: str = Query(...), end: str = Query(...), groups: str = Query("all"), brands: str = Query("all"), sentiments: str = Query("all"), advisor_filter: str = Query("all"),
                    granularity: str = Query("week"), insurance_lines: str = Query("all")):
    return get_crisis_management_spec(start, end, groups, brands, sentiments, advisor_filter, granularity, BASE_DIR, insurance_lines)


@app.post("/api/crisis/status")
def api_update_crisis_status(payload: CrisisStatusUpdate):
    return update_crisis_status(payload.comment_id, payload.status, BASE_DIR)


@app.get("/api/temp/intent_topic_showcase")
def api_temp_intent_topic_showcase(brand_id: str = Query("all"), insurance_lines: str = Query("all")):
    return get_intent_topic_showcase(BASE_DIR, brand_id=brand_id, insurance_lines=insurance_lines)



@app.get("/api/community/prudential_report")
def api_community_prudential_report(start: str | None = Query(None), end: str | None = Query(None),
                                    groups: str = Query("all"), sentiments: str = Query("all"),
                                    brand_id: str = Query("prudential")):
    return get_community_report_spec(BASE_DIR, brand_id=brand_id, start=start, end=end, groups=groups, sentiments=sentiments)

@app.get("/api/community/prudential_table_report")
def api_community_prudential_table_report(start: str | None = Query(None), end: str | None = Query(None),
                                          groups: str = Query("all"), sentiments: str = Query("all"),
                                          brand_id: str = Query("prudential"), limit: int = Query(500)):
    return get_community_table_report(BASE_DIR, brand_id=brand_id, start=start, end=end, groups=groups, sentiments=sentiments, limit=limit)


@app.get("/community-report-table")
def community_report_table():
    return FileResponse("frontend/community_report_table.html")


@app.get("/temp-intent")
def temp_intent_dashboard():
    return FileResponse("frontend/temp_intent_dashboard.html")


@app.get("/")
def dashboard():
    return FileResponse("frontend/index.html")


if __name__ == "__main__":
    uvicorn.run("dashboard:app", host="0.0.0.0", port=8000, reload=True)
