from d_agent_client import Text2SQLIPCClient

_D_AGENT_HOST    = "127.0.0.1"
_D_AGENT_PORT    = 61001
_D_AGENT_AUTHKEY = "secret123"

_MAX_THREAD_CHARS = 8000


def _cap(text: str, limit: int) -> str:
    if len(text) > limit:
        return text[:limit] + f"\n... [truncated at {limit} chars, {len(text)} total]"
    return text


def _get_client() -> Text2SQLIPCClient:
    return Text2SQLIPCClient.tcp(_D_AGENT_HOST, _D_AGENT_PORT, authkey=_D_AGENT_AUTHKEY)


def search_database(query: str, cutoff_time: str, doc_type: str = "", subreddit: str = "", start_time: str = "") -> str:
    client = None
    try:
        lines = [f"Question: {query}", f"Cutoff_time: {cutoff_time}"]
        if doc_type:
            lines.append(f"Doc_type: {doc_type}")
        if subreddit:
            lines.append(f"Subreddits: {subreddit}")
        if start_time:
            lines.append(f"Start_time: {start_time}")
        client = _get_client()
        return client.query("\n".join(lines), engine="hybrid")
    except Exception as e:
        return f"TOOL ERROR: The search failed. Reason: {str(e)}."
    finally:
        if client:
            client.close()


def get_post_core_info(post_id: str, cutoff_time: str) -> str:
    client = None
    try:
        client = _get_client()
        return client.query(f'getpostcoreinfo("{post_id}", "{cutoff_time}")', engine="sql")
    except Exception as e:
        return f"TOOL ERROR: Could not retrieve post {post_id}. Reason: {str(e)}."
    finally:
        if client:
            client.close()


def get_comment_core_info(comment_id: str, cutoff_time: str) -> str:
    client = None
    try:
        client = _get_client()
        return client.query(f'getcommentcoreinfo("{comment_id}", "{cutoff_time}")', engine="sql")
    except Exception as e:
        return f"TOOL ERROR: Could not retrieve comment {comment_id}. Reason: {str(e)}."
    finally:
        if client:
            client.close()


def get_post_comments_list(comment_id: str, cutoff_time: str, up: int = 0, down: int = 0, max_comments: int = 100) -> str:
    client = None
    try:
        client = _get_client()
        result = client.query(
            f'getpostcommentslist("{comment_id}", "{cutoff_time}", {up}, {down}, {max_comments})',
            engine="sql"
        )
        return _cap(result, _MAX_THREAD_CHARS)
    except Exception as e:
        return f"TOOL ERROR: Could not retrieve thread for comment {comment_id}. Reason: {str(e)}."
    finally:
        if client:
            client.close()


def get_author_history_list(author_id: str, cutoff_time: str, max_posts: int = 20, max_comments: int = 20) -> str:
    client = None
    try:
        client = _get_client()
        result = client.query(
            f'getauthorhistorylist("{author_id}", "{cutoff_time}", {max_posts}, {max_comments})',
            engine="sql"
        )
        return _cap(result, _MAX_THREAD_CHARS)
    except Exception as e:
        return f"TOOL ERROR: Could not retrieve history for author {author_id}. Reason: {str(e)}."
    finally:
        if client:
            client.close()


# ── OpenAI function-calling schemas ──────────────────────────────────────────

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_database",
        "description": (
            "Search the Reddit database for posts and comments using hybrid semantic + full-text search. "
            "PRIMARY evidence-gathering tool. Returns ranked results with IDs, snippets, and metadata. "
            "Available subreddits: CryptoCurrency, Economics, personalfinance, Entrepreneur, worldnews, "
            "politics, science, technology, space, Health, ChatGPT, hardware, music, movies, sports, "
            "gaming, weather, todayilearned, AskReddit"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query":       {"type": "string", "description": "Natural language question or keywords"},
                "cutoff_time": {"type": "string", "description": "Upper date bound YYYY-MM-DD"},
                "doc_type":    {"type": "string", "description": "'submission' or 'comment', or '' for both", "default": ""},
                "subreddit":   {"type": "string", "description": "Single subreddit name, or '' for all", "default": ""},
                "start_time":  {"type": "string", "description": "Lower date bound YYYY-MM-DD, or ''", "default": ""},
            },
            "required": ["query", "cutoff_time"],
        },
    }},
    {"type": "function", "function": {
        "name": "get_post_core_info",
        "description": "Get full content and metadata for a specific Reddit post ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "post_id":     {"type": "string", "description": "Post ID, e.g. '1am42ts'"},
                "cutoff_time": {"type": "string", "description": "Upper date bound YYYY-MM-DD"},
            },
            "required": ["post_id", "cutoff_time"],
        },
    }},
    {"type": "function", "function": {
        "name": "get_comment_core_info",
        "description": "Get full content and metadata for a specific Reddit comment ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "comment_id":  {"type": "string", "description": "Comment ID, e.g. 'kpyo44t'"},
                "cutoff_time": {"type": "string", "description": "Upper date bound YYYY-MM-DD"},
            },
            "required": ["comment_id", "cutoff_time"],
        },
    }},
    {"type": "function", "function": {
        "name": "get_post_comments_list",
        "description": (
            "Get comments from the post containing a given comment ID. "
            "With up=0 and down=0: returns ALL comments under the same post. "
            "With up/down set: returns ancestor and descendant comment IDs around that comment."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "comment_id":   {"type": "string", "description": "A comment ID"},
                "cutoff_time":  {"type": "string", "description": "Upper date bound YYYY-MM-DD"},
                "up":           {"type": "integer", "description": "Ancestor context levels (default 0)", "default": 0},
                "down":         {"type": "integer", "description": "Descendant reply levels (default 0)", "default": 0},
                "max_comments": {"type": "integer", "description": "Max comments to return (default 100)", "default": 100},
            },
            "required": ["comment_id", "cutoff_time"],
        },
    }},
    {"type": "function", "function": {
        "name": "get_author_history_list",
        "description": "Get recent posts and comments by a specific Reddit author, ordered chronologically.",
        "parameters": {
            "type": "object",
            "properties": {
                "author_id":    {"type": "string", "description": "Username, 'u/username', or 't2_...' fullname"},
                "cutoff_time":  {"type": "string", "description": "Upper date bound YYYY-MM-DD"},
                "max_posts":    {"type": "integer", "description": "Max posts to return (default 20)", "default": 20},
                "max_comments": {"type": "integer", "description": "Max comments to return (default 20)", "default": 20},
            },
            "required": ["author_id", "cutoff_time"],
        },
    }},
]

TOOL_MAP = {
    "search_database":        search_database,
    "get_post_core_info":     get_post_core_info,
    "get_comment_core_info":  get_comment_core_info,
    "get_post_comments_list": get_post_comments_list,
    "get_author_history_list": get_author_history_list,
}
