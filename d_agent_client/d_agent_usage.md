Now it supports 5 functions:
• getpostcoreinfo(post_id, [timestamp])
Returns the core info for one post: post id, title, author, subreddit, created time, and main text.
• getcommentcoreinfo(comment_id, [timestamp])
Returns the core info for one comment: comment id, author, subreddit, created time, comment text, and its parent post id.
• getpostcommentslist(comment_id, [timestamp], [up], [down], [max_comments])
Given a comment id:
with no up/down, returns all comments under the same post
with up/down, returns ancestor and descendant comment ids around that comment
• getauthorhistorylist(author, [timestamp|max_posts], [max_comments])
Returns recent post ids and comment ids for one author.
It accepts:
username, e.g. AlexWasTakenWasTaken
u/username
t2_... author fullname

Search modes
• --vector
Pure pgvector semantic search
• --hybrid
Hybrid search: semantic search + PostgreSQL full-text search
• no flag
Uses server auto mode; the current server defaults to hybrid
Filters supported by pgvector / hybrid search
Use these inside the query text:

Month: 2024-02
Doc_type: submission or Doc_type: comment
Subreddits: 
AskReddit
gaming
worldnews
todayilearned
science
technology
movies
sports
Health
space
politics
CryptoCurrency
Economics
music
weather
personalfinance
Entrepreneur
hardware
ChatGPT

Authors: user1,user2
Start_time: 2024-02-01
Cutoff_time: 2024-03-01
Engine: vector or Engine: hybrid
Best practice: put each filter on its own line.

Here is an example usage:

# python client.py --tcp 0.0.0.0:61001 --authkey secret123 --getpostcoreinfo 1am42ts 2025-10-01
# python client.py --tcp 0.0.0.0:61001 --authkey secret123 --getcommentcoreinfo kpyo44t 2025-10-01
# python client.py --tcp 0.0.0.0:61001 --authkey secret123 --getpostcommentslist kpyo44t 2026-10-01
# python client.py --tcp 0.0.0.0:61001 --authkey secret123 --getauthorhistorylist iguessmynameischris 2025-10-01
python client.py \
  --tcp 127.0.0.1:61001 \
  --authkey secret123 \
  --q $'Question: Find posts about bitcoin\nCutoff_time: 2025-01-01\nDoc_type: comment'