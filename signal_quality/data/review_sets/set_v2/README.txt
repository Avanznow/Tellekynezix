Review set rs-783b8cb434530570: 300 windows from 30 participants.
Send ONLY review_set.html to reviewers. Keep manifest.json: it contains the rule decisions.
Reviewers open the page, enter their name, label every window and click 'Export review record'.
Then: python -m signal_quality.review_set import <record.json> --set <this folder> --out reviews/<name>.jsonl
