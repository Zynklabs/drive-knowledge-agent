"""Keep document text from steering the prompt, and answers from carrying what it planted.

Indexed files are untrusted: anyone who can edit a file in the folder can write text meant for
the model. Two defences sit on either side of the model:

- ``as_data`` neutralises the tags that frame passages in the prompt, so a document can't
  close its ``<source>`` and pose as another source or as instructions;
- ``safe_answer`` keeps only links to the retrieved Drive files and drops images, so an answer
  can't carry a phishing link, or an image URL that leaks the answer when a browser loads it.
"""

from __future__ import annotations

import re
from collections.abc import Collection

_FRAME_TAG = re.compile(r"<(\s*/?\s*)(sources?|passage|conversation)\b", re.IGNORECASE)
_IMAGE = re.compile(r"!\[([^\]\n]*)\]\([^)\n]*\)")
_REF_IMAGE = re.compile(r"!\[([^\]\n]*)\]\[[^\]\n]*\]")
_HTML_IMAGE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_LINK = re.compile(r"\[([^\]\n]+)\]\((\S+?)(?:\s+\"[^\"\n]*\")?\)")
_REF_DEFINITION = re.compile(
    r"^ {0,3}\[[^\]\n]+\]:[ \t]*<?(?:[a-z][a-z0-9+.-]*:|//)\S*.*$\n?", re.I | re.M
)


def as_data(text: str) -> str:
    """Make prompt-framing tags inside document text inert: ``<source`` becomes ``\u2039source``
    (a look-alike angle quote), so the text reads the same but closes no tag."""
    return _FRAME_TAG.sub("\u2039\\1\\2", text)


def safe_answer(markdown: str, allowed_urls: Collection[str] = ()) -> str:
    """Drop images and turn links to anything but ``allowed_urls`` into their plain text.

    Bare URLs are left as they are: they show where they go, so they can't disguise a link.
    """
    text = _IMAGE.sub(r"\1", markdown)
    text = _REF_IMAGE.sub(r"\1", text)
    text = _HTML_IMAGE.sub("", text)
    text = _REF_DEFINITION.sub("", text)
    return _LINK.sub(lambda m: m.group(0) if m.group(2) in allowed_urls else m.group(1), text)
