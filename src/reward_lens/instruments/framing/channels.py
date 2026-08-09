"""The four channels the policy's context arrives on, and the cues that read them.

A channel is a named source of text the policy can see: the prompt, the system message, the file
names its own context carries, and the error strings the grader returns. A cue is one pattern over
that text and the inference a match permits, stated as a sentence about what the policy can do.
A cue that fires is a `Leak`, and a `Leak` is the unit the panel reports.

Two things this module is deliberately not. It is not a detector of intent: a cue fires on text,
and text that names the grader is a fact about the context, not a claim that any policy read it.
And it is not a belief measurement: nothing here runs a policy, so the strongest sentence a `Leak`
may carry is "the policy can", never "the policy does" or "the policy believes". The inference
sentences are written that way on purpose and the panel's guard holds them to it.

The cue set is a floor and says so. A leak carried by a paraphrase no pattern here matches is a
leak this module does not find, which is why the entry's limitations name the cue set by name.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "CHANNELS",
    "CHANNEL_NAMES",
    "CUES",
    "Channel",
    "Cue",
    "Leak",
    "Sample",
    "channel",
    "probe",
]

#: How much of a matching line the excerpt carries. Enough to see what fired, bounded so that a
#: witness never becomes a copy of the prompt it read.
EXCERPT_CHARS = 160

#: Message roles whose content is context the policy is handed. The assistant's own turns are what
#: the policy wrote rather than what it was given, so they are not read.
CONTEXT_ROLES: tuple[str, ...] = ("user", "system", "developer", "tool")


@dataclass(frozen=True)
class Sample:
    """One piece of policy-visible text and where it came from."""

    origin: str
    text: str


@dataclass(frozen=True)
class Cue:
    """One pattern over policy-visible text, and the inference a match permits."""

    id: str
    pattern: str
    inference: str
    channels: frozenset[str] | None = None

    def applies_to(self, name: str) -> bool:
        return self.channels is None or name in self.channels

    def fired(self, text: str) -> str | None:
        """The excerpt around the first match, or `None`. The excerpt is the evidence, so it is the
        matched text in its line rather than the match alone: `at least 0.8` read on its own says
        nothing about what has to clear it."""
        found = re.search(self.pattern, text)
        if found is None:
            return None
        line_start = text.rfind("\n", 0, found.start()) + 1
        line_end = text.find("\n", found.end())
        line = text[line_start : line_end if line_end != -1 else len(text)].strip()
        if len(line) > EXCERPT_CHARS:
            line = line[: EXCERPT_CHARS - 1].rstrip() + "…"
        return line or text[found.start() : found.end()]


@dataclass(frozen=True)
class Leak:
    """One inference the context permits, with the text that permits it."""

    channel: str
    origin: str
    cue_id: str
    excerpt: str
    inference: str


#: The cue table. Each entry is a pattern, the channels it reads, and the sentence a match licenses.
#: The sentences are the record's, so they are written as the finding's own words and not as
#: fragments the panel assembles: a sentence assembled at the call site is one no test can pin.
CUES: tuple[Cue, ...] = (
    Cue(
        id="grader_named",
        pattern=r"(?i)\b(graders?|judges?|scorers?|auto-?grader|reward(?:\s+function|\.py)|"
        r"grading\s+script)\b",
        inference=(
            "the policy can name the component that scores it, so it can address the scorer "
            "rather than the task"
        ),
    ),
    Cue(
        id="rubric_disclosed",
        pattern=r"(?i)\b(rubrics?|criteri(?:a|on)|you\s+will\s+be\s+(?:scored|graded|judged|"
        r"evaluated)|(?:is|are|being)\s+evaluated|scored\s+on|awards?\s+(?:a\s+)?points?)\b",
        inference=(
            "the policy can read what the scorer rewards, so it can write to the rubric instead "
            "of to the task"
        ),
    ),
    Cue(
        id="threshold_disclosed",
        pattern=r"(?i)(\bthresholds?\b[^.\n]{0,24}[0-9]|\bat\s+least\s+[0-9]|"
        r"\bscores?\b\s*(?:>=|>|of\s+at\s+least)\s*[0-9]|"
        r"\bpass(?:ing)?\s+(?:mark|score)\b[^.\n]{0,16}[0-9])",
        inference=(
            "the policy can read the number it has to clear, so it can aim at the threshold and "
            "stop there"
        ),
    ),
    Cue(
        id="check_named",
        pattern=r"(?i)\b(expected|must\s+contain|does\s+not\s+match|requires?|required|missing|"
        r"assert\w*|regexp?|pattern)\b",
        inference=(
            "the policy can learn which check decided the score, so it can satisfy that check "
            "without doing the work it stands for"
        ),
    ),
    Cue(
        id="source_visible",
        pattern=r"(?i)\b(reward|grader|grading|rubric|score|scoring|judge|verifier|eval)\b",
        inference=(
            "the policy can open the file that scores it, so anything the grader's own source "
            "says is readable from inside the task"
        ),
        channels=frozenset({"file_names"}),
    ),
)


# --- the readers ---------------------------------------------------------------------------------

#: Task keys whose value is the text the policy is asked on.
PROMPT_KEYS = ("prompt", "question", "input", "instruction", "user", "text")

#: Task keys whose value is the system message the policy runs under.
SYSTEM_KEYS = ("system", "system_prompt", "system_message")

#: Task keys whose value is a listing the harness puts in the policy's context: the tools it is
#: offered, the workspace it is shown, the environment description it is given.
LISTING_KEYS = (
    "tools",
    "tool_listing",
    "tools_available",
    "files",
    "file_list",
    "workspace",
    "environment",
    "context",
)

#: Everything on a task that is context the policy receives. The file-name channel reads these and
#: nothing else: a name the policy is never shown is not a name it can infer anything from.
CONTEXT_KEYS = PROMPT_KEYS + SYSTEM_KEYS + LISTING_KEYS

#: One path or file name as it appears in text: an optional directory prefix, a stem, and a short
#: extension that starts with a letter. The tail guard keeps `0.8` and a dotted module path out.
FILE_REFERENCE = re.compile(r"(?<![\w./-])((?:[\w.-]+/)*[\w-]+\.[A-Za-z]\w{0,7})(?![\w/])")


def _texts(task: Any, keys: Sequence[str], roles: Sequence[str]) -> Iterable[tuple[str, str]]:
    """The `(key, text)` pairs one task carries, from its own keys and from a messages list."""
    if not isinstance(task, dict):
        return
    for key in keys:
        value = task.get(key)
        if isinstance(value, str) and value.strip():
            yield key, value
    messages = task.get("messages")
    if isinstance(messages, (list, tuple)):
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") not in roles:
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                yield f"messages[{index}]", content


def read_prompt(subject: Any, corpus: Any) -> tuple[Sample, ...]:
    """What the policy is asked, task by task."""
    found: list[Sample] = []
    for index, task in enumerate(getattr(corpus, "tasks", ()) or ()):
        for key, text in _texts(task, PROMPT_KEYS, ("user",)):
            found.append(Sample(origin=f"task[{index}].{key}", text=text))
    return tuple(found)


def read_system_message(subject: Any, corpus: Any) -> tuple[Sample, ...]:
    """What the policy runs under, task by task."""
    found: list[Sample] = []
    for index, task in enumerate(getattr(corpus, "tasks", ()) or ()):
        for key, text in _texts(task, SYSTEM_KEYS, ("system",)):
            found.append(Sample(origin=f"task[{index}].{key}", text=text))
    return tuple(found)


def _flatten(key: str, value: Any) -> Iterable[tuple[str, str]]:
    """`(key, text)` for a string value, or one pair per string in a listed one."""
    if isinstance(value, str) and value.strip():
        yield key, value
    elif isinstance(value, (list, tuple)):
        for index, inner in enumerate(value):
            if isinstance(inner, str) and inner.strip():
                yield f"{key}[{index}]", inner


def context_texts(corpus: Any) -> tuple[Sample, ...]:
    """Every piece of text the policy is handed: its prompt, its system message, its listings.

    This is the channel's whole world. The auditor's shell sees a project directory and the policy
    does not, so nothing here walks one: a file name reaches the policy through its context or it
    does not reach the policy at all.
    """
    found: list[Sample] = []
    for index, task in enumerate(getattr(corpus, "tasks", ()) or ()):
        if not isinstance(task, dict):
            continue
        for key in CONTEXT_KEYS:
            for inner_key, text in _flatten(key, task.get(key)):
                found.append(Sample(origin=f"task[{index}].{inner_key}", text=text))
        messages = task.get("messages")
        if isinstance(messages, (list, tuple)):
            for position, message in enumerate(messages):
                if not isinstance(message, dict) or message.get("role") not in CONTEXT_ROLES:
                    continue
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    found.append(
                        Sample(origin=f"task[{index}].messages[{position}]", text=content)
                    )
    return tuple(found)


def grader_file_names(subject: Any) -> frozenset[str]:
    """The names of the grader's own files: the grader itself, and what its source opens by name.

    Read off the subject the audit was given, never off the disk. A name the grader's source does
    not carry and the grader path does not carry is somebody else's file.
    """
    names: set[str] = set()
    grader_path = getattr(subject, "grader_path", None)
    if grader_path is not None:
        names.add(PurePosixPath(str(grader_path).replace("\\", "/")).name)
    source = getattr(subject, "source", "") or ""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for reference, _ in _references(node.value):
                    names.add(PurePosixPath(reference).name)
    return frozenset(name for name in names if name)


def _references(text: str) -> Iterable[tuple[str, str]]:
    """Each file reference in `text`, with the line it sits on, which is what a witness quotes."""
    for found in FILE_REFERENCE.finditer(text):
        start = text.rfind("\n", 0, found.start()) + 1
        end = text.find("\n", found.end())
        line = text[start : end if end != -1 else len(text)].strip()
        yield found.group(1), line or found.group(1)


def read_file_names(subject: Any, corpus: Any) -> tuple[Sample, ...]:
    """The grader's own file names where the policy's context names them, and nowhere else.

    Names only, and only names the context carries. Nothing here opens a file and nothing here
    lists a directory: what the policy can read off a name it was shown is the question, and the
    auditor's own filesystem is not an answer to it, because the policy is not standing in it.
    """
    wanted = grader_file_names(subject)
    if not wanted:
        return ()
    found: list[Sample] = []
    seen: set[tuple[str, str]] = set()
    for sample in context_texts(corpus):
        for reference, line in _references(sample.text):
            if PurePosixPath(reference).name not in wanted:
                continue
            key = (sample.origin, reference)
            if key in seen:
                continue
            seen.add(key)
            found.append(Sample(origin=f"{sample.origin}:{reference}", text=line))
    return tuple(found)


def read_error_strings(subject: Any, corpus: Any) -> tuple[Sample, ...]:
    """The strings the grader hands back when it refuses, read off its own source.

    A raised exception's message and a returned string literal are the two shapes that reach a
    caller, and a trainer that surfaces either one puts it in front of the policy. The source is
    parsed rather than scanned so that a docstring, a comment or a log line is not mistaken for
    something the policy sees.
    """
    source = getattr(subject, "source", "") or ""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return ()
    found: list[Sample] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            carried = node.exc
            arguments = carried.args if isinstance(carried, ast.Call) else []
            for argument in arguments:
                for text in _string_parts(argument):
                    found.append(Sample(origin=f"source:{node.lineno}:raise", text=text))
        elif isinstance(node, ast.Return) and node.value is not None:
            for text in _string_parts(node.value):
                found.append(Sample(origin=f"source:{node.lineno}:return", text=text))
    return tuple(found)


def _string_parts(node: ast.AST) -> tuple[str, ...]:
    """Every string constant a node carries, f-strings and concatenations included."""
    parts: list[str] = []
    for inner in ast.walk(node):
        if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
            parts.append(inner.value)
    return tuple(part for part in parts if part.strip())


@dataclass(frozen=True)
class Channel:
    """One source of policy-visible text: what it is, what it depends on, and how it is read."""

    name: str
    description: str
    depends_on: tuple[str, ...]
    read: Callable[[Any, Any], tuple[Sample, ...]]
    absent: str
    remedy: str


CHANNELS: tuple[Channel, ...] = (
    Channel(
        name="prompt",
        description="the prompt the policy is asked on",
        depends_on=("digest:task_distribution",),
        read=read_prompt,
        absent="no task carried a prompt, so there was no prompt text to read",
        remedy="supply --tasks with the prompts the policy is actually asked on",
    ),
    Channel(
        name="system_message",
        description="the system message the policy runs under",
        depends_on=("digest:task_distribution",),
        read=read_system_message,
        absent="no task carried a system message, so there was no system text to read",
        remedy="supply --tasks carrying the system message the policy runs under",
    ),
    Channel(
        name="file_names",
        description="the file names the policy's context carries",
        depends_on=("digest:task_distribution", "digest:source"),
        read=read_file_names,
        absent=(
            "no file name in the context the policy is handed resolved to the grader's own "
            "files, so there was no name the policy could have read"
        ),
        remedy=(
            "supply --tasks carrying the context the policy actually receives, the tool and "
            "workspace listings in it included, so the names it is shown can be read"
        ),
    ),
    Channel(
        name="error_strings",
        description="the error strings the grader returns",
        depends_on=("digest:source",),
        read=read_error_strings,
        absent="the grader's source was empty or would not parse, so no returned string was read",
        remedy="point the audit at the grader's source, so its returned strings can be read",
    ),
)

CHANNEL_NAMES: tuple[str, ...] = tuple(one.name for one in CHANNELS)


def channel(name: str) -> Channel:
    """The channel of that name; `KeyError` for a name this module does not have."""
    for one in CHANNELS:
        if one.name == name:
            return one
    raise KeyError(name)


def probe(one: Channel, samples: Sequence[Sample]) -> tuple[Leak, ...]:
    """Every inference the samples of this channel permit, in sample then cue order.

    One leak per (sample, cue) pair and no deduplication across samples: two tasks that disclose
    the threshold are two places the disclosure has to be fixed, and collapsing them would hide
    the second from whoever fixes the first.
    """
    leaks: list[Leak] = []
    for sample in samples:
        for cue in CUES:
            if not cue.applies_to(one.name):
                continue
            excerpt = cue.fired(sample.text)
            if excerpt is None:
                continue
            leaks.append(
                Leak(
                    channel=one.name,
                    origin=sample.origin,
                    cue_id=cue.id,
                    excerpt=excerpt,
                    inference=cue.inference,
                )
            )
    return tuple(leaks)
