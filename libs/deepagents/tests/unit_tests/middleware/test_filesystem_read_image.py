"""Image payload validation in `read_file` (#3864).

Syntactically invalid base64 and over-limit image payloads used to be
returned as successful multimodal tool results, which were then persisted in
the checkpoint and replayed to the model on every later turn. These tests pin
the invariant that an invalid image payload never reaches the model as a
successful `image` content block, under both the default `error` contract and
the opt-in `notice` contract.
"""

import base64
from typing import Any

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage

from deepagents.backends import StateBackend
from deepagents.backends.protocol import ReadResult
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemState

_PNG_PATH = "/photo.png"
_INVALID_BASE64 = "!!!not-base64!!!"
_OVER_LIMIT_PAYLOAD = base64.b64encode(b"\x00" * (5 * 1024 * 1024 + 1)).decode("ascii")


def _build_runtime(state: FilesystemState, tool_call_id: str) -> ToolRuntime[None, FilesystemState]:
    return ToolRuntime(
        state=state,
        context=None,
        tool_call_id=tool_call_id,
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


def _binary_backend(payload: str) -> StateBackend:
    """Backend stub that returns `payload` base64-encoded for any binary read."""

    class BinaryBackend(StateBackend):
        def read(self, file_path: str, offset: int = 0, limit: int = 100) -> ReadResult:
            return ReadResult(file_data={"content": payload, "encoding": "base64"})

    return BinaryBackend()


def _read_image(path: str, payload: str, **middleware_options: Any) -> ToolMessage:
    """Invoke `read_file` against a backend serving `payload` as base64.

    `middleware_options` are forwarded to `FilesystemMiddleware` (e.g.
    `max_image_input_bytes`, `invalid_binary_read`); omitted options use the
    middleware defaults.
    """
    middleware = FilesystemMiddleware(backend=_binary_backend(payload), **middleware_options)
    state = FilesystemState(messages=[], files={})
    runtime = _build_runtime(state, "image-read-1")
    read_file_tool = next(tool for tool in middleware.tools if tool.name == "read_file")
    result = read_file_tool.invoke({"file_path": path, "runtime": runtime})
    assert isinstance(result, ToolMessage)
    return result


def _block_types(result: ToolMessage) -> list[str]:
    """Content block types carried by a tool result (`[]` for text results)."""
    blocks = result.content if isinstance(result.content, list) else []
    return [str(block.get("type")) for block in blocks if isinstance(block, dict)]


def test_read_file_invalid_base64_image_rejected_by_default() -> None:
    """Syntactically invalid base64 fails the read instead of attaching a block."""
    result = _read_image(_PNG_PATH, _INVALID_BASE64)

    assert result.status == "error"
    assert isinstance(result.content, str)
    assert "not valid base64" in result.content
    assert "image" not in _block_types(result)


def test_read_file_oversized_image_rejected_by_default() -> None:
    """A payload above the default 5 MiB cap fails the read by default."""
    result = _read_image(_PNG_PATH, _OVER_LIMIT_PAYLOAD)

    assert result.status == "error"
    assert isinstance(result.content, str)
    assert "maximum input size" in result.content
    assert "image" not in _block_types(result)


def test_read_file_empty_image_payload_never_attached() -> None:
    """An empty payload never surfaces as an image block.

    Whole-file empty content is intercepted by the existing empty-content
    warning before image validation runs; this pins that the warning path
    keeps a degenerate empty image block out of the conversation.
    """
    result = _read_image(_PNG_PATH, "")

    assert isinstance(result.content, str)
    assert "image" not in _block_types(result)


def test_read_file_valid_small_image_still_succeeds() -> None:
    """A healthy payload still returns the standard image content block."""
    payload = base64.b64encode(b"\x89PNG\r\n\x1a\n image bytes").decode("ascii")

    result = _read_image(_PNG_PATH, payload)

    assert result.status == "success"
    assert _block_types(result) == ["image"]
    assert isinstance(result.content, list)
    block = result.content[0]
    assert block["base64"] == payload
    assert block["mime_type"] == "image/png"
    assert result.additional_kwargs["read_file_path"] == _PNG_PATH
    assert result.additional_kwargs["read_file_media_type"] == "image/png"


def test_read_file_non_image_binary_still_returns_file_block() -> None:
    """Generic binary reads (e.g. `.docx`) are outside image validation."""
    payload = base64.b64encode(b"PK\x03\x04 docx bytes").decode("ascii")

    result = _read_image("/report.docx", payload)

    assert result.status == "success"
    assert _block_types(result) == ["file"]


def test_read_file_max_image_input_bytes_none_disables_cap() -> None:
    """`max_image_input_bytes=None` opts out of the size cap."""
    result = _read_image(_PNG_PATH, _OVER_LIMIT_PAYLOAD, max_image_input_bytes=None)

    assert result.status == "success"
    assert _block_types(result) == ["image"]


def test_read_file_custom_cap_rejects_oversized_image() -> None:
    """A custom cap below the payload size rejects the read."""
    payload = base64.b64encode(b"x" * 11).decode("ascii")

    result = _read_image(_PNG_PATH, payload, max_image_input_bytes=10)

    assert result.status == "error"
    assert isinstance(result.content, str)
    assert "image" not in _block_types(result)


@pytest.mark.parametrize("payload", [_INVALID_BASE64, _OVER_LIMIT_PAYLOAD], ids=["invalid-base64", "oversized"])
def test_read_file_notice_mode_returns_text_without_image_block(payload: str) -> None:
    """`invalid_binary_read="notice"` completes the turn with a text notice."""
    result = _read_image(_PNG_PATH, payload, invalid_binary_read="notice")

    assert result.status == "success"
    assert isinstance(result.content, str)
    assert _PNG_PATH in result.content
    assert "image" not in _block_types(result)


@pytest.mark.parametrize("invalid_binary_read", ["error", "notice"])
@pytest.mark.parametrize("payload", [_INVALID_BASE64, _OVER_LIMIT_PAYLOAD], ids=["invalid-base64", "oversized"])
def test_invalid_image_payload_never_persists_as_successful_image_block(payload: str, invalid_binary_read: str) -> None:
    """The #3864 invariant holds under both the error and notice contracts.

    An invalid payload must not be persisted as a successful multimodal tool
    result: the read either fails outright or comes back as plain text.
    """
    result = _read_image(_PNG_PATH, payload, invalid_binary_read=invalid_binary_read)

    assert result.status != "success" or "image" not in _block_types(result)


@pytest.mark.parametrize("value", [0, -1])
def test_init_rejects_nonpositive_max_image_input_bytes(value: int) -> None:
    """A non-positive cap is a configuration error, matching `grep_max_count`."""
    with pytest.raises(ValueError, match="max_image_input_bytes must be positive or None"):
        FilesystemMiddleware(backend=StateBackend(), max_image_input_bytes=value)


def test_init_rejects_unknown_invalid_binary_read_mode() -> None:
    """Only the `error` and `notice` contracts are accepted."""
    with pytest.raises(ValueError, match="invalid_binary_read must be"):
        FilesystemMiddleware(backend=StateBackend(), invalid_binary_read="warn")  # type: ignore[arg-type]
