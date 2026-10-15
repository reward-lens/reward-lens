"""Historical records remain renderable rather than silently becoming contemporary claims."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "campaign-v3-cards"
EXPECTED_SHA256 = {
    "armorm.html": "1260e43296450179f6451cc11494b77876a1b5c323af7c7c78fe05652c784e37",
    "grm-gemma2-2b.html": "646fd6fc1311c1945ce4f39fee61f6220ebeed1667a2358427d5f1d5de112112",
    "skywork-v01.html": "d4f7f5e09afc4793a11432f59ec6ed66aebdeca05e5615619448bae4c109726a",
    "skywork-v02.html": "f6bfca776b0bdc49fb7a3934acbc14815d27570b77c1ef6758029b725e2f480a",
    "skywork-v2-llama31-8b.html": "9ab5684ac5184133c890a73db1992cbe57a0ff26d781706961389b95ba4614e9",
    "skywork-v2-qwen3-0.6b.html": "8017cf772df22053bfd784b2c574573cedd47e752431d8b999207246a553583d",
    "skywork-v2-qwen3-1.7b.html": "f95a9dfe49bff02148021c91ccb801702545427a8aaa87e73b8ffc4022a7462e",
    "skywork-v2-qwen3-4b.html": "8b16ee1e282d92f3520747fe0523281e321b95bfdc5bff5a42690a9ef7be77ff",
    "skywork-v2-qwen3-8b.html": "baf3e0e9a593c11df44446c55ba7527d787cd220d90d3e757fb2ca692916c915",
    "tulu-rm.html": "adeea773966b31c9ee852269ca2d6843b1bf9e2eed946f5993f67f46872f47e4",
}


def test_ten_v3_cards_migrate_to_explicit_unknowns(tmp_path: Path) -> None:
    from reward_lens.product.ingest import migrate_v3_card

    for index in range(10):
        source = tmp_path / f"v3-{index}.json"
        source.write_text(json.dumps({"version": 3, "id": f"card-{index}"}), encoding="utf-8")
        migration = migrate_v3_card(source)
        assert migration.preview["id"] == f"card-{index}"
        assert migration.preview["intent"] == "unknown"


def test_every_emitted_campaign_v3_card_renders_and_migrates_without_inventing_semantics(
    tmp_path: Path,
) -> None:
    from reward_lens.product.ingest import migrate_v3_card

    cards = sorted(FIXTURES.glob("*.html"))
    assert [card.name for card in cards] == sorted(EXPECTED_SHA256)
    for card in cards:
        raw = card.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256[card.name]
        copied = tmp_path / card.name
        shutil.copyfile(card, copied)
        migration = migrate_v3_card(copied)
        assert migration.preview["rendered_html"] == raw.decode("utf-8")
        assert migration.preview["intent"] == "unknown"
        assert migration.preview["level_of_detail"] == "unknown"
        assert migration.preview["composite"] == "unknown"
        assert migration.validation_report["source_format"] == "campaign-rm-card/html"
