from yomibench.student import clean_target, format_prompt, gate


def test_format_prompt():
    assert format_prompt("辛い", "ツライ") == "文: 辛い\n下書き: ツライ\n読み: "


def test_clean_target_keeps_katakana_and_long_vowel_only():
    assert clean_target("ショーリオオサメタ。") == "ショーリオオサメタ"
    assert clean_target("ソーチ、デアル") == "ソーチデアル"
    assert clean_target("ひらがな") == "ヒラガナ"


def test_gate_accepts_katakana():
    assert gate("ツライ", "カライ", False) == ("カライ", False)


def test_gate_rejects_non_kana():
    assert gate("ツライ", "辛い", False) == ("ツライ", True)


def test_gate_rejects_truncated():
    assert gate("ツライ", "カライ", True) == ("ツライ", True)


def test_gate_rejects_runaway():
    assert gate("アイウエオカキクケコ", "アイウエオカキクケコ" * 3, False) == ("アイウエオカキクケコ", True)


def test_gate_rejects_too_short():
    assert gate("アイウエオカキクケコ", "アイウ", False) == ("アイウエオカキクケコ", True)


def test_gate_rejects_empty():
    assert gate("ツライ", "  ", False) == ("ツライ", True)


def test_generate_pads_left_and_flags_outputs_without_eos():
    torch = __import__("pytest").importorskip("torch")
    from yomibench.student import generate

    class Tok:
        bos_token_id, pad_token_id, eos_token_id = 1, 0, 2

        def __call__(self, text, add_special_tokens):
            return {"input_ids": [10 + i for i in range(len(text))]}

        def decode(self, row, skip_special_tokens):
            return "".join(chr(0x30A0 + t) for t in row)

    class Model:
        device = torch.device("cpu")

        def generate(self, input_ids, attention_mask, **kw):
            self.ids, self.mask = input_ids, attention_mask
            new = torch.tensor([[0x0B, 0x0B, 2], [0x0B, 0x0B, 0x0B]])  # row 1 ends with eos, row 2 does not
            return torch.cat([input_ids, new], dim=1)

    model = Model()
    out = generate(model, Tok(), ["ab", "abcd"], max_new_tokens=3)
    assert model.ids.tolist() == [[0, 0, 1, 10, 11], [1, 10, 11, 12, 13]]
    assert model.mask.tolist() == [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]
    assert out == [("カカ", False), ("カカカ", True)]
