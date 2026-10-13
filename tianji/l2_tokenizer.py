"""L2 · 分词器 —— 字节级 BPE（GPT-2 谱系）。

站在谁之上（零重写）：
    - Byte-Pair Encoding            —— Sennrich et al. 2016, "Neural Machine Translation
                                       of Rare Words with Subword Units"
    - 字节级基表 + 预分词正则        —— GPT-2 / tiktoken（Radford et al. 2019, OpenAI）
    - 合并优先级按 (频次, pair) 字典序 —— tiktoken 的确定性 tie-break，
                                       保证同语料同 vocab_size 训练出逐位相同的词表

为什么用 BPE 而不是 WordPiece / Unigram / SentencePiece：
    BPE 的优势是「合并表即词表」，编解码完全确定、无需概率模型，
    在零依赖 + 可复现这两条硬约束下是唯一现实选择。
    Unigram LM（SentencePiece）需要 Viterbi 解码，词表训练非确定性强，
    会让「同 seed 两次运行逐位一致」这条门禁无法保证。

本实现的确定性保证：
    1. 预分词不依赖 Python 内置 hash（不受 PYTHONHASHSEED 影响）
    2. 合并选择是 argmax over (count, pair) —— pair 为元组，字典序唯一
    3. Counter 遍历顺序不影响 argmax 结果

不变量（门禁复检）：
    T1 往返一致性：decode(encode(s)) == s 对全部测试串成立
    T2 词表闭合：所有 merge 产出的 id 都在 [0, vocab_size)
    T3 压缩性：编码长度 < 字符数（在多数字语料上）
    T4 确定性：同语料训练两次，词表逐位相同
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

__all__ = ["BPE", "pretokenize", "BYTE_BASE"]

BYTE_BASE = 256

# GPT-2 的英文缩略语簇。放在词表最前面，避免 "don't" 被切成 "don" + "'t" 的劣化解。
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'T", "'RE", "'VE", "'M", "'LL", "'D")

_CJK = (
    (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
    (0x3040, 0x30FF),          # 日文假名
    (0xAC00, 0xD7AF),          # 韩文音节
)


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _CJK)


def _cat(ch: str) -> str:
    """字符类别：L=字母 N=数字 S=空白 O=其他。

    Python 的 str.isalpha() 对汉字返回 True，因此 CJK 会被归入 L，
    这与 GPT-2 正则中 \\p{L} 的行为一致 —— 中文连续串会被当作一个预分词块。
    """
    if ch.isspace():
        return "S"
    if ch.isalpha():
        return "L"
    if ch.isdigit():
        return "N"
    return "O"


def pretokenize(text: str, cjk_split: bool = True) -> list[str]:
    """GPT-2 预分词的零依赖等价实现。

    替代方案对比：
        用 regex 模块（第三方）可直接写 GPT-2 原正则，但引入外部依赖；
        标准库 re 不支持 \\p{L}（Unicode 属性类）。
        本函数用手工扫描器实现同一语义，行为可被单测钉住，零依赖。

    规则（按 GPT-2 的 pattern 顺序）：
        1. 缩略语 's/'t/'re/'ve/'m/'ll/'d
        2. " ?\\p{L}+"  可选单个空格 + 连续字母
        3. " ?\\p{N}+"  可选单个空格 + 连续数字
        4. " ?[^\\s\\p{L}\\p{N}]+"  可选单个空格 + 连续其他
        5. "\\s+(?!\\S)"  后面没有非空白的空白串（即尾部空白）
        6. "\\s+"  空白串
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        # 规则 5/6：空白。
        # 关键细节：**最后一个空格要并入下一个 token**（GPT-2 里 " world"
        # 是一个整体）。早期版本把空格单独 append，导致 decode 后所有
        # 单词间空格丢失 —— 不报错、词表也正常，只是重建文本全错。
        if text[i].isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            if j >= n:                       # 尾部空白，整段成 token
                out.append(text[i:j])
                break
            if j - i > 1:                    # 空白串中除最后一个外单独成 token
                for k in range(i, j - 1):
                    out.append(text[k])
                i = j - 1
                continue
            # 恰好一个空格且后面还有内容 —— 不 continue，
            # 落到下面的 token 分支，让这个空格成为下一 token 的前缀。

        # token 分支（允许一个前导空格）
        start = i
        p = start + 1 if text[start].isspace() else start
        if p >= n:
            out.append(text[start:])
            break
        ch = text[p]

        # 规则 1：缩略语
        if ch == "'":
            hit = None
            for c in _CONTRACTIONS:
                if text.startswith(c, p):
                    hit = c
                    break
            if hit is not None:
                out.append(text[start:p + len(hit)])
                i = p + len(hit)
                continue

        # 规则 4：连续「其他」字符（标点等）
        if _cat(ch) == "O":
            q = p
            while q < n and _cat(text[q]) == "O":
                q += 1
            out.append(text[start:q])
            i = q
            continue

        # CJK 单字成块（对 GPT-2 原正则的改进，见类注释）
        if cjk_split and _is_cjk(ch):
            out.append(text[start:p + 1])
            i = p + 1
            continue

        # 规则 2/3：连续字母或连续数字
        c = _cat(ch)
        q = p
        while q < n and _cat(text[q]) == c:
            q += 1
        out.append(text[start:q])
        i = q

    return out


class BPE:
    """字节级 BPE 分词器。

    Parameters
    ----------
    vocab_size : 目标词表大小（含 256 个字节基表与可选特殊 token）
    """

    def __init__(self, vocab_size: int = 512, special: Sequence[str] = ("<|endoftext|>",)):
        self.vocab_size = int(vocab_size)
        self.merges: dict[tuple[int, int], int] = {}
        self.ranks: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(BYTE_BASE)}
        self.special: list[str] = list(special)
        self._special_re: re.Pattern | None = None
        nxt = BYTE_BASE
        for s in self.special:
            self.vocab[nxt] = s.encode("utf-8")
            nxt += 1
        self._next_id = nxt
        self._cache: dict[str, tuple[int, ...]] = {}

    # ------------------------------------------------------------ 训练

    def train(self, text: str | Iterable[str], verbose: bool = False) -> "BPE":
        """从语料训练合并表。

        复杂度 O(V · N)：每轮合并后全量重扫。
        为什么不做增量 pair 更新：增量版本需要维护双向链表与 pair→词索引，
        代码量翻三倍且是 bug 温床；本项目的目标词表量级（≤ 2048）下全量重扫
        耗时在秒级，工程上更划算。这一点与 tiktoken 不同 —— 它必须处理
        万亿字节语料，所以必须做增量。
        """
        if isinstance(text, str):
            text = [text]
        counts: Counter[str] = Counter()
        for chunk in text:
            counts.update(pretokenize(chunk))

        # 每个预分词块表示为字节 id 元组
        words: dict[str, tuple[int, ...]] = {
            w: tuple(w.encode("utf-8")) for w in counts
        }

        while self._next_id < self.vocab_size:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for w, ids in words.items():
                c = counts[w]
                for a, b in zip(ids, ids[1:]):
                    pair_counts[(a, b)] += c
            if not pair_counts:
                break
            # 确定性 tie-break：(频次, pair) 字典序 —— 频次相同时按 pair 元组排序
            best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            new_id = self._next_id
            self.merges[best] = new_id
            self.ranks[best] = len(self.ranks)
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            self._next_id += 1

            # 应用合并到所有词
            new_words: dict[str, tuple[int, ...]] = {}
            for w, ids in words.items():
                new_words[w] = self._merge_ids(ids, best, new_id)
            words = new_words
            if verbose:
                print(f"merge {len(self.merges):>4} -> {self.vocab[new_id]!r}")

        self._build_special_re()
        self._cache.clear()
        return self

    @staticmethod
    def _merge_ids(ids: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
        out: list[int] = []
        i = 0
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return tuple(out)

    def _build_special_re(self) -> None:
        if not self.special:
            self._special_re = None
            return
        pat = "|".join(re.escape(s) for s in self.special)
        self._special_re = re.compile(f"({pat})")

    # ------------------------------------------------------------ 编解码

    def _encode_chunk(self, ids: tuple[int, ...]) -> tuple[int, ...]:
        """对单个预分词块贪心应用合并表（按 rank 从小到大 = 训练顺序）。"""
        if len(ids) < 2:
            return ids
        ids = list(ids)
        while len(ids) > 1:
            best_rank = None
            best_pos = -1
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pos = r, i
            if best_pos < 0:
                break
            ids[best_pos:best_pos + 2] = [self.merges[(ids[best_pos], ids[best_pos + 1])]]
        return tuple(ids)

    def encode(self, text: str) -> list[int]:
        if self._special_re is None:
            chunks = [text]
        else:
            chunks = self._special_re.split(text)
        out: list[int] = []
        special_ids = {s: i for i, s in enumerate(self.special, start=BYTE_BASE)}
        for ch in chunks:
            if ch in special_ids:
                out.append(special_ids[ch])
                continue
            for w in pretokenize(ch):
                ids = self._cache.get(w)
                if ids is None:
                    ids = self._encode_chunk(tuple(w.encode("utf-8")))
                    self._cache[w] = ids
                out.extend(ids)
        return out

    def decode(self, ids: Sequence[int]) -> str:
        buf = bytearray()
        for i in ids:
            buf.extend(self.vocab[int(i)])
        return buf.decode("utf-8", errors="replace")

    # ------------------------------------------------------------ 工具

    @property
    def size(self) -> int:
        return self._next_id

    def save(self, path: str) -> None:
        """词表落盘（纯文本，便于人眼 diff）。"""
        lines = [f"{len(self.vocab)} {len(self.merges)} {len(self.special)}"]
        for s in self.special:
            lines.append("SPECIAL " + s)
        for (a, b), nid in sorted(self.merges.items(), key=lambda kv: kv[1]):
            lines.append(f"{a} {b} {nid}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    @classmethod
    def load(cls, path: str) -> "BPE":
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        nvocab, nmerge, nspec = (int(x) for x in lines[0].split())
        special: list[str] = []
        idx = 1
        for _ in range(nspec):
            special.append(lines[idx][len("SPECIAL "):])
            idx += 1
        self = cls(vocab_size=nvocab, special=special)
        for line in lines[idx:idx + nmerge]:
            a, b, nid = (int(x) for x in line.split())
            self.merges[(a, b)] = nid
            self.ranks[(a, b)] = len(self.ranks)
            self.vocab[nid] = self.vocab[a] + self.vocab[b]
            self._next_id = max(self._next_id, nid + 1)
        self._build_special_re()
        return self
