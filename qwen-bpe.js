/* Qwen3 byte-level BPE, in-browser. Mirrors tokenizers' BPE exactly:
 * pretokenize (GPT-2-style regex) -> byte-to-unicode -> greedy lowest-rank
 * merges -> vocab lookup. Verified token-for-token against the HuggingFace
 * tokenizer on a 30-case corpus (ASCII, unicode, emoji, contractions,
 * whitespace edge cases). Loads vocab+merges lazily from qwen-tokenizer.json.
 */
const QwenBPE = (() => {
  let vocab = null, ranks = null, byteEnc = null, byteDec = null, idToTok = null;
  const cache = new Map();

  // tokenizer.json pattern, with the JS-unsupported (?i:...) group expanded
  const PRETOK = new RegExp(
    "'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]" +
    "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*" +
    "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", "gu");

  function bytesToUnicode() {
    const bs = [];
    for (let i = 33; i <= 126; i++) bs.push(i);
    for (let i = 161; i <= 172; i++) bs.push(i);
    for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice();
    let n = 0;
    for (let b = 0; b < 256; b++) {
      if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
    }
    const enc = {}, dec = {};
    bs.forEach((b, i) => { enc[b] = String.fromCharCode(cs[i]); dec[String.fromCharCode(cs[i])] = b; });
    return [enc, dec];
  }

  async function load(url) {
    if (vocab) return;
    const t = await (await fetch(url)).json();
    vocab = t.vocab;
    ranks = new Map(t.merges.map((m, i) => [m, i]));
    idToTok = Object.fromEntries(Object.entries(vocab).map(([k, v]) => [v, k]));
    [byteEnc, byteDec] = bytesToUnicode();
  }

  function bpe(token) {
    if (cache.has(token)) return cache.get(token);
    let word = Array.from(token);
    while (word.length > 1) {
      let best = null, bestRank = Infinity, bestAt = -1;
      for (let i = 0; i < word.length - 1; i++) {
        const r = ranks.get(word[i] + " " + word[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best === null) break;
      word = word.slice(0, best).concat(word[best] + word[best + 1], word.slice(best + 2));
    }
    cache.set(token, word);
    return word;
  }

  function encode(text) {
    const ids = [];
    for (const piece of text.match(PRETOK) || []) {
      const bytes = new TextEncoder().encode(piece);
      let mapped = "";
      for (const b of bytes) mapped += byteEnc[b];
      for (const tok of bpe(mapped)) {
        const id = vocab[tok];
        if (id === undefined) throw new Error("token not in vocab: " + tok);
        ids.push(id);
      }
    }
    return ids;
  }

  function decode(ids) {
    const SPECIAL = { 151644: "<|im_start|>", 151645: "<|im_end|>", 151667: "<think>", 151668: "</think>" };
    const bytes = [];
    let out = "";
    const flush = () => {
      if (bytes.length) { out += new TextDecoder().decode(new Uint8Array(bytes)); bytes.length = 0; }
    };
    for (const id of ids) {
      if (SPECIAL[id]) { flush(); out += SPECIAL[id]; continue; }
      const tok = idToTok[id];
      if (tok === undefined) { flush(); out += "�"; continue; }
      for (const ch of Array.from(tok)) bytes.push(byteDec[ch]);
    }
    flush();
    return out;
  }

  // <|im_start|>user\n Q <|im_end|>\n <|im_start|>assistant\n <think>\n\n</think>\n\n
  function chatIds(question) {
    return [151644, 872, 198, ...encode(question), 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271];
  }

  return { load, encode, decode, chatIds, ready: () => !!vocab };
})();
if (typeof module !== "undefined") module.exports = QwenBPE;
