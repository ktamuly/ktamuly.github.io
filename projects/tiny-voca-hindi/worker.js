// Runs the Hindi front end (Python, through Pyodide) and the model (ONNX Runtime Web) off the main thread.
// Messages in: {type: "load"} and {type: "speak", id, text, speed, pitch}. Messages out: status, ready,
// sentence (with its audio, as soon as it is made), done (with the whole clip), error (plain: true when the
// message is meant for the reader as it is). Adapted from tiny-voca's web demo (Apache-2.0).

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

importScripts(PYODIDE + "pyodide.js", ORT + "ort.wasm.min.js");

let phonemize, session, config;

const status = (text, progress) => postMessage({ type: "status", text, progress });

// Bump when the files under assets/ change, so browsers fetch them again.
const VERSION = "hi-2";

// Downloaded files are kept in the browser's cache storage: a second visit starts without the network.
async function download(url, label) {
  const key = `${url}?v=${VERSION}`;
  let store = null;
  try {
    store = await caches.open("tiny-voca-" + VERSION);
    const hit = await store.match(key);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch (error) { store = null; }
  const bytes = await fetchBytes(key, label);
  if (store) store.put(key, new Response(bytes)).catch(() => {});
  return bytes;
}

async function fetchBytes(url, label) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !total) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    status(label, received / total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes;
}

async function loadFrontEnd() {
  status("Loading the Hindi front end");
  const pyodide = await loadPyodide({ indexURL: PYODIDE });
  const source = await download("assets/frontend.zip", "Loading the Hindi front end");
  pyodide.unpackArchive(source.buffer, "zip", { extractDir: "/frontend" });
  pyodide.runPython(`
import json, os, sys
os.environ["TINY_VOCA_FRONTEND"] = "hindi"
os.environ["TINY_VOCA_HINGLISH"] = "/frontend/hinglish_lexicon.json"
sys.path.insert(0, "/frontend")
from tiny_voca.text import phonemize

def phonemize_json(text):
    try:
        result = phonemize(text)
    except KeyError as error:
        return json.dumps({"error": str(error.args[0]) if error.args else "unknown word"})
    except ValueError:
        return "null"
    symbols = [s for s in result.symbols if s not in ("<bos>", "<eos>")]
    if not any(s != "<sp>" and any(ch.isalpha() for ch in s) for s in symbols):
        return "null"      # nothing to pronounce (punctuation only): the sentence is skipped
    return json.dumps({"ids": list(result.ids), "stress": list(result.stress),
                       "text": " ".join(symbols).replace("<sp>", "·")})

phonemize_json("नमस्ते।")
`);
  phonemize = pyodide.globals.get("phonemize_json");
}

async function loadModel() {
  const answer = await fetch(`assets/config.json?v=${VERSION}`);
  if (!answer.ok) throw new Error(`The model's files could not be loaded (${answer.status}). Reload the page.`);
  config = await answer.json();
  const bytes = await download("assets/model.onnx", "Loading the model");
  status("Starting the model");
  ort.env.wasm.wasmPaths = ORT;
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
}

// Sentences end at . ! ? … or the Devanagari danda ।, followed by a space and a capital, a digit or Devanagari.
// tiny_voca.infer.sentences splits the same way; keep the two alike.
const ABBREVIATIONS = /(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|no|e\.g|i\.e|a\.m|p\.m)|\b[A-Z])\.$/i;
function sentences(text) {
  const parts = [];
  let current = "";
  for (const piece of text.replace(/\s+/g, " ").trim().split(/(?<=[.!?…।॥]["'”’)\]]?)\s+(?=["'“‘(\[]?[A-Z0-9ऀ-ॿ])/)) {
    current = current ? current + " " + piece : piece;
    if (!ABBREVIATIONS.test(current)) { parts.push(current); current = ""; }
  }
  if (current) parts.push(current);
  return parts.filter((part) => /[\p{L}\p{N}]/u.test(part));
}

// A text that ends without a mark gets a danda, so the voice falls at the end instead of breaking off.
function finished(text) {
  const body = text.replace(/\s+$/, "");
  if (!body) return text;
  return /[.!?…,;:।॥]["'”’)\]»]*$/.test(body) ? body : body + "।";
}

// One utterance holds at most max_tokens phonemes; longer sentences are cut at commas, then by words.
function fit(sentence) {
  const result = JSON.parse(phonemize(sentence));
  if (!result) return [];
  if (result.error) throw Object.assign(new Error(result.error), { plain: true });
  if (result.ids.length <= config.max_tokens - 12) return [result];
  const pieces = sentence.split(/(?<=[,;:—])\s+/);
  if (pieces.length > 1) return pieces.flatMap(fit);
  const words = sentence.split(" ");
  if (words.length < 2) return [];
  const half = Math.ceil(words.length / 2);
  return [...fit(words.slice(0, half).join(" ")), ...fit(words.slice(half).join(" "))];
}

async function synthesize(phones, speed, pitch) {
  const count = phones.ids.length;
  const integers = (values) => new ort.Tensor("int64", BigInt64Array.from(values.map(BigInt)), [1, count]);
  const scalar = (value) => new ort.Tensor("float32", Float32Array.from([value]), [1]);
  const output = await session.run({
    tokens: integers(phones.ids), stress: integers(phones.stress),
    voice: new ort.Tensor("int64", BigInt64Array.from([0n]), [1]),
    speed: scalar(speed), pitch_shift: scalar(pitch * 0.058), pitch_range: scalar(1),
  });
  return output.audio.data;
}

// The model sometimes peaks above 1.0, which would clip in the 16-bit file. Scale the whole clip down then.
const PEAK = 0.99;
function limitPeak(samples) {
  let loudest = 0;
  for (let i = 0; i < samples.length; i++) loudest = Math.max(loudest, Math.abs(samples[i]));
  if (loudest > PEAK) for (let i = 0; i < samples.length; i++) samples[i] *= PEAK / loudest;
}

async function speak({ id, text, speed, pitch }) {
  const started = performance.now();
  const list = sentences(finished(text));
  const chunks = [];
  let phonemeTime = 0;
  for (let index = 0; index < list.length; index++) {
    let mark = performance.now();
    const utterances = fit(list[index]);
    phonemeTime += performance.now() - mark;
    for (const phones of utterances) {
      const audio = await synthesize(phones, speed, pitch);
      chunks.push(audio);
      const heard = audio.slice();
      limitPeak(heard);
      postMessage({ type: "sentence", id, index, total: list.length, phonemes: phones.text, audio: heard,
                    sampleRate: config.sample_rate }, [heard.buffer]);
    }
  }
  if (!chunks.length) {
    postMessage({ type: "error", id, message: "Type some Hindi text (Devanagari) to hear it.", plain: true });
    return;
  }
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const audio = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) { audio.set(chunk, offset); offset += chunk.length; }
  limitPeak(audio);
  postMessage({ type: "done", id, audio, sampleRate: config.sample_rate, phonemeTime,
                totalTime: performance.now() - started }, [audio.buffer]);
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      await Promise.all([loadFrontEnd(), loadModel()]);
      postMessage({ type: "ready", sampleRate: config.sample_rate });
    } else if (data.type === "speak") {
      await speak(data);
    }
  } catch (error) {
    const message = String(error && error.message || error);
    const unknown = message.match(/Latin word '([^']+)'/);
    if (unknown) postMessage({ type: "error", id: data.id, plain: true,
      message: `“${unknown[1]}” isn't in the model's English word list. Write it in Devanagari and try again.` });
    else postMessage({ type: "error", id: data.id, message, plain: Boolean(error && error.plain) });
  }
};
