import Engine
import Foundation
import KeylimeCore
import os

/// The decoder and its model session. Everything engine-side runs on one serial queue (never the event-tap thread);
/// session creation runs on a loader queue and is swapped in on the engine queue.
final class EngineHost {
    let queue = DispatchQueue(label: "ac.engine", qos: .userInteractive)
    private let loadQueue = DispatchQueue(label: "ac.load", qos: .utility)
    private let log = Logger(subsystem: "dev.mostlime.autocorrect-mac", category: "engine")

    // engine queue only
    private var corrector: Corrector?
    private var scorer: NeuralScorer?
    /// m3 (models_v2 m3dict): replaces the scorer + corrector path while loaded
    private var m3: M3Corrector?
    /// loader queue only: the m3 lexicon + delete index, kept across reloads of the same file
    private var m3Lex: (path: String, lex: M3Lexicon)?
    private var params = DecodeParams.fallback
    private var spec: ModelSpec?

    // any thread
    private let stateLock = NSLock()
    private var _ready = false
    private var _status = "Starting…"
    private var _modelLabel = ""
    private var _lastMs = 0.0

    var ready: Bool { stateLock.lock(); defer { stateLock.unlock() }; return _ready }
    var status: String { stateLock.lock(); defer { stateLock.unlock() }; return _status }
    var modelLabel: String { stateLock.lock(); defer { stateLock.unlock() }; return _modelLabel }
    private var _modelId = "none"
    /// manifest id of the loaded model ("none" before one is loaded): upload events carry it
    var modelId: String { stateLock.lock(); defer { stateLock.unlock() }; return _modelId }
    var lastMs: Double { stateLock.lock(); defer { stateLock.unlock() }; return _lastMs }
    /// jobs queued or running on the engine queue (the usage meter's backlog)
    private var _pending = 0
    private var _maxPos = 0
    private var _ctx = MAX_CTX
    /// "graph limit 1023, using 960": what the context setting actually gives
    var contextInfo: String {
        stateLock.lock(); defer { stateLock.unlock() }
        if _maxPos == 0 { return "graph not probed yet" }
        return "graph accepts positions up to \(_maxPos), using \(min(_ctx, _maxPos + 1 - NeuralScorer.TARGET_ROOM))"
    }

    /// context-length setting (tokens): applied on the engine queue, cache dropped
    func setContext(_ tokens: Int) {
        stateLock.lock(); _ctx = tokens; stateLock.unlock()
        queue.async {
            guard let sc = self.scorer, sc.maxCtx != tokens else { return }
            sc.maxCtx = tokens
            sc.clearCache()
        }
    }
    // style picker: the saved choice, and what's in effect for the loaded model (standard when it has no style tokens)
    private var _style = WritingStyle.defaultStyle
    private var _effectiveStyle = WritingStyle.defaultStyle
    private var _supportsStyle = false
    /// the loaded model was trained with style tokens (the Settings picker is shown only then)
    var supportsStyle: Bool { stateLock.lock(); defer { stateLock.unlock() }; return _supportsStyle }
    /// the style decoding uses now
    var effectiveStyle: String { stateLock.lock(); defer { stateLock.unlock() }; return _effectiveStyle }

    /// Style picker: takes effect on the engine queue (control token + params; the scorer's cache is dropped when the
    /// token changes), no reload.
    func setStyle(_ style: String) {
        stateLock.lock(); _style = WritingStyle.stored(style); stateLock.unlock()
        queue.async { self.applyStyleOnQueue() }
    }

    /// engine queue only
    private func applyStyleOnQueue() {
        guard let sp = spec else { return }
        stateLock.lock()
        let st = WritingStyle.effective(_style, supportsStyle: sp.supportsStyle)
        _effectiveStyle = st; _supportsStyle = sp.supportsStyle
        stateLock.unlock()
        if let sc = scorer {
            let id = sp.styleToken(st)
            if sc.styleId != id { sc.styleId = id; sc.clearCache() }
        }
        params = sp.paramsFor(st)
        log.info("style \(st, privacy: .public) (token \(sp.styleToken(st).map(String.init) ?? "none", privacy: .public))")
        DispatchQueue.main.async { self.onChange?() }
    }

    /// the formal style's per-word capitals and commas: auto mode only (suggest-only never changes text)
    private var formalWordPass: Bool { effectiveStyle == "formal" && Settings.shared.mode == .auto }

    var pending: Int { stateLock.lock(); defer { stateLock.unlock() }; return _pending }
    private func begin() { stateLock.lock(); _pending += 1; stateLock.unlock() }
    private func end() { stateLock.lock(); _pending -= 1; stateLock.unlock() }
    var onChange: (() -> Void)?

    let user: UserData

    init(user: UserData) { self.user = user }

    func setStatus(_ s: String, ready r: Bool? = nil) {
        stateLock.lock(); _status = s; if let r = r { _ready = r }; stateLock.unlock()
        DispatchQueue.main.async { self.onChange?() }
    }

    /// Lexicon + candidate generators (engine queue).
    func start(lexicon: URL) {
        queue.async {
            let t0 = Date()
            do {
                let lex = try Lexicon(path: lexicon.path)
                let gen = CandidateGenerator(lex: lex, sp: SpatialModel(sigma: 0.3))
                self.corrector = Corrector(gen: gen, gen2: CandidateGeneratorV2(gen: gen), user: self.user)
                self.log.info("lexicon \(lex.size) words in \(Date().timeIntervalSince(t0))s")
            } catch {
                self.setStatus("Lexicon failed: \(error)", ready: false)
            }
        }
    }

    /// engine queue only
    func inLexiconOnQueue(_ w: String) -> Bool { corrector?.lex.index[w.lowercased()] != nil }

    /// engine queue only: frequency rank in the lexicon (0 = most common)
    func rankOnQueue(_ w: String) -> Int? { corrector?.lex.index[w.lowercased()] }

    /// Open a model's session (loader queue), then swap it in and warm it up.
    func load(spec: ModelSpec, url: @escaping (ModelFile) -> URL, completion: ((Error?) -> Void)? = nil) {
        setStatus("Loading \(spec.label)…", ready: false)
        loadQueue.async {
            do {
                let t0 = Date()
                let threads = min(4, max(1, ProcessInfo.processInfo.activeProcessorCount / 2))
                if let m3s = spec.m3 {
                    let c = try self.openM3(m3s, url: url, threads: threads)
                    _ = try c.correct("hello tehre", style: "standard")
                    let secs = Date().timeIntervalSince(t0)
                    self.queue.async {
                        self.scorer?.step.close(); self.scorer = nil
                        self.m3?.model.close(); self.m3 = c
                        self.spec = spec
                        self.applyStyleOnQueue()
                        self.stateLock.lock(); self._modelLabel = spec.label; self._modelId = spec.id; self._maxPos = 0; self.stateLock.unlock()
                        self.setStatus("Ready · \(spec.label)", ready: true)
                        self.log.info("m3 \(spec.id) ready in \(secs)s, \(threads) threads")
                        completion?(nil)
                    }
                    return
                }
                let tok = try SPTokenizer.fromJSON(Data(contentsOf: url(spec.vocab)))
                let step = try StepModel.open(path: url(spec.graph).path, meta: spec.meta, threads: threads)
                self.stateLock.lock(); let st = WritingStyle.effective(self._style, supportsStyle: spec.supportsStyle); self.stateLock.unlock()
                let sc = NeuralScorer(tok: tok, step: step, styleId: spec.styleToken(st))
                // how far its positions go (rope table = max_seq): the context setting is clamped under it
                let mp = step.maxPosition()
                sc.maxPos = mp
                self.stateLock.lock(); self._maxPos = mp; sc.maxCtx = self._ctx; self.stateLock.unlock()
                self.log.info("graph accepts positions up to \(mp)")
                // warm-up: the first ORT run allocates; do it before the first word
                _ = try sc.score([LmReq("hello there", " friend"), LmReq("hello there", " fiend")])
                sc.clearCache()
                let secs = Date().timeIntervalSince(t0)
                self.queue.async {
                    self.scorer?.step.close()
                    self.m3?.model.close(); self.m3 = nil
                    self.scorer = sc
                    self.spec = spec
                    self.applyStyleOnQueue()
                    self.stateLock.lock(); self._modelLabel = spec.label; self._modelId = spec.id; self.stateLock.unlock()
                    self.setStatus("Ready · \(spec.label)", ready: true)
                    self.log.info("session \(spec.id) ready in \(secs)s, \(threads) threads")
                    completion?(nil)
                }
            } catch {
                self.setStatus("Model failed to load: \(error)", ready: false)
                self.log.error("load failed: \(String(describing: error))")
                completion?(error)
            }
        }
    }

    /// loader queue: tokenizer, lexicon (+ delete index, cached), encoder + decoder sessions
    private func openM3(_ s: M3Spec, url: (ModelFile) -> URL, threads: Int) throws -> M3Corrector {
        let tok = try SPTokenizer.fromJSON(Data(contentsOf: url(s.vocab)))
        let lp = url(s.lexicon).path
        let lex: M3Lexicon
        if let c = m3Lex, c.path == lp { lex = c.lex } else {
            let t0 = Date()
            lex = try M3Lexicon(path: lp, symspell: s.symspell.map { url($0).path })
            m3Lex = (lp, lex)
            log.info("m3 lexicon \(lex.lex.size) words + delete index (\(lex.indexSource)) in \(Date().timeIntervalSince(t0))s")
        }
        let model = try M3Model(encoder: url(s.encoder).path, decoder: url(s.decoder).path, meta: s.meta, threads: threads)
        // the n-gram term (params b): m3_ngram.bin, memory-mapped
        var lm: M3WordLM? = nil
        if let ng = s.ngram, s.params.b != 0 { lm = try M3Trigram(path: url(ng).path) }
        return M3Corrector(model: model, tok: tok, lexicon: lex, params: s.params, lm: lm)
    }

    /// Correct one word end (engine queue). The precheck runs first on the engine queue (focused field checks via AX).
    func correct(_ job: CorrectionJob, precheck: @escaping () -> Bool, done: @escaping (WordResult?, Bool) -> Void) {
        queue.async {
            guard precheck() else { done(nil, false); return }
            // the keystroke fallback holds the separator (200 ms at most): never more than 256 tokens there
            let full = self.scorer?.maxCtx
            if let f = full, f > Self.KEYS_MAX_CTX { self.scorer?.maxCtx = Self.KEYS_MAX_CTX }
            self.correctOnQueue(job.input, job.prev, job.merge, done)
            if let f = full { self.scorer?.maxCtx = f }
        }
    }

    static let KEYS_MAX_CTX = 256

    /// One finished word (AX path). done(result, typed word is in the lexicon), on the engine queue.
    func correctWord(_ input: WordInput, prev: PrevWord?, merge: MergeInput?, done: @escaping (WordResult?, Bool) -> Void) {
        begin()
        queue.async { self.correctOnQueue(input, prev, merge, done); self.end() }
    }

    private func correctOnQueue(_ input: WordInput, _ prev: PrevWord?, _ merge: MergeInput?, _ done: (WordResult?, Bool) -> Void) {
        if let m = m3 {
            do {
                // the previous word as it stands in the field (one plain word, or no re-check)
                var prevTyped: String? = nil
                if prev != nil {
                    let l = input.left
                    let start = l.lastIndex(where: { $0.isWhitespace }).map { l.index(after: $0) } ?? l.startIndex
                    let last = String(l[start...])
                    if M3Text.segWords(last) == [last] { prevTyped = last }
                }
                let r = try m.correctWord(input, prevTyped: prevTyped, style: effectiveStyle, user: user.all)
                stateLock.lock(); _lastMs = r.ms; stateLock.unlock()
                done(r, m.D.inLex(M3Text.lower(input.typed)))
            } catch {
                log.error("m3 correction failed: \(String(describing: error))")
                done(nil, false)
            }
            return
        }
        guard let c = corrector, let sc = scorer else { done(nil, false); return }
        do {
            sc.lastCtxTokens = 0
            let r = try c.correctWord(input, scorer: sc, params: params, prev: prev, merge: merge, formal: formalWordPass)
            stateLock.lock(); _lastMs = r.ms; stateLock.unlock()
            done(r, c.lex.index[input.typed.lowercased()] != nil)
        } catch {
            log.error("correction failed: \(String(describing: error))")
            done(nil, false)
        }
    }

    /// The idle pass: sentence checks, nearest first. done(fixes per request with "is a spelling fix", nil = cancelled /
    /// no model), on the engine queue.
    func checkSentences(_ reqs: [SentenceReq], cancelled: @escaping () -> Bool, done: @escaping ([[(SentenceFix, Bool)]]?) -> Void) {
        begin()
        queue.async {
            defer { self.end() }
            if let m = self.m3 {
                var out: [[(SentenceFix, Bool)]] = []
                for r in reqs {
                    if cancelled() { done(nil); return }
                    guard let fx = try? m.checkSentence(r, style: self.effectiveStyle, user: self.user) else { done(nil); return }
                    out.append(fx.map { ($0, Policy.isSpellingEdit($0.from, $0.to, typedOOV: !m.D.inLex(M3Text.lower($0.from)))) })
                }
                if cancelled() { done(nil); return }
                done(out); return
            }
            guard let c = self.corrector, let sc = self.scorer else { done(nil); return }
            var out: [[(SentenceFix, Bool)]] = []
            for r in reqs {
                if cancelled() { done(nil); return }
                guard let fx = try? SentenceCheck.check(c, r, scorer: sc, params: self.params, minGap: Policy.FLAG_MARGIN, cancelled: cancelled) else {
                    done(nil); return
                }
                out.append(fx.map { ($0, Policy.isSpellingEdit($0.from, $0.to, typedOOV: c.lex.index[$0.from.lowercased()] == nil)) })
            }
            if cancelled() { done(nil); return }
            done(out)
        }
    }

    /// The formal style's pause pass: commas and sentence ends (Rewind). done(changes, nil = cancelled / no model), on the
    /// engine queue.
    func checkPunct(_ targets: [PunctTarget], cancelled: @escaping () -> Bool, done: @escaping ([PunctChange]?) -> Void) {
        begin()
        queue.async {
            defer { self.end() }
            guard let sc = self.scorer else { done(nil); return }
            let p = self.params
            guard let ch = try? Rewind.check(targets, scorer: sc, lam: p.lamRe, user: self.user, pen: p.pen, cancelled: cancelled),
                  !cancelled() else { done(nil); return }
            done(ch)
        }
    }

    /// engine queue: the typed word is a lexicon word
    func inLexicon(_ w: String, done: @escaping (Bool) -> Void) {
        queue.async { done(self.inLexiconOnQueue(w)) }
    }

    /// Layout change: letter centres for the spatial model (engine queue).
    func setLayout(_ centers: [Character: (Double, Double)]) {
        queue.async { KeyGeom.setCenters(centers) }
    }
}
