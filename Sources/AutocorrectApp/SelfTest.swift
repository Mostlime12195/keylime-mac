import Engine
import Foundation
import KeylimeCore
import Darwin

/// `Keylime --selftest <model dir>`: the bundled app end to end without the UI (CI runs it on the built .app):
/// lexicon from the bundle, onnxruntime from Contents/Frameworks, the model from the dir (manifest.json + files).
enum SelfTest {
    /// `Keylime --rsstest <model dir> <rows per run>`: fresh-process memory of scoring 240 candidates of one slot with
    /// the m2s int4 graph. Prints resident / peak sizes and a checksum of the scores (equal across row limits).
    static func rss(_ args: [String]) -> Int32 {
        guard let i = args.firstIndex(of: "--rsstest"), i + 2 < args.count, let rows = Int(args[i + 2]) else { print("usage: --rsstest <model dir> <rows>"); return 2 }
        func mb(_ b: Int) -> String { String(format: "%.0f", Double(b) / 1_048_576) }
        func cur() -> Int {
            var info = mach_task_basic_info()
            var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
            let kr = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count) } }
            return kr == KERN_SUCCESS ? Int(info.resident_size) : 0
        }
        func peak() -> Int { var u = rusage(); getrusage(RUSAGE_SELF, &u); return u.ru_maxrss }   // bytes on macOS
        do {
            let dir = URL(fileURLWithPath: args[i + 1])
            let manifest = try Manifest.parse(Data(contentsOf: dir.appendingPathComponent("manifest.json")))
            guard let spec = manifest.models.first(where: { $0.graph.file.contains("int4m") }) else { print("no int4m model"); return 1 }
            let step = try StepModel.open(path: dir.appendingPathComponent(spec.graph.file).path, meta: spec.meta, threads: 4)
            let tok = try SPTokenizer.fromJSON(Data(contentsOf: dir.appendingPathComponent(spec.vocab.file)))
            let sc = NeuralScorer(tok: tok, step: step, styleId: spec.styleToken("standard"))
            sc.maxRows = rows
            let left = "hey are you coming to the party tonight or are you staying home because i think everyone else is going and"
            let words = ["not", "now", "nit", "know", "note", "noted", "nothing", "never", "new", "next", "nice", "night", "nine", "no",
                         "yes", "yeah", "yep", "maybe", "might", "mind", "mine", "miss", "missed", "more", "most", "much", "must", "my",
                         "the", "then", "there", "their", "they", "them", "this", "that", "those", "though", "through", "thought"]
            var reqs = words.flatMap { w in ["", "s", "ed", "ing", "er"].map { LmReq(left, " " + w + $0) } }
            reqs += words.map { LmReq(left, " " + $0 + ".") }
            let r0 = cur()
            var sum: UInt64 = 1469598103934665603
            for _ in 0..<3 {
                for s in try sc.score(reqs) { sum = (sum ^ s.bitPattern) &* 1099511628211 }
            }
            print("RSSTEST rows/run \(rows): session open \(mb(r0)) MB, after 3 scorings of \(reqs.count) candidates: resident \(mb(cur())) MB, peak \(mb(peak())) MB, checksum \(String(sum, radix: 16))")
            return 0
        } catch { print("FAIL: \(error)"); return 1 }
    }

    static func run(_ args: [String]) -> Int32 {
        guard let i = args.firstIndex(of: "--selftest"), i + 1 < args.count else { print("usage: --selftest <model dir>"); return 2 }
        let dir = URL(fileURLWithPath: args[i + 1])
        do {
            print("onnxruntime \(OrtSessionHandle.ortVersion)")
            guard let lexURL = Bundle.main.url(forResource: "lexicon", withExtension: "tsv") else { print("FAIL: no lexicon.tsv in bundle"); return 1 }
            var t0 = Date()
            let lex = try Lexicon(path: lexURL.path)
            let gen = CandidateGenerator(lex: lex, sp: SpatialModel(sigma: 0.3))
            let corr = Corrector(gen: gen, gen2: CandidateGeneratorV2(gen: gen))
            print(String(format: "lexicon %d words + generators in %.2f s", lex.size, Date().timeIntervalSince(t0)))
            let manifest = try Manifest.parse(Data(contentsOf: dir.appendingPathComponent("manifest.json")))
            var failed = 0
            for spec in manifest.models {
                if spec.m3 != nil { failed += try m3(spec, dir); continue }
                let g = dir.appendingPathComponent(spec.graph.file), v = dir.appendingPathComponent(spec.vocab.file)
                guard FileManager.default.fileExists(atPath: g.path) else { print("skip \(spec.id): no \(spec.graph.file)"); continue }
                t0 = Date()
                let step = try StepModel.open(path: g.path, meta: spec.meta, threads: 4)
                let sc = NeuralScorer(tok: try SPTokenizer.fromJSON(Data(contentsOf: v)), step: step, styleId: spec.styleToken("standard"))
                print(String(format: "%@: session in %.2f s", spec.id, Date().timeIntervalSince(t0)))
                failed += try styles(spec, corr, sc)
                // type each sentence through the same state machine the app uses
                let cases = [("i thnik it is going to rain tomorow ", "I think it is going to rain tomorrow "),
                             ("can you send me teh file ", "can you send me the file "),
                             ("thanks for the help ", "thanks for the help ")]
                for (typed, want) in cases {
                    let core = TypingCore()
                    var shown = ""
                    var ms: [Double] = []
                    for c in typed {
                        switch core.key(.text(String(c)), canCorrect: true) {
                        case .pass: shown.append(c)
                        case .hold(let job):
                            let r = try corr.correctWord(job.input, scorer: sc, params: spec.paramsFor("standard"), prev: job.prev, merge: job.merge)
                            ms.append(r.ms)
                            if let inj = core.resolve(job, r) { shown = String(shown.dropLast(inj.deletes)) + inj.text }
                            shown += job.sep
                        case .undo: break
                        }
                    }
                    let ok = shown == want
                    if !ok { failed += 1 }
                    print("  \(ok ? "ok  " : "FAIL") '\(typed)' -> '\(shown)' (want '\(want)'); ms/word \(ms.map { String(format: "%.0f", $0) }.joined(separator: " "))")
                }
                // stage (b): what a finished word does in each mode
                for (left, typed, want) in [("can you send me", "teh", "auto=apply suggest=flag"), ("thanks for the", "help", "auto=keep suggest=keep")] {
                    let r = try corr.correctWord(WordInput(typed: typed, left: left, sep: " "), scorer: sc, params: spec.paramsFor("standard"), prev: nil)
                    let o = WordOutcome(r, typedOOV: lex.index[typed] == nil)
                    func name(_ m: CorrectionMode) -> String {
                        switch Policy.boundary(o, mode: m, hasPrev: false, prevOut: nil, userWord: false) {
                        case .keep: return "keep"
                        case .apply: return "apply"
                        case .flag: return "flag"
                        }
                    }
                    let got = "auto=\(name(.auto)) suggest=\(name(.suggest))"
                    let ok = got == want
                    if !ok { failed += 1 }
                    print("  \(ok ? "ok  " : "FAIL") word end '\(left) \(typed)': \(got) (want \(want)); margin \(String(format: "%.1f", o.margin))")
                }
                // non-word typos are fixed in every context; slang, names and odd casing are left alone
                let consistent: [(String, String, String?)] = [
                    ("can you", "hlep", "help"), ("I need some", "hlep", "help"), ("thanks for the", "hlep", "help"), ("", "hlep", "help"),
                    ("I went home", "isnstead", "instead"), ("use this one", "isnstead", "instead"), ("we could walk", "isnstead", "instead"),
                    ("", "Isnstead", "Instead"),
                    ("haha", "lmao", nil), ("that's so funny", "lol", nil), ("it's cold", "brrr", nil), ("I'm", "gonna", nil),
                    ("I love my", "iPhone", nil), ("I talked to", "Mazinga", nil), ("that was", "sooo", nil), ("lmao", "lllmo", nil),
                ]
                for (left, typed, want) in consistent {
                    let r = try corr.correctWord(WordInput(typed: typed, left: left, sep: left.isEmpty ? "" : " "), scorer: sc,
                                                 params: spec.paramsFor("standard"), prev: nil)
                    let o = WordOutcome(r, typedOOV: lex.index[typed.lowercased()] == nil)
                    let d = Policy.boundary(o, mode: .auto, hasPrev: false, prevOut: nil, userWord: false, nameLike: Policy.nameLike(typed, left: left))
                    var got = "kept"
                    if case .apply(let out, _, _) = d { got = out } else if case .flag = d { got = "underlined" }
                    let ok = want.map { got == $0 } ?? (got == "kept" || got == "underlined")
                    if !ok { failed += 1 }
                    print("  \(ok ? "ok  " : "FAIL") auto '\(left) \(typed)' -> \(got) (want \(want ?? "left alone")); margin \(String(format: "%.1f", o.margin))")
                }
                // stage (c): the idle sentence pass, with right context
                t0 = Date()
                let fixes = try SentenceCheck.check(corr, SentenceReq(left: "", sentence: "can you send me teh file when you get a chance."),
                                                    scorer: sc, params: spec.paramsFor("standard"), minGap: Policy.FLAG_MARGIN)
                let ok = fixes.contains { $0.from == "teh" && $0.to == "the" } && fixes.count <= 2
                if !ok { failed += 1 }
                print("  \(ok ? "ok  " : "FAIL") idle pass: \(fixes.map { "\($0.from)->\($0.to) (\(String(format: "%.1f", $0.gap)))" }) in \(String(format: "%.0f", Date().timeIntervalSince(t0) * 1000)) ms")
                step.close()
            }
            failed += checkPieces()
            return failed == 0 ? 0 : 1
        } catch {
            print("FAIL: \(error)")
            return 1
        }
    }

    /// The style picker on a real model: each style's token and params, and that switching the token changes the scores
    /// (the scorer is left on the standard token). The formal word pass is printed, not checked (model-dependent).
    static func styles(_ spec: ModelSpec, _ corr: Corrector, _ sc: NeuralScorer) throws -> Int {
        var failed = 0
        func check(_ ok: Bool, _ what: String) { if !ok { failed += 1 }; print("  \(ok ? "ok  " : "FAIL") \(what)") }
        defer { sc.styleId = spec.styleToken("standard"); sc.clearCache() }
        let req = [LmReq("i think it is going to", " rain"), LmReq("i think it is going to", " rian")]
        var lp: [String: [Double]] = [:]
        for st in WritingStyle.all {
            let p = spec.paramsFor(st)
            print("  style \(st): token \(spec.styleToken(st).map(String.init) ?? "none"), lam \(p.lam) bias \(p.bias) lam_r \(p.lamRe), pen \(p.pen.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }.joined(separator: ", "))")
            sc.styleId = spec.styleToken(st); sc.clearCache()
            lp[st] = try sc.score(req)
        }
        if spec.supportsStyle {
            check(lp.values.allSatisfy { $0.allSatisfy(\.isFinite) }, "every style token scores")
            check(lp["formal"] != lp["standard"] && lp["casual"] != lp["standard"], "the style token changes the scores")
            check(spec.paramsFor("formal").pen["caps"] != nil && spec.paramsFor("standard").pen["caps"] == nil, "formal uses pen_formal")
            sc.styleId = spec.styleToken("formal"); sc.clearCache()
            let p = spec.paramsFor("formal")
            // sentence-start capital, i -> I, a comma after the previous word ("come" before "let")
            for (left, typed, prevTyped) in [("", "hello", nil), ("well", "i", nil), ("if you want to come", "let", "come")] as [(String, String, String?)] {
                var pw: PrevWord? = nil
                if let pt = prevTyped {
                    let s = corr.candidates(pt, nil)
                    pw = PrevWord(slot: s, idx: max(s.typedIdx, 0), left: String(left.dropLast(pt.count + 1)), sep: " ", lmScored: true)
                }
                let r = try corr.correctWord(WordInput(typed: typed, left: left, sep: left.isEmpty ? "" : " "), scorer: sc, params: p, prev: pw, formal: true)
                print("  info formal '\(left)' + '\(typed)' -> '\(r.out)'" + (r.prevText.map { ", previous word -> '\($0)'" } ?? ""))
            }
        }
        return failed
    }

    /// m3 (models_v2 m3dict) end to end from the bundle's onnxruntime: whole messages, a word end through Policy, the
    /// pause pass, and ms per word.
    static func m3(_ spec: ModelSpec, _ dir: URL) throws -> Int {
        guard let s = spec.m3 else { return 0 }
        var failed = 0
        func check(_ ok: Bool, _ what: String) { if !ok { failed += 1 }; print("  \(ok ? "ok  " : "FAIL") \(what)") }
        func u(_ f: ModelFile) -> String { dir.appendingPathComponent(f.file).path }
        for f in s.files where !FileManager.default.fileExists(atPath: u(f)) { print("skip \(spec.id): no \(f.file)"); return 0 }
        var t0 = Date()
        let lex = try M3Lexicon(path: u(s.lexicon))
        print(String(format: "%@: lexicon %d words + delete index in %.2f s", spec.id, lex.lex.size, Date().timeIntervalSince(t0)))
        t0 = Date()
        let model = try M3Model(encoder: u(s.encoder), decoder: u(s.decoder), meta: s.meta, threads: 4)
        let c = M3Corrector(model: model, tok: try SPTokenizer.fromJSON(Data(contentsOf: URL(fileURLWithPath: u(s.vocab)))), lexicon: lex, params: s.params)
        print(String(format: "%@: sessions in %.2f s; params th_auto %.4f th_sugg %.4f b %.2f", spec.id, Date().timeIntervalSince(t0),
                     s.params.thAuto, s.params.thSugg, s.params.b))
        defer { model.close() }
        let texts = ["can you send me teh file when you get a chance", "thanks for the help", "I may take the oppurtunity to run some errands.",
                     "we shoudl meet at 3pm on Fridya", "dont worry ill be there in a minuite", "Thsi is teh best plce in toen"]
        var perWord: [Double] = []
        for t in texts {
            let r = try c.correct(t)
            let words = max(1, M3Text.segWords(t).count)
            perWord.append(r.ms / Double(words))
            print("  info '\(t)' -> '\(r.text)' auto \(r.auto.map { "\($0.typed)->\($0.to)" }) suggest \(r.suggest.map { "\($0.typed)->\($0.to)" }) \(String(format: "%.0f ms", r.ms))")
            if t == "thanks for the help" { check(r.text == t && r.suggest.isEmpty, "a clean sentence is left alone") }
        }
        let r0 = try c.correct(texts[0])
        check(r0.auto.contains { $0.typed == "teh" && $0.to == "the" } || r0.suggest.contains { $0.typed == "teh" && $0.to == "the" },
              "teh -> the is AUTO or SUGGEST")
        let w = try c.correctWord(WordInput(typed: "teh", left: "can you send me", sep: " "), prevTyped: "me", style: "standard", user: [])
        let o = WordOutcome(w, typedOOV: true)
        func name(_ m: CorrectionMode) -> String {
            switch Policy.boundary(o, mode: m, hasPrev: true, prevOut: "me", userWord: false) {
            case .keep: return "keep"
            case .apply(let out, _, _): return "apply \(out)"
            case .flag(let fx, _): return "flag \(fx)"
            }
        }
        print("  info word end 'teh': tier \(w.m3Tier?.rawValue ?? "nil"), auto mode \(name(.auto)), suggest mode \(name(.suggest))")
        check(w.m3Tier != nil && w.m3Tier != .keep && name(.suggest).hasPrefix("flag"), "word end: teh is flagged in suggest mode")
        let fx = try c.checkSentence(SentenceReq(left: "", sentence: " can you send me teh file."), style: "standard", user: nil)
        check(fx.contains { $0.from == "teh" && $0.to == "the" && $0.start == 17 }, "pause pass finds teh at its offset: \(fx.map { "\($0.from)->\($0.to)@\($0.start) \($0.tier?.rawValue ?? "")" })")
        // ms per word over more text (whole messages, the warm session)
        var ms: [Double] = []
        for _ in 0..<3 { for t in texts { let r = try c.correct(t); ms.append(r.ms / Double(max(1, M3Text.segWords(t).count))) } }
        ms.sort()
        print(String(format: "  info m3 ms/word over %d messages: p50 %.1f p95 %.1f (decoder runs %d, ORT %.0f ms total)", ms.count,
                     ms[ms.count / 2], ms[min(ms.count - 1, ms.count * 95 / 100)], model.decRuns, model.ortSeconds * 1000))
        _ = perWord
        return failed
    }

    /// marks, update manifest, this bundle's signature (the updater's identity check)
    static func checkPieces() -> Int {
        var failed = 0
        func check(_ ok: Bool, _ what: String) { if !ok { failed += 1 }; print("  \(ok ? "ok  " : "FAIL") \(what)") }
        let t = MarkTracker()
        t.add(start: 4, word: "teh", fixes: ["the"], spelling: true)
        t.reconcile("oh, see teh cat")
        check(t.marks.first?.start == 8, "marks follow the text")
        let m = #"{"version":"9.9.9","url":"https://example.com/Keylime-mac-9.9.9.zip","sha256":"\#(String(repeating: "a", count: 64))","size":1}"#
        check((try? UpdateCore.parse(Data(m.utf8)))?.version == "9.9.9" && UpdateCore.newer("9.9.9", than: Updater.currentVersion),
              "update manifest parses and is newer than \(Updater.currentVersion)")
        let req = CodeIdentity.ownRequirement() ?? "unsigned"
        print("  designated requirement: \(req)")
        let signed = !req.contains("cdhash") && req.contains("certificate leaf")
        if ProcessInfo.processInfo.environment["KL_REQUIRE_SIGNED"] == "1" {
            check(signed, "signed with the Keylime certificate")
            check(CodeIdentity.mismatch(Bundle.main.bundleURL) == nil, "this bundle passes the updater's identity check")
        }
        print("  system autocorrect is \(SystemAutocorrect.isOn ? "on" : "off")")
        return failed
    }
}
