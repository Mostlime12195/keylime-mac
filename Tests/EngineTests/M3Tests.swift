import Foundation
import XCTest
@testable import Engine

/// m3 decoder port: the text helpers and the candidate generator against the Python reference (no model needed), then
/// the shared end-to-end test vectors (m3dict_cpt/test_vectors.json) on the int4 graphs from AC_MODEL_DIR.
final class M3Tests: XCTestCase {
    static let lexicon = M3Lexicon(lex: Lexicon(tsv: Fixtures.data("lexicon_m3.tsv")))

    func testText() {
        let fx = Fixtures.json("m3_text.json") as! [String: Any]
        var bad = 0
        for c in fx["align"] as! [[String: Any]] {
            let a = c["a"] as! [String], b = c["b"] as! [String]
            let want = (c["idx"] as! [[NSNumber]]).map { $0.map { $0.intValue } }
            let got = M3Text.alignIdx(a, b)
            if got != want { bad += 1; if bad < 5 { print("align \(a) / \(b): \(got) want \(want)") } }
        }
        XCTAssertEqual(bad, 0, "alignments")
        for c in fx["prot"] as! [[String: Any]] {
            XCTAssertEqual(M3Text.protectedSlots(M3Text.scalars(c["t"] as! String)), c["prot"] as! [Bool], c["t"] as! String)
        }
        for c in fx["case"] as! [[String: Any]] {
            XCTAssertEqual(M3Text.applyCase(c["w"] as! String, c["p"] as! String), c["out"] as! String)
        }
        XCTAssertEqual(M3Text.segWords("don't  stop-me now'"), ["don't", "stop", "me", "now"])
        XCTAssertEqual(M3Text.round4(0.123449999), 0.1234)
        XCTAssertEqual(M3Text.round4(-1e9), -1e9)
        XCTAssertEqual(M3Text.npSum(ArraySlice<Float>([1, 2, 3])), 6)
    }

    /// CandGen.cands (kprep.GEN) for 753 typed words: same candidates, kinds and order; features to 1e-6
    func testCands() {
        let ref = Fixtures.json("cands_m3.json") as! [String: [[Any]]]
        let D = Self.lexicon
        var bad = 0, n = 0
        for (typed, want) in ref.sorted(by: { $0.key < $1.key }) {
            n += 1
            let got = D.cands(typed, k: 8)
            let ok = got.count == want.count && zip(got, want).allSatisfy { g, w in
                g.text == w[0] as! String && g.kind == w[3] as! String && abs(g.sp - (w[1] as! NSNumber).doubleValue) < 1e-6
                    && abs(g.logp - (w[2] as! NSNumber).doubleValue) < 1e-9
            }
            if !ok {
                bad += 1
                if bad <= 8 { print("cands '\(typed)':\n  got  \(got.map { "\($0.text)/\($0.kind)/\($0.sp)" })\n  want \(want.map { "\($0[0])/\($0[3])/\($0[1])" })") }
            }
        }
        print("m3 cands: \(n - bad)/\(n) identical")
        XCTAssertEqual(bad, 0)
    }

    func testConstraint() {
        let ctx = M3Ctx(Self.lexicon, typed: "southern forces shot down theee northern warplanes", user: [], injected: ["three", "thee"])
        func run(_ pieces: [String]) -> (Bool, Bool) {
            var st = M3State()
            var split = false
            for p in pieces {
                let (ok, sp, ns) = M3Constraint.feed(ctx, st, p)
                if !ok { return (false, split) }
                split = sp; st = ns
            }
            return (M3Constraint.finishOk(ctx, st), split)
        }
        XCTAssertTrue(run([" the", " three"]).0)
        XCTAssertFalse(run([" thxq"]).0, "not a prefix of any valid word")
        XCTAssertTrue(run([" war", " pl"]).1, "splitting the typed lexicon word warplanes")
        XCTAssertTrue(run([" don", "'t"]).0)
        XCTAssertTrue(run([" dont"]).0, "casual contractions stay legal")
    }

    // MARK: end to end

    struct Paths { let enc: String; let dec: String }

    func paths() throws -> Paths {
        Paths(enc: try Fixtures.modelPath("m3.encoder.int4.onnx"), dec: try Fixtures.modelPath("m3.decoder.int4.onnx"))
    }

    /// The shared test vectors: Mac must produce the Python reference's (and Android's) output on every one. Texts,
    /// kinds, decisions and lists exactly; features to 2e-3 (onnxruntime on another CPU).
    func testVectors() throws {
        guard Bundle.module.url(forResource: "m3_vectors.json", withExtension: nil, subdirectory: "Fixtures") != nil else {
            throw XCTSkip("no m3_vectors.json yet")
        }
        let p = try paths()
        let vf = Fixtures.json("m3_vectors.json") as! [String: Any]
        let P = M3Params.from(json: vf["params"] as! [String: Any])
        let model = try M3Model(encoder: p.enc, decoder: p.dec, meta: M3Meta(), threads: 2)
        defer { model.close() }
        let lm: M3WordLM? = try (P.b != 0 ? M3Trigram(path: Fixtures.modelPath("m3_ngram.bin")) : nil)
        // the shipped delete index when the model dir has it (the candidate test covers the one built on device)
        let ss = ((ProcessInfo.processInfo.environment["AC_MODEL_DIR"] ?? "") as NSString).appendingPathComponent("m3_symspell.bin")
        let lexicon = FileManager.default.fileExists(atPath: ss)
            ? M3Lexicon(lex: Self.lexicon.lex, symspell: try Data(contentsOf: URL(fileURLWithPath: ss))) : Self.lexicon
        print("m3 vectors: delete index \(lexicon.indexSource)")
        let c = M3Corrector(model: model, tok: Fixtures.tokM2, lexicon: lexicon, params: P, lm: lm)
        var fails = 0, n = 0
        var worst = 0.0
        var lines: [String] = []
        for v in vf["vectors"] as! [[String: Any]] {
            n += 1
            let input = v["input"] as! String, style = v["style"] as? String ?? "standard"
            let r = try c.correct(input, style: style)
            var why: [String] = []
            if r.greedy != v["constrained_greedy"] as! String { why.append("greedy '\(r.greedy)' want '\(v["constrained_greedy"]!)'") }
            if r.blocked != (v["blocked_steps"] as! NSNumber).intValue { why.append("blocked \(r.blocked) want \(v["blocked_steps"]!)") }
            if r.fast != (v["fast_path"] as! Bool) { why.append("fast \(r.fast)") }
            if r.text != v["auto_text"] as! String { why.append("auto_text '\(r.text)' want '\(v["auto_text"]!)'") }
            func pairs(_ x: Any?) -> [String] { (x as! [[String: Any]]).map { "\($0["typed"]!)->\($0["to"]!)" } }
            if r.auto.map({ "\($0.typed)->\($0.to)" }) != pairs(v["auto"]) { why.append("auto \(r.auto.map { "\($0.typed)->\($0.to)" }) want \(pairs(v["auto"]))") }
            if r.suggest.map({ "\($0.typed)->\($0.to)" }) != pairs(v["suggest"]) { why.append("suggest \(r.suggest.map { "\($0.typed)->\($0.to)" }) want \(pairs(v["suggest"]))") }
            let wp = v["plans"] as! [[String: Any]]
            if wp.count != r.plans.count { why.append("plans \(r.plans.count) want \(wp.count)") } else {
                for (g, w) in zip(r.plans, wp) {
                    if g.typed != w["typed"] as! String || g.outWord != w["greedy_word"] as! String || g.nonword != w["nonword"] as! Bool
                        || g.nameLike != w["name_like"] as! Bool {
                        why.append("plan \(g.typed)/\(g.outWord) nonword \(g.nonword) name \(g.nameLike) want \(w["typed"]!)/\(w["greedy_word"]!)"); continue
                    }
                    if let wj = w["j"] as? NSNumber, let ws = w["span"] as? [NSNumber], let wc = w["changed"] as? Bool, let wr = w["rare"] as? Bool,
                       let wk = w["rank"] as? NSNumber, let wl = w["wlp"] as? NSNumber {
                        if g.j != wj.intValue || [g.span.0, g.span.1] != ws.map({ $0.intValue }) || g.changed != wc || g.rare != wr
                            || g.rank != wk.intValue || abs(g.wlp - wl.doubleValue) > 2e-3 {
                            why.append("plan \(g.typed): j \(g.j) span \(g.span) changed \(g.changed) rare \(g.rare) rank \(g.rank) wlp \(g.wlp) want \(wj) \(ws) \(wc) \(wr) \(wk) \(wl)")
                        }
                    }
                    let wa = w["alts"] as! [[Any]]
                    if wa.count != g.alts.count { why.append("plan \(g.typed): \(g.alts.map { $0.text }) want \(wa.map { $0[0] })"); continue }
                    for (ga, x) in zip(g.alts, wa) {
                        if ga.text != x[0] as! String || ga.kind != x[1] as! String { why.append("alt \(ga.text)/\(ga.kind) want \(x[0])/\(x[1])"); continue }
                        let feats = [ga.dm3, ga.sp, ga.pr, ga.ng]
                        for k in 0..<4 {
                            let d = abs(feats[k] - (x[2 + k] as! NSNumber).doubleValue)
                            worst = max(worst, d)
                            if d > 2e-3 { why.append("alt \(ga.text) feature \(k) \(feats[k]) want \(x[2 + k])") }
                        }
                    }
                }
            }
            if !why.isEmpty { fails += 1 }
            lines.append("\(why.isEmpty ? "ok  " : "FAIL") '\(input)' -> '\(r.text)' (\(String(format: "%.0f", r.ms)) ms)" + (why.isEmpty ? "" : "\n      " + why.joined(separator: "\n      ")))
        }
        print(lines.joined(separator: "\n"))
        print("m3 vectors: \(n - fails)/\(n) identical, worst feature diff \(String(format: "%.2e", worst)), decoder runs \(model.decRuns), ORT \(String(format: "%.0f", model.ortSeconds * 1000)) ms")
        XCTAssertEqual(fails, 0)
    }
}
