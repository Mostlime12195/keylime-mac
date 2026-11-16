import Foundation

/// Decision params (m3dict params json): S(alt) = dm3 + a*sp + b*ng + d*prior (+ copy biases); AUTO / SUGGEST thresholds.
public struct M3Params: Equatable {
    public var k = 8
    public var rare = 1_000_000_000.0
    public var lowconf = -3.0
    public var a = 0.5, b = 0.0, d = 0.0
    public var cName = 0.0, cReal = 0.0, cRare = 0.0
    public var thAuto = 0.0, thSugg = 0.0
    public init() {}

    public static func from(json o: [String: Any]) -> M3Params {
        var p = M3Params()
        func d(_ k: String) -> Double? { (o[k] as? NSNumber)?.doubleValue }
        if let v = d("k") { p.k = Int(v) }
        if let v = d("rare") { p.rare = v }
        if let v = d("lowconf") { p.lowconf = v }
        if let v = d("a") { p.a = v }
        if let v = d("b") { p.b = v }
        if let v = d("d") { p.d = v }
        if let v = d("c_name") { p.cName = v }
        if let v = d("c_real") { p.cReal = v }
        if let v = d("c_rare") { p.cRare = v }
        if let v = d("th_auto") { p.thAuto = v }
        if let v = d("th_sugg") { p.thSugg = v }
        return p
    }
}

/// A word-sequence LM for the n-gram term (b * ng): sentence log-prob of lowercase words.
public protocol M3WordLM: AnyObject {
    func sentLp(_ words: [String]) -> Double
}

public struct M3Alt: Equatable {
    public let text: String
    public let kind: String
    public var dm3 = 0.0, sp = 0.0, pr = 0.0, ng = 0.0
}

/// One uncertain slot (decode.slot_plan): typed word j, its output words' span in the greedy text, the alternatives.
public struct M3Plan {
    public let j: Int
    public let typed: String
    public let span: (Int, Int)
    public let outWord: String
    public let nonword: Bool
    public let prot: Bool
    public let changed: Bool
    public let rare: Bool
    public let low: Bool
    public let wlp: Double
    public let rank: Int
    public let nameLike: Bool
    public var alts: [M3Alt]
}

public enum M3Tier: String { case keep, suggest, auto }

public struct M3Change: Equatable {
    public let j: Int
    public let typed: String
    public let to: String
    public let margin: Double
}

public struct M3Result {
    public let text: String
    public let greedy: String
    public let auto: [M3Change]
    public let suggest: [M3Change]
    public let fast: Bool
    public let blocked: Int
    public let plans: [M3Plan]
    /// typed SEG words and the output word indices aligned to each
    public let typedWords: [String]
    public let asg: [[Int]]
    public let ms: Double
}

/// One typed word's verdict for the app: AUTO / SUGGEST / keep, the best fix and the ranked fixes.
public struct M3WordVerdict {
    public let j: Int
    public let typed: String
    public let tier: M3Tier
    public let best: String
    public let margin: Double
    /// considered fixes other than the typed word, best first (at most 3)
    public let fixes: [String]
    public let nonword: Bool
}

/// The m3 dictionary-aware corrector (port of onnx_backend.M3Dict.correct + tune.decide / apply / casual_filter).
public final class M3Corrector {
    public let model: M3Model
    public let tok: SPTokenizer
    public let D: M3Lexicon
    public var params: M3Params
    public var lm: M3WordLM?
    /// piece text per token for the constraint (decode.piece_texts)
    let ptext: [String?]
    static let RARE = 10000

    public init(model: M3Model, tok: SPTokenizer, lexicon: M3Lexicon, params: M3Params, lm: M3WordLM? = nil) {
        self.model = model; self.tok = tok; self.D = lexicon; self.params = params; self.lm = lm
        var pt: [String?] = []
        pt.reserveCapacity(tok.pieces.count)
        for i in 0..<tok.pieces.count {
            let p = tok.pieces[i]
            let u = Array(p.unicodeScalars)
            if p.hasPrefix("<0x") && p.hasSuffix(">") && u.count == 6, let b = Int(M3Text.string(u[3..<5]), radix: 16) {
                pt.append(b < 128 ? String(Character(Unicode.Scalar(UInt8(b)))) : "\u{80}")
            } else if tok.types[i] == SPTokenizer.T_CONTROL || tok.types[i] == SPTokenizer.T_UNKNOWN {
                pt.append(nil)
            } else {
                pt.append(p.replacingOccurrences(of: SPTokenizer.WS, with: " "))
            }
        }
        ptext = pt
    }

    // MARK: features

    func spFeat(_ typed: String, _ alt: String) -> Double {
        let t = M3Text.lower(typed).unicodeScalars.filter { Character($0).isLetter || $0 == "'" }
        let a = M3Text.lower(alt).unicodeScalars.filter { ($0.value >= 97 && $0.value <= 122) || $0 == "'" }
        if a.isEmpty { return -50.0 }
        let sp = D.sp
        guard let code = sp.wordCode(M3Text.string(Array(a))) else { return -50.0 }
        return sp.score(sp.emitTable(KeyGeom.keyTouches(M3Text.string(Array(t)))), code)
    }

    func prFeat(_ alt: String) -> Double {
        var s = 0.0
        for w in M3Text.segWords(alt) { let l = M3Text.lower(w); s += D.lex.index[l].map { D.lex.logp[$0] } ?? D.floor }
        return s
    }

    /// decode.word_lps: greedy log-prob over each output word's span (piece-text offsets, as the reference)
    func wordLps(_ out: Scalars, _ toks: [Int], _ lps: [Double]) -> [Double] {
        var full: Scalars = []
        for t in toks { full += Array((ptext[t] ?? "").unicodeScalars) }
        var lead = 0
        while lead < full.count && full[lead] == " " { lead += 1 }
        var spans: [(Int, Int)] = []
        var pos = 0
        for t in toks {
            let n = (ptext[t] ?? "").unicodeScalars.count
            spans.append((pos - lead, pos + n - lead)); pos += n
        }
        return M3Text.seg(out).map { (a, b) in
            var s = 0.0
            for (k, (x, y)) in spans.enumerated() where x < b && y > a { s += lps[k] }
            return s
        }
    }

    // MARK: plan

    func slotPlan(_ typed: Scalars, _ out: Scalars, k: Int, cache: inout [String: [M3Cand]], rareRank: Int, wlp: [Double]?,
                  lowconf: Double?) -> (plans: [M3Plan], tw: [String], asg: [[Int]]) {
        let tw = M3Text.seg(typed).map { M3Text.string(typed[$0.0..<$0.1]) }
        let ow = M3Text.seg(out).map { (M3Text.string(out[$0.0..<$0.1]), $0.0, $0.1) }
        let asg = M3Text.alignIdx(tw, ow.map { $0.0 })
        let prot = M3Text.protectedSlots(typed)
        var plans: [M3Plan] = []
        for (j, t) in tw.enumerated() {
            let idx = asg[j]
            guard let i0 = idx.first, let i1 = idx.last, idx == Array(i0...i1) else { continue }
            let o = idx.map { ow[$0].0 }.joined(separator: " ")
            let tl = M3Text.lower(t)
            let tu = Array(tl.unicodeScalars)
            let isPoss = tl.hasSuffix("'s")
            let base = isPoss ? M3Text.string(tu.dropLast(2)) : ""
            let pj = j < prot.count ? prot[j] : false
            let nonword = !pj && !(D.lex.index[tl] != nil || (isPoss && D.lex.index[base] != nil))
            let changed = M3Text.lower(o) != tl
            let rank = D.lex.index[tl] ?? (isPoss ? (D.lex.index[base] ?? -1) : -1)
            var rare = !pj && !nonword && rank >= rareRank && tu.count >= 3
            var lpW = 0.0
            if let w = wlp, i1 < w.count { for i in idx { lpW += w[i] } }
            let low = lowconf != nil && !pj && !nonword && !changed && lpW < lowconf! && tu.count >= 2
            rare = rare || low
            if !(nonword || changed || rare) { continue }
            let pat = ow[i0].0
            let t0 = t.unicodeScalars.first!, p0 = pat.unicodeScalars.first!
            let copy = M3Text.isUpper(p0) && M3Text.isLower(t0) ? M3Text.upper(String(Character(t0))) + M3Text.string(Array(t.unicodeScalars).dropFirst()) : t
            var alts = [M3Alt(text: copy, kind: "copy")]
            if changed { alts.append(M3Alt(text: o, kind: "greedy")) }
            if nonword || rare {
                let cl: [M3Cand]
                if let c = cache[tl] { cl = c } else { cl = D.cands(tl, k: 8); cache[tl] = cl }
                var have = Set(alts.map { M3Text.lower($0.text) })
                for c in cl.prefix(k) {
                    let ct = M3Text.applyCase(c.text, pat)
                    if !have.contains(M3Text.lower(ct)) { alts.append(M3Alt(text: ct, kind: c.kind)); have.insert(M3Text.lower(ct)) }
                }
            }
            let nameLike = (M3Text.isUpper(t0) && j > 0 && t != "I") || (tu.count > 1 && M3Text.pyIsUpper(t))
            plans.append(M3Plan(j: j, typed: t, span: (ow[i0].1, ow[i1].2), outWord: o, nonword: nonword, prot: pj, changed: changed,
                                rare: rare, low: low, wlp: M3Text.round4(lpW), rank: rank, nameLike: nameLike, alts: alts))
        }
        return (plans, tw, asg)
    }

    // MARK: decision (tune.py)

    static func candsOn(_ p: M3Plan, _ P: M3Params) -> Bool {
        p.nonword || (p.rare && (Double(p.rank) >= P.rare || p.wlp < P.lowconf))
    }

    static func decideAlts(_ p: M3Plan, _ P: M3Params) -> [Int] {
        var out: [Int] = []
        var ncand = 0
        for (i, a) in p.alts.enumerated() {
            if a.kind != "copy" && a.kind != "greedy" {
                if !candsOn(p, P) { continue }
                ncand += 1
                if ncand > P.k { continue }
            }
            out.append(i)
        }
        return out
    }

    static func score(_ a: M3Alt, _ P: M3Params) -> Double { a.dm3 + P.a * a.sp + P.b * a.ng + P.d * a.pr }

    static func copyScore(_ p: M3Plan, _ P: M3Params) -> Double {
        score(p.alts[0], P) + (p.nameLike ? P.cName : 0.0) + (p.nonword ? 0.0 : p.changed ? P.cReal : P.cRare)
    }

    /// tune.decide -> (best non-copy alt, margin over the copy, kind, index); no alternative -> (copy, -1e18, "copy", 0)
    public static func decide(_ p: M3Plan, _ P: M3Params) -> (String, Double, String, Int) {
        var best: M3Alt? = nil
        var bs = -1e18, scCopy = 0.0, bi = 0
        for i in decideAlts(p, P) {
            let a = p.alts[i]
            if a.kind == "copy" { scCopy = copyScore(p, P) } else {
                let s = score(a, P)
                if s > bs { best = a; bs = s; bi = i }
            }
        }
        guard let b = best else { return (p.alts[0].text, -1e18, "copy", 0) }
        return (b.text, bs - scCopy, b.kind, bi)
    }

    /// the considered alternatives other than the copy, best first (ties: list order)
    public static func ranked(_ p: M3Plan, _ P: M3Params) -> [(String, Double)] {
        let ks = decideAlts(p, P).filter { p.alts[$0].kind != "copy" }
        let sc = ks.map { score(p.alts[$0], P) }
        let order = ks.indices.sorted { sc[$0] != sc[$1] ? sc[$0] > sc[$1] : $0 < $1 }
        return order.map { (p.alts[ks[$0]].text, sc[$0]) }
    }

    /// tune.casual_filter: casual keeps apostrophe-less contractions. A typed one (CASUAL_OK, no apostrophe) is never a
    /// correction site (its slot is dropped); elsewhere alternatives that only add or move apostrophes are dropped.
    static func casualFilter(_ plans: [M3Plan]) -> [M3Plan] {
        plans.compactMap { p -> M3Plan? in
            var q = p
            let t = M3Text.lower(M3Text.string(p.typed.unicodeScalars.filter { $0 != "'" }))
            if !p.typed.contains("'") && M3_CASUAL_OK.contains(t) { return nil }
            q.alts = p.alts.filter { $0.kind == "copy" || $0.kind == "greedy" || M3Text.lower(M3Text.string($0.text.unicodeScalars.filter { $0 != "'" })) != t }
            return q
        }
    }

    static func splice(_ s: Scalars, _ span: (Int, Int), _ rep: String) -> Scalars {
        Array(s[0..<span.0]) + Array(rep.unicodeScalars) + Array(s[span.1...])
    }

    /// tune.apply: every change with margin >= th applied to the greedy text, every other slot back to its copy
    static func apply(_ o: Scalars, _ plans: [M3Plan], _ P: M3Params, _ th: Double) -> String {
        var edits: [((Int, Int), String)] = []
        for p in plans {
            let (a, mg, kd, _) = decide(p, P)
            let tgt = kd != "copy" && mg >= th ? a : p.alts[0].text
            if tgt != p.outWord { edits.append((p.span, tgt)) }
        }
        var s = o
        for (sp, t) in edits.sorted(by: { $0.0.0 > $1.0.0 }) { s = splice(s, sp, t) }
        return M3Text.string(s)
    }

    func enc(_ s: String) -> [Int] { tok.encode(s.replacingOccurrences(of: "\n", with: "\n ")) }

    // MARK: end to end

    public func correct(_ text: String, style: String = "standard", user: [String] = []) throws -> M3Result {
        let t0 = Date()
        let P = params
        let typed = M3Text.scalars(text)
        let src = model.srcIds(text, style: style)
        let e = try model.encode(src)
        var cache: [String: [M3Cand]] = [:]
        var inj: [String] = []
        for w in M3Text.segWords(text) {
            let l = M3Text.lower(w)
            if D.lex.index[l] != nil { continue }
            let cl: [M3Cand]
            if let c = cache[l] { cl = c } else { cl = D.cands(l, k: 8); cache[l] = cl }
            inj += cl.map { $0.text }
        }
        let ctx = M3Ctx(D, typed: text, user: user, injected: inj)
        let g = try model.constrainedGreedy(src, e, ctx, ptext)
        let o = tok.decode(g.toks).replacingOccurrences(of: "\n ", with: "\n")
        let os = M3Text.scalars(o)
        let wl = wordLps(os, g.toks, g.lps)
        var (plans, tw, asg) = slotPlan(typed, os, k: 8, cache: &cache, rareRank: min(Self.RARE, Int(min(P.rare, 1e9))), wlp: wl,
                                        lowconf: P.lowconf > -1e8 ? P.lowconf : nil)
        let base = enc(o)
        let ow = M3Text.segWords(o).map(M3Text.lower)
        let owLp = (lm != nil && P.b != 0) ? lm!.sentLp(ow) : 0
        var fast = true
        for pi in plans.indices {
            let p = plans[pi]
            let keep = Self.decideAlts(p, P)
            var ds: [Double] = []
            if keep.count > 1 {
                fast = false
                let vs = keep.filter { p.alts[$0].kind != "greedy" }.map { enc(M3Text.string(Self.splice(os, p.span, p.alts[$0].text))) }
                ds = try model.deltaScores(e, base: base, variants: vs)
            }
            var di = 0
            var alts: [M3Alt] = []
            for (i, a0) in p.alts.enumerated() {
                var a = a0
                var dm3: Double
                if !keep.contains(i) || keep.count == 1 { dm3 = a.kind != "copy" || !p.changed ? 0.0 : -1e9 } else if a.kind == "greedy" { dm3 = 0 } else { dm3 = ds[di]; di += 1 }
                var ng = 0.0
                if let lm = lm, P.b != 0, a.kind != "greedy" {
                    ng = lm.sentLp(M3Text.segWords(M3Text.string(Self.splice(os, p.span, a.text))).map(M3Text.lower)) - owLp
                }
                a.dm3 = M3Text.round4(dm3); a.sp = M3Text.round4(spFeat(p.typed, a.text)); a.pr = M3Text.round4(prFeat(a.text))
                a.ng = M3Text.round4(ng)
                alts.append(a)
            }
            plans[pi].alts = alts
        }
        if style == "casual" { plans = Self.casualFilter(plans) }
        var auto: [M3Change] = [], sugg: [M3Change] = []
        for p in plans {
            let (a, mg, kd, _) = Self.decide(p, P)
            if kd == "copy" { continue }
            let c = M3Change(j: p.j, typed: p.typed, to: a, margin: Double(String(format: "%.3f", mg)) ?? mg)
            if mg >= P.thAuto { auto.append(c) } else if mg >= P.thSugg { sugg.append(c) }
        }
        return M3Result(text: Self.apply(os, plans, P, P.thAuto), greedy: o, auto: auto, suggest: sugg, fast: fast, blocked: g.blocked,
                        plans: plans, typedWords: tw, asg: asg, ms: Date().timeIntervalSince(t0) * 1000)
    }

    // MARK: app side

    /// The verdict for typed word j (nil: no uncertain slot, i.e. keep). A slot whose output also swallowed the next
    /// typed word(s) (a merge) is never usable on its own: keep.
    public func verdict(_ r: M3Result, j: Int) -> M3WordVerdict? {
        guard let p = r.plans.first(where: { $0.j == j }) else { return nil }
        if j + 1 < r.asg.count && r.asg[j + 1].isEmpty { return nil }
        let P = params
        let (best, mg, kd, _) = Self.decide(p, P)
        let tier: M3Tier = kd == "copy" ? .keep : mg >= P.thAuto ? .auto : mg >= P.thSugg ? .suggest : .keep
        var fixes: [String] = []
        for (t, _) in Self.ranked(p, P) where M3Text.lower(t) != M3Text.lower(p.typed) && !fixes.contains(t) {
            fixes.append(t)
            if fixes.count == 3 { break }
        }
        return M3WordVerdict(j: j, typed: p.typed, tier: tier, best: kd == "copy" ? p.typed : best, margin: kd == "copy" ? 0 : mg,
                             fixes: fixes, nonword: p.nonword)
    }
}
