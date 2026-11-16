import Foundation

// The dictionary side of the m3 decoder (port of m3dict_cpt/lexicon.py + decode.Ctx / feed): the 100k shipping
// lexicon with its prefix table, the SymSpell delete index (m3_symspell.bin, or built here from the lexicon: high 32 bits of
// FNV-1a 64 of every delete; every hit is verified by its delete strings, so key collisions drop out),
// the candidate generator, and the per-message word constraint of the constrained greedy.

/// One generator candidate: lowercase text, spatial log-lik, log prior, kind (word / contr / gdrop / split / space).
public struct M3Cand: Equatable {
    public let text: String
    public let sp: Double
    public let logp: Double
    public let kind: String
}

public final class M3Lexicon {
    public let lex: Lexicon
    /// prefix (with apostrophes) -> letter count of its shortest completion
    let prefixMin: [String: Int]
    /// SymSpell index: high 32 bits of the delete hashes, sorted by (key, word id)
    let symH: [UInt32]
    let symIds: [UInt32]
    /// where the index came from ("file" / "built")
    public let indexSource: String
    /// apostrophe-less form -> lexicon words with apostrophes (lexicon order)
    let expand: [String: [String]]
    let gen2: CandidateGeneratorV2
    let sp: SpatialModel
    /// log prior of a word outside the lexicon (min log prior - 1)
    public let floor: Double

    // lexicon.CandGen with kprep.GEN = (qd3 7, short 2, bag_n 300) and the defaults
    static let MAXD = 2, SHORT = 2, QD3 = 7, BAG_N = 300, BAG_MIN = 6, MAX_HITS = 4000, PRIOR_W = 0.5

    public convenience init(path: String, symspell: String? = nil) throws {
        self.init(lex: try Lexicon(path: path), symspell: try symspell.map { try Data(contentsOf: URL(fileURLWithPath: $0), options: .alwaysMapped) })
    }

    /// m3_symspell.bin: b'KLSS', u32 version 1, n_words, n_entries, maxd, short (LE), u32 key[n], u32 word_id[n]
    static func readIndex(_ d: Data, nWords: Int) -> ([UInt32], [UInt32])? {
        guard d.count >= 24, d.prefix(4) == Data("KLSS".utf8) else { return nil }
        func u32(_ o: Int) -> UInt32 { d.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: o, as: UInt32.self)) } }
        let ver = u32(4), nw = Int(u32(8)), n = Int(u32(12)), maxd = Int(u32(16)), short = Int(u32(20))
        guard ver == 1, nw == nWords, maxd == MAXD, short == SHORT, d.count == 24 + 8 * n else { return nil }
        var keys = [UInt32](repeating: 0, count: n), ids = [UInt32](repeating: 0, count: n)
        d.withUnsafeBytes { raw in
            keys.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[24..<(24 + 4 * n)])) }
            ids.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[(24 + 4 * n)..<(24 + 8 * n)])) }
        }
        if UInt32(littleEndian: 1) != 1 { keys = keys.map { UInt32(littleEndian: $0) }; ids = ids.map { UInt32(littleEndian: $0) } }
        return (keys, ids)
    }

    public init(lex: Lexicon, symspell: Data? = nil) {
        self.lex = lex
        let n = lex.size
        var pm: [String: Int] = [:]
        pm.reserveCapacity(260_000)
        var ex: [String: [String]] = [:]
        for w in lex.words {
            let u = Array(w.unicodeScalars)
            let letters = u.filter { $0 != "'" }.count
            var v = String.UnicodeScalarView()
            for k in 0..<u.count {
                v.append(u[k])
                let q = String(v)
                if (pm[q] ?? 99) > letters { pm[q] = letters }
            }
            if u.contains("'") { ex[M3Text.string(u.filter { $0 != "'" }), default: []].append(w) }
        }
        prefixMin = pm
        expand = ex
        // SymSpell: every delete at distance <= 2 (<= 1 for words of <= 2 letters) of the apostrophe-less word
        if let d = symspell, let r = Self.readIndex(d, nWords: n) {
            symH = r.0; symIds = r.1; indexSource = "file"
        } else {
            var pairs: [UInt64] = []   // key << 32 | id: sorting it sorts by (key, id)
            pairs.reserveCapacity(3_500_000)
            var hs: [UInt64] = []
            for i in 0..<n {
                let s = Array(lex.words[i].utf8.filter { $0 != 39 })
                Self.deleteHashes(s, s.count > Self.SHORT ? Self.MAXD : 1, into: &hs)
                for h in hs { pairs.append((h >> 32) << 32 | UInt64(i)) }
            }
            pairs.sort()
            symH = pairs.map { UInt32($0 >> 32) }
            symIds = pairs.map { UInt32($0 & 0xffff_ffff) }
            indexSource = "built"
        }
        var mn = Double.infinity
        for x in lex.logp where x < mn { mn = x }
        floor = mn - 1
        sp = SpatialModel(sigma: 0.3)
        gen2 = CandidateGeneratorV2(gen: CandidateGenerator(lex: lex, sp: sp))
    }

    @inline(__always) static func fnv(_ s: [UInt8], skip a: Int, _ b: Int, _ c: Int) -> UInt64 {
        var h: UInt64 = 0xcbf29ce484222325
        for (k, x) in s.enumerated() where k != a && k != b && k != c {
            h = (h ^ UInt64(x)) &* 0x100000001b3
        }
        return h
    }

    /// FNV-1a 64 of every distinct string left after removing at most d characters (lexicon.deletes), sorted
    static func deleteHashes(_ s: [UInt8], _ d: Int, into out: inout [UInt64]) {
        out.removeAll(keepingCapacity: true)
        let L = s.count
        out.append(fnv(s, skip: -1, -1, -1))
        if d >= 1 { for a in 0..<L { out.append(fnv(s, skip: a, -1, -1)) } }
        if d >= 2 && L >= 2 { for a in 0..<L { for b in (a + 1)..<L { out.append(fnv(s, skip: a, b, -1)) } } }
        if d >= 3 && L >= 3 {
            for a in 0..<L { for b in (a + 1)..<L { for c in (b + 1)..<L where c < L { out.append(fnv(s, skip: a, b, c)) } } }
        }
        out.sort()
        var w = 0
        for r in 0..<out.count where r == 0 || out[r] != out[r - 1] { out[w] = out[r]; w += 1 }
        out.removeLast(out.count - w)
    }

    // MARK: lookups

    public func inLex(_ w: String) -> Bool { lex.index[w] != nil }
    public func rank(_ w: String) -> Int? { lex.index[w] }

    /// every distinct string left after removing at most d characters (lexicon.deletes)
    static func deleteStrings(_ s: [UInt8], _ d: Int) -> Set<[UInt8]> {
        var out: Set<[UInt8]> = [s]
        var frontier: Set<[UInt8]> = [s]
        for _ in 0..<d {
            var nxt = Set<[UInt8]>()
            for x in frontier { for i in 0..<x.count { var y = x; y.remove(at: i); nxt.insert(y) } }
            out.formUnion(nxt); frontier = nxt
        }
        return out
    }

    @inline(__always) static func fnvHi(_ s: [UInt8]) -> UInt32 {
        var h: UInt64 = 0xcbf29ce484222325
        for x in s { h = (h ^ UInt64(x)) &* 0x100000001b3 }
        return UInt32(h >> 32)
    }

    /// q is one of the word's deletes: a subsequence of its apostrophe-less form, at most d letters shorter
    func isDelete(_ q: [UInt8], of id: Int) -> Bool {
        var s: [UInt8] = []
        s.reserveCapacity(24)
        for b in lex.words[id].utf8 where b != 39 { s.append(b) }
        let d = s.count > Self.SHORT ? Self.MAXD : 1
        let diff = s.count - q.count
        if diff < 0 || diff > d { return false }
        var i = 0
        for b in s where i < q.count && b == q[i] { i += 1 }
        return i == q.count
    }

    /// lexicon ids that share a delete with the typed word (FORMAT.md lookup: index hits on the high 32 bits of the
    /// delete hash, each checked against the word, so key collisions drop out), sorted, unique
    func lookup(_ tl: String) -> [Int] {
        let s = Array(tl.utf8.filter { $0 != 39 })
        let d = s.count > Self.SHORT ? (s.count >= Self.QD3 ? 3 : Self.MAXD) : 1
        var got: [Int] = []
        for q in Self.deleteStrings(s, d) {
            let h = Self.fnvHi(q)
            var lo = 0, hi = symH.count
            while lo < hi { let m = (lo + hi) / 2; if symH[m] < h { lo = m + 1 } else { hi = m } }
            var k = lo
            while k < symH.count && symH[k] == h {
                let id = Int(symIds[k])
                if isDelete(q, of: id) { got.append(id) }
                k += 1
            }
        }
        got.sort()
        var w = 0
        for r in 0..<got.count where r == 0 || got[r] != got[r - 1] { got[w] = got[r]; w += 1 }
        return Array(got.prefix(w))
    }

    /// m2's bag-of-keys prefilter (CandGen.bag_top): top N by (score desc, id asc)
    func bagTop(_ tl: String, _ N: Int) -> [Int] {
        var q = [Float](repeating: 0, count: 26)
        for u in tl.unicodeScalars where u.value >= 97 && u.value <= 122 { q[Int(u.value) - 97] += 1 }
        var nl: Float = 0
        for x in q { nl += x }
        let n = lex.size
        var cheap = [Double](repeating: 0, count: n)
        lex.bag.withUnsafeBufferPointer { B in
            for i in 0..<n {
                var dist: Float = 0, cnt: Float = 0
                for j in 0..<26 { let b = B[i * 26 + j]; dist += abs(b - q[j]); cnt += b }
                let dn = abs(cnt - nl)
                cheap[i] = dn > 2 ? -1e9 : Double(-(dist + Float(1.5) * dn)) + 0.15 * lex.logp[i]
            }
        }
        // only words within 2 letters can make the cut (the rest are -1e9 and dropped anyway)
        let idx = (0..<n).filter { cheap[$0] > -1e8 }.sorted { cheap[$0] != cheap[$1] ? cheap[$0] > cheap[$1] : $0 < $1 }
        return Array(idx.prefix(N))
    }

    func spatial(_ tl: String, _ words: [String]) -> [Double] {
        let em = sp.emitTable(KeyGeom.keyTouches(tl))
        return words.map { w in sp.wordCode(w).map { sp.score(em, $0) } ?? -1e9 }
    }

    /// CandGen.cands: the k best lexicon words by spatial + 0.5 * log prior, then apostrophe expansions, the g-drop and
    /// the two best space splits. The typed word itself is never a candidate.
    public func cands(_ typed: String, k: Int = 8) -> [M3Cand] {
        let tl = M3Text.lower(typed)
        var ids = lookup(tl)
        let n = tl.unicodeScalars.filter { $0 != "'" }.count
        if n >= Self.BAG_MIN {
            ids = Array(Set(ids).union(bagTop(tl, Self.BAG_N))).sorted()
        }
        if ids.count > Self.MAX_HITS { ids = Array(ids.prefix(Self.MAX_HITS)) }
        ids = ids.filter { lex.words[$0] != tl }
        var res: [M3Cand] = []
        if !ids.isEmpty {
            let em = sp.emitTable(KeyGeom.keyTouches(tl))
            let sc = ids.map { sp.score(em, lex.codes[$0]) }
            let rank = ids.indices.map { sc[$0] + Self.PRIOR_W * lex.logp[ids[$0]] }
            let order = ids.indices.sorted { rank[$0] != rank[$1] ? rank[$0] > rank[$1] : ids[$0] < ids[$1] }
            res = order.prefix(k).map { M3Cand(text: lex.words[ids[$0]], sp: sc[$0], logp: lex.logp[ids[$0]], kind: "word") }
        }
        let have = Set(res.map { $0.text })
        var extra: [(String, String)] = []
        if !tl.contains("'") { extra += (expand[tl] ?? []).map { ($0, "contr") } }
        if tl.hasSuffix("in") && tl.unicodeScalars.count >= 4 && lex.index[tl + "g"] != nil { extra.append((tl + "g", "gdrop")) }
        extra = extra.filter { !have.contains($0.0) }
        if !extra.isEmpty {
            let sc = spatial(tl, extra.map { $0.0 })
            for (j, e) in extra.enumerated() { res.append(M3Cand(text: e.0, sp: sc[j], logp: lex.logp[lex.index[e.0]!], kind: e.1)) }
        }
        let alpha = !tl.isEmpty && tl.unicodeScalars.allSatisfy { M3Text.isL($0) }
        if alpha && lex.index[tl] == nil {
            for c in gen2.splits(KeyGeom.keyTouches(tl), tl).prefix(2) where !have.contains(c.text) {
                var lp = 0.0
                for x in c.text.split(separator: " ") { lp += lex.logp[lex.index[String(x)]!] }
                res.append(M3Cand(text: c.text, sp: c.sp, logp: lp, kind: c.kind))
            }
        }
        return res
    }
}

/// apostrophe-less contractions: legal output words though the lexicon dropped them (decode.CASUAL_OK)
let M3_CASUAL_OK: Set<String> = Set("""
dont cant wont im ive youre theyre isnt doesnt didnt wasnt werent arent hasnt havent hadnt
shouldnt couldnt wouldnt thats whats theres heres wheres whos youve youll youd theyve theyll theyd weve itll thatll
mustnt hows whys whens shes hes
""".split(whereSeparator: { $0 == " " || $0 == "\n" }).map(String.init))

/// decode.Ctx: per-message valid words (typed copies, user dictionary, injected candidates) and the split rule's words.
final class M3Ctx {
    let D: M3Lexicon
    let words: Set<String>
    let prefixes: Set<String>
    let cap: Int
    let splitWords: [String]

    init(_ D: M3Lexicon, typed: String, user: [String] = [], injected: [String] = []) {
        self.D = D
        let tw = Set(M3Text.segWords(typed).map(M3Text.lower))
        let ul = Set(user.map(M3Text.lower))
        var ws = tw.union(ul).union(M3_CASUAL_OK)
        for c in injected { for x in M3Text.segWords(c) { ws.insert(M3Text.lower(x)) } }
        words = ws
        var ps = Set<String>()
        for w in ws {
            var v = String.UnicodeScalarView()
            for u in w.unicodeScalars { v.append(u); ps.insert(String(v)) }
        }
        prefixes = ps
        cap = (tw.map { $0.unicodeScalars.filter { $0 != "'" }.count } + [4]).max()! + 4
        splitWords = Array(tw).sorted().filter { $0.unicodeScalars.count >= 3 && (D.lex.index[$0] != nil || ul.contains($0)) }
    }

    func validWord(_ w: String) -> Bool {
        if words.contains(w) { return true }
        if D.lex.index[w] != nil { return w.unicodeScalars.filter { $0 != "'" }.count <= cap }
        let u = Array(w.unicodeScalars)
        for s in ["'s", "'"] {
            let ls = s.unicodeScalars.count
            if w.hasSuffix(s) && u.count > ls {
                let b = M3Text.string(u.dropLast(ls))
                if D.lex.index[b] != nil || words.contains(b) { return true }
            }
        }
        return false
    }

    func validPrefix(_ p: String) -> Bool {
        if prefixes.contains(p) { return true }
        if (D.prefixMin[p] ?? 99) <= cap { return true }
        let u = Array(p.unicodeScalars)
        if let i = u.lastIndex(of: "'"), i > 0 {
            let tail = M3Text.string(u[(i + 1)...])
            if tail == "" || tail == "s" { return validWord(M3Text.string(u[..<i])) }
        }
        return false
    }

    func rems(_ w: String) -> [String]? {
        let n = w.unicodeScalars.count
        let r = splitWords.filter { $0.unicodeScalars.count > n && $0.hasPrefix(w) }.map { M3Text.string(Array($0.unicodeScalars).dropFirst(n)) }
        return r.isEmpty ? nil : r
    }
}

/// decode.feed state: open word (lowercase, may end in '), split remainders active when it started, remainders after
/// the last completed word, only spaces since that word
struct M3State {
    var cur: String = ""
    var srem: [String]? = nil
    var rem: [String]? = nil
    var gap = false
}

enum M3Constraint {
    static func rstripApos(_ s: String) -> String {
        var u = Array(s.unicodeScalars)
        while u.last == "'" { u.removeLast() }
        return M3Text.string(u)
    }

    /// decode.feed: append a piece's text -> (legal, splits a typed lexicon word, new state)
    static func feed(_ ctx: M3Ctx, _ st: M3State, _ s: String) -> (Bool, Bool, M3State) {
        var cur = st.cur, srem = st.srem, rem = st.rem, gap = st.gap
        for ch in s.unicodeScalars {
            if M3Text.isL(ch) {
                if cur.isEmpty { srem = gap ? rem : nil }
                cur.unicodeScalars.append(M3Text.isUpper(ch) ? Unicode.Scalar(ch.value + 32)! : ch)
            } else if ch == "'" && !cur.isEmpty && cur.unicodeScalars.last != "'" {
                cur.unicodeScalars.append("'")
            } else {
                if !cur.isEmpty {
                    let w = rstripApos(cur)
                    if !ctx.validWord(w) { return (false, false, st) }
                    rem = ctx.rems(w); gap = true; cur = ""; srem = nil
                    if ch != " " { gap = false }
                } else if ch != " " {
                    gap = false
                }
            }
        }
        var split = false
        if !cur.isEmpty {
            if !ctx.validPrefix(cur) { return (false, false, st) }
            if let sr = srem, !sr.isEmpty {
                let c = M3Text.string(cur.unicodeScalars.filter { $0 != "'" })
                split = sr.contains { $0.hasPrefix(c) }
            }
        }
        return (true, split, M3State(cur: cur, srem: srem, rem: rem, gap: gap))
    }

    static func finishOk(_ ctx: M3Ctx, _ st: M3State) -> Bool {
        st.cur.isEmpty || ctx.validWord(rstripApos(st.cur))
    }
}
