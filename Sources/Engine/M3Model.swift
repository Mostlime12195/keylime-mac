import Foundation

/// m3 graph facts from the manifest (m3.manifest.json): decoder start / end token, style ids, length limits.
public struct M3Meta {
    public var eos = 2
    public var styleIds: [String: Int] = ["casual": 256, "standard": 257, "formal": 258]
    public var maxSrc = 384
    public var maxTgt = 112
    public var nCross = 3
    public var nKvHeads = 2
    public var headDim = 64
    public var convKernel = 7
    public var dModel = 512
    public init() {}

    public static func from(json o: [String: Any]) -> M3Meta {
        var m = M3Meta()
        func i(_ k: String) -> Int? { (o[k] as? NSNumber)?.intValue }
        let c = o["config"] as? [String: Any] ?? o
        func ci(_ k: String) -> Int? { (c[k] as? NSNumber)?.intValue ?? i(k) }
        if let v = i("eos") { m.eos = v }
        if let s = (o["style_ids"] as? [String: Any])?.compactMapValues({ ($0 as? NSNumber)?.intValue }), !s.isEmpty { m.styleIds = s }
        if let v = i("max_src") { m.maxSrc = v }
        if let v = i("max_tgt") { m.maxTgt = v }
        if let v = i("n_cross") { m.nCross = v }
        if let v = ci("n_kv_heads") { m.nKvHeads = v }
        if let v = ci("head_dim") { m.headDim = v }
        if let v = ci("conv_kernel") { m.convKernel = v }
        if let v = ci("d_model") { m.dModel = v }
        return m
    }
}

/// The int4 encoder-decoder through onnxruntime (port of m3dict_cpt/onnx_backend.OrtDict): batch 1, KV cache, the
/// constrained greedy and teacher-forced deltas with the shared prefix reused.
public final class M3Model {
    public let meta: M3Meta
    let enc: OrtSessionHandle
    let dec: OrtSessionHandle
    /// decoder state inputs without the past_ prefix, in graph order (k0, v0, c0, c1)
    let stateNames: [String]
    /// timing
    public var decRuns = 0
    public var ortSeconds = 0.0

    public struct Encoded { let cross: [Feed]; let S: Int }

    struct DState {
        var data: [String: [Float]]
        var shapes: [String: [Int64]]
        var mask: [Bool]
        var pos: Int
    }

    /// MatMulNBits accuracy_level the graphs run with: 4 = int8 activations against the same int4 weights (the graphs are
    /// stored with 0, fp32 dequantization, several times slower per step); test vectors are made with 4
    public static let ACCURACY_LEVEL = 4
    /// MatMulNBits nodes patched at load (65 encoder, 28 decoder for m3)
    public private(set) var patchedEncoder = 0
    public private(set) var patchedDecoder = 0

    /// the serialized attribute: name "accuracy_level" (field 1), then its int value 0 (field 3)
    static let ACC_PATTERN: [UInt8] = [0x0a, 0x0e] + Array("accuracy_level".utf8) + [0x18, 0x00]

    /// graph bytes with every MatMulNBits accuracy_level 0 set to `level` (one byte; protobuf lengths unchanged) and the
    /// number of nodes changed (onnx_backend.with_accuracy_level / Android M3Model.withAccuracyLevel)
    public static func withAccuracyLevel(_ graph: Data, _ level: Int) -> (Data, Int) {
        var b = Data(graph)   // one heap copy of the mapped file
        let p = ACC_PATTERN, n = b.count, m = p.count
        var count = 0
        var i = 0
        b.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            let buf = raw.bindMemory(to: UInt8.self)
            while i <= n - m {
                if buf[i] == p[0] && buf[i + m - 1] == p[m - 1] && buf[i + 2] == p[2] {
                    var ok = true
                    for j in 1..<m where buf[i + j] != p[j] { ok = false; break }
                    if ok { buf[i + m - 1] = UInt8(level); count += 1; i += m; continue }
                }
                i += 1
            }
        }
        return (b, count)
    }

    public init(encoder: String, decoder: String, meta: M3Meta, threads: Int, accuracyLevel: Int = M3Model.ACCURACY_LEVEL) throws {
        self.meta = meta
        if accuracyLevel != 0 {
            let (e, ne) = Self.withAccuracyLevel(try Data(contentsOf: URL(fileURLWithPath: encoder), options: .alwaysMapped), accuracyLevel)
            enc = try OrtSessionHandle(bytes: e, threads: threads)
            let (d, nd) = Self.withAccuracyLevel(try Data(contentsOf: URL(fileURLWithPath: decoder), options: .alwaysMapped), accuracyLevel)
            dec = try OrtSessionHandle(bytes: d, threads: threads)
            patchedEncoder = ne; patchedDecoder = nd
        } else {
            enc = try OrtSessionHandle(path: encoder, threads: threads, ortFormat: encoder.hasSuffix(".ort"))
            dec = try OrtSessionHandle(path: decoder, threads: threads, ortFormat: decoder.hasSuffix(".ort"))
        }
        stateNames = dec.inputNames.filter { $0.hasPrefix("past_") && $0 != "past_mask" }.map { String($0.dropFirst(5)) }
    }

    public func close() { enc.close(); dec.close() }

    public func srcIds(_ text: String, style: String) -> [Int] {
        let st = meta.styleIds[style] ?? meta.styleIds["standard"] ?? 257
        return Array(([st] + text.utf8.map { Int($0) }).prefix(meta.maxSrc))
    }

    public func encode(_ src: [Int]) throws -> Encoded {
        let S = src.count
        let names = (0..<meta.nCross).flatMap { ["cross_k\($0)", "cross_v\($0)"] }
        let t0 = Date()
        let r = try enc.run([Feed("src_ids", int64: src.map(Int64.init), [1, Int64(S)]),
                             Feed("taps", float: [Float](repeating: 0, count: S * 2), [1, Int64(S), 2])], outputs: names)
        ortSeconds += Date().timeIntervalSince(t0)
        return Encoded(cross: names.map { Feed($0, float: r[$0]!.0, r[$0]!.1) }, S: S)
    }

    func fresh() -> DState {
        var d: [String: [Float]] = [:], s: [String: [Int64]] = [:]
        for n in stateNames {
            if n.hasPrefix("k") || n.hasPrefix("v") {
                d[n] = [Float](repeating: 0, count: meta.nKvHeads * meta.headDim)
                s[n] = [1, Int64(meta.nKvHeads), 1, Int64(meta.headDim)]
            } else {
                d[n] = [Float](repeating: 0, count: (meta.convKernel - 1) * meta.dModel)
                s[n] = [1, Int64(meta.convKernel - 1), Int64(meta.dModel)]
            }
        }
        return DState(data: d, shapes: s, mask: [false], pos: 0)
    }

    /// one decoder call over ids -> (last-position log-probs if asked, log-prob of each next id, new state if asked)
    func run(_ e: Encoded, _ st: DState, _ ids: [Int], next: [Int]? = nil, wantLast: Bool = false, wantState: Bool = false) throws
        -> (last: [Float]?, tok: [Float], state: DState?) {
        let n = ids.count
        var feeds: [Feed] = [
            Feed("ids", int64: ids.map(Int64.init), [1, Int64(n)]),
            Feed("next_ids", int64: (next ?? [Int](repeating: 0, count: n)).map(Int64.init), [1, Int64(n)]),
            Feed("pos", int64: (0..<n).map { Int64(st.pos + $0) }, [Int64(n)]),
            Feed("past_mask", bool: st.mask, [Int64(st.mask.count)]),
            Feed("src_mask", bool: [Bool](repeating: true, count: e.S), [Int64(e.S)]),
        ]
        for nm in stateNames { feeds.append(Feed("past_\(nm)", float: st.data[nm]!, st.shapes[nm]!)) }
        feeds += e.cross
        var want = ["tok_lp"]
        if wantLast { want.append("last_lp") }
        if wantState { want += stateNames.map { "present_\($0)" } }
        let t0 = Date()
        let r = try dec.run(feeds, outputs: want)
        ortSeconds += Date().timeIntervalSince(t0); decRuns += 1
        var ns: DState? = nil
        if wantState {
            var d: [String: [Float]] = [:], s: [String: [Int64]] = [:]
            for nm in stateNames { let o = r["present_\(nm)"]!; d[nm] = o.0; s[nm] = o.1 }
            ns = DState(data: d, shapes: s, mask: st.mask + [Bool](repeating: true, count: n), pos: st.pos + n)
        }
        return (wantLast ? r["last_lp"]!.0 : nil, r["tok_lp"]!.0, ns)
    }

    /// the k largest values' indices, best first (ties: lower index first)
    static func topK(_ v: [Float], _ k: Int) -> [Int] {
        var best: [Int] = []
        best.reserveCapacity(k + 1)
        for i in 0..<v.count {
            if best.count == k {
                let w = best[k - 1]
                if !(v[i] > v[w]) { continue }
                best.removeLast()
            }
            var p = best.count
            while p > 0 && v[i] > v[best[p - 1]] { p -= 1 }
            best.insert(i, at: p)
        }
        return best
    }

    /// OrtDict.constrained_greedy -> output tokens, their log-probs, steps where the argmax was blocked
    func constrainedGreedy(_ src: [Int], _ e: Encoded, _ ctx: M3Ctx, _ ptext: [String?], K: Int = 8, tauSplit: Double = 4.0) throws
        -> (toks: [Int], lps: [Double], blocked: Int) {
        let eos = meta.eos
        var state = fresh()
        var cs = M3State()
        var out: [Int] = [], lps: [Double] = []
        var nblk = 0
        var ids = [eos]
        for _ in 0..<min(meta.maxTgt, src.count / 2 + 24) {
            let r = try run(e, state, ids, wantLast: true, wantState: true)
            let lp = r.last!
            state = r.state!
            let top = Self.topK(lp, K)
            var choice: (Int, M3State)? = nil
            var splitTok: (Int, M3State, Double)? = nil
            var scanned = false
            var cand = top
            while choice == nil {
                for tok in cand {
                    let v = Double(lp[tok])
                    if tok == eos {
                        if M3Constraint.finishOk(ctx, cs) { choice = (tok, cs); break }
                        continue
                    }
                    guard tok < ptext.count, let pt = ptext[tok] else { continue }
                    let (ok, split, ns) = M3Constraint.feed(ctx, cs, pt)
                    if !ok { continue }
                    if split {
                        if splitTok == nil { splitTok = (tok, ns, v) }
                        continue
                    }
                    choice = (tok, ns)
                    if let s = splitTok, s.2 - v >= tauSplit { choice = (s.0, s.1) }
                    break
                }
                if choice == nil {
                    if scanned { choice = splitTok.map { ($0.0, $0.1) } ?? (eos, cs); break }
                    let all = (0..<lp.count).sorted { lp[$0] != lp[$1] ? lp[$0] > lp[$1] : $0 < $1 }
                    cand = Array(all.dropFirst(K))
                    scanned = true
                }
            }
            let (tok, ns) = choice!
            cs = ns
            if tok != top[0] { nblk += 1 }
            if tok == eos { break }
            out.append(tok); ids = [tok]; lps.append(Double(lp[tok]))
        }
        return (out, lps, nblk)
    }

    /// OrtDict.delta_scores: log P(v + EOS) - log P(base + EOS) per variant (decoder state after the common prefix shared)
    func deltaScores(_ e: Encoded, base: [Int], variants: [[Int]]) throws -> [Double] {
        let eos = meta.eos
        var res: [Double] = []
        var cache: [Int: (DState, Double)] = [:]
        for v in variants {
            var L = 0
            while L < min(v.count, base.count) && v[L] == base[L] { L += 1 }
            let pre = [eos] + base[0..<L]
            if cache[L] == nil {
                let st = try (L == 0 ? fresh() : run(e, fresh(), Array(pre.dropLast()), wantState: true).state!)
                let tl = try run(e, st, [pre.last!] + base[L...], next: Array(base[L...]) + [eos]).tok
                cache[L] = (st, Double(M3Text.npSum(tl[...])))
            }
            let (st, bl) = cache[L]!
            let tl = try run(e, st, [pre.last!] + v[L...], next: Array(v[L...]) + [eos]).tok
            res.append(Double(M3Text.npSum(tl[...])) - bl)
        }
        return res
    }
}
