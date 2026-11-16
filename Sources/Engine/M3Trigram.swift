import Foundation

/// The pruned, quantized KN word trigram (m3_ngram.bin; reader = m3dict_cpt/qkn.py QKN): backoff form, every value a
/// multiple of 1/1024, so sums are exact in Double and identical on every platform.
///   log P(w | u v) = T(u,v,w) if stored, else G3(u,v) + B(v,w) if the history is stored, else B(v,w)
///   B(v,w)         = Q2(v,w) if stored, else G2(v) + P1(w)
public final class M3Trigram: M3WordLM {
    static let SCALE = 1024.0
    static let LP_OOV = -16.0
    let data: Data
    let ids: [String: Int]
    let V: Int
    // offsets (in bytes) of the arrays inside data
    let p1Off, g2Off, boffOff, bentOff, huoffOff, hentOff, toffOff, tentOff: Int

    public enum Failure: Error { case bad(String) }

    public init(path: String) throws {
        let d = try Data(contentsOf: URL(fileURLWithPath: path), options: .alwaysMapped)
        data = d
        guard d.count >= 28, d.prefix(4) == Data("KLQN".utf8) else { throw Failure.bad("not a KLQN file") }
        func u32(_ o: Int) -> Int { Int(d.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: o, as: UInt32.self)) }) }
        let ver = u32(4), v = u32(8), nb = u32(12), nh = u32(16), nt = u32(20), nv = u32(24)
        guard ver == 1 else { throw Failure.bad("version \(ver)") }
        V = v
        var o = 28
        let vocab = String(decoding: d[(d.startIndex + o)..<(d.startIndex + o + nv)], as: UTF8.self).components(separatedBy: "\n")
        guard vocab.count == v else { throw Failure.bad("vocab \(vocab.count) != \(v)") }
        var m: [String: Int] = [:]
        m.reserveCapacity(v)
        for (i, w) in vocab.enumerated() where m[w] == nil { m[w] = i }
        ids = m
        o += nv + (4 - nv % 4) % 4
        func take(_ bytes: Int) -> Int { let at = o; o += bytes + (4 - bytes % 4) % 4; return at }
        p1Off = take(2 * v); g2Off = take(2 * v)
        boffOff = take(4 * (v + 1)); bentOff = take(4 * nb)
        huoffOff = take(4 * (v + 1)); hentOff = take(4 * nh)
        toffOff = take(4 * (nh + 1)); tentOff = take(4 * nt)
        guard o == d.count else { throw Failure.bad("size \(d.count), layout wants \(o)") }
    }

    @inline(__always) private func u16(_ off: Int, _ i: Int) -> UInt16 {
        data.withUnsafeBytes { UInt16(littleEndian: $0.loadUnaligned(fromByteOffset: off + 2 * i, as: UInt16.self)) }
    }
    @inline(__always) private func u32(_ off: Int, _ i: Int) -> UInt32 {
        data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: off + 4 * i, as: UInt32.self)) }
    }

    /// index of the entry in [lo, hi) of a (word << 15 | q) array whose word is x, or -1
    private func find(_ off: Int, _ lo: Int, _ hi: Int, _ x: UInt32) -> Int {
        var a = lo, b = hi
        while a < b { let mid = (a + b) / 2; if (u32(off, mid) >> 15) < x { a = mid + 1 } else { b = mid } }
        return a < hi && (u32(off, a) >> 15) == x ? a : -1
    }

    @inline(__always) private func q(_ e: UInt32) -> Double { -Double(e & 0x7FFF) / Self.SCALE }

    func wid(_ w: String) -> Int {
        if let f = w.unicodeScalars.first, let t = f.properties.numericType, t == .decimal || t == .digit { return 2 }
        return ids[M3Text.lower(w)] ?? 0
    }

    /// QKN.lp: log P(w | u v)
    func lp(_ u: Int, _ v: Int, _ w: Int) -> Double {
        let h = find(hentOff, Int(u32(huoffOff, u)), Int(u32(huoffOff, u + 1)), UInt32(v))
        var back = 0.0
        if h >= 0 {
            let t = find(tentOff, Int(u32(toffOff, h)), Int(u32(toffOff, h + 1)), UInt32(w))
            if t >= 0 { return q(u32(tentOff, t)) }
            back = q(u32(hentOff, h))
        }
        let bi = find(bentOff, Int(u32(boffOff, v)), Int(u32(boffOff, v + 1)), UInt32(w))
        let b2 = bi >= 0 ? q(u32(bentOff, bi)) : -Double(u16(g2Off, v)) / Self.SCALE + -Double(u16(p1Off, w)) / Self.SCALE
        return back + b2
    }

    /// QKN.sent_lp: history starts at <s> <s>; an out-of-vocabulary word costs LP_OOV
    public func sentLp(_ words: [String]) -> Double {
        let ids = [1, 1] + words.map(wid)
        var s = 0.0
        if ids.count > 2 { for i in 2..<ids.count { s += ids[i] == 0 ? Self.LP_OOV : lp(ids[i - 2], ids[i - 1], ids[i]) } }
        return s
    }
}
