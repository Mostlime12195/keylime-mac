import Foundation

// Text helpers of the m3 dictionary-aware decoder (port of m3dict_cpt/decode.py): the scorer's SEG words, protected
// tokens, case patterns, the scorer's Needleman-Wunsch alignment and Python/numpy rounding and sums. Text is handled
// as arrays of Unicode scalars, so offsets are Python string indices.

public typealias Scalars = [Unicode.Scalar]

public enum M3Text {
    @inline(__always) static func isL(_ u: Unicode.Scalar) -> Bool {
        let v = u.value
        return (v >= 65 && v <= 90) || (v >= 97 && v <= 122)
    }
    @inline(__always) static func isUpper(_ u: Unicode.Scalar) -> Bool { u.value >= 65 && u.value <= 90 }
    @inline(__always) static func isLower(_ u: Unicode.Scalar) -> Bool { u.value >= 97 && u.value <= 122 }
    @inline(__always) static func isDigit(_ u: Unicode.Scalar) -> Bool { u.value >= 48 && u.value <= 57 }

    /// Python str.isspace (what re's \s matches in a str pattern)
    static func isSpace(_ u: Unicode.Scalar) -> Bool {
        switch u.value {
        case 0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000: return true
        default: return false
        }
    }

    public static func scalars(_ s: String) -> Scalars { Array(s.unicodeScalars) }
    public static func string(_ u: Scalars) -> String { var v = String.UnicodeScalarView(); v.append(contentsOf: u); return String(v) }
    public static func string(_ u: ArraySlice<Unicode.Scalar>) -> String { var v = String.UnicodeScalarView(); v.append(contentsOf: u); return String(v) }

    /// ASCII lowercase (Python str.lower on the ASCII words this decoder handles; other scalars kept)
    public static func lower(_ s: String) -> String {
        var v = String.UnicodeScalarView()
        for u in s.unicodeScalars { v.append(isUpper(u) ? Unicode.Scalar(u.value + 32)! : u) }
        return String(v)
    }
    public static func upper(_ s: String) -> String {
        var v = String.UnicodeScalarView()
        for u in s.unicodeScalars { v.append(isLower(u) ? Unicode.Scalar(u.value - 32)! : u) }
        return String(v)
    }

    /// SEG = [A-Za-z]+(?:'[A-Za-z]+)* : (start, end) scalar offsets of every match, left to right
    public static func seg(_ t: Scalars) -> [(Int, Int)] {
        var out: [(Int, Int)] = []
        var i = 0
        let n = t.count
        while i < n {
            if !isL(t[i]) { i += 1; continue }
            let s = i
            while i < n && isL(t[i]) { i += 1 }
            while i + 1 < n && t[i] == "'" && isL(t[i + 1]) {
                i += 1
                while i < n && isL(t[i]) { i += 1 }
            }
            out.append((s, i))
        }
        return out
    }

    public static func segWords(_ s: String) -> [String] {
        let t = scalars(s)
        return seg(t).map { string(t[$0.0..<$0.1]) }
    }

    /// PROT = [/@_#:\\]|\.[A-Za-z]|[A-Za-z][0-9]|[0-9][A-Za-z]|^&|www|http, searched in one whitespace token
    static func protectedToken(_ t: ArraySlice<Unicode.Scalar>) -> Bool {
        let a = Array(t)
        if a.first == "&" { return true }
        for i in 0..<a.count {
            let c = a[i]
            if c == "/" || c == "@" || c == "_" || c == "#" || c == ":" || c == "\\" { return true }
            if i + 1 < a.count {
                let d = a[i + 1]
                if c == "." && isL(d) { return true }
                if isL(c) && isDigit(d) { return true }
                if isDigit(c) && isL(d) { return true }
            }
            if i + 2 < a.count && c == "w" && a[i + 1] == "w" && a[i + 2] == "w" { return true }
            if i + 3 < a.count && c == "h" && a[i + 1] == "t" && a[i + 2] == "t" && a[i + 3] == "p" { return true }
        }
        return false
    }

    /// decode.protected_slots: one flag per SEG word of the text (its whitespace token looks like a URL / handle / code)
    public static func protectedSlots(_ t: Scalars) -> [Bool] {
        var out: [Bool] = []
        var i = 0
        let n = t.count
        while i < n {
            if isSpace(t[i]) { i += 1; continue }
            let s = i
            while i < n && !isSpace(t[i]) { i += 1 }
            let tok = t[s..<i]
            let p = protectedToken(tok)
            for _ in seg(Array(tok)) { out.append(p) }
        }
        return out
    }

    static let I_WORDS: Set<String> = ["i", "i'm", "i'll", "i'd", "i've"]

    /// decode.apply_case: candidate (lowercase) cased like the output word
    public static func applyCase(_ word: String, _ pattern: String) -> String {
        let w = scalars(word)
        let firstPart = word.split(separator: "'", omittingEmptySubsequences: false).first.map(String.init) ?? ""
        if I_WORDS.contains(word) || (firstPart == "i" && word.contains("'")) { return "I" + string(w.dropFirst()) }
        let letters = pattern.unicodeScalars.filter { isL($0) }
        if letters.count > 1 && letters.allSatisfy(isUpper) { return upper(word) }
        if let f = letters.first, isUpper(f) { return upper(string(w.prefix(1))) + string(w.dropFirst()) }
        return word
    }

    /// Python str.isupper on ASCII text: some cased letter, none lowercase
    static func pyIsUpper(_ s: String) -> Bool {
        var cased = false
        for u in s.unicodeScalars { if isLower(u) { return false }; if isUpper(u) { cased = true } }
        return cased
    }

    // MARK: alignment

    static func lev(_ a: Scalars, _ b: Scalars) -> Int {
        if a.isEmpty { return b.count }
        if b.isEmpty { return a.count }
        var prev = Array(0...b.count), cur = [Int](repeating: 0, count: b.count + 1)
        for i in 1...a.count {
            cur[0] = i
            for j in 1...b.count {
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1))
            }
            swap(&prev, &cur)
        }
        return prev[b.count]
    }

    /// rapidfuzz Levenshtein.normalized_distance (unit weights): distance / max length
    static func ned(_ a: Scalars, _ b: Scalars) -> Double {
        let m = max(a.count, b.count)
        return m == 0 ? 0 : Double(lev(a, b)) / Double(m)
    }

    /// decode.align_idx: for each typed word the output word indices assigned to it (merges go to the first typed word)
    public static func alignIdx(_ aWords: [String], _ bWords: [String]) -> [[Int]] {
        let GAP = 0.6, MULTI = 0.45, SUBW = 0.6
        let a = aWords.map { scalars(lower($0)) }, b = bWords.map { scalars(lower($0)) }
        let n = a.count, m = b.count
        let INF = 1e9
        var D = [[Double]](repeating: [Double](repeating: INF, count: m + 1), count: n + 1)
        var B = [[(Int, Int, Int)?]](repeating: [(Int, Int, Int)?](repeating: nil, count: m + 1), count: n + 1)
        // op codes: 0 del, 1 ins, 2 sub, 3 merge, 4 split
        D[0][0] = 0
        for i in 0...n {
            for j in 0...m {
                if i == 0 && j == 0 { continue }
                var best = INF
                var bp: (Int, Int, Int)? = nil
                if i > 0 && D[i - 1][j] + GAP < best { best = D[i - 1][j] + GAP; bp = (0, 1, 0) }
                if j > 0 && D[i][j - 1] + GAP < best { best = D[i][j - 1] + GAP; bp = (1, 0, 1) }
                if i > 0 && j > 0 {
                    let c = D[i - 1][j - 1] + SUBW * ned(a[i - 1], b[j - 1])
                    if c < best { best = c; bp = (2, 1, 1) }
                    for k in 2...3 {
                        if i >= k {
                            let c = D[i - k][j - 1] + SUBW * ned(Array(a[(i - k)..<i].joined()), b[j - 1]) + MULTI * Double(k - 1)
                            if c < best { best = c; bp = (3, k, 1) }
                        }
                        if j >= k {
                            let c = D[i - 1][j - k] + SUBW * ned(a[i - 1], Array(b[(j - k)..<j].joined())) + MULTI * Double(k - 1)
                            if c < best { best = c; bp = (4, 1, k) }
                        }
                    }
                }
                D[i][j] = best; B[i][j] = bp
            }
        }
        var ops: [(Int, Int, Int, Int, Int)] = []
        var i = n, j = m
        while i > 0 || j > 0 {
            let (op, di, dj) = B[i][j]!
            ops.append((op, i - di, i, j - dj, j)); i -= di; j -= dj
        }
        var out = [[Int]](repeating: [], count: n)
        for (op, i1, _, j1, j2) in ops.reversed() where op >= 2 { out[i1] += Array(j1..<j2) }
        return out
    }

    // MARK: numbers

    /// Python round(x, 4): correctly rounded from the exact binary value (printf does exactly that)
    public static func round4(_ x: Double) -> Double {
        if !x.isFinite { return x }
        return Double(String(format: "%.4f", x)) ?? x
    }

    /// numpy float32 sum (pairwise: < 8 sequential, <= 128 eight accumulators, else halves)
    public static func npSum(_ a: ArraySlice<Float>) -> Float {
        let n = a.count, s = a.startIndex
        if n < 8 {
            var r: Float = 0
            for x in a { r += x }
            return r
        }
        if n <= 128 {
            var r = [Float](a[s..<(s + 8)])
            var i = 8
            while i < n - (n % 8) {
                for k in 0..<8 { r[k] += a[s + i + k] }
                i += 8
            }
            var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
            while i < n { res += a[s + i]; i += 1 }
            return res
        }
        var n2 = n / 2
        n2 -= n2 % 8
        return npSum(a[s..<(s + n2)]) + npSum(a[(s + n2)..<(s + n)])
    }
}
