import Foundation

/// The manifest's "models_v2" list: model types the released apps don't know (m3 = "m3dict"). Old apps ignore the
/// key; this app shows each entry it can run, in the place of the models[] entry it "replaces" (m3 takes m1's slot).
public struct V2File: Equatable {
    public let role: String
    public let file: String
    public let size: Int64
    public let sha256: String
}

public struct V2Model {
    public let id: String
    public let type: String
    public let label: String
    public let replaces: String?
    public let files: [V2File]
    public let config: [String: Any]
    public let params: [String: Any]
    public let baseUrls: [String]

    public func file(_ role: String) -> V2File? { files.first { $0.role == role } }
}

public enum ModelListV2 {
    /// model types this app can run
    public static let supported: Set<String> = ["m3dict"]

    /// role of a file from its name when the manifest doesn't say
    public static func role(of name: String) -> String {
        let n = name.lowercased()
        if n.contains("encoder_ctx") { return "encoder_ctx" }
        if n.contains("encoder") { return "encoder" }
        if n.contains("decoder") { return "decoder" }
        if n.contains("vocab") || n.hasSuffix(".json") && n.contains("tok") { return "vocab" }
        if n.contains("ngram") || n.contains("trigram") || n.contains("kn") && n.hasSuffix(".bin") { return "ngram" }
        if n.contains("symspell") { return "symspell" }
        if n.contains("lexicon") || n.hasSuffix(".tsv") { return "lexicon" }
        return n
    }

    static func files(_ x: Any?) -> [V2File] {
        func one(_ role: String?, _ name: String?, _ f: [String: Any]) -> V2File? {
            guard let file = (f["file"] as? String) ?? name, let sha = f["sha256"] as? String else { return nil }
            return V2File(role: (f["role"] as? String) ?? role ?? Self.role(of: file), file: file,
                          size: (f["size"] as? NSNumber)?.int64Value ?? 0, sha256: sha.lowercased())
        }
        if let arr = x as? [[String: Any]] { return arr.compactMap { one(nil, nil, $0) } }
        if let d = x as? [String: Any] {
            return d.keys.sorted().compactMap { k in
                guard let f = d[k] as? [String: Any] else { return nil }
                // {role: {file, sha256}} or {file name: {sha256, size}}
                return f["file"] != nil ? one(k, nil, f) : one(nil, k, f)
            }
        }
        return []
    }

    /// the entries this app (appVersion) can run: a supported type, min_app.mac not newer than the app (dev builds
    /// 0.0.x run everything)
    public static func parse(_ manifest: [String: Any], appVersion: String) -> [V2Model] {
        guard let arr = manifest["models_v2"] as? [[String: Any]] else { return [] }
        var out: [V2Model] = []
        for m in arr {
            guard let id = m["id"] as? String, let type = m["type"] as? String, supported.contains(type) else { continue }
            if let min = (m["min_app"] as? [String: Any])?["mac"] as? String, !appVersion.hasPrefix("0.0."),
               UpdateCore.newer(min, than: appVersion) { continue }
            let fs = files(m["files"])
            guard !fs.isEmpty else { continue }
            out.append(V2Model(id: id, type: type, label: m["label"] as? String ?? id, replaces: m["replaces"] as? String, files: fs,
                               config: m["config"] as? [String: Any] ?? m["m3"] as? [String: Any] ?? [:], params: m["params"] as? [String: Any] ?? [:],
                               baseUrls: ModelMirrors.baseUrls(m)))
        }
        return out
    }

    /// Picker order: models[] ids with each replaced one swapped for its v2 entry; v2 entries that replace nothing (or
    /// an id that isn't there) go last.
    public static func order(_ ids: [String], _ v2: [V2Model]) -> [String] {
        var out: [String] = []
        var used = Set<String>()
        for id in ids {
            if let r = v2.first(where: { $0.replaces == id }) {
                if used.insert(r.id).inserted { out.append(r.id) }
            } else { out.append(id) }
        }
        for r in v2 where used.insert(r.id).inserted { out.append(r.id) }
        return out
    }
}
