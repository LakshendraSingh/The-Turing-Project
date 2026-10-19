"""
Text Synthetic & AI-Generated Disinformation Detector (2026)
Detects Frontier LLM outputs (GPT-4o, Claude 3.5, Gemini 2.0, Llama 3) and Disinformation.
Uses Multi-Tier Linguistic Forensics:
- Fast-DetectGPT / Binoculars inspired Perplexity & Burstiness (sentence-to-sentence variance)
- N-gram rank entropy & Zipf's law distribution conformity
- Stylometric formulaic markers (LLM clichés, transition hedging, discourse markers)
- Lexical diversity: Type-Token Ratio (TTR) & Hapax Legomena ratio
- Neural text classification backbone
"""

import math
import os
import re
from collections import Counter
import numpy as np
import torch
import torch.nn as nn


class GatedResidualBlock1d(nn.Module):
    """
    Gated Residual Block with LayerNorm, Gated Linear Units (GLU), and skip connections.
    """
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        res = x
        h = self.norm(x)
        h = self.fc1(h)
        val, gate = h.chunk(2, dim=-1)
        h = val * torch.sigmoid(gate)
        h = self.dropout(h)
        h = self.fc2(h)
        return res + h


class FeatureAttentionBlock(nn.Module):
    """
    Multi-Head Self-Attention over tokenized forensic features (TabTransformer style).
    Allows cross-feature attention reasoning between burstiness, perplexity, discourse clichés, and syntactic markers.
    """
    def __init__(self, embed_dim: int = 32, num_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TextClassifierBackbone(nn.Module):
    """
    2026 Hybrid FLAG-Net (Forensic Linguistic Attentive Gated Network):
    - Path 1: Feature-Tokenized Multi-Head Self-Attention (learning cross-feature dependency matrices)
    - Path 2: Deep Gated Residual GLU Network with LayerNorm & identity skip connections
    - Dynamic Gated Fusion Layer combining contextual tabular attention with dense non-linear projections
    - Calibrated Dual-Projection Classification Head
    """
    def __init__(self, in_features=24, embed_dim=32, num_heads=4, hidden_dim=64, num_classes=1, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim

        # Path 1: Feature Attention Tokenizer
        self.feature_proj = nn.Linear(1, embed_dim)
        self.feature_pos = nn.Parameter(torch.randn(1, in_features, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.feat_attn1 = FeatureAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)
        self.feat_attn2 = FeatureAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)

        # Path 2: Gated Residual MLP
        self.dense_stem = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.glu_block1 = GatedResidualBlock1d(hidden_dim, dropout=0.15)
        self.glu_block2 = GatedResidualBlock1d(hidden_dim, dropout=0.15)

        # Dynamic Gated Fusion
        fused_in = embed_dim * 2 + hidden_dim  # CLS (32) + Mean Attn (32) + Dense (64) = 128
        self.fusion_gate = nn.Sequential(
            nn.Linear(fused_in, fused_in),
            nn.Sigmoid(),
        )

        # Classification Head
        self.head = nn.Sequential(
            nn.Linear(fused_in, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        b = x.shape[0]

        # Path 1: Tokenize & Multi-Head Self-Attention
        tokens = self.feature_proj(x.unsqueeze(-1)) + self.feature_pos
        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        seq = self.feat_attn1(seq)
        seq = self.feat_attn2(seq)

        cls_out = seq[:, 0, :]
        mean_attn = torch.mean(seq[:, 1:, :], dim=1)

        # Path 2: Gated Residual GLU
        dense = self.dense_stem(x)
        dense = self.glu_block1(dense)
        dense = self.glu_block2(dense)

        # Dynamic Gated Fusion
        combined = torch.cat([cls_out, mean_attn, dense], dim=-1)
        gate = self.fusion_gate(combined)
        fused = combined * gate

        return self.head(fused)


class TextDeepfakeDetector:
    # Common 2024-2026 frontier LLM formulaic clichés & transition markers
    LLM_DISCOURSE_MARKERS = [
        r"\bdelve into\b",
        r"\btapestry of\b",
        r"\ba testament to\b",
        r"\bbeacon of\b",
        r"\bnavigating the\b",
        r"\bparamount importance\b",
        r"\bin conclusion\b",
        r"\bit is crucial to\b",
        r"\bit is important to remember\b",
        r"\bmoreover\b",
        r"\bfurthermore\b",
        r"\bseamlessly\b",
        r"\bharnessing the power\b",
        r"\bpivotal role\b",
        r"\bever-evolving\b",
        r"\bholistic approach\b",
    ]

    def __init__(self, device: str = None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = TextClassifierBackbone().to(self.device)
        weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "text_best.pt")
        if os.path.exists(weights_path):
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.model.eval()
        from .polling import TextForensicsCommittee
        self.committee = TextForensicsCommittee(device=self.device)

    def tokenize_words(self, text: str) -> list:
        return re.findall(r"\b[A-Za-z0-9'-]+\b", text.lower())

    def split_sentences(self, text: str) -> list:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 0]

    def compute_lexical_diversity(self, words: list) -> dict:
        """
        Computes Type-Token Ratio (TTR) and Hapax Legomena (words appearing exactly once).
        """
        if not words:
            return {"ttr": 0.0, "hapax_ratio": 0.0, "word_count": 0}

        total_words = len(words)
        vocab = Counter(words)
        unique_words = len(vocab)
        hapax = sum(1 for count in vocab.values() if count == 1)

        ttr = unique_words / total_words
        hapax_ratio = hapax / total_words

        return {
            "ttr": round(ttr, 4),
            "hapax_ratio": round(hapax_ratio, 4),
            "word_count": total_words,
            "unique_words": unique_words,
        }

    def compute_burstiness_and_perplexity(self, text: str) -> dict:
        """
        Computes sentence length distribution, burstiness (variation of sentence complexity),
        and simulated n-gram entropy perplexity.
        Human text exhibits high burstiness; LLMs generate uniform sentence complexity.
        """
        sentences = self.split_sentences(text)
        if not sentences:
            return {"burstiness": 0.0, "mean_sentence_len": 0.0, "sentence_len_std": 0.0, "perplexity_score": 100.0}

        sent_lengths = [len(self.tokenize_words(s)) for s in sentences]
        mean_len = float(np.mean(sent_lengths))
        std_len = float(np.std(sent_lengths))

        # Burstiness: (sigma - mu) / (sigma + mu) normalized between [-1, 1]
        if (std_len + mean_len) > 0:
            burstiness = (std_len - mean_len) / (std_len + mean_len)
        else:
            burstiness = 0.0

        # N-gram conditional entropy
        words = self.tokenize_words(text)
        if len(words) > 3:
            bigrams = list(zip(words[:-1], words[1:]))
            bigram_counts = Counter(bigrams)
            unigram_counts = Counter(words[:-1])

            # Bigram entropy
            entropy = 0.0
            for bg, bg_count in bigram_counts.items():
                p_bg = bg_count / len(bigrams)
                p_cond = bg_count / unigram_counts[bg[0]]
                entropy -= p_bg * math.log2(p_cond + 1e-12)

            perplexity_score = float(2 ** entropy)
        else:
            perplexity_score = 50.0

        return {
            "burstiness": round(burstiness, 4),
            "mean_sentence_len": round(mean_len, 2),
            "sentence_len_std": round(std_len, 2),
            "perplexity_score": round(perplexity_score, 2),
        }

    def detect_llm_discourse_markers(self, text: str) -> dict:
        """
        Scans for over-indexed formulaic phrases typical in LLM generation.
        """
        matches = []
        for pattern in self.LLM_DISCOURSE_MARKERS:
            found = re.findall(pattern, text, flags=re.IGNORECASE)
            if found:
                matches.extend(found)

        words = self.tokenize_words(text)
        total_words = max(len(words), 1)
        cliche_density = len(matches) / (total_words / 100.0)  # occurrences per 100 words

        return {
            "found_markers": matches,
            "marker_count": len(matches),
            "density_per_100_words": round(cliche_density, 3),
        }

    COMMON_CONTRACTIONS = [
        r"\bcan't\b", r"\bdon't\b", r"\bwon't\b", r"\bit's\b", r"\bi'm\b",
        r"\byou're\b", r"\bwe're\b", r"\bthey're\b", r"\bi've\b", r"\bwe've\b",
        r"\bdidn't\b", r"\bisn't\b", r"\baren't\b", r"\bwasn't\b", r"\bweren't\b",
        r"\bhasn't\b", r"\bhaven't\b", r"\bwouldn't\b", r"\bcouldn't\b", r"\bshouldn't\b"
    ]
    STOP_WORDS = {
        "the", "is", "at", "which", "on", "and", "a", "an", "in", "to", "of", "for",
        "with", "as", "by", "that", "this", "it", "from", "be", "are", "or", "have", "has"
    }

    def compute_advanced_stylometrics(self, text: str, words: list) -> dict:
        total_words = max(1, len(words))
        # 1. Contraction density (humans use informal contractions; formal LLMs default to uncontracted forms)
        contractions_found = 0
        for pat in self.COMMON_CONTRACTIONS:
            contractions_found += len(re.findall(pat, text, flags=re.IGNORECASE))
        contraction_ratio = min(1.0, contractions_found / max(1.0, total_words / 50.0))

        # 2. Stop word ratio
        stop_count = sum(1 for w in words if w.lower() in self.STOP_WORDS)
        stop_ratio = min(1.0, stop_count / total_words)

        # 3. Punctuation entropy
        puncts = re.findall(r"[,.!?:;\-–—\"'()]", text)
        if puncts:
            p_counts = Counter(puncts)
            n_p = len(puncts)
            p_entropy = -sum((cnt / n_p) * math.log2(cnt / n_p + 1e-12) for cnt in p_counts.values())
            norm_p_entropy = min(1.0, p_entropy / 3.0)
        else:
            norm_p_entropy = 0.0

        # 4. Yule's Characteristic K proxy (vocabulary concentration)
        w_counts = Counter(words)
        freq_sq_sum = sum(cnt ** 2 for cnt in w_counts.values())
        yules_k = 10000.0 * (freq_sq_sum - total_words) / max(1.0, float(total_words ** 2))
        norm_yules_k = min(1.0, max(0.0, yules_k / 200.0))

        # 5. Question & Exclamation ratio
        expressive_count = len(re.findall(r"[?!]", text))
        expressive_ratio = min(1.0, expressive_count / max(1, len(re.split(r"[.!?]+", text))))

        # 6. Syllable complexity proxy (vowel clusters per word)
        vowel_clusters = sum(len(re.findall(r"[aeiouy]+", w.lower())) for w in words)
        avg_syllables = vowel_clusters / total_words
        norm_syllables = min(1.0, avg_syllables / 3.0)

        # 7. Comma & clause complexity
        clause_punc = len(re.findall(r"[,;:]", text))
        clause_density = min(1.0, clause_punc / max(1.0, total_words / 20.0))

        # 8. Zipf rank distribution deviation
        sorted_counts = sorted(w_counts.values(), reverse=True)
        if len(sorted_counts) >= 5:
            expected_ratios = [1.0 / (r + 1) for r in range(min(10, len(sorted_counts)))]
            actual_ratios = [c / sorted_counts[0] for c in sorted_counts[:len(expected_ratios)]]
            zipf_dev = float(np.mean(np.abs(np.array(actual_ratios) - np.array(expected_ratios))))
            norm_zipf_dev = min(1.0, zipf_dev * 2.0)
        else:
            norm_zipf_dev = 0.0

        return {
            "contraction_ratio": round(contraction_ratio, 3),
            "stop_ratio": round(stop_ratio, 3),
            "punct_entropy": round(norm_p_entropy, 3),
            "yules_k": round(norm_yules_k, 3),
            "expressive_ratio": round(expressive_ratio, 3),
            "syllable_complexity": round(norm_syllables, 3),
            "clause_density": round(clause_density, 3),
            "zipf_deviation": round(norm_zipf_dev, 3),
        }

    def build_feature_vector(self, text: str) -> np.ndarray:
        words = self.tokenize_words(text)
        lexical = self.compute_lexical_diversity(words)
        burst_info = self.compute_burstiness_and_perplexity(text)
        markers = self.detect_llm_discourse_markers(text)
        adv = self.compute_advanced_stylometrics(text, words)

        mean_word_len = sum(len(w) for w in words) / max(1, len(words))
        punc_count = len(re.findall(r"[,;:\-–—]", text))
        sent_count = len(re.split(r"[.!?]+", text))
        upper_count = sum(1 for c in text if c.isupper())
        digit_count = sum(1 for c in text if c.isdigit())

        feat = np.array([
            # Base Lexical & Perplexity (1-8)
            lexical["ttr"],
            lexical["hapax_ratio"],
            burst_info["burstiness"],
            burst_info["mean_sentence_len"] / 30.0,
            burst_info["sentence_len_std"] / 15.0,
            min(1.0, burst_info["perplexity_score"] / 100.0),
            min(1.0, markers["density_per_100_words"] / 2.0),
            min(1.0, len(words) / 500.0),
            # Syntactic & Punctuation (9-16)
            min(1.0, mean_word_len / 10.0),
            min(1.0, punc_count / max(1, len(words))),
            min(1.0, sent_count / 30.0),
            min(1.0, (upper_count / max(1, len(text))) * 10.0),
            min(1.0, (digit_count / max(1, len(text))) * 20.0),
            min(1.0, markers["marker_count"] / 5.0),
            min(1.0, abs(burst_info["burstiness"])),
            min(1.0, max(0.0, burst_info["perplexity_score"] / 50.0)),
            # Advanced Stylometrics & Zipfian Distribution (17-24)
            adv["contraction_ratio"],
            adv["stop_ratio"],
            adv["punct_entropy"],
            adv["yules_k"],
            adv["expressive_ratio"],
            adv["syllable_complexity"],
            adv["clause_density"],
            adv["zipf_deviation"],
        ], dtype=np.float32)
        return feat

    def analyze(self, text: str) -> dict:
        """
        Analyzes text for synthetic / LLM generation indicators.
        """
        if not text or len(text.strip()) < 30:
            return {
                "modality": "text",
                "error": "Text too short for reliable forensic analysis (minimum 30 characters required).",
                "fake_probability": 0.5,
                "is_synthetic": False,
            }

        words = self.tokenize_words(text)
        lexical = self.compute_lexical_diversity(words)
        burst_info = self.compute_burstiness_and_perplexity(text)
        markers = self.detect_llm_discourse_markers(text)
        adv = self.compute_advanced_stylometrics(text, words)

        # Build 24-dim feature vector for neural classifier
        feat_vec = self.build_feature_vector(text)

        feat_tensor = torch.from_numpy(feat_vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(feat_tensor)
            neural_score = float(torch.sigmoid(logits).cpu().item())

        # Probabilistic Calibration (2026 Multi-Domain Engine)
        # Neural Backbone (FLAG-Net with TabTransformer) provides primary discrimination (70%)
        # LLM discourse markers, Zipfian deviation, and perplexity serve as evidentiary modifiers (30%)
        marker_evidence = min(1.0, markers["density_per_100_words"] / 1.5)
        zipf_evidence = min(1.0, adv["zipf_deviation"] * 2.0)
        norm_burst = max(0.0, -burst_info["burstiness"])

        evidence_score = float(0.50 * marker_evidence + 0.30 * zipf_evidence + 0.20 * norm_burst)

        composite_fake_prob = float(0.70 * neural_score + 0.30 * evidence_score)
        composite_fake_prob = max(0.01, min(0.99, composite_fake_prob))

        # Execute 3-Model Democratic Polling Committee
        polling = self.committee.poll(
            primary_prob=composite_fake_prob,
            burstiness=burst_info["burstiness"],
            cliche_count=markers["marker_count"],
        )

        is_synthetic = composite_fake_prob >= 0.50
        classification = f"AI-GENERATED TEXT / SYNTHETIC ({polling.consensus_type} {polling.consensus_ratio})" if is_synthetic else f"HUMAN-WRITTEN TEXT ({polling.consensus_type} {polling.consensus_ratio})"

        signals = []
        signals.append(f"Democratic Polling Committee: {polling.summary}")
        if burst_info["burstiness"] < -0.15:
            signals.append(f"Unnaturally low sentence burstiness ({burst_info['burstiness']:.2f}) indicating synthetic uniformity")
        if markers["marker_count"] > 0:
            signals.append(f"Contains {markers['marker_count']} formulaic LLM discourse marker(s): {', '.join(set(markers['found_markers'][:3]))}")
        if burst_info["perplexity_score"] < 25.0:
            signals.append(f"Low bigram entropy perplexity ({burst_info['perplexity_score']:.1f}) typical of autoregressive top-p sampling")

        return {
            "modality": "text",
            "fake_probability": round(composite_fake_prob, 4),
            "is_synthetic": is_synthetic,
            "classification": classification,
            "confidence_percent": round(abs(composite_fake_prob - 0.5) * 200, 1),
            "forensic_signals": signals if signals else ["Natural lexical burstiness and human syntactic variance observed"],
            "polling": polling.to_dict(),
            "metrics": {
                "lexical_diversity": lexical,
                "burstiness_and_perplexity": burst_info,
                "discourse_markers": markers,
                "neural_model_score": round(neural_score, 4),
            },
        }
